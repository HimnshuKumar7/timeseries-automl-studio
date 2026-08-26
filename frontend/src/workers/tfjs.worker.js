import * as Comlink from 'comlink';
import * as tf from '@tensorflow/tfjs';
// WebGL backend imported for tfjs
import '@tensorflow/tfjs-backend-webgl';

// Enable prod mode to disable debug checks overhead
try { tf.enableProdMode(); } catch { /* Production mode may already be enabled. */ }

let isInitialized = false;
let initPromise = null;

// --- Minimal linear algebra helpers (no new dependencies) used solely to fit
// a VAR(1) model on the exogenous feature history and forecast it forward,
// so the recursive DL forecast loop below can feed the network realistic
// future exogenous values instead of freezing the last observed row. ---

function transpose(A) {
  const rows = A.length, cols = A[0].length;
  const T = Array.from({ length: cols }, () => new Array(rows).fill(0));
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) T[j][i] = A[i][j];
  return T;
}

function matMul(A, B) {
  const n = A.length, m = B.length, p = B[0].length;
  const C = Array.from({ length: n }, () => new Array(p).fill(0));
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < m; k++) {
      const a = A[i][k];
      if (a === 0) continue;
      for (let j = 0; j < p; j++) C[i][j] += a * B[k][j];
    }
  }
  return C;
}

// Gauss-Jordan inversion for small square matrices. Returns null if singular
// (caller falls back to freezing the last observed exogenous row in that case).
function invertMatrix(M) {
  const n = M.length;
  const A = M.map((row, i) => [...row, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxVal = Math.abs(A[col][col]);
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(A[r][col]) > maxVal) { maxVal = Math.abs(A[r][col]); pivotRow = r; }
    }
    if (maxVal < 1e-10) return null; // singular
    [A[col], A[pivotRow]] = [A[pivotRow], A[col]];
    const pivot = A[col][col];
    for (let j = 0; j < 2 * n; j++) A[col][j] /= pivot;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = A[r][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) A[r][j] -= factor * A[col][j];
    }
  }
  return A.map(row => row.slice(n));
}

// Fits a VAR(1) model (Z_t = c + A * Z_{t-1}) on scaled exogenous history via
// OLS (normal equations), then recursively forecasts `steps` future scaled
// exogenous rows. Returns null if there isn't enough history or the
// regression is singular; the caller falls back to freezing the last
// observed exogenous row in that case (previous behavior).
function fitAndForecastExogenousVAR1(exogScaledHistory, steps) {
  const k = exogScaledHistory[0]?.length || 0;
  const n = exogScaledHistory.length;
  if (k === 0) return null;
  if (n < k + 5) return null; // need reasonably more rows than parameters

  const Y = exogScaledHistory.slice(1);
  const Xraw = exogScaledHistory.slice(0, n - 1);
  const Xaug = Xraw.map(row => [1, ...row]);

  const Xt = transpose(Xaug);
  const XtX = matMul(Xt, Xaug);
  const XtXinv = invertMatrix(XtX);
  if (!XtXinv) return null;

  const XtY = matMul(Xt, Y);
  const W = matMul(XtXinv, XtY); // (k+1) x k coefficient matrix

  const forecasts = [];
  let current = exogScaledHistory[n - 1].slice();
  for (let s = 0; s < steps; s++) {
    const xAug = [1, ...current];
    const next = new Array(k).fill(0);
    for (let j = 0; j < k; j++) {
      let val = 0;
      for (let i = 0; i < xAug.length; i++) val += xAug[i] * W[i][j];
      next[j] = val;
    }
    forecasts.push(next);
    current = next;
  }
  return forecasts;
}

function prepareTimeSeriesData(data, exogData = [], timesteps = 10, splitFraction = 0.2) {
  const n = data.length;
  const numExog = (exogData && exogData.length === n && exogData[0]) ? exogData[0].length : 0;
  const C = 1 + numExog; // channel 0 = target, channels 1..C-1 = exogenous features

  // Build the raw multivariate matrix: [target, ...exogenous] per row.
  const rawMatrix = [];
  for (let i = 0; i < n; i++) {
    const row = [data[i]];
    if (numExog > 0) row.push(...exogData[i]);
    rawMatrix.push(row);
  }

  // Independent min-max scaling per channel (same approach as before, just
  // applied per-column instead of to a single series).
  const mins = new Array(C).fill(Infinity);
  const maxs = new Array(C).fill(-Infinity);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < C; c++) {
      const v = rawMatrix[i][c];
      if (v < mins[c]) mins[c] = v;
      if (v > maxs[c]) maxs[c] = v;
    }
  }
  const ranges = maxs.map((max, c) => (max - mins[c]) || 1);

  const scaledMatrix = rawMatrix.map(row => row.map((v, c) => (v - mins[c]) / ranges[c]));

  const X = [];
  const y = [];

  // Build one sliding window per timestep (stride = 1) so training actually uses
  // 100% of the dataset's timeline instead of an arbitrarily downsampled subset.
  // A safety cap still exists purely to protect the browser tab from hanging/OOMing
  // on truly massive datasets (hundreds of thousands of rows); when the dataset is
  // that large, the stride grows just enough to stay under the cap while still
  // spanning the full timeline evenly, rather than silently truncating to a fixed
  // small number of windows regardless of dataset size.
  // A window starting at i predicts the value at i + timesteps. Therefore the
  // final valid start is n - timesteps - 1 (not n - timesteps): the latter has
  // no next target and previously appended an undefined label to y.
  const totalPossibleWindows = Math.max(0, scaledMatrix.length - timesteps);
  if (totalPossibleWindows < 2) {
    throw new Error(`At least ${timesteps + 2} valid observations are required for a ${timesteps}-step input window and train/test evaluation.`);
  }
  const MAX_WINDOWS_SAFETY_CAP = 20000;
  const stride = totalPossibleWindows > MAX_WINDOWS_SAFETY_CAP
    ? Math.ceil(totalPossibleWindows / MAX_WINDOWS_SAFETY_CAP)
    : 1;

  for (let i = 0; i < scaledMatrix.length - timesteps; i += stride) {
    const window = scaledMatrix.slice(i, i + timesteps); // [timesteps][C]
    X.push(window);
    y.push([scaledMatrix[i + timesteps][0]]); // predict target channel only
  }

  // Respect the user-configured train/test split from Model Config (same as the
  // classical/pyodide models), instead of a hardcoded 80/20 split.
  const testFraction = (typeof splitFraction === 'number' && splitFraction > 0 && splitFraction < 1)
    ? splitFraction
    : 0.2;
  const splitIdx = Math.min(X.length - 1, Math.max(1, Math.floor(X.length * (1 - testFraction))));

  return {
    X_train: X.slice(0, splitIdx),
    y_train: y.slice(0, splitIdx),
    X_test: X.slice(splitIdx),
    y_test: y.slice(splitIdx),
    mins,
    ranges,
    scaledMatrix,
    timesteps,
    C,
    numExog
  };
}

async function evaluateAndForecast(model, X_test, y_test, scaledMatrix, mins, ranges, timesteps, C, numExog, forecastSteps = 2) {
  // 1. Evaluate on test set (unchanged logic; target is always channel 0)
  let rmse = 0;
  let mae = 0;

  if (X_test.length > 0) {
    const xs = tf.tensor3d(X_test);
    const preds = model.predict(xs);
    const predsArray = await preds.array();

    let sumSqErr = 0;
    let sumAbsErr = 0;
    let evaluatedCount = 0;

    for (let i = 0; i < predsArray.length; i++) {
      // Keep evaluation resilient to malformed caller-supplied labels. Normal
      // windows always have both values after the corrected bounds above.
      const scaledPred = predsArray[i]?.[0];
      const scaledTrue = y_test[i]?.[0];
      if (!Number.isFinite(scaledPred) || !Number.isFinite(scaledTrue)) continue;

      const predUnscaled = (scaledPred * ranges[0]) + mins[0];
      const trueUnscaled = (scaledTrue * ranges[0]) + mins[0];

      const err = trueUnscaled - predUnscaled;
      sumSqErr += err * err;
      sumAbsErr += Math.abs(err);
      evaluatedCount++;
    }

    if (evaluatedCount > 0) {
      rmse = Math.sqrt(sumSqErr / evaluatedCount);
      mae = sumAbsErr / evaluatedCount;
    }

    xs.dispose();
    preds.dispose();
  }

  // 2. Forecast future exogenous channels (scaled) via VAR(1) so the recursive
  // loop below feeds the model realistic future exogenous inputs instead of
  // freezing the last observed row.
  let exogForecast = null;
  if (numExog > 0) {
    const exogHistoryScaled = scaledMatrix.map(row => row.slice(1)); // drop target channel
    exogForecast = fitAndForecastExogenousVAR1(exogHistoryScaled, forecastSteps);
    // null => not enough history / singular fit; falls back to freezing below.
  }

  // 3. Iterative forecast for forecastSteps
  const forecast = [];
  const currentWindow = scaledMatrix.slice(-timesteps).map(row => row.slice());
  const lastExogRow = scaledMatrix[scaledMatrix.length - 1].slice(1);

  for (let i = 0; i < forecastSteps; i++) {
    const xs = tf.tensor3d([currentWindow]);
    const pred = model.predict(xs);
    const predArray = await pred.array();
    const scaledPred = predArray[0][0];

    forecast.push((scaledPred * ranges[0]) + mins[0]);

    const nextExog = exogForecast ? exogForecast[i] : lastExogRow;
    const nextRow = [scaledPred, ...nextExog];

    // Slide window
    currentWindow.push(nextRow);
    currentWindow.shift();

    xs.dispose();
    pred.dispose();
  }

  return { metrics: { rmse, mae }, forecast };
}


/**
 * Fits a freshly-built model, automatically retrying from scratch with a more
 * conservative learning rate if training diverges (NaN/Infinite loss) partway
 * through. This protects against gradient explosion on small or highly
 * volatile datasets, regardless of which backend (CPU/WebGL) ends up being
 * used, instead of letting a diverged model silently produce NaN metrics.
 */
async function fitWithDivergenceGuard(buildModelFn, X_train, y_train, epochs, batchSize, onEpoch) {
  const learningRates = [0.001, 0.0003, 0.0001];

  for (let attempt = 0; attempt < learningRates.length; attempt++) {
    const lr = learningRates[attempt];
    const model = buildModelFn(lr);

    const xs = tf.tensor3d(X_train);
    const ys = tf.tensor2d(y_train);

    let diverged = false;

    await model.fit(xs, ys, {
      epochs,
      batchSize,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if (logs && (Number.isNaN(logs.loss) || !Number.isFinite(logs.loss))) {
            diverged = true;
            model.stopTraining = true;
          }
          if (onEpoch) onEpoch(epoch, logs);
          await new Promise(r => setTimeout(r, 0));
        }
      }
    });

    xs.dispose();
    ys.dispose();

    const isLastAttempt = attempt === learningRates.length - 1;
    if (!diverged || isLastAttempt) {
      return model;
    }

    // Diverged partway through: discard this model and retry from scratch
    // with a lower learning rate rather than surfacing NaN metrics.
    model.dispose();
  }
}

const tfjsAPI = {
  async init() {
    if (isInitialized) return true;

    if (!initPromise) {
      initPromise = (async () => {
        try {
          // Explicitly set CPU backend in Web Worker context to prevent WebGL promise hanging
          await tf.setBackend('cpu');
          await tf.ready();
          isInitialized = true;
          return true;
        } catch (e) {
          console.warn("TFJS CPU backend init fallback:", e);
          isInitialized = true;
          return true;
        }
      })();
    }

    try {
      await initPromise;
      return true;
    } catch (e) {
      initPromise = null;
      throw e;
    }
  },

  // `exogData` (optional, appended last to preserve existing call sites): a
  // 2D array [nRows][numExogenousFeatures] aligned row-for-row with `data`.
  // Omit or pass null/[] for the original univariate (target-only) behavior.
  async trainLSTM(data, epochs = 4, units = 8, forecastSteps = 2, splitFraction = 0.2, progressCallback, exogData = null) {
    await this.init();

    if (typeof forecastSteps === 'function') {
      progressCallback = forecastSteps;
      forecastSteps = 2;
      splitFraction = 0.2;
    }
    if (typeof splitFraction === 'function') {
      progressCallback = splitFraction;
      splitFraction = 0.2;
    }

    const { X_train, y_train, X_test, y_test, mins, ranges, scaledMatrix, timesteps, C, numExog } =
      prepareTimeSeriesData(data, exogData || [], 10, splitFraction);

    const batchSize = Math.max(1, Math.min(32, X_train.length));
    const model = await fitWithDivergenceGuard(
      (lr) => {
        const m = tf.sequential();
        m.add(tf.layers.lstm({ units: units, inputShape: [timesteps, C], returnSequences: false }));
        m.add(tf.layers.dense({ units: 1 }));
        m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });
        return m;
      },
      X_train, y_train, epochs, batchSize,
      (epoch, logs) => { if (progressCallback) progressCallback(epoch, logs); }
    );

    const { metrics, forecast } = await evaluateAndForecast(model, X_test, y_test, scaledMatrix, mins, ranges, timesteps, C, numExog, forecastSteps);

    // Save model to indexedDB
    const modelName = 'lstm-model-' + Date.now();
    await model.save('indexeddb://' + modelName);

    return {
      modelName: modelName,
      metrics,
      forecast
    };
  },

  async predictLSTM(modelName, X_test) {
    await this.init();
    const model = await tf.loadLayersModel('indexeddb://' + modelName);
    const xs = tf.tensor3d(X_test);
    const preds = model.predict(xs);
    const predictions = await preds.array();

    // Cleanup
    xs.dispose();
    preds.dispose();

    return predictions;
  },

  async trainGRU(data, epochs = 4, units = 8, forecastSteps = 2, splitFraction = 0.2, progressCallback, exogData = null) {
    await this.init();

    if (typeof forecastSteps === 'function') {
      progressCallback = forecastSteps;
      forecastSteps = 2;
      splitFraction = 0.2;
    }
    if (typeof splitFraction === 'function') {
      progressCallback = splitFraction;
      splitFraction = 0.2;
    }

    const { X_train, y_train, X_test, y_test, mins, ranges, scaledMatrix, timesteps, C, numExog } =
      prepareTimeSeriesData(data, exogData || [], 10, splitFraction);

    const batchSize = Math.max(1, Math.min(32, X_train.length));
    const model = await fitWithDivergenceGuard(
      (lr) => {
        const m = tf.sequential();
        m.add(tf.layers.gru({ units: units, inputShape: [timesteps, C], returnSequences: false }));
        m.add(tf.layers.dense({ units: 1 }));
        m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });
        return m;
      },
      X_train, y_train, epochs, batchSize,
      (epoch, logs) => { if (progressCallback) progressCallback(epoch, logs); }
    );

    const { metrics, forecast } = await evaluateAndForecast(model, X_test, y_test, scaledMatrix, mins, ranges, timesteps, C, numExog, forecastSteps);

    const modelName = 'gru-model-' + Date.now();
    await model.save('indexeddb://' + modelName);

    return { modelName, metrics, forecast };
  },

  async predictGRU(modelName, X_test) {
    await this.init();
    const model = await tf.loadLayersModel('indexeddb://' + modelName);
    const xs = tf.tensor3d(X_test);
    const preds = model.predict(xs);
    const predictions = await preds.array();
    xs.dispose();
    preds.dispose();
    return predictions;
  },

  async trainLSTMGRUHybrid(data, epochs = 4, units = 8, forecastSteps = 2, splitFraction = 0.2, progressCallback, exogData = null) {
    await this.init();

    if (typeof forecastSteps === 'function') {
      progressCallback = forecastSteps;
      forecastSteps = 2;
      splitFraction = 0.2;
    }
    if (typeof splitFraction === 'function') {
      progressCallback = splitFraction;
      splitFraction = 0.2;
    }

    const { X_train, y_train, X_test, y_test, mins, ranges, scaledMatrix, timesteps, C, numExog } =
      prepareTimeSeriesData(data, exogData || [], 10, splitFraction);

    const batchSize = Math.max(1, Math.min(32, X_train.length));
    const model = await fitWithDivergenceGuard(
      (lr) => {
        const m = tf.sequential();
        // LSTM layer extracts long-range temporal patterns, passing a full sequence to GRU
        m.add(tf.layers.lstm({ units: units, inputShape: [timesteps, C], returnSequences: true }));
        // GRU layer compresses that sequence into a final representation
        m.add(tf.layers.gru({ units: Math.max(4, Math.round(units / 2)), returnSequences: false }));
        m.add(tf.layers.dense({ units: 1 }));
        m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });
        return m;
      },
      X_train, y_train, epochs, batchSize,
      (epoch, logs) => { if (progressCallback) progressCallback(epoch, logs); }
    );

    const { metrics, forecast } = await evaluateAndForecast(model, X_test, y_test, scaledMatrix, mins, ranges, timesteps, C, numExog, forecastSteps);

    const modelName = 'lstm-gru-hybrid-model-' + Date.now();
    await model.save('indexeddb://' + modelName);

    return { modelName, metrics, forecast };
  },

  async predictLSTMGRUHybrid(modelName, X_test) {
    await this.init();
    const model = await tf.loadLayersModel('indexeddb://' + modelName);
    const xs = tf.tensor3d(X_test);
    const preds = model.predict(xs);
    const predictions = await preds.array();
    xs.dispose();
    preds.dispose();
    return predictions;
  },

  async trainBiLSTM(data, epochs = 4, units = 8, forecastSteps = 2, splitFraction = 0.2, progressCallback, exogData = null) {
    await this.init();

    if (typeof forecastSteps === 'function') {
      progressCallback = forecastSteps;
      forecastSteps = 2;
      splitFraction = 0.2;
    }
    if (typeof splitFraction === 'function') {
      progressCallback = splitFraction;
      splitFraction = 0.2;
    }

    const { X_train, y_train, X_test, y_test, mins, ranges, scaledMatrix, timesteps, C, numExog } =
      prepareTimeSeriesData(data, exogData || [], 10, splitFraction);

    const batchSize = Math.max(1, Math.min(32, X_train.length));
    const model = await fitWithDivergenceGuard(
      (lr) => {
        const m = tf.sequential();
        m.add(tf.layers.bidirectional({
          layer: tf.layers.lstm({ units: units, returnSequences: false }),
          inputShape: [timesteps, C],
          mergeMode: 'concat'
        }));
        m.add(tf.layers.dense({ units: 1 }));
        m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });
        return m;
      },
      X_train, y_train, epochs, batchSize,
      (epoch, logs) => { if (progressCallback) progressCallback(epoch, logs); }
    );

    const { metrics, forecast } = await evaluateAndForecast(model, X_test, y_test, scaledMatrix, mins, ranges, timesteps, C, numExog, forecastSteps);

    const modelName = 'bilstm-model-' + Date.now();
    await model.save('indexeddb://' + modelName);

    return { modelName, metrics, forecast };
  },

  async predictBiLSTM(modelName, X_test) {
    await this.init();
    const model = await tf.loadLayersModel('indexeddb://' + modelName);
    const xs = tf.tensor3d(X_test);
    const preds = model.predict(xs);
    const predictions = await preds.array();
    xs.dispose();
    preds.dispose();
    return predictions;
  },

  async trainTransformer(data, epochs = 4, headDim = 8, forecastSteps = 2, splitFraction = 0.2, progressCallback, exogData = null) {
    await this.init();

    if (typeof forecastSteps === 'function') {
      progressCallback = forecastSteps;
      forecastSteps = 2;
      splitFraction = 0.2;
    }
    if (typeof splitFraction === 'function') {
      progressCallback = splitFraction;
      splitFraction = 0.2;
    }

    const { X_train, y_train, X_test, y_test, mins, ranges, scaledMatrix, timesteps, C, numExog } =
      prepareTimeSeriesData(data, exogData || [], 10, splitFraction);

    const features = C;
    const ffDim = headDim * 2;

    const buildTransformer = (lr) => {
      const input = tf.input({ shape: [timesteps, features] });
      const query = tf.layers.dense({ units: headDim }).apply(input);
      const key = tf.layers.dense({ units: headDim }).apply(input);
      const value = tf.layers.dense({ units: headDim }).apply(input);

      // Custom dot-product self-attention since tf.layers.attention is not available in tfjs
      const scores = tf.layers.dot({ axes: [2, 2] }).apply([query, key]);
      const attnWeights = tf.layers.activation({ activation: 'softmax' }).apply(scores);
      const attnOutput = tf.layers.dot({ axes: [2, 1] }).apply([attnWeights, value]);

      const attnProjected = tf.layers.dense({ units: features }).apply(attnOutput);

      const add1 = tf.layers.add().apply([input, attnProjected]);
      const norm1 = tf.layers.layerNormalization().apply(add1);

      const ff1 = tf.layers.dense({ units: ffDim, activation: 'relu' }).apply(norm1);
      const ff2 = tf.layers.dense({ units: features }).apply(ff1);
      const add2 = tf.layers.add().apply([norm1, ff2]);
      const norm2 = tf.layers.layerNormalization().apply(add2);

      const pooled = tf.layers.globalAveragePooling1d().apply(norm2);
      const output = tf.layers.dense({ units: 1 }).apply(pooled);

      const m = tf.model({ inputs: input, outputs: output });
      m.compile({ optimizer: tf.train.adam(lr), loss: 'meanSquaredError' });
      return m;
    };

    const batchSize = Math.max(1, Math.min(32, X_train.length));
    const model = await fitWithDivergenceGuard(
      buildTransformer,
      X_train, y_train, epochs, batchSize,
      (epoch, logs) => { if (progressCallback) progressCallback(epoch, logs); }
    );

    const { metrics, forecast } = await evaluateAndForecast(model, X_test, y_test, scaledMatrix, mins, ranges, timesteps, C, numExog, forecastSteps);

    const modelName = 'transformer-model-' + Date.now();
    await model.save('indexeddb://' + modelName);

    return { modelName, metrics, forecast };
  },

  async predictTransformer(modelName, X_test) {
    await this.init();
    const model = await tf.loadLayersModel('indexeddb://' + modelName);
    const xs = tf.tensor3d(X_test);
    const preds = model.predict(xs);
    const predictions = await preds.array();
    xs.dispose();
    preds.dispose();
    return predictions;
  }
};

Comlink.expose(tfjsAPI);
