import React, { useEffect, useState } from 'react';
import { useStore } from '../store/useStore';
import { storage } from '../lib/storage';
import { getPyodideAPI, getTfjsAPI, resetPyodideWorker, resetTfjsWorker } from '../lib/workerHelper';
import { Loader2, CheckCircle2 } from 'lucide-react';
import * as Comlink from 'comlink';

const DEEP_MODEL_DISPLAY_NAMES = {
  lstm: 'LSTM',
  gru: 'GRU',
  lstmGruHybrid: 'LSTM-GRU Hybrid',
  bilstm: 'Bi-LSTM',
  transformer: 'Transformer'
};

const DEEP_MODEL_TRAINERS = {
  lstm: 'trainLSTM',
  gru: 'trainGRU',
  lstmGruHybrid: 'trainLSTMGRUHybrid',
  bilstm: 'trainBiLSTM',
  transformer: 'trainTransformer'
};

function extractDeepLearningData(csvString, targetCol, featureCols = []) {
  const lines = csvString.trim().split('\n');
  if (lines.length < 2) return { target: [], exogData: [] };
  
  const header = lines[0].split(',').map(s => s.trim().replace(/^"|"$/g, ''));
  const selectedColumns = [targetCol, ...featureCols];
  const columnIndexes = selectedColumns.map((column) => header.indexOf(column));
  if (columnIndexes.some((index) => index === -1)) return { target: [], exogData: [] };
  
  const target = [];
  const exogData = [];
  const lastValid = new Array(selectedColumns.length);
  const hasValidValue = new Array(selectedColumns.length).fill(false);

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const cols = line.split(',');
    if (cols.length <= Math.max(...columnIndexes)) continue;

    for (let column = 0; column < columnIndexes.length; column++) {
      const value = Number.parseFloat(cols[columnIndexes[column]].trim().replace(/^"|"$/g, ''));
      if (Number.isFinite(value)) {
        lastValid[column] = value;
        hasValidValue[column] = true;
      }
    }

    // Preserve target/feature row alignment. Leading rows with missing values
    // are skipped; later gaps are forward-filled independently per column.
    if (hasValidValue.every(Boolean)) {
      target.push(lastValid[0]);
      exogData.push(lastValid.slice(1));
    }
  }

  return { target, exogData };
}


export default function StepTrain() {
  const { datasetId, modelsToTrain, targetColumn, deepFeatureColumns, setStep, currentStep, setModelResults, datasetPreview, modelParams } = useStore();
  
  const [trainingStatus, setTrainingStatus] = useState({});
  const [allDone, setAllDone] = useState(false);

  // Stable, primitive-only keys so the effect only re-runs when something that
  // actually changes training actually changes — not on every store update that
  // happens to produce a new object/array reference (e.g. datasetPreview, modelParams).
  const modelsKey = modelsToTrain.join(',');
  const deepFeatureKey = deepFeatureColumns.join(',');
  const { splitFraction, forecastHorizon, epochs, rfEstimators, rfMaxDepth, rfMinSamplesSplit } = modelParams;

  useEffect(() => {
    if (currentStep !== 'TRAIN') return;

    // Guards against a second, overlapping invocation of this effect (e.g. React
    // StrictMode's dev double-invoke, or any dependency tick mid-run) firing a
    // second training run against the same shared singleton worker. Without this,
    // two runs can race and interleave setState calls, corrupting the live UI even
    // though the underlying data (persisted separately) is fine.
    let cancelled = false;

    // Initialize state
    const initStatus = {};
    modelsToTrain.forEach(m => {
      initStatus[m] = { status: 'pending', progress: 'Waiting to start...' };
    });
    setTrainingStatus(initStatus);
    setAllDone(false);

    const runTraining = async () => {
      if (!datasetId || !targetColumn) return;
      const rawData = await storage.getRawDataset(datasetId);
      if (!rawData || cancelled) return;
      
      // Free up Web Worker memory by resetting Pyodide before loading massive ML libraries.
      // Also reset the TF.js worker so each training session starts from a clean
      // WebGL/GPU/backend state — otherwise retraining without a full page reload
      // can silently hang on the first epoch of the next run.
      resetPyodideWorker();
      resetTfjsWorker();

      const finalResults = {};

      // Run sequentially to prevent race conditions on Pyodide globals
      for (const model of modelsToTrain) {
        if (cancelled) return;
        try {
          setTrainingStatus(prev => ({ ...prev, [model]: { status: 'training', progress: 'Training in progress...' } }));
          
          if (model === 'arima' || model === 'varma' || model === 'boosting' || model === 'linear' || model === 'randomForest') {
            const pyodide = getPyodideAPI();
            if (!pyodide) throw new Error("Pyodide worker unavailable");
            
            const result = await pyodide.trainModel(
              model, 
              rawData.rawContent, 
              targetColumn,
              modelParams,
              Comlink.proxy((msg) => {
                if (cancelled) return;
                setTrainingStatus(prev => ({ ...prev, [model]: { status: 'training', progress: msg } }));
              })
            );
            
            if (cancelled) return;

            if (result && result.metrics && result.forecast) {
              finalResults[model] = { metrics: result.metrics, forecast: result.forecast };
            }
            
            setTrainingStatus(prev => ({ 
              ...prev, 
              [model]: { status: 'done', progress: 'Complete', metrics: result.metrics, forecast: result.forecast } 
            }));
          } else if (model in DEEP_MODEL_TRAINERS) {
            const tfjs = getTfjsAPI();
            if (!tfjs) throw new Error("TF.js worker unavailable");
            
            const { target: timeSeriesData, exogData } = extractDeepLearningData(
              rawData.rawContent,
              targetColumn,
              deepFeatureColumns
            );
            if (timeSeriesData.length === 0) throw new Error("Could not extract aligned numeric data for the target and selected deep-learning features.");
            
            const trainFn = tfjs[DEEP_MODEL_TRAINERS[model]];
            const forecastSteps = modelParams.forecastHorizon || 5;
            const deepSplitFraction = modelParams.splitFraction ?? 0.2;
            const totalEpochs = modelParams.epochs || 4;
            const result = await trainFn(
              timeSeriesData,
              totalEpochs,
              8,
              forecastSteps,
              deepSplitFraction,
              Comlink.proxy((epoch, logs) => {
                if (cancelled) return;
                setTrainingStatus(prev => ({
                  ...prev,
                  [model]: { status: 'training', progress: `Epoch ${epoch + 1}/${totalEpochs} (Loss: ${logs.loss.toFixed(4)})` }
                }));
              }),
              exogData
            );
            
            if (cancelled) return;

            if (result && result.metrics && result.forecast) {
              finalResults[model] = { metrics: result.metrics, forecast: result.forecast };
            }

            setTrainingStatus(prev => ({ 
              ...prev, 
              [model]: { status: 'done', progress: `Completed (${totalEpochs} Epochs)`, metrics: result.metrics, forecast: result.forecast } 
            }));
          }
        } catch (e) {
          if (cancelled) return;
          let errorMsg = e.message || 'Unknown error';
          if (errorMsg.includes('ValueError:')) {
            errorMsg = errorMsg.split('ValueError:').pop().trim();
          } else if (errorMsg.includes('Exception:')) {
            errorMsg = errorMsg.split('Exception:').pop().trim();
          } else {
            const lines = errorMsg.split('\\n').filter((l) => l.trim().length > 0);
            errorMsg = lines[lines.length - 1] || errorMsg;
          }
          if (errorMsg.includes('contains no valid numeric data')) {
            errorMsg = `${errorMsg} Please go back to Step 2 (EDA) and select a numerical column like 'Open' or 'Close'.`;
          }
          setTrainingStatus(prev => ({ ...prev, [model]: { status: 'error', progress: errorMsg } }));
        }
      }

      if (cancelled) return;

      setAllDone(true);

      // Store results as soon as training finishes, but stay on this page —
      // only navigate to Results when the user explicitly clicks
      // "View Results & Forecast" below.
      if (Object.keys(finalResults).length > 0) {
        setModelResults(finalResults);
      }
    };

    runTraining();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep, datasetId, modelsKey, deepFeatureKey, targetColumn, splitFraction, forecastHorizon, epochs, rfEstimators, rfMaxDepth, rfMinSamplesSplit]);

  const handleContinue = () => {
    const results = {};
    Object.keys(trainingStatus).forEach(m => {
      const stat = trainingStatus[m];
      if (stat.status === 'done' && stat.metrics && stat.forecast) {
        results[m] = { metrics: stat.metrics, forecast: stat.forecast };
      }
    });
    setModelResults(results);
    setStep('RESULTS');
  };

  return (
    <div className="max-w-4xl mx-auto space-y-8">
      <div className="bg-white dark:bg-gray-900 p-6 rounded-xl shadow-sm border border-gray-100 dark:border-gray-800">
        <h2 className="text-2xl font-bold mb-4 text-gray-800 dark:text-gray-100">Training Models</h2>
        
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6 p-4 rounded-lg bg-blue-50/50 dark:bg-blue-950/30 border border-blue-100 dark:border-blue-900/40">
          <div>
            <h3 className="font-semibold text-gray-800 dark:text-gray-200 text-sm">Full Dataset Volume Trained</h3>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              100% of the uploaded dataset's rows are used ({Math.round((1 - (modelParams.splitFraction ?? 0.2)) * 100)}% Train / {Math.round((modelParams.splitFraction ?? 0.2) * 100)}% Test split).
            </p>
          </div>
          {typeof datasetPreview?.shape?.[0] === 'number' && (() => {
            // Use the true total row count (datasetPreview.shape[0], from the full
            // dataframe) rather than datasetPreview.preview.length, which is capped
            // at 100 rows by the backend/preview generator and understates the
            // actual dataset size for anything larger than that.
            const totalRows = datasetPreview.shape[0];
            const testFraction = modelParams.splitFraction ?? 0.2;
            const trainRows = Math.floor(totalRows * (1 - testFraction));
            const testRows = totalRows - trainRows;
            return (
              <div className="flex flex-wrap items-center gap-2 text-xs font-mono font-medium">
                <span className="px-2.5 py-1 rounded bg-blue-100 dark:bg-blue-900/60 text-blue-800 dark:text-blue-200">
                  Train ({Math.round((1 - testFraction) * 100)}%): {trainRows.toLocaleString()} rows
                </span>
                <span className="px-2.5 py-1 rounded bg-purple-100 dark:bg-purple-900/60 text-purple-800 dark:text-purple-200">
                  Test ({Math.round(testFraction * 100)}%): {testRows.toLocaleString()} rows
                </span>
              </div>
            );
          })()}
        </div>
        
        <div className="space-y-4 mb-8">
          {modelsToTrain.map(model => {
            const stat = trainingStatus[model];
            return (
              <div key={model} className="p-4 rounded-lg border border-gray-200 dark:border-gray-800 bg-gray-50 dark:bg-gray-800/50 flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  {stat?.status === 'pending' && <div className="w-6 h-6 rounded-full bg-gray-200 dark:bg-gray-700" />}
                  {stat?.status === 'training' && <Loader2 className="w-6 h-6 text-blue-500 animate-spin" />}
                  {stat?.status === 'done' && <CheckCircle2 className="w-6 h-6 text-green-500" />}
                  {stat?.status === 'error' && <div className="w-6 h-6 rounded-full bg-red-500" />}
                  
                  <div>
                    <h3 className="font-semibold text-gray-800 dark:text-gray-200">
                      {DEEP_MODEL_DISPLAY_NAMES[model] || (model === 'boosting' ? 'Gradient Boosting' : model === 'randomForest' ? 'Random Forest' : model === 'linear' ? 'Linear Regression' : model === 'varma' ? 'VARMA' : model.toUpperCase())}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-gray-400">{stat?.progress || 'Waiting...'}</p>
                  </div>
                </div>
                
                {stat?.metrics && typeof stat.metrics.rmse === 'number' && (
                  <div className="text-right">
                    <div className="text-xs text-gray-500">RMSE</div>
                    <div className="font-mono font-medium dark:text-white">{stat.metrics.rmse.toFixed(3)}</div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="flex justify-between items-center pt-4 border-t border-gray-100 dark:border-gray-800">
          <button 
            onClick={() => setStep('MODEL')}
            className="px-6 py-2 text-sm font-medium text-gray-600 bg-gray-100 rounded-md hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300 dark:hover:bg-gray-700 transition-colors"
          >
            Back
          </button>
          <button 
            onClick={handleContinue}
            disabled={!allDone}
            className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed shadow-sm transition-colors"
          >
            View Results & Forecast
          </button>
        </div>
      </div>
    </div>
  );
}
