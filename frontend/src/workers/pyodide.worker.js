import * as Comlink from 'comlink';

let pyodideInstance = null;
let initPromise = null;

const pyodideAPI = {
    async init(progressCallback) {
        if (pyodideInstance) return true;

        if (!initPromise) {
            initPromise = (async () => {
                const CDNS = [
                    "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
                    "https://fastly.jsdelivr.net/pyodide/v0.26.4/full/",
                    "https://gcore.jsdelivr.net/pyodide/v0.26.4/full/",
                    "https://unpkg.com/pyodide@0.26.4/"
                ];

                let lastError = null;
                for (let i = 0; i < CDNS.length; i++) {
                    const cdn = CDNS[i];
                    try {
                        console.log(`[Pyodide Worker] Attempting initialization from ${cdn}...`);
                        if (progressCallback) await progressCallback(`Loading Pyodide engine (source ${i + 1}/${CDNS.length})...`);

                        const pyodideModule = await import(/* @vite-ignore */ `${cdn}pyodide.mjs`);

                        if (progressCallback) await progressCallback("Initializing Pyodide WASM runtime...");
                        console.log("[Pyodide Worker] Loading Pyodide WASM from " + cdn);
                        const instance = await pyodideModule.loadPyodide({
                            indexURL: cdn
                        });

                        if (progressCallback) await progressCallback("Loading Python libraries (pandas)...");
                        console.log("[Pyodide Worker] Loading Python packages...");
                        await instance.loadPackage(['pandas']);

                        pyodideInstance = instance;
                        console.log("[Pyodide Worker] Pyodide initialized successfully from " + cdn);
                        if (progressCallback) await progressCallback("Pyodide ready.");
                        return true;
                    } catch (err) {
                        console.warn(`[Pyodide Worker] Failed to load Pyodide from ${cdn}:`, err);
                        lastError = err;
                    }
                }

                initPromise = null;
                const errorDetail = lastError?.message || String(lastError);
                throw new Error(
                    `Failed to load Pyodide engine from CDNs (${errorDetail}). ` +
                    `If you have an ad-blocker (e.g. Brave Shields, uBlock Origin) or firewall enabled, please whitelist or disable it and try again.`
                );
            })();
        }

        return await initPromise;
    },

    async runPython(code) {
        if (!pyodideInstance) await this.init();
        const result = await pyodideInstance.runPythonAsync(code);
        return result;
    },

    async getPreview(csvContent) {
        if (!pyodideInstance) await this.init();

        // Set the variable in Python global scope
        pyodideInstance.globals.set("csv_content", csvContent);

        const code = `
import pandas as pd
import io
import json

def generate_preview():
    try:
        df = pd.read_csv(io.StringIO(csv_content))
        
        preview_df = df.head(100)
        
        preview_json = preview_df.to_json(orient='records')
        
        # Only select numeric columns for correlation and stats
        numeric_df = df.select_dtypes(include=['number'])
        
        corr_matrix = {}
        summary_stats = {}
        
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr().to_dict()
            summary_stats = numeric_df.describe().to_dict()
        
        result = {
            "columns": list(df.columns),
            "shape": list(df.shape),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "preview": json.loads(preview_json),
            "correlation_matrix": corr_matrix,
            "summary_statistics": summary_stats
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

generate_preview()
`;
        const resultJson = await pyodideInstance.runPythonAsync(code);
        return JSON.parse(resultJson);
    },

    async trainModel(modelType, csvContent, targetCol, modelParams = {}, progressCallback) {
        if (!pyodideInstance) {
            if (progressCallback) progressCallback("Loading Pyodide runtime...");
            await new Promise(resolve => setTimeout(resolve, 100));
            await this.init(progressCallback);
        }

        // Lazy load massive ML libraries only when training, to keep uploads snappy!
        if (modelType === 'arima') {
            if (progressCallback) progressCallback("Downloading statsmodels library (10-20MB)...");
            await new Promise(resolve => setTimeout(resolve, 100));
            await pyodideInstance.loadPackage(['statsmodels']);
        } else if (modelType === 'boosting' || modelType === 'linear' || modelType === 'randomForest') {
            if (progressCallback) progressCallback("Downloading scikit-learn library (30-50MB)...");
            await new Promise(resolve => setTimeout(resolve, 100));
            await pyodideInstance.loadPackage(['scikit-learn']);
        }

        if (progressCallback) progressCallback("Fitting model...");
        await new Promise(resolve => setTimeout(resolve, 100));

        const horizon = (typeof modelParams.forecastHorizon === 'number' && modelParams.forecastHorizon > 0)
            ? Math.floor(modelParams.forecastHorizon)
            : (parseInt(modelParams.forecastHorizon, 10) || 5);

        pyodideInstance.globals.set("csv_content", csvContent);
        pyodideInstance.globals.set("target_col", targetCol);
        pyodideInstance.globals.set("model_type", modelType);
        pyodideInstance.globals.set("forecast_horizon", horizon);
        pyodideInstance.globals.set("split_fraction", modelParams.splitFraction || 0.2);
        pyodideInstance.globals.set("rf_estimators", modelParams.rfEstimators || 100);
        pyodideInstance.globals.set("rf_max_depth", modelParams.rfMaxDepth || 0);
        pyodideInstance.globals.set("rf_min_samples_split", modelParams.rfMinSamplesSplit || 2);
        pyodideInstance.globals.set("arima_p", parseInt(modelParams.arima_p, 10) || 10);
        pyodideInstance.globals.set("arima_d", parseInt(modelParams.arima_d, 10) || 1);
        pyodideInstance.globals.set("arima_q", parseInt(modelParams.arima_q, 10) || 5);

        const code = `
import pandas as pd
import io
import json
import numpy as np

def run_training():
    df = pd.read_csv(io.StringIO(csv_content))
    
    # Enforce target column is numeric
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # basic preprocessing: ffill
    df = df.ffill().bfill()
    
    # Drop any remaining NaNs
    df = df.dropna(subset=[target_col])
    if len(df) == 0:
        raise ValueError(f"Target column '{target_col}' contains no valid numeric data to forecast.")
        
    # LIMIT DATASET SIZE FOR IN-BROWSER PERFORMANCE (1500 rows max)
    #if len(df) > 1500:
    #    df = df.tail(1500).reset_index(drop=True)
        
    y = df[target_col].values
    
    metrics = {}
    forecast = []
    
    forecast_steps = int(forecast_horizon) if 'forecast_horizon' in globals() else 5
    
    if model_type == 'arima':
        from statsmodels.tsa.arima.model import ARIMA
        # Univariate ARIMA model order (p, d, q) = (10, 1, 5)
        # p=10 (autoregressive order), d=1 (differencing degree), q=5 (moving-average order)
        p = int(arima_p) if 'arima_p' in globals() else 10
        d = int(arima_d) if 'arima_d' in globals() else 1
        q = int(arima_q) if 'arima_q' in globals() else 5
        model = ARIMA(y, order=(p, d, q))
        fitted = model.fit(method_kwargs={'method': 'nm', 'maxiter': 50})
        preds = fitted.fittedvalues
        forecast = fitted.forecast(steps=forecast_steps).tolist()
        mse = np.mean((y - preds)**2)
        metrics = {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - preds)))}
    elif model_type == 'varma':
        from statsmodels.tsa.statespace.varmax import VARMAX
        from statsmodels.tsa.arima.model import ARIMA
        
        num_df = df.select_dtypes(include=['number']).copy()
        if target_col in num_df.columns:
            cols = [target_col] + [c for c in num_df.columns if c != target_col][:3]
            endog = num_df[cols]
        else:
            endog = pd.DataFrame({target_col: y})

        if endog.shape[1] >= 2:
            try:
                # Multivariate VARMA(1, 1) model
                model = VARMAX(endog, order=(1, 1), trend='c')
                fitted = model.fit(disp=False, maxiter=40)
                preds_df = fitted.fittedvalues
                preds = preds_df[target_col].values if target_col in preds_df.columns else preds_df.iloc[:, 0].values
                
                raw_fc = fitted.forecast(steps=forecast_steps)
                if hasattr(raw_fc, target_col):
                    forecast = raw_fc[target_col].values.tolist()
                elif hasattr(raw_fc, 'iloc'):
                    forecast = raw_fc.iloc[:, 0].values.tolist()
                else:
                    forecast = list(raw_fc)

                mse = np.mean((y - preds)**2)
                metrics = {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - preds)))}
            except Exception:
                # Fallback to ARMA(1, 0, 1) if solver encounters matrix singularity
                model = ARIMA(y, order=(1, 0, 1))
                fitted = model.fit(method_kwargs={'method': 'nm', 'maxiter': 30})
                preds = fitted.fittedvalues
                forecast = fitted.forecast(steps=forecast_steps).tolist()
                mse = np.mean((y - preds)**2)
                metrics = {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - preds)))}
        else:
            # Single-variable dataset: Univariate VARMA(1,1) is ARMA(1, 0, 1)
            model = ARIMA(y, order=(1, 0, 1))
            fitted = model.fit(method_kwargs={'method': 'nm', 'maxiter': 40})
            preds = fitted.fittedvalues
            forecast = fitted.forecast(steps=forecast_steps).tolist()
            mse = np.mean((y - preds)**2)
            metrics = {"rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(y - preds)))}
    elif model_type in ['linear', 'randomForest', 'boosting']:
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        num_lags = 5
        for i in range(1, num_lags + 1):
            df[f'lag{i}'] = df[target_col].shift(i)
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("Not enough data to create lag features.")
            
        X = df.drop(columns=[target_col])
        X = X.select_dtypes(include=['number'])
        y = df[target_col]
        
        split_idx = int(len(df) * (1 - split_fraction))
        if split_idx <= 0 or split_idx >= len(df):
            split_idx = len(df) - 1 # Fallback to 1 element test set if too small
            
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        if model_type == 'linear':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
        elif model_type == 'boosting':
            from sklearn.ensemble import HistGradientBoostingRegressor
            model = HistGradientBoostingRegressor(random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=int(rf_estimators),
                max_depth=int(rf_max_depth) if rf_max_depth > 0 else None,
                min_samples_split=int(rf_min_samples_split),
                random_state=42
            )
            
        model.fit(X_train, y_train)
        
        if len(y_test) > 0:
            preds = model.predict(X_test)
            MAE = mean_absolute_error(y_test, preds)
            MSE = mean_squared_error(y_test, preds)
            metrics = {"rmse": float(np.sqrt(MSE)), "mae": float(MAE)}
        else:
            metrics = {"rmse": 0.0, "mae": 0.0}

        # --- Forecast future exogenous feature values via a joint VARMA model ---
        # Instead of freezing exogenous columns (temp/humidity/etc.) at their
        # last observed value for every future step, forecast them forward
        # so the recursive AR loop below feeds the trained model realistic
        # future inputs rather than stale ones.
        lag_cols = {f'lag{i}' for i in range(1, num_lags + 1)}
        exog_cols = [c for c in X.columns if c not in lag_cols]
        exog_forecast_df = None
        if len(exog_cols) > 0 and len(df) > 10:
            try:
                from statsmodels.tsa.statespace.varmax import VARMAX
                exog_hist = df[exog_cols].astype(float)
                exog_model = VARMAX(exog_hist, order=(1, 0), trend='c')
                exog_fitted = exog_model.fit(disp=False, maxiter=40)
                raw_exog_fc = exog_fitted.forecast(steps=forecast_steps)
                exog_forecast_df = pd.DataFrame(
                    np.asarray(raw_exog_fc), columns=exog_cols
                ).reset_index(drop=True)
            except Exception:
                # If VARMA fails (e.g. too little data, singular matrix),
                # fall back to the previous behavior (last observed values).
                exog_forecast_df = None

        # Forecast steps iteratively using auto-regressive lags
        last_row = df.iloc[-1].copy()
        current_y = last_row[target_col]
        current_lags = [last_row[f'lag{i}'] for i in range(1, num_lags)]
        
        forecast = []
        for step in range(forecast_steps):
            next_x = last_row.copy()
            if exog_forecast_df is not None:
                for col in exog_cols:
                    next_x[col] = exog_forecast_df.iloc[step][col]
            next_x['lag1'] = current_y
            for i in range(1, num_lags):
                next_x[f'lag{i+1}'] = current_lags[i - 1]
            next_x_df = pd.DataFrame([next_x])[X.columns]
            
            pred = model.predict(next_x_df)[0]
            forecast.append(float(pred))
            
            current_lags = [current_y] + current_lags[:-1]
            current_y = pred
    
    return json.dumps({
        "metrics": metrics,
        "forecast": forecast
    })

run_training()
`;
        const resultJson = await pyodideInstance.runPythonAsync(code);
        return JSON.parse(resultJson);
    }
};

Comlink.expose(pyodideAPI);