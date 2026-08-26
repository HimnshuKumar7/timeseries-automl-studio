# Time-Series AutoML Studio

Time-Series AutoML Studio is a browser-based forecasting application that lets users upload a CSV, inspect a series, choose forecasting models, train them locally, compare error metrics, and generate multi-step forecasts.

It was built as a research-oriented internship project to explore practical time-series model comparison in a privacy-conscious, client-side workflow.

## Highlights

- Upload and explore time-series CSV data with summary statistics, correlations, and interactive charts.
- Train classical statistical, machine-learning, and deep-learning forecasting models from one workflow.
- Compare RMSE and MAE across selected models.
- Generate recursive multi-step forecasts.
- Select extra numeric inputs for deep-learning models; their future values are forecast with a lightweight VAR(1) routine.
- Run Pyodide/scikit-learn/statsmodels and TensorFlow.js work inside browser Web Workers, keeping training responsive.
- Store uploaded datasets and trained TensorFlow.js models locally in IndexedDB.
- Authenticate users through a FastAPI + PostgreSQL backend with JWT-backed HTTP-only session cookies.

## Models

| Family | Models |
| --- | --- |
| Classical statistical | ARIMA(1,1,1), VARMAX / VARMA-style multivariate forecasting |
| Machine learning | Linear Regression, HistGradientBoostingRegressor, Random Forest Regressor |
| Deep learning | LSTM, GRU, LSTM-GRU Hybrid, Bidirectional LSTM, Transformer |

### Forecasting design

- Classical models use the target series, or a small multivariate numeric set for VARMAX.
- ML models use three autoregressive target lags plus available numeric columns. Future numeric inputs are forecast with VARMAX when possible.
- Deep models use a 10-step sliding window. By default they are univariate; optional numeric inputs selected in EDA are appended as feature channels.
- Deep forecasts recursively feed each predicted target back into the next input window. Future selected feature values are generated with an in-worker VAR(1) forecast, with a last-observation fallback if fitting fails.

## Architecture

```text
React + Vite UI
  |- EDA, model configuration, training status, results
  |- IndexedDB: raw CSV data and TensorFlow.js model storage
  |- Pyodide Web Worker: pandas, scikit-learn, statsmodels
  `- TensorFlow.js Web Worker: sequence models and recursive forecasts

FastAPI API
  `- PostgreSQL: users and authentication records
```

The frontend proxies `/api` requests to FastAPI during local development, so browser session cookies remain same-origin.

## Quick start

### Prerequisites

- Node.js 18+
- Python 3.10+
- PostgreSQL database (local PostgreSQL, Neon, or another compatible provider)

### 1. Start the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `DATABASE_URL` and `JWT_SECRET` in `backend/.env`, then run:

```powershell
uvicorn app.main:app --reload --port 8000
```

The backend creates its `users` table automatically on startup.

### 2. Start the frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL shown by Vite, usually `http://localhost:5173`.

## Typical workflow

1. Create an account or sign in.
2. Upload a CSV ordered by time.
3. Select the numeric target column in EDA.
4. Optionally select non-target numeric columns as deep-learning input features.
5. Choose models, test split, forecast horizon, and training epochs.
6. Train selected models and compare RMSE/MAE.
7. Review the multi-step forecast in the results view.

## Validation notes

The UI uses a chronological holdout split for ML and deep models; the default is 80% train and 20% test. Classical models report fitted-value errors. For a production-grade experiment, prefer rolling-origin validation and ensure future covariates are genuinely available or separately forecast at each validation cutoff.

## Technology stack

- Frontend: React 19, Vite, Tailwind CSS, Zustand, Plotly
- Browser ML: TensorFlow.js, Pyodide, pandas, scikit-learn, statsmodels
- Backend: FastAPI, asyncpg, PostgreSQL, PyJWT, bcrypt

## Limitations and next steps

- CSV parsing in the current browser training path expects simple comma-delimited numeric fields; complex quoted CSV fields should be handled with a dedicated CSV parser.
- Deep feature scaling currently uses the complete uploaded sequence; fitting scalers on the training partition would make evaluation stricter.
- Unknown future exogenous features are forecast for the final horizon, but test-time evaluation can be extended with rolling-origin feature forecasts.
- Fixed baseline model configurations can be expanded with automated hyperparameter search and experiment tracking.

## Project summary

> Built a browser-based Time-Series AutoML Studio using React, FastAPI, Pyodide, and TensorFlow.js. Implemented local training and comparison of statistical, ML, and deep sequence models, recursive multi-step forecasting, browser-worker execution, interactive EDA, authentication, and multivariate exogenous-feature support.
