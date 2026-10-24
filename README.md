# Industrial Predictive Maintenance Platform

A portfolio-grade starter project for predicting machine failure risk from sensor snapshots.
It includes a reproducible ML training pipeline, FastAPI inference API, Isolation Forest anomaly scoring,
and a React dashboard.

## Important scope note
The UCI AI4I 2020 dataset is **synthetic**, not live factory telemetry. It supports a failure-risk classification
demo, but it does not contain per-machine run-to-failure trajectories suitable for honest Remaining Useful Life
(RUL) prediction. Do not label the current model as an RUL model. Add a separate NASA C-MAPSS RUL model as a
later extension if you want RUL forecasting.

## Stack
- Python 3.11
- pandas, scikit-learn, joblib
- FastAPI + Uvicorn
- React + Vite
- Optional later: PostgreSQL, Docker, MLflow, GitHub Actions

## 1. Environment setup (Windows PowerShell)
Install Python 3.11+ and Node.js LTS. Then:

```powershell
cd industrial-predictive-maintenance
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Download the dataset
Get `ai4i2020.csv` from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Place it at `data/ai4i2020.csv`. Keep the original source and cite the UCI dataset in your README/report.
The dataset is CC BY 4.0; preserve attribution.

## 3. Train the models
From the project root, with `.venv` active:

```powershell
python -m backend.train
```

The script creates:
- `artifacts/failure_model.joblib`
- `artifacts/anomaly_model.joblib`
- `artifacts/metadata.json`
- `artifacts/evaluation.json`
- `artifacts/test_predictions.csv`

Read `artifacts/evaluation.json` and record the actual metrics. Do not make up accuracy numbers.

## 4. Start the API
```powershell
uvicorn backend.app:app --reload
```
Open http://127.0.0.1:8000/docs to try the API.
Health check: http://127.0.0.1:8000/health

## 5. Start the React dashboard
Open a second terminal:
```powershell
cd frontend
npm install
npm run dev
```
Vite prints the local URL, usually http://localhost:5173.

The frontend expects the API at `http://127.0.0.1:8000`. To change it, create `frontend/.env`:
```env
VITE_API_URL=http://127.0.0.1:8000
```

## 6. End-to-end mechanism
1. A user selects or edits sensor readings in the React UI.
2. React sends JSON to `POST /predict`.
3. FastAPI validates the request with Pydantic.
4. The API builds a one-row DataFrame using the model's expected feature names.
5. The classifier returns a failure probability.
6. A configurable threshold converts the probability into a risk label. The default is 0.35 for a demo, not a safety-certified threshold.
7. Isolation Forest returns an anomaly score. An anomaly is not the same as a predicted failure.
8. The API returns probability, risk label, anomaly status, and global model evaluation metadata.
9. The UI renders the result. The current starter does not persist predictions; PostgreSQL persistence is a later milestone.

## 7. How to evaluate properly
The dataset has rare failure cases. Accuracy alone is not enough. Inspect:
- Precision and recall for the failure class
- F1 score
- PR-AUC / average precision
- ROC-AUC
- Confusion matrix
- False negatives, because missed failures matter

The split in this starter is stratified random split because AI4I is a row-level synthetic benchmark. For real temporal sensor streams, use chronological/group-based evaluation to avoid leakage. Do not use `TWF`, `HDF`, `PWF`, `OSF`, or `RNF` as input features: they leak the target.

## 8. Deployment plan
Recommended simple split deployment:
- API: Render or Railway as a Python web service
- Frontend: Vercel
- Set `VITE_API_URL` to the deployed API URL.
- Set API CORS origins to your deployed frontend URL through `CORS_ORIGINS`.
- Commit `artifacts/` only if your model file is small and the repository policy allows it. Alternatively train during deployment from a bundled dataset, but avoid relying on network downloads during the build.

This starter allows local CORS for development. Configure production origins before public release.

## 9. Industry-level extensions
Implement in this order:
1. Prediction history and maintenance work orders in PostgreSQL.
2. Time-series sensor simulator that emits readings every few seconds.
3. Alert rules with deduplication/cooldowns.
4. Model monitoring: feature drift, prediction distribution, and periodic evaluation.
5. MLflow experiment tracking and model versioning.
6. Docker Compose for API + PostgreSQL + frontend.
7. A separate NASA C-MAPSS sequence model for RUL regression, with MAE/RMSE and NASA scoring.

## API example
```json
{
  "machine_type": "M",
  "air_temperature_k": 300.0,
  "process_temperature_k": 310.0,
  "rotational_speed_rpm": 1500,
  "torque_nm": 40.0,
  "tool_wear_min": 120
}
```

Dataset source: UCI Machine Learning Repository, AI4I 2020 Predictive Maintenance Dataset,
https://doi.org/10.24432/C5HS5C
