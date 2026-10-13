# AI-Powered Predictive Maintenance for IoT Devices

A student-friendly machine-learning project that estimates machine-failure risk from operating-condition data. It includes data loading, preprocessing, model training, evaluation, an interactive Streamlit dashboard, and saved output artifacts.

> **Important:** This is a software demonstration, not a certified industrial safety system. It does not connect to physical sensors by default. The code attempts to download the UCI AI4I 2020 dataset; if internet access is unavailable, it generates clearly labelled synthetic demo data.

## Features
- Loads the UCI AI4I 2020 Predictive Maintenance dataset or offline synthetic data
- Preprocessing in a Scikit-learn pipeline (missing-value handling, scaling, categorical encoding)
- Random Forest classifier with class weighting
- Accuracy, precision, recall, F1 score, and ROC-AUC where available
- Confusion matrix and classification report
- Streamlit dashboard with interactive machine-condition inputs
- Saved model and metrics

## Tech stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Streamlit, Flask, Joblib, pytest.

## Dataset
UCI AI4I 2020 Predictive Maintenance Dataset: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

The dataset is synthetic industrial data created for predictive-maintenance research. If the dataset cannot be downloaded, the application creates a separate synthetic fallback dataset. Do not describe fallback data as real sensor data.

## Project structure
```text
AI-Powered-Predictive-Maintenance/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/raw/
├── models/
├── outputs/
└── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
    ├── predict.py
    ├── visualization.py\n    └── api.py
└── tests/
    ├── test_api.py
    └── test_data_pipeline.py
└── .github/workflows/python-ci.yml
```

## Run on Windows (PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.visualization
streamlit run app.py
```

If PowerShell blocks activation, run this for the current terminal only:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run on macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
python -m src.visualization
streamlit run app.py
```

Open the local URL Streamlit prints (usually http://localhost:8501).

## Outputs
- `models/predictive_maintenance_model.joblib`: trained model pipeline
- `outputs/metrics.json`: evaluation metrics and dataset source
- `outputs/confusion_matrix.png`: confusion matrix
- `outputs/classification_report.txt`: class-wise precision/recall/F1
- `outputs/failure_counts.png`: failure label counts
- `outputs/correlation_heatmap.png`: numeric feature correlations

These generated files are local artifacts and are ignored by Git by default. Commit screenshots under `images/` if you want to include proof in the repository.

## How to interpret metrics
- **Accuracy:** fraction of all test predictions that were correct.
- **Precision:** among records flagged as failures, fraction that were failures.
- **Recall:** among actual failures, fraction detected by the model.
- **F1 score:** balance between precision and recall.
- **ROC-AUC:** how well model scores separate positive and negative examples, when both classes are present.

Do not rely on accuracy alone because failure labels are often imbalanced.

## Limitations
- The dataset is not a live machine feed.
- Predictions are estimates and do not guarantee a real machine will or will not fail.
- This baseline uses tabular classification, not a sequence model such as LSTM.
- Before real deployment, data quality, calibration, time-based validation, drift monitoring, and safety review are required.

## GitHub
```bash
git init
git add .
git commit -m "Build predictive maintenance ML pipeline and dashboard"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

## Flask REST API

The project also provides a small HTTP API so another program can submit machine readings and receive the model's prediction. The API uses the same saved model pipeline as the Streamlit dashboard.

### Start the API

First install the dependencies and train the model if you have not already done so:

```powershell
pip install -r requirements.txt
python -m src.train
python -m src.api
```

The development API runs at `http://127.0.0.1:5000`. Keep this terminal open while using the API. The Streamlit dashboard can run separately in another terminal with `streamlit run app.py`.

### Endpoints

- `GET /` — API name, endpoint list, and required feature names.
- `GET /health` — basic status and whether the saved model file exists.
- `POST /predict` — validates machine readings and returns the model prediction.

Example PowerShell request:

```powershell
$body = @{
  "Type" = "L"
  "Air temperature [K]" = 300.0
  "Process temperature [K]" = 310.0
  "Rotational speed [rpm]" = 1500
  "Torque [Nm]" = 40.0
  "Tool wear [min]" = 100
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body $body
```

The response includes `prediction` (0 or 1), a readable `status`, estimated `risk_percent`, and `data_source`. These are model estimates, not guarantees of machine safety. The API accepts machine type `L`, `M`, or `H`; all five numeric sensor fields are required.

### Run automated tests

```powershell
python -m pytest -q
```

The tests use Flask's test client and mock the model prediction where appropriate, so most API tests do not require a trained model or internet access. They cover successful responses, missing fields, invalid input, and missing-model handling. Data-pipeline tests check reproducible demo data, required columns, preprocessing, and a small model-fit smoke test.

GitHub Actions also runs the test suite and Python syntax checks on pushes and pull requests using Python 3.11 and 3.12.

This Flask development server is intended for local demonstration. Do not expose it directly to the public internet or treat it as a production industrial control service.
