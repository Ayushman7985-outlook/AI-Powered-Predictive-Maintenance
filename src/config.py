from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_PATH = DATA_DIR / "ai4i2020.csv"
MODEL_PATH = MODEL_DIR / "predictive_maintenance_model.joblib"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
PLOT_PATH = OUTPUT_DIR / "confusion_matrix.png"

FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
TARGET = "Machine failure"

for folder in (DATA_DIR, MODEL_DIR, OUTPUT_DIR):
    folder.mkdir(parents=True, exist_ok=True)
