import joblib
import pandas as pd
from src.config import MODEL_PATH, FEATURES

def predict_machine(values: dict) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python -m src.train")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    row = {feature: values[feature] for feature in FEATURES}
    frame = pd.DataFrame([row], columns=FEATURES)
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][1])
    return {
        "prediction": prediction,
        "status": "Potential failure risk" if prediction == 1 else "No failure predicted",
        "risk_percent": round(probability * 100, 2),
        "data_source": bundle.get("data_source", "Unknown"),
    }

if __name__ == "__main__":
    example = {
        "Type": "L", "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0, "Rotational speed [rpm]": 1400,
        "Torque [Nm]": 55.0, "Tool wear [min]": 220,
    }
    print(predict_machine(example))
