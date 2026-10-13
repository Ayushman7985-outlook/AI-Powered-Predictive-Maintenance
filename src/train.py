import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.config import MODEL_PATH, METRICS_PATH, FEATURES, TARGET
from src.data_loader import load_data
from src.preprocessing import prepare_data

def train():
    df, source = load_data()
    X, y, model, clean = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "data_source": source,
        "rows_used": int(len(clean)),
        "failure_rows": int(y.sum()),
        "failure_rate_percent": round(float(y.mean() * 100), 3),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, probs)), 4)
    except ValueError:
        metrics["roc_auc"] = None
    joblib.dump({"model": model, "features": FEATURES, "target": TARGET, "data_source": source}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("\nTraining complete")
    print(f"Data source: {source}")
    print(f"Rows used: {len(clean)} | Failure rows: {int(y.sum())}")
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{key}: {metrics.get(key)}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    return model, X_test, y_test, pred, metrics

if __name__ == "__main__":
    train()
