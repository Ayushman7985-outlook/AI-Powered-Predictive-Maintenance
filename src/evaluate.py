import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from src.config import PLOT_PATH, OUTPUT_DIR
from src.train import train

def evaluate():
    model, X_test, y_test, pred, metrics = train()
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["No failure", "Failure"]).plot(ax=ax, colorbar=False)
    ax.set_title("Predictive Maintenance — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=160)
    plt.close(fig)
    report = classification_report(y_test, pred, labels=[0, 1], target_names=["No failure", "Failure"], zero_division=0)
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved confusion matrix: {PLOT_PATH}")

if __name__ == "__main__":
    evaluate()
