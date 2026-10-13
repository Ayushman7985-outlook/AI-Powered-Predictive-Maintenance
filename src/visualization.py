import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import OUTPUT_DIR
from src.data_loader import load_data

def save_dataset_charts():
    df, source = load_data()
    if "Machine failure" not in df.columns:
        raise ValueError("Machine failure target not found.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="Machine failure", ax=ax)
    ax.set_title(f"Machine Failure Label Counts\nSource: {source}")
    ax.set_xlabel("Machine failure (0 = no, 1 = yes)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "failure_counts.png", dpi=160)
    plt.close(fig)
    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(numeric.corr(numeric_only=True), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric Feature Correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close(fig)
    print(f"Charts saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    save_dataset_charts()
