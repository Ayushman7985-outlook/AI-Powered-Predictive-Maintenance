from __future__ import annotations
import zipfile
from io import BytesIO
from urllib.request import urlopen
import numpy as np
import pandas as pd
from src.config import DATA_PATH

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"

def make_demo_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create clearly synthetic data so the project can run offline."""
    rng = np.random.default_rng(seed)
    machine_type = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    air = rng.normal(300.0, 2.0, n)
    process = air + rng.normal(10.0, 1.5, n)
    rpm = np.clip(rng.normal(1538, 180, n), 900, 2900)
    torque = np.clip(rng.normal(40, 10, n), 3, 80)
    wear = rng.integers(0, 254, n)
    # Simulated risk signal: this is a demo label, not a real industrial measurement.
    risk = (-5.0 + 0.045 * (torque - 40) + 0.012 * (wear - 120)
            + 0.0018 * (rpm - 1500) + 0.65 * (machine_type == "L")
            + 0.9 * (machine_type == "H"))
    probability = 1 / (1 + np.exp(-risk))
    failure = rng.binomial(1, probability)
    return pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "Product ID": [f"SIM-{i:05d}" for i in range(1, n + 1)],
        "Type": machine_type,
        "Air temperature [K]": air,
        "Process temperature [K]": process,
        "Rotational speed [rpm]": rpm.round().astype(int),
        "Torque [Nm]": torque.round(2),
        "Tool wear [min]": wear,
        "Machine failure": failure,
        "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    })

def load_data(force_download: bool = False) -> tuple[pd.DataFrame, str]:
    """Try UCI data first; fall back to synthetic demo data when offline."""
    if DATA_PATH.exists() and not force_download:
        try:
            df = pd.read_csv(DATA_PATH)
            if "Machine failure" in df.columns:
                return df, "Local CSV"
        except Exception:
            pass
    try:
        with urlopen(UCI_ZIP_URL, timeout=8) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(BytesIO(archive_bytes)) as zf:
            csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            with zf.open(csv_name) as csv_file:
                df = pd.read_csv(csv_file)
        df.to_csv(DATA_PATH, index=False)
        return df, "UCI AI4I 2020 dataset"
    except Exception:
        df = make_demo_data()
        df.to_csv(DATA_PATH, index=False)
        return df, "Synthetic demo data (offline fallback)"
