import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from src.config import FEATURES, TARGET

def prepare_data(df: pd.DataFrame):
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))
    clean = df[FEATURES + [TARGET]].copy()
    clean = clean.dropna(subset=[TARGET])
    clean[TARGET] = pd.to_numeric(clean[TARGET], errors="coerce")
    clean = clean.dropna(subset=[TARGET])
    clean[TARGET] = clean[TARGET].astype(int)
    clean = clean[clean[TARGET].isin([0, 1])]
    X = clean[FEATURES]
    y = clean[TARGET]
    categorical = ["Type"]
    numeric = [c for c in FEATURES if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformer = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    model = Pipeline([
        ("preprocess", transformer),
        ("classifier", RandomForestClassifier(
            n_estimators=250, class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    return X, y, model, clean
