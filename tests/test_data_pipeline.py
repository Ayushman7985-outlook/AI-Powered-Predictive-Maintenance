import pandas as pd
import pytest

from src.config import FEATURES, TARGET
from src.data_loader import make_demo_data
from src.preprocessing import prepare_data


def test_demo_data_is_reproducible_and_has_required_columns():
    first = make_demo_data(n=120, seed=17)
    second = make_demo_data(n=120, seed=17)

    pd.testing.assert_frame_equal(first, second)
    assert set(FEATURES + [TARGET]).issubset(first.columns)
    assert len(first) == 120
    assert set(first[TARGET].unique()).issubset({0, 1})


def test_preprocessing_returns_features_target_and_pipeline():
    df = make_demo_data(n=180, seed=21)
    X, y, model, clean = prepare_data(df)

    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == len(clean)
    assert set(y.unique()).issubset({0, 1})
    # Smoke-test fitting and prediction on a small local synthetic dataset.
    model.fit(X, y)
    predictions = model.predict(X.iloc[:8])
    assert len(predictions) == 8
    assert set(predictions).issubset({0, 1})


def test_preprocessing_explains_missing_required_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        prepare_data(pd.DataFrame({"Type": ["L"], TARGET: [0]}))
