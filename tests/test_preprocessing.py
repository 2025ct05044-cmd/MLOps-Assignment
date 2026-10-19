"""Unit tests for dataset loading and schema validation."""

import pandas as pd
import pytest

from src.data_preprocessing import (
    EXPECTED_COLUMNS, FEATURE_COLUMNS, load_dataset, split_features_target,
)


def sample_dataframe():
    rows = []
    for i, target in enumerate([0, 1, 0, 1]):
        rows.append({
            "age": 45 + i, "sex": i % 2, "cp": i % 4,
            "trestbps": 120 + i, "chol": 220 + i, "fbs": i % 2,
            "restecg": i % 3, "thalach": 150 - i, "exang": i % 2,
            "oldpeak": 0.1 * i, "slope": i % 3, "ca": i % 4,
            "thal": i % 3, "target": target,
        })
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def test_load_dataset_validates_columns_and_labels(tmp_path):
    path = tmp_path / "heart.csv"
    sample_dataframe().to_csv(path, index=False)
    df = load_dataset(path)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert set(df["target"]) == {0, 1}


def test_split_features_target_returns_expected_shapes():
    X, y = split_features_target(sample_dataframe())
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (4, 13)
    assert y.tolist() == [0, 1, 0, 1]


def test_load_dataset_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    sample_dataframe().drop(columns=["thal"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_dataset(path)


def test_load_dataset_rejects_non_binary_target(tmp_path):
    path = tmp_path / "bad_target.csv"
    df = sample_dataframe()
    df.loc[0, "target"] = 2
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="only 0 and 1"):
        load_dataset(path)
