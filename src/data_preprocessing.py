"""Dataset loading and validation utilities for the heart disease project."""

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "target"
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the CSV, validate required columns/labels, and remove duplicates."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    df = pd.read_csv(path)
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    # Ignore unrelated columns and enforce a stable column order.
    df = df.loc[:, EXPECTED_COLUMNS].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="raise")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")
    if not df[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target column must contain only 0 and 1.")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int64")

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Dataset must contain both target classes (0 and 1).")
    if df[FEATURE_COLUMNS].isna().all().all():
        raise ValueError("All feature values are missing.")

    return df


def split_features_target(df: pd.DataFrame):
    """Split a validated dataframe into feature matrix X and labels y."""
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")
    return df.loc[:, FEATURE_COLUMNS].copy(), df[TARGET_COLUMN].copy()
