%cd /content/MLOps-Assignment

from pathlib import Path

Path("src").mkdir(exist_ok=True)

code = r'''
"""Load and validate the heart disease dataset."""

from pathlib import Path

import pandas as pd

TARGET_COLUMN = "target"

FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

EXPECTED_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def load_dataset(path):
    """Load the CSV, validate its schema and target, and remove duplicates."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    # Keep the assignment's expected columns and their order.
    df = df[EXPECTED_COLUMNS].copy()

    # Remove duplicate rows and normalize the target to numeric values.
    df = df.drop_duplicates().reset_index(drop=True)
    df[TARGET_COLUMN] = pd.to_numeric(
        df[TARGET_COLUMN], errors="raise"
    ).astype(int)

    target_values = set(df[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"Target must contain only 0 and 1; found {target_values}"
        )

    if df[TARGET_COLUMN].nunique() != 2:
        raise ValueError("Dataset must contain both target classes: 0 and 1.")

    if df[FEATURE_COLUMNS].isna().all(axis=None):
        raise ValueError("All feature values are missing.")

    return df


def split_features_target(df):
    """Return features (X) and target labels (y)."""
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(f"DataFrame is missing columns: {missing_columns}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y
'''

Path("src/data_preprocessing.py").write_text(code.strip() + "\n")

print("Created src/data_preprocessing.py")
print("File size:", Path("src/data_preprocessing.py").stat().st_size, "bytes")
