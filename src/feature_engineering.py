"""Reusable preprocessing and estimator pipeline builders."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
]


def build_preprocessor():
    """Build the same numeric/categorical preprocessing strategy as the notebooks."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(model_name: str = "logistic_regression", random_state: int = 42):
    """Create a complete sklearn Pipeline, including preprocessing and classifier."""
    name = model_name.strip().lower()
    if name in {"logistic_regression", "logistic", "lr"}:
        classifier = LogisticRegression(max_iter=2000, random_state=random_state)
    elif name in {"random_forest", "randomforest", "rf"}:
        classifier = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
    else:
        raise ValueError(
            "Unsupported model_name. Use 'logistic_regression' or 'random_forest'."
        )

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
