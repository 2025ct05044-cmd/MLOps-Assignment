"""Unit tests for the model pipeline and persisted-model prediction interface."""

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_preprocessing import FEATURE_COLUMNS
from src.feature_engineering import build_pipeline


def sample_features():
    return pd.DataFrame([
        [45, 1, 0, 130, 230, 0, 1, 150, 0, 1.0, 1, 0, 2],
        [60, 0, 2, 140, 260, 1, 0, 120, 1, 2.2, 2, 1, 3],
        [52, 1, 1, 125, 210, 0, 1, 160, 0, 0.2, 0, 0, 1],
        [67, 0, 3, 150, 280, 1, 2, 110, 1, 3.1, 2, 2, 2],
        [48, 1, 0, 118, 205, 0, 0, 165, 0, 0.0, 1, 0, 2],
        [58, 0, 2, 135, 245, 0, 1, 130, 1, 1.5, 1, 1, 3],
    ], columns=FEATURE_COLUMNS)


def test_logistic_pipeline_fits_and_predicts():
    X = sample_features()
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = build_pipeline("logistic_regression")
    model.fit(X, y)
    assert isinstance(model, Pipeline)
    assert model.predict(X).shape == (len(X),)
    assert model.predict_proba(X).shape == (len(X), 2)


def test_pipeline_can_be_saved_and_reloaded(tmp_path):
    X = sample_features()
    y = pd.Series([0, 1, 0, 1, 0, 1])
    model = build_pipeline("logistic_regression")
    model.fit(X, y)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    assert loaded.predict(X).tolist() == model.predict(X).tolist()


def test_unknown_model_name_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="Unsupported model_name"):
        build_pipeline("not_a_model")
