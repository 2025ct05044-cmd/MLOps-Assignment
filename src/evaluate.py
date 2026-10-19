"""Evaluate a saved sklearn pipeline on a labelled CSV dataset."""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_preprocessing import load_dataset, split_features_target


def evaluate(model_path, data_path, output_path="reports/evaluation_metrics.json"):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    df = load_dataset(data_path)
    X, y = split_features_target(df)
    model = joblib.load(model_path)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y, predictions)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "rows_evaluated": len(y),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("Saved metrics to:", output_path)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/heart_disease_pipeline.joblib")
    parser.add_argument("--data", default="data/processed/cleaned_heart_disease.csv")
    parser.add_argument("--output", default="reports/evaluation_metrics.json")
    args = parser.parse_args()
    evaluate(args.model, args.data, args.output)


if __name__ == "__main__":
    main()
