"""Train candidate heart-disease models and save the validation-selected pipeline."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.data_preprocessing import load_dataset, split_features_target
from src.feature_engineering import build_pipeline

RANDOM_STATE = 42


def train(data_path, model_dir="models", reports_dir="reports"):
    df = load_dataset(data_path)
    X, y = split_features_target(df)

    # Split into train/validation/test. The test set is not used for model selection.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )

    model_dir, reports_dir = Path(model_dir), Path(reports_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows, fitted = [], {}
    for name in ("logistic_regression", "random_forest"):
        pipeline = build_pipeline(name, random_state=RANDOM_STATE)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        predictions = pipeline.predict(X_val)
        auc = float(roc_auc_score(y_val, probabilities))
        accuracy = float(accuracy_score(y_val, predictions))
        rows.append({
            "model": name,
            "validation_roc_auc": auc,
            "validation_accuracy": accuracy,
        })
        fitted[name] = pipeline

    comparison = pd.DataFrame(rows).sort_values(
        "validation_roc_auc", ascending=False
    )
    selected_name = str(comparison.iloc[0]["model"])
    selected = fitted[selected_name]

    # Refit the selected model on train + validation before final test evaluation.
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    selected = build_pipeline(selected_name, random_state=RANDOM_STATE)
    selected.fit(X_train_full, y_train_full)

    test_probabilities = selected.predict_proba(X_test)[:, 1]
    test_predictions = selected.predict(X_test)
    test_metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "test_accuracy": float(accuracy_score(y_test, test_predictions)),
        "selected_model": selected_name,
        "random_state": RANDOM_STATE,
        "selection_metric": "validation_roc_auc",
        "training_rows": int(len(X_train_full)),
        "test_rows": int(len(X_test)),
    }

    joblib.dump(selected, model_dir / "heart_disease_pipeline.joblib")
    comparison.to_csv(reports_dir / "model_comparison.csv", index=False)
    (model_dir / "heart_disease_pipeline_metadata.json").write_text(
        json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8"
    )
    (reports_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8"
    )

    print("Selected model:", selected_name)
    print("Test ROC-AUC:", f'{test_metrics["test_roc_auc"]:.4f}')
    print("Test accuracy:", f'{test_metrics["test_accuracy"]:.4f}')
    print("Saved model:", model_dir / "heart_disease_pipeline.joblib")
    print("Saved comparison:", reports_dir / "model_comparison.csv")
    return selected, test_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/processed/cleaned_heart_disease.csv",
        help="Path to the processed heart disease CSV.",
    )
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    train(args.data, args.model_dir, args.reports_dir)


if __name__ == "__main__":
    main()
