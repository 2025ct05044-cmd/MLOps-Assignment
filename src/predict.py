"""Make predictions from a saved pipeline using a JSON or CSV input file."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_preprocessing import FEATURE_COLUMNS


def predict(model_path, input_path):
    model_path, input_path = Path(model_path), Path(input_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    if input_path.suffix.lower() == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = [payload]
        data = pd.DataFrame(payload)
    elif input_path.suffix.lower() == ".csv":
        data = pd.read_csv(input_path)
    else:
        raise ValueError("Input file must have a .json or .csv extension.")

    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Input is missing required feature columns: {missing}")
    data = data.loc[:, FEATURE_COLUMNS]

    model = joblib.load(model_path)
    predictions = model.predict(data).astype(int).tolist()
    probabilities = model.predict_proba(data)[:, 1].astype(float).tolist()
    results = [
        {"prediction": prediction, "probability_class_1": probability}
        for prediction, probability in zip(predictions, probabilities)
    ]
    print(json.dumps(results, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/heart_disease_pipeline.joblib")
    parser.add_argument("--input", required=True, help="Path to JSON or CSV feature data.")
    args = parser.parse_args()
    predict(args.model, args.input)


if __name__ == "__main__":
    main()
