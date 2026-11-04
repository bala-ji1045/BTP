from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib

from src.features import feature_schema, full_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict same-minute PV power from one sky image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()
    timestamp = datetime.fromisoformat(args.timestamp)
    schema = __import__("json").loads(args.schema.read_text())
    if schema["feature_names"] != feature_schema()["feature_names"]:
        raise ValueError("schema feature order does not match this inference code")
    features = full_features(args.image.read_bytes(), timestamp)
    model = joblib.load(args.model)
    prediction = float(model.predict(features.reshape(1, -1))[0])
    print(f"{prediction:.9f} kW")


if __name__ == "__main__":
    main()