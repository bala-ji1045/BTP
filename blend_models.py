from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import feature_schema
from src.pipeline import split_indices


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_kw": float(mean_absolute_error(actual, predicted)),
        "rmse_kw": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend existing PV model predictions.")
    parser.add_argument("--samples", type=Path, default=Path("artifacts/run_001/matched_samples.csv"))
    parser.add_argument("--model-a", type=Path, default=Path("artifacts/tuning_001/sky_cloud_pv_regressor_tuned.joblib"))
    parser.add_argument("--model-b", type=Path, default=Path("artifacts/tuning_mae_001/sky_cloud_pv_regressor_mae_tuned.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/blend_001"))
    args = parser.parse_args()

    names = feature_schema()["feature_names"]
    frame = pd.read_csv(args.samples, usecols=["timestamp", "target"] + names)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    x = frame[names].to_numpy(dtype=np.float32)
    y = frame["target"].to_numpy(dtype=np.float64)
    train_slice, validation_slice, test_slice = split_indices(len(y))
    model_a = joblib.load(args.model_a)
    model_b = joblib.load(args.model_b)
    prediction_a = model_a.predict(x)
    prediction_b = model_b.predict(x)

    results = []
    for weight_a in np.linspace(0.0, 1.0, 101):
        weight_b = 1.0 - weight_a
        validation_prediction = weight_a * prediction_a[validation_slice] + weight_b * prediction_b[validation_slice]
        results.append({"weight_model_a": float(weight_a), "weight_model_b": float(weight_b),
                        "validation_mae_kw": float(mean_absolute_error(y[validation_slice], validation_prediction))})
    results.sort(key=lambda result: (result["validation_mae_kw"], -result["weight_model_a"]))
    best = results[0]
    weight_a = best["weight_model_a"]
    weight_b = best["weight_model_b"]
    validation_prediction = weight_a * prediction_a[validation_slice] + weight_b * prediction_b[validation_slice]
    correction = float(np.median(y[validation_slice] - validation_prediction))
    corrected_validation_prediction = validation_prediction + correction
    correction_improves_validation = mean_absolute_error(
        y[validation_slice], corrected_validation_prediction
    ) < best["validation_mae_kw"]
    if not correction_improves_validation:
        correction = 0.0
    corrected_validation_mae = float(mean_absolute_error(
        y[validation_slice], validation_prediction + correction
    ))
    test_prediction = weight_a * prediction_a[test_slice] + weight_b * prediction_b[test_slice]
    corrected_test_prediction = test_prediction + correction
    test_metrics = metrics(y[test_slice], test_prediction)
    corrected_test_metrics = metrics(y[test_slice], corrected_test_prediction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "blend_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["weight_model_a", "weight_model_b", "validation_mae_kw"])
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "best_config.json").write_text(json.dumps({
        "model_a": str(args.model_a),
        "model_b": str(args.model_b),
        "best_weights": {"model_a": weight_a, "model_b": weight_b},
        "validation_mae_kw": best["validation_mae_kw"],
        "validation_mae_corrected_kw": corrected_validation_mae,
        "residual_correction_kw": correction,
        "correction_improves_validation": bool(correction_improves_validation),
        "test_metrics": test_metrics,
        "corrected_test_metrics": corrected_test_metrics,
        "note": "Blend weight selected on validation only; test used once after selection.",
    }, indent=2) + "\n")
    predictions = pd.DataFrame({
        "timestamp": frame.iloc[test_slice]["timestamp"].to_numpy(),
        "actual_pv_kw": y[test_slice],
        "predicted_pv_kw": test_prediction,
        "corrected_predicted_pv_kw": corrected_test_prediction,
    })
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(json.dumps({"best": best, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()