from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import feature_schema
from src.pipeline import split_indices


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_kw": float(mean_absolute_error(actual, predicted)),
        "rmse_kw": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "samples": int(actual.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train separate daytime and nighttime PV regressors.")
    parser.add_argument("--samples", type=Path, default=Path("artifacts/run_001/matched_samples.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/day_night_001"))
    parser.add_argument("--day-start-hour", type=int, default=6)
    parser.add_argument("--day-end-hour", type=int, default=18)
    args = parser.parse_args()
    if not 0 <= args.day_start_hour < args.day_end_hour <= 24:
        raise SystemExit("daytime hours must satisfy 0 <= start < end <= 24")

    names = feature_schema()["feature_names"]
    frame = pd.read_csv(args.samples, usecols=["timestamp", "target"] + names)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    daytime = frame["timestamp"].dt.hour.between(args.day_start_hour, args.day_end_hour - 1)
    x = frame[names].to_numpy(dtype=np.float32)
    y = frame["target"].to_numpy(dtype=np.float64)
    timestamps = frame["timestamp"]
    train_slice, validation_slice, test_slice = split_indices(len(frame))

    model_params = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 800,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 50,
        "l2_regularization": 0.1,
        "random_state": 42,
    }
    models = {}
    train_counts = {}
    for label, selector in (("day", daytime.to_numpy()), ("night", (~daytime).to_numpy())):
        train_mask = selector.copy()
        train_mask[:train_slice.start] = False
        train_mask[train_slice.stop:] = False
        if not train_mask.any():
            raise SystemExit(f"no training samples for {label} model")
        model = HistGradientBoostingRegressor(**model_params)
        model.fit(x[train_mask], y[train_mask])
        models[label] = model
        train_counts[label] = int(train_mask.sum())

    def routed_predictions(indices: slice) -> tuple[np.ndarray, np.ndarray]:
        positions = np.arange(len(frame))[indices]
        predictions = np.empty(len(positions), dtype=np.float64)
        for label, selector in (("day", daytime.to_numpy()), ("night", (~daytime).to_numpy())):
            selected = selector[positions]
            if selected.any():
                predictions[selected] = models[label].predict(x[positions[selected]])
        return y[indices], predictions

    split_metrics = {}
    for label, indices in (("validation", validation_slice), ("test", test_slice)):
        actual, predicted = routed_predictions(indices)
        split_metrics[label] = metrics(actual, predicted)
        for period, subgroup_selector in (("day", daytime.to_numpy()), ("night", (~daytime).to_numpy())):
            subgroup = subgroup_selector[np.arange(len(frame))[indices]]
            if subgroup.any():
                split_metrics[f"{label}_{period}"] = metrics(actual[subgroup], predicted[subgroup])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for label, model in models.items():
        joblib.dump(model, args.output_dir / f"sky_cloud_pv_regressor_{label}.joblib")
    config = {
        "day_start_hour": args.day_start_hour,
        "day_end_hour_exclusive": args.day_end_hour,
        "model_parameters": model_params,
        "train_counts": train_counts,
        "split_counts": {"train": train_slice.stop - train_slice.start,
                         "validation": validation_slice.stop - validation_slice.start,
                         "test": test_slice.stop - test_slice.start},
        "metrics": split_metrics,
        "note": "Routing uses timestamp hour only; test metrics were not used for model selection.",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(split_metrics, indent=2) + "\n")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()