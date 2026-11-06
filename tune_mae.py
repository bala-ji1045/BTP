from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features import feature_schema
from src.pipeline import split_indices

FEATURES: np.ndarray
TARGETS: np.ndarray
TRAIN_SLICE: slice
VALIDATION_SLICE: slice


def evaluate(item: tuple[int, tuple[str, ...], tuple[object, ...]]) -> dict:
    index, keys, values = item
    params = dict(zip(keys, values))
    model = HistGradientBoostingRegressor(**params, random_state=42)
    model.fit(FEATURES[TRAIN_SLICE], TARGETS[TRAIN_SLICE])
    prediction = model.predict(FEATURES[VALIDATION_SLICE])
    return {"combination": index, **params,
            "validation_mae_kw": float(mean_absolute_error(TARGETS[VALIDATION_SLICE], prediction))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused MAE-oriented tuning using existing features.")
    parser.add_argument("--samples", type=Path, default=Path("artifacts/run_001/matched_samples.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tuning_mae_001"))
    args = parser.parse_args()

    names = feature_schema()["feature_names"]
    frame = pd.read_csv(args.samples, usecols=["timestamp", "target"] + names)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    x = frame[names].to_numpy(dtype=np.float32)
    y = frame["target"].to_numpy(dtype=np.float64)
    train_slice, validation_slice, test_slice = split_indices(len(y))
    global FEATURES, TARGETS, TRAIN_SLICE, VALIDATION_SLICE
    FEATURES, TARGETS = x, y
    TRAIN_SLICE, VALIDATION_SLICE = train_slice, validation_slice

    search_space = {
        "loss": ("squared_error", "absolute_error"),
        "max_depth": (3, 6, 10),
        "max_bins": (128, 255),
        "early_stopping": (True, False),
    }
    fixed = {
        "learning_rate": 0.05,
        "max_iter": 800,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 50,
        "l2_regularization": 0.1,
    }
    keys = list(search_space)
    combinations = []
    for index, values in enumerate(itertools.product(*(search_space[key] for key in keys)), start=1):
        combinations.append((index, tuple([*keys, *fixed.keys()]), tuple([*values, *fixed.values()])))

    results = []
    with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
        for completed, result in enumerate(pool.imap(evaluate, combinations), start=1):
            results.append(result)
            print(f"[{completed}/{len(combinations)}] {result}", flush=True)
    results.sort(key=lambda result: (result["validation_mae_kw"], result["combination"]))
    best = results[0]
    best_params = {key: best[key] for key in [*keys, *fixed.keys()]}
    best_model = HistGradientBoostingRegressor(**best_params, random_state=42)
    best_model.fit(x[train_slice], y[train_slice])
    test_prediction = best_model.predict(x[test_slice])
    test_metrics = {
        "mae_kw": float(mean_absolute_error(y[test_slice], test_prediction)),
        "rmse_kw": float(np.sqrt(mean_squared_error(y[test_slice], test_prediction))),
        "r2": float(r2_score(y[test_slice], test_prediction)),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["combination", *keys, *fixed.keys(), "validation_mae_kw"]
    with (args.output_dir / "all_combinations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "best_config.json").write_text(json.dumps({
        "best_parameters": best_params,
        "validation_mae_kw": best["validation_mae_kw"],
        "test_metrics": test_metrics,
        "combinations_evaluated": len(results),
    }, indent=2) + "\n")
    joblib.dump(best_model, args.output_dir / "sky_cloud_pv_regressor_mae_tuned.joblib")
    print(json.dumps({"best": best, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()