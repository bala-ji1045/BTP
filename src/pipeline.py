from __future__ import annotations

import csv
import json
import logging
import re
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

from .features import feature_schema, full_features

TIMESTAMP_RE = re.compile(r"(\d{14})")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
TARGET = "Huang_E4102_kW"


def parse_member_timestamp(name: str) -> datetime | None:
    match = TIMESTAMP_RE.search(name)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def load_pv(path: Path) -> tuple[dict[datetime, float], dict]:
    raw: dict[datetime, list[float]] = {}
    invalid_timestamp = invalid_target = 0
    total = 0
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "Date" not in reader.fieldnames or TARGET not in reader.fieldnames:
            raise ValueError(f"CSV must contain Date and {TARGET}")
        for row in reader:
            total += 1
            try:
                timestamp = datetime.fromisoformat(row["Date"].strip()).replace(second=0, microsecond=0)
            except (TypeError, ValueError):
                invalid_timestamp += 1
                continue
            try:
                target = float(row[TARGET])
                if not np.isfinite(target):
                    raise ValueError
            except (TypeError, ValueError):
                invalid_target += 1
                continue
            raw.setdefault(timestamp, []).append(target)
    values = {timestamp: float(np.mean(targets)) for timestamp, targets in raw.items()}
    return values, {"rows": total, "invalid_timestamp": invalid_timestamp, "invalid_target": invalid_target,
                    "duplicate_minutes": sum(len(v) - 1 for v in raw.values() if len(v) > 1)}


def discover_members(images_dir: Path, archives_pattern: str, pv: dict[datetime, float], max_samples: int | None,
                     logger: logging.Logger) -> tuple[list[dict], dict]:
    candidates: list[dict] = []
    skipped = Counter()
    archives = sorted(images_dir.glob(archives_pattern))
    if not archives:
        raise FileNotFoundError(f"no archives matching {archives_pattern} in {images_dir}")
    for archive_path in archives:
        try:
            with tarfile.open(archive_path, mode="r:*") as archive:
                for member in archive:
                    if not member.isfile() or Path(member.name).suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    timestamp = parse_member_timestamp(member.name)
                    if timestamp is None:
                        skipped["timestamp"] += 1
                        continue
                    minute = timestamp.replace(second=0, microsecond=0)
                    if minute not in pv:
                        skipped["unmatched_image"] += 1
                        continue
                    candidates.append({"timestamp": minute, "archive": str(archive_path), "member": member.name})
                    if max_samples is not None and len(candidates) >= max_samples * 3:
                        break
        except (tarfile.TarError, OSError) as exc:
            logger.warning("Skipping archive %s: %s", archive_path, exc)
        if max_samples is not None and len(candidates) >= max_samples * 3:
            break
    candidates.sort(key=lambda row: (row["timestamp"], row["archive"], row["member"]))
    unique = []
    seen = set()
    for row in candidates:
        if row["timestamp"] in seen:
            skipped["duplicate_image"] += 1
            continue
        seen.add(row["timestamp"])
        unique.append(row)
        if max_samples is not None and len(unique) >= max_samples:
            break
    return unique, {"archives": len(archives), "candidate_images": len(candidates), "skipped": dict(skipped)}


def build_dataset(records: list[dict], pv: dict[datetime, float], logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    rows = []
    failures = 0
    records_by_archive: dict[str, list[dict]] = {}
    for record in records:
        records_by_archive.setdefault(record["archive"], []).append(record)
    for archive_path in sorted(records_by_archive):
        try:
            with tarfile.open(archive_path, mode="r:*") as archive:
                for record in tqdm(records_by_archive[archive_path], desc=f"Extracting {Path(archive_path).name}", leave=False):
                    try:
                        member = archive.getmember(record["member"])
                        extracted = archive.extractfile(member)
                        if extracted is None:
                            raise ValueError("member could not be read")
                        features = full_features(extracted, record["timestamp"])
                        row = {"timestamp": record["timestamp"].isoformat(), "archive": record["archive"],
                               "member": record["member"], "target": pv[record["timestamp"]]}
                        row.update(dict(zip(feature_schema()["feature_names"], features.tolist())))
                        rows.append(row)
                    except Exception as exc:
                        failures += 1
                        logger.warning("Skipping %s:%s: %s", record["archive"], record["member"], exc)
        except (tarfile.TarError, OSError) as exc:
            failures += len(records_by_archive[archive_path])
            logger.warning("Skipping archive %s during feature extraction: %s", archive_path, exc)
    rows.sort(key=lambda row: row["timestamp"])
    names = feature_schema()["feature_names"]
    x = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    y = np.asarray([row["target"] for row in rows], dtype=np.float64)
    return x, y, rows, {"feature_failures": failures}


def split_indices(size: int) -> tuple[slice, slice, slice]:
    if size < 3:
        raise ValueError("at least three matched samples are required")
    train_end = max(1, int(size * 0.70))
    val_end = max(train_end + 1, int(size * 0.85))
    val_end = min(val_end, size - 1)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, size)


def train_model(x: np.ndarray, y: np.ndarray, rows: list[dict], output_dir: Path, config: dict) -> dict:
    train_slice, val_slice, test_slice = split_indices(len(y))
    model = HistGradientBoostingRegressor(learning_rate=0.05, max_depth=6, max_iter=300,
                                          min_samples_leaf=20, random_state=42)
    model.fit(x[train_slice], y[train_slice])
    metrics = {}
    predictions = []
    for split_name, split in (("train", train_slice), ("validation", val_slice), ("test", test_slice)):
        actual = y[split]
        predicted = model.predict(x[split])
        metrics[split_name] = {"mae_kw": float(mean_absolute_error(actual, predicted)),
                               "rmse_kw": float(np.sqrt(mean_squared_error(actual, predicted))),
                               "r2": float(r2_score(actual, predicted))}
        for row, actual_value, predicted_value in zip(rows[split], actual, predicted):
            predictions.append({"timestamp": row["timestamp"], "split": split_name, "actual_pv_kw": float(actual_value),
                                "predicted_pv_kw": float(predicted_value), "absolute_error_kw": abs(float(actual_value - predicted_value)),
                                "signed_error_kw": float(predicted_value - actual_value)})
    output_dir.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, output_dir / "sky_cloud_pv_regressor.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (output_dir / "predictions.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=predictions[0].keys())
        writer.writeheader(); writer.writerows(predictions)
    with (output_dir / "matched_samples.csv").open("w", newline="") as handle:
        fields = ["timestamp", "archive", "member", "target"] + feature_schema()["feature_names"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (output_dir / "feature_schema.json").write_text(json.dumps(feature_schema(), indent=2) + "\n")
    summary = [
        f"Samples: {len(y)}",
        f"Features: {x.shape[1]}",
        f"Split counts: train={len(y[train_slice])}, validation={len(y[val_slice])}, test={len(y[test_slice])}",
        f"Model: HistGradientBoostingRegressor({model.get_params()})",
        f"Metrics: {json.dumps(metrics, sort_keys=True)}",
        "Limitations: chronological evaluation does not guarantee robustness across sites or years; handcrafted features are not a CNN.",
    ]
    (output_dir / "training_summary.txt").write_text("\n".join(summary) + "\n")
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "logs" / "README.txt").write_text("Run completed with Python logging output on stderr.\n")
    config.update({"feature_count": int(x.shape[1]), "sample_count": int(x.shape[0]), "model_parameters": model.get_params(),
                   "split_counts": {"train": len(y[train_slice]), "validation": len(y[val_slice]), "test": len(y[test_slice])}})
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    return metrics