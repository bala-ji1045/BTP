from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.pipeline import build_dataset, discover_members, load_pv, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the sky/cloud PV regressor.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--archives-pattern", default="2019_*.tar")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger("pv_train")
    if not args.images_dir.is_dir():
        raise SystemExit(f"image directory does not exist: {args.images_dir}")
    if not args.csv.is_file():
        raise SystemExit(f"CSV does not exist: {args.csv}")
    pv, csv_stats = load_pv(args.csv)
    records, match_stats = discover_members(args.images_dir, args.archives_pattern, pv, args.max_samples, logger)
    x, y, rows, feature_stats = build_dataset(records, pv, logger)
    if len(y) < 3:
        raise SystemExit("fewer than three valid matched samples were extracted")
    config = {"images_dir": str(args.images_dir), "csv": str(args.csv), "archives_pattern": args.archives_pattern,
              "max_samples": args.max_samples, "timestamp_regex": r"(\d{14})", "pv_duplicate_policy": "mean",
              "image_duplicate_policy": "first_by_archive_and_member", "split": [0.70, 0.15, 0.15],
              "csv_stats": csv_stats, "matching_stats": match_stats, "feature_stats": feature_stats}
    metrics = train_model(x, y, rows, args.output_dir, config)
    print(json.dumps({"samples": len(y), "features": int(x.shape[1]), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()