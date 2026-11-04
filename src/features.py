from __future__ import annotations

import io
import json
import math
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image

PERCENTILES = (10, 25, 50, 75, 90)
HIST_EDGES = tuple(np.linspace(0.0, 256.0, 9).tolist())


def feature_schema() -> dict:
    names = []
    for channel in ("r", "g", "b"):
        names.extend(f"{channel}_{stat}" for stat in ("mean", "std", "min", "max"))
    names.extend(("gray_brightness", "gray_contrast"))
    names.extend(f"gray_p{p}" for p in PERCENTILES)
    names.extend(("grad_x_mean", "grad_x_std", "grad_y_mean", "grad_y_std", "grad_mag_mean"))
    for channel in ("rb", "rg", "gb"):
        names.extend(f"diff_{channel}_{stat}" for stat in ("mean", "std", "min", "max"))
    names.extend(f"hist_{channel}_{index:02d}" for channel in ("r", "g", "b", "gray") for index in range(8))
    names.extend(("hour_sin", "hour_cos", "day_sin", "day_cos", "month_sin", "month_cos"))
    return {
        "feature_names": names,
        "dtype": "float32",
        "image": {
            "pixel_range": "[0, 255]",
            "rgb_conversion": "Pillow convert('RGB'); alpha composited by Pillow conversion",
            "grayscale": "0.299*R + 0.587*G + 0.114*B",
            "percentiles": list(PERCENTILES),
            "gradient": "absolute np.diff on grayscale; zero padding is not used",
            "color_difference_statistics": ["mean", "std", "min", "max"],
            "histogram_bins": 8,
            "histogram_edges": list(HIST_EDGES),
            "histogram_normalization": "proportion",
            "resize": None,
        },
        "time": {
            "hour_cycle": 24,
            "day_cycle": 365,
            "month_cycle": 12,
            "hour_component_only": True,
        },
    }


def _stats(values: np.ndarray) -> list[float]:
    return [float(values.mean()), float(values.std()), float(values.min()), float(values.max())]


def image_features(source: bytes | BinaryIO | Image.Image) -> np.ndarray:
    if isinstance(source, Image.Image):
        image = source
    else:
        image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    with image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[0] == 0 or rgb.shape[1] == 0:
        raise ValueError("image has no pixels")
    red, green, blue = (rgb[:, :, index] for index in range(3))
    gray = 0.299 * red + 0.587 * green + 0.114 * blue
    values: list[float] = []
    for channel in (red, green, blue):
        values.extend(_stats(channel))
    values.extend([float(gray.mean()), float(gray.std())])
    values.extend(float(value) for value in np.percentile(gray, PERCENTILES))
    grad_x = np.abs(np.diff(gray, axis=1))
    grad_y = np.abs(np.diff(gray, axis=0))
    grad_mag = math.sqrt(float(np.mean(grad_x ** 2)) + float(np.mean(grad_y ** 2)))
    values.extend([float(grad_x.mean()), float(grad_x.std()), float(grad_y.mean()), float(grad_y.std()), grad_mag])
    for difference in (red - blue, red - green, green - blue):
        values.extend(_stats(difference))
    for channel in (red, green, blue, gray):
        values.extend(float(value) for value in np.histogram(channel, bins=HIST_EDGES)[0] / channel.size)
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("non-finite image features")
    return result


def time_features(timestamp: datetime) -> np.ndarray:
    values = [
        math.sin(2 * math.pi * timestamp.hour / 24),
        math.cos(2 * math.pi * timestamp.hour / 24),
        math.sin(2 * math.pi * timestamp.timetuple().tm_yday / 365),
        math.cos(2 * math.pi * timestamp.timetuple().tm_yday / 365),
        math.sin(2 * math.pi * timestamp.month / 12),
        math.cos(2 * math.pi * timestamp.month / 12),
    ]
    return np.asarray(values, dtype=np.float32)


def full_features(source: bytes | BinaryIO, timestamp: datetime) -> np.ndarray:
    result = np.concatenate((image_features(source), time_features(timestamp))).astype(np.float32)
    if result.size != len(feature_schema()["feature_names"]):
        raise ValueError("feature schema length mismatch")
    return result


def write_schema(path: Path) -> None:
    path.write_text(json.dumps(feature_schema(), indent=2) + "\n")