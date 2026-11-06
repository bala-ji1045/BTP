# Sky/Cloud Image-Based PV Power Prediction

## 1. Purpose

This project estimates photovoltaic (PV) power output in kilowatts from timestamped sky/cloud images and timestamp-derived features.

The implemented task is **same-minute PV estimation**. An image timestamp is matched with the PV measurement from the same minute. This is not future forecasting because the target was not shifted into the future.

The implementation follows the project specification and uses a handcrafted-feature baseline with `sklearn.ensemble.HistGradientBoostingRegressor`.

## 2. Dataset Inspection

The real dataset was available on the mounted SD card:

```text
/images directory: /media/aidslab/Research
PV CSV:            /media/aidslab/Research/2019_pv_raw.csv
Archive pattern:   2019_*.tar
```

Observed input files:

- 10 monthly image TAR archives matched by `2019_*.tar`
- 1 PV CSV file
- Archive members used `.jpg`, `.jpeg`, and `.png` image formats
- Image filenames contained 14-digit timestamps such as `20190124060000`

CSV inspection:

- Rows: `431,041`
- Timestamp column: `Date`
- Target column: `Huang_E4102_kW`
- Invalid timestamps: `0`
- Invalid target values: `0`
- Duplicate PV minutes: `0`

The target remained in kilowatts. Valid nighttime and negative values were not automatically removed.

## 3. Timestamp Matching

The implementation extracts the first 14-digit sequence from an image member path:

```python
r"(\d{14})"
```

The timestamp is parsed using:

```python
datetime.strptime(value, "%Y%m%d%H%M%S")
```

Both image and CSV timestamps are normalized to minute precision by setting seconds and microseconds to zero. Matching is performed by normalized timestamp, not row order.

Duplicate policies:

- Duplicate CSV minute: mean target value
- Duplicate image minute: first image after sorting by archive path and member path
- Final dataset: one sample per minute

Matching statistics:

- Candidate image members: `137,026`
- Unmatched images: `50,327`
- Duplicate image minutes discarded: `613`
- Feature extraction failures: `16,750`
- Final matched samples: `119,663`

The feature failures were caused primarily by corrupt or truncated image data in part of the October archive. Recoverable image errors were skipped rather than terminating the entire run.

## 4. TAR Processing

Archives were processed with Python's `tarfile` module.

Processing behavior:

- Archives were sorted deterministically
- Members were processed sequentially
- Images were read with `extractfile`
- Full archive extraction to disk was not performed
- Original image arrays were not accumulated in memory
- Invalid image members were logged and skipped
- TAR archives were kept open while their selected members were decoded to avoid reopening a large archive for every image
- `tqdm` progress tracking was used during feature extraction

## 5. Feature Engineering

The final feature vector contains **74 float32 features**.

### 5.1 RGB statistics

For each RGB channel, the following were calculated:

- Mean
- Standard deviation
- Minimum
- Maximum

Total: `12` features.

Pixel values remained in the original `[0, 255]` range.

Images were converted to RGB with Pillow. The grayscale conversion was:

```text
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

### 5.2 Grayscale features

The following grayscale values were calculated:

- Brightness: grayscale mean
- Contrast: grayscale standard deviation
- Percentiles: 10th, 25th, 50th, 75th, and 90th

Total: `7` features.

### 5.3 Gradient features

Texture was calculated using absolute `numpy.diff` gradients:

- Horizontal gradient mean
- Horizontal gradient standard deviation
- Vertical gradient mean
- Vertical gradient standard deviation
- Mean gradient magnitude summary

Total: `5` features.

No padding or image resizing was applied.

### 5.4 Color differences

Signed floating-point difference maps were calculated:

```text
R - B
R - G
G - B
```

For each difference map, mean, standard deviation, minimum, and maximum were calculated.

Total: `12` features.

### 5.5 Histograms

Normalized 8-bin histograms were calculated for:

- Red
- Green
- Blue
- Grayscale

Histogram edges were:

```text
0, 32, 64, 96, 128, 160, 192, 224, 256
```

Total: `32` features.

Each histogram sums approximately to 1.

### 5.6 Time features

Six cyclical timestamp features were added:

- `hour_sin`
- `hour_cos`
- `day_sin`
- `day_cos`
- `month_sin`
- `month_cos`

Cycles:

- Hour: 24
- Day of year: 365
- Month: 12

The hour component uses the parsed timestamp hour only.

## 6. Dataset Matrix and Split

The final matrix was constructed as:

```text
X: two-dimensional NumPy float32 array
Y: one-dimensional numeric target array in kW
```

Feature count: `74`

The samples were sorted chronologically and split without shuffling:

| Split | Samples |
|---|---:|
| Train | 83,764 |
| Validation | 17,949 |
| Test | 17,950 |

The test set was not used for hyperparameter selection.

## 7. Baseline Model

The baseline estimator was:

```python
HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_depth=6,
    max_iter=300,
    min_samples_leaf=20,
    random_state=42,
)
```

The original baseline metrics were:

| Split | MAE (kW) | RMSE (kW) | R2 |
|---|---:|---:|---:|
| Train | 0.5344 | 0.9035 | 0.9886 |
| Validation | 0.8756 | 1.2418 | 0.9775 |
| Test | 1.0427 | 1.4872 | 0.9690 |

## 8. Hyperparameter Tuning

### 8.1 First grid search

A complete grid of `243` configurations was evaluated using validation MAE.

The search varied:

```text
learning_rate:     0.02, 0.05, 0.1
max_iter:          300, 500, 800
max_leaf_nodes:    15, 31, 63
min_samples_leaf:  10, 20, 50
l2_regularization: 0, 0.1, 1.0
```

The best configuration was:

```text
learning_rate     = 0.05
max_iter          = 800
max_leaf_nodes    = 63
min_samples_leaf  = 50
l2_regularization = 0.1
max_depth         = 6
```

This was combination number `161`.

Metrics:

- Validation MAE: `0.826735 kW`
- Test MAE: `1.020569 kW`
- Test RMSE: `1.473111 kW`
- Test R2: `0.969609`

The full results are stored in:

```text
artifacts/tuning_001/all_combinations.csv
artifacts/tuning_001/best_config.json
```

### 8.2 MAE-focused search

A second search evaluated `24` configurations using the same data and features. It varied:

```text
loss:           squared_error, absolute_error
max_depth:      3, 6, 10
max_bins:       128, 255
early_stopping: True, False
```

The other parameters were fixed to the first tuning result.

The best validation configuration was:

```text
loss              = absolute_error
max_depth         = 10
max_bins          = 255
early_stopping    = False
learning_rate     = 0.05
max_iter          = 800
max_leaf_nodes    = 63
min_samples_leaf  = 50
l2_regularization = 0.1
```

Metrics:

- Validation MAE: `0.779334 kW`
- Test MAE: `1.069529 kW`
- Test RMSE: `1.525401 kW`
- Test R2: `0.967413`

Although validation MAE improved, test MAE became worse. This indicates overfitting to the single validation period, so this model was not selected as the most reliable final model.

Results are stored in:

```text
artifacts/tuning_mae_001/all_combinations.csv
artifacts/tuning_mae_001/best_config.json
```

## 9. Daytime and Nighttime Models

Separate models were trained using the existing timestamp hour feature:

```text
Daytime: 06:00 through 17:59
Nighttime: 18:00 through 05:59
```

Training sample counts:

- Daytime: `72,420`
- Nighttime: `11,344`

Combined metrics:

| Split | MAE (kW) | RMSE (kW) | R2 |
|---|---:|---:|---:|
| Validation | 0.8301 | 1.1831 | 0.9795 |
| Test | 1.0796 | 1.5084 | 0.9681 |

The separate models did not improve the overall test MAE compared with the single tuned model.

Artifacts:

```text
artifacts/day_night_001/sky_cloud_pv_regressor_day.joblib
artifacts/day_night_001/sky_cloud_pv_regressor_night.joblib
artifacts/day_night_001/metrics.json
artifacts/day_night_001/config.json
```

## 10. Model Blending

The two strongest existing tuned models were blended:

- Model A: first 243-run tuned squared-error model
- Model B: MAE-focused absolute-error model

Blend weights were selected using validation MAE over weights from 0.00 to 1.00.

Selected weights:

```text
Model A: 0.29
Model B: 0.71
```

Uncorrected blend metrics:

- Validation MAE: `0.769144 kW`
- Test MAE: `1.022495 kW`
- Test RMSE: `1.469732 kW`
- Test R2: `0.969749`

The blend improved validation MAE but was slightly worse on test MAE than the single first-tuning model, whose test MAE was `1.020569 kW`.

Artifacts:

```text
artifacts/blend_001/blend_results.csv
artifacts/blend_001/best_config.json
artifacts/blend_001/test_predictions.csv
```

The blend is represented by two model files plus weights rather than by a single estimator file.

## 11. Validation Residual Correction

For the selected blend, the validation residual correction was calculated as:

```python
correction = np.median(y_validation - prediction_validation)
```

The measured correction was:

```text
-0.296008 kW
```

Metrics:

| Version | Validation MAE (kW) | Test MAE (kW) |
|---|---:|---:|
| Blend without correction | 0.7691 | 1.0225 |
| Blend with correction | 0.7167 | 1.0250 |

The correction was kept in the saved experiment because it reduced validation MAE. However, it slightly increased test MAE, so the uncorrected blend is safer when prioritizing generalization.

The complete correction details are in:

```text
artifacts/blend_001/best_config.json
```

## 12. Inference

The inference interface is implemented in `predict.py`.

Example:

```bash
.venv/bin/python predict.py \
  --image /path/to/image.jpg \
  --timestamp 2019-06-01T12:30:00 \
  --model artifacts/run_001/sky_cloud_pv_regressor.joblib \
  --schema artifacts/run_001/feature_schema.json
```

The inference path:

1. Loads the image bytes
2. Converts the image to RGB
3. Recomputes the same image features
4. Computes the six timestamp features
5. Validates feature ordering against the saved schema
6. Loads the Joblib model
7. Returns a continuous PV estimate in kW

The prediction represents the supplied image timestamp's same-minute PV estimate.

## 13. Testing

Automated tests cover:

- 14-digit timestamp extraction
- Invalid timestamp handling
- Feature schema length
- Float32 feature output
- Finite feature values
- Normalized histogram sums
- Six time features
- Chronological split boundaries

The final test run result was:

```text
4 passed
```

## 14. Reproducible Commands

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a 300-sample smoke test:

```bash
.venv/bin/python train.py \
  --images-dir /media/aidslab/Research \
  --csv /media/aidslab/Research/2019_pv_raw.csv \
  --output-dir artifacts/smoke \
  --max-samples 300
```

Run full feature extraction and baseline training:

```bash
.venv/bin/python train.py \
  --images-dir /media/aidslab/Research \
  --csv /media/aidslab/Research/2019_pv_raw.csv \
  --output-dir artifacts/run_001
```

Run the 243-combination grid search:

```bash
.venv/bin/python tune.py \
  --samples artifacts/run_001/matched_samples.csv \
  --output-dir artifacts/tuning_001
```

Run the MAE-focused search:

```bash
.venv/bin/python tune_mae.py \
  --samples artifacts/run_001/matched_samples.csv \
  --output-dir artifacts/tuning_mae_001
```

Train separate daytime/nighttime models:

```bash
.venv/bin/python train_day_night.py \
  --samples artifacts/run_001/matched_samples.csv \
  --output-dir artifacts/day_night_001
```

Run blending and residual correction:

```bash
.venv/bin/python blend_models.py \
  --samples artifacts/run_001/matched_samples.csv \
  --model-a artifacts/tuning_001/sky_cloud_pv_regressor_tuned.joblib \
  --model-b artifacts/tuning_mae_001/sky_cloud_pv_regressor_mae_tuned.joblib \
  --output-dir artifacts/blend_001
```

Run tests:

```bash
.venv/bin/pytest -q
```

## 15. Final Model Decision

The model with the lowest validation MAE was the residual-corrected blend:

```text
Validation MAE: 0.7167 kW
```

However, test performance is the more important generalization check. The best measured test MAE was produced by the first 243-run tuned single model:

```text
Test MAE: 1.0206 kW
Test RMSE: 1.4731 kW
Test R2: 0.9696
```

The residual-corrected blend was not selected as the final general-purpose model because its test MAE was `1.0250 kW`, slightly worse than the single tuned model.

## 16. Limitations

- The baseline uses handcrafted image statistics rather than a CNN or Vision Transformer.
- The model estimates same-minute output, not future output.
- Some image members were unreadable because of corrupted or truncated archive data.
- A single chronological validation block can lead to validation overfitting.
- The data represents one site and one year, limiting cross-site and cross-year conclusions.
- No weather variables were used.
- Image quality, sensor noise, occlusion, and lighting changes may affect performance.
- The project artifacts directory is ignored by Git because it contains large generated files and trained models.

## 17. Recommended Next Step

For a more reliable improvement, use multiple chronological validation windows and select a model by average validation MAE. This should be done before using the test set for final evaluation. The existing single-period validation results should not be described as proof that the model will generalize to another year, site, or weather regime.
