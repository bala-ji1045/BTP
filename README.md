# Sky/Cloud PV Power Regression

This repository implements the documented handcrafted-feature baseline using `HistGradientBoostingRegressor`. It estimates PV output for the same minute as the supplied image; it is not future forecasting.

Implementation decisions not specified by the project brief are recorded in `feature_schema.json` and `run_config.json`: first image by sorted archive/member for duplicate minutes, mean PV target for duplicate CSV minutes, original `[0,255]` pixels, five grayscale percentiles (`10,25,50,75,90`), absolute `np.diff` gradients, and normalized 8-bin histograms.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Smoke test and training

The SD-card dataset used here is mounted at `/media/aidslab/Research`.

```bash
python train.py --images-dir /media/aidslab/Research --csv /media/aidslab/Research/2019_pv_raw.csv --output-dir artifacts/smoke --max-samples 300
python train.py --images-dir /media/aidslab/Research --csv /media/aidslab/Research/2019_pv_raw.csv --output-dir artifacts/run_001
```

Archives are streamed with `tarfile`; they are not extracted to disk. The model uses a chronological 70/15/15 split without shuffling.

## Inference

```bash
python predict.py --image /path/to/image.jpg --timestamp 2019-06-01T12:30:00 --model artifacts/run_001/sky_cloud_pv_regressor.joblib --schema artifacts/run_001/feature_schema.json
```

## Tests

```bash
pytest -q
```