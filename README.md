# TS01 — M4 classical benchmark forecaster

Classical time-series benchmark on the M4 Forecasting Competition data:
Theta, ETS, and ARIMA combinations against naive baselines. No neural networks.

Part of the time-series systems portfolio plan. Status: phase 0 (evaluation harness).

## Layout

- `src/forecaster/evaluation/` — MASE, sMAPE, and rolling-origin backtest folds
- `tests/` — split and metric tests
- `data/` — empty; the M4 data is read from Kaggle, not committed

## Data

Dataset: [yogesh94/m4-forecasting-competition-dataset](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset) (CC0 on Kaggle; check the original M4 terms).

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```
