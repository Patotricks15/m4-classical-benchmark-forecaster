import numpy as np
import pytest

from forecaster.evaluation.metrics import mase, smape


def test_mase_of_naive_forecast_on_train_is_one():
    y_train = np.array([1.0, 2.0, 4.0, 7.0, 11.0])
    y_true = np.array([16.0, 22.0])
    y_pred = np.array([11.0, 11.0])
    expected = np.mean(np.abs(y_true - y_pred)) / np.mean(np.abs(np.diff(y_train)))
    assert mase(y_true, y_pred, y_train) == pytest.approx(expected)


def test_mase_zero_scale_raises():
    with pytest.raises(ValueError):
        mase(np.array([1.0]), np.array([1.0]), np.array([3.0, 3.0, 3.0]))


def test_smape_perfect_forecast_is_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y) == pytest.approx(0.0)
