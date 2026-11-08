import pytest

from forecaster.evaluation.backtest import rolling_origin_folds


def test_folds_are_ordered_and_end_at_n():
    folds = list(rolling_origin_folds(n=100, horizon=10, n_folds=3))
    assert [f.test_end for f in folds] == [80, 90, 100]
    assert all(f.test_end - f.test_start == 10 for f in folds)


def test_train_never_overlaps_test():
    for f in rolling_origin_folds(n=100, horizon=5, n_folds=4, gap=2):
        assert f.train_end <= f.test_start
        assert f.test_start - f.train_end == 2


def test_too_short_series_raises():
    with pytest.raises(ValueError):
        list(rolling_origin_folds(n=20, horizon=10, n_folds=3))
