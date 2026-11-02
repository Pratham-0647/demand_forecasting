import numpy as np
import pandas as pd
import pytest

from m5fc import metrics


def test_mase_scale_matches_manual_lag7():
    rng = np.random.default_rng(1)
    y = rng.poisson(3, 50).astype(float)
    y[0] = 1.0
    expected = np.abs(y[7:] - y[:-7]).mean()
    assert metrics.mase_scale(y[None, :])[0] == pytest.approx(expected)


def test_mase_scale_ignores_leading_zeros():
    tail = np.array([1, 2, 3, 4, 5, 6, 7, 2, 2, 3, 4, 5, 6, 7], dtype=float)
    y = np.concatenate([np.zeros(30), tail])
    expected = np.abs(tail[7:] - tail[:-7]).mean()  # = 1/7
    assert metrics.mase_scale(y[None, :])[0] == pytest.approx(expected)


def test_mase_scale_undefined_cases():
    constant = np.ones(30)
    too_short = np.r_[np.zeros(25), np.ones(5)]
    scale = metrics.mase_scale(np.vstack([constant, too_short]))
    assert np.isnan(scale).all()


def test_mase_one_when_forecast_error_equals_scale():
    y_true = np.array([[2.0, 4.0]])
    y_pred = np.array([[3.0, 3.0]])  # MAE = 1
    assert metrics.mase(y_true, y_pred, np.array([1.0]))[0] == pytest.approx(1.0)
    assert metrics.mase(y_true, y_pred, np.array([2.0]))[0] == pytest.approx(0.5)


def test_wape_is_volume_weighted():
    y_true = np.array([[100.0, 100.0], [1.0, 1.0]])
    y_pred = np.array([[90.0, 110.0], [0.0, 0.0]])
    # (10 + 10 + 1 + 1) / 202
    assert metrics.wape(y_true, y_pred) == pytest.approx(22 / 202)
    np.testing.assert_allclose(metrics.series_wape(y_true, y_pred), [0.1, 1.0])
    assert np.isnan(metrics.wape(np.zeros((1, 3)), np.ones((1, 3))))


def test_mape_breaks_on_zero_actuals():
    y_true = np.array([[0.0, 2.0, 0.0]])
    assert not np.isfinite(metrics.mape(y_true, np.ones((1, 3)))[0])
    assert metrics.zero_actual_share(y_true) == pytest.approx(2 / 3)


def test_evaluate_level_perfect_forecast_and_reference():
    rng = np.random.default_rng(2)
    y_train = rng.poisson(5, (4, 60)).astype(float)
    y_true = rng.poisson(5, (4, 28)).astype(float)
    snaive = np.tile(y_train[:, -7:], 4)
    summary, per_series = metrics.evaluate_level(
        y_train, y_true, {"seasonal_naive": snaive, "oracle": y_true}, level=3)
    oracle = summary.set_index("model").loc["oracle"]
    assert oracle["mase_mean"] == 0 and oracle["wape"] == 0
    assert oracle["share_beats_snaive"] == 1.0
    assert summary.set_index("model").loc["seasonal_naive", "share_beats_snaive"] == 0.0
    assert per_series.shape == (4, 2)
    ov = metrics.overall(pd.concat([summary, summary.assign(level=4)]))
    assert list(ov["model"]) == ["seasonal_naive", "oracle"]


def test_pinball_and_coverage():
    y = np.array([[0.0, 10.0]])
    q = np.array([[2.0, 2.0]])
    # alpha=0.9: over by 2 -> 0.1*2, under by 8 -> 0.9*8
    assert metrics.pinball(y, q, 0.9)[0] == pytest.approx((0.2 + 7.2) / 2)
    assert metrics.pinball(y, q, 0.5)[0] == pytest.approx(metrics.mae(y, q)[0] / 2)
    assert metrics.coverage(y, q) == 0.5
    assert metrics.interval_coverage(y, np.zeros_like(y), q) == 0.5


def test_pinball_minimised_by_true_quantile():
    y = np.random.default_rng(4).exponential(1.0, (1, 20000))
    losses = {c: metrics.pinball(y, np.full_like(y, c), 0.9)[0] for c in np.linspace(1.5, 3.1, 33)}
    best = min(losses, key=losses.get)
    assert abs(best - np.log(10)) < 0.15  # true 0.9-quantile of Exp(1)
