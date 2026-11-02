import numpy as np
import pandas as pd

from m5fc import baselines
from m5fc.data import Hierarchy, first_nonzero, to_long


def _hier(Y):
    Y = np.asarray(Y, dtype=np.float32)
    ids = pd.DataFrame({"unique_id": [f"s{i}" for i in range(len(Y))], "level": 12})
    return Hierarchy(ids, Y, pd.date_range("2020-01-01", periods=Y.shape[1], freq="D"))


def test_seasonal_naive_repeats_last_week():
    y = np.arange(21)
    fc = baselines.seasonal_naive(_hier([y]), horizon=10)
    np.testing.assert_array_equal(fc[0], [14, 15, 16, 17, 18, 19, 20, 14, 15, 16])


def test_first_nonzero_and_trim():
    Y = [[0, 0, 3] + [1] * 20, [0] * 23]
    np.testing.assert_array_equal(first_nonzero(np.array(Y)), [2, 23])
    long = to_long(_hier(Y))
    counts = long.groupby("unique_id", observed=True).size()
    assert counts["s0"] == 21  # leading zeros dropped
    assert counts["s1"] == 14  # all-zero series keeps two seasons


def test_to_long_max_days():
    long = to_long(_hier([np.ones(100)]), max_days=30)
    assert len(long) == 30
    assert long["ds"].iloc[-1] == pd.Timestamp("2020-01-01") + pd.Timedelta(days=99)


def test_ets_and_arima_shapes_and_nonnegative():
    rng = np.random.default_rng(0)
    week = np.array([5, 4, 4, 5, 7, 9, 8], dtype=float)
    Y = np.vstack([np.tile(week, 20) + rng.normal(0, 0.5, 140), rng.poisson(0.3, 140)])
    h = _hier(Y)
    for fn in (baselines.ets, baselines.arima):
        fc = fn(h, horizon=14, n_jobs=1)
        assert fc.shape == (2, 14)
        assert np.isfinite(fc).all() and (fc >= 0).all()
        # A strongly weekly series should be forecast close to its weekly pattern.
        assert np.abs(fc[0, :7] - week).mean() < 1.5
