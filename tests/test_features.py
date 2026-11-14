import numpy as np
import pandas as pd

from m5fc.data import HORIZON, ID_COLS
from m5fc.features import Panel, build_features


def _panel(n=4, n_days=500, seed=0):
    rng = np.random.default_rng(seed)
    Y = rng.poisson(2, (n, n_days)).astype(np.float32)
    Y[0, :50] = 0  # late launch
    total = n_days + HORIZON
    dates = pd.date_range("2015-01-01", periods=total, freq="D")
    cal = pd.DataFrame({
        "date": dates, "wday": dates.dayofweek + 1, "month": dates.month, "year": dates.year,
        "wm_yr_wk": np.arange(total) // 7,
        "event_name_1": "none", "event_type_1": "none",
    })
    meta = pd.DataFrame({c: [f"{c}{i % 2}" for i in range(n)] for c in ID_COLS})
    price = np.full((n, total), 2.0, np.float32)
    price[1, 300:] = 1.5
    codes = {c: pd.factorize(meta[c], sort=True)[0].astype(np.int32) for c in ID_COLS}
    return Panel(meta, Y, price, cal, np.zeros((n, total), np.int8), codes)


def test_features_do_not_depend_on_future_sales():
    p = _panel()
    origin = 400
    X1, _, y1, r1 = build_features(p, origin, train_days=200)
    p2 = _panel()
    p2.Y[:, origin:] = 999.0  # scramble everything at/after the origin
    X2, _, y2, r2 = build_features(p2, origin, train_days=200)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_lag_and_target_alignment():
    p = _panel()
    origin = 400
    X, names, y, rows = build_features(p, origin, train_days=200)
    X = pd.DataFrame(X, columns=names)
    assert rows["day"].is_monotonic_increasing  # day-major layout
    r = rows.reset_index(drop=True)
    i = np.flatnonzero((r["series"] == 2) & (r["day"] == origin - 1))[0]
    assert y[i] == p.Y[2, origin - 1]
    assert X.loc[i, "lag_28"] == p.Y[2, origin - 1 - 28]
    assert X.loc[i, "rmean_7"] == np.float32(p.Y[2, origin - 1 - 28 - 6: origin - 1 - 28 + 1].mean())
    # Forecast rows: target unknown, last horizon day's lag_28 is the last observed day.
    j = np.flatnonzero((r["series"] == 2) & (r["day"] == origin + HORIZON - 1))[0]
    assert np.isnan(y[j]) and X.loc[j, "lag_28"] == p.Y[2, origin - 1]


def test_rows_before_first_sale_dropped_and_price_causal():
    p = _panel()
    X, names, y, rows = build_features(p, 400, train_days=380)
    X = pd.DataFrame(X, columns=names)
    assert rows.loc[rows["series"] == 0, "day"].min() >= 50
    s1 = X[rows["series"].to_numpy() == 1]
    d1 = rows.loc[rows["series"] == 1, "day"].to_numpy()
    assert np.allclose(s1.loc[d1 >= 300, "price_rel_max"], 0.75)
    assert np.allclose(s1.loc[d1 < 300, "price_rel_max"], 1.0)
