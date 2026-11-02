"""Phase 4 features for the global LightGBM model (bottom level, 30,490 series).

Leakage guards
--------------
* The sales matrix is truncated at the forecast origin T **before** any feature
  is computed, so no feature can see d >= T, whatever the bug.
* Sales features use lags of at least 28 days (the horizon). That lets one model
  forecast all 28 days directly with no recursion: for a test day T+27 the
  lag-28 value is day T-1, the last observed day. Lags 7 and 14 would need
  recursive forecasting (feeding predictions back in) or would leak actuals.
* Price features are causal: price relative to the running max of past prices,
  never a full-series mean or max. Prices for the forecast window are assumed
  to be known in advance (planned prices). Event and SNAP flags come from the
  calendar, which is published ahead of time in M5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DATA_DIR, HORIZON, ID_COLS, load_calendar

SALES_LAGS = (28, 35, 42, 49, 56, 91, 364)
ROLL_WINDOWS = (7, 28, 56, 112)
CAT_FEATURES = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1"]


@dataclass
class Panel:
    """Static inputs shared by every origin."""

    meta: pd.DataFrame  # ID_COLS for the 30,490 bottom series
    Y: np.ndarray  # (n_series, n_days) float32 sales
    price: np.ndarray  # (n_series, n_days + 28) float32 daily sell price (NaN = not on sale)
    cal: pd.DataFrame  # calendar, one row per day (1,969 rows)
    snap: np.ndarray  # (n_series, n_days + 28) SNAP flag for the series' state
    codes: dict[str, np.ndarray]  # categorical codes per series

    @property
    def n_series(self) -> int:
        return self.Y.shape[0]


def load_panel(data_dir=DATA_DIR) -> Panel:
    sales = pd.read_parquet(data_dir / "sales_train_evaluation.parquet")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    meta = sales[ID_COLS].reset_index(drop=True)
    Y = sales[day_cols].to_numpy(np.float32)
    cal = load_calendar(data_dir)
    n_days = len(cal)

    prices = pd.read_parquet(data_dir / "sell_prices.parquet")
    key = meta["item_id"] + "|" + meta["store_id"]
    prices["key"] = prices["item_id"] + "|" + prices["store_id"]
    weeks = np.sort(cal["wm_yr_wk"].unique())
    wk_idx = pd.Index(weeks)
    pw = np.full((len(meta), len(weeks)), np.nan, np.float32)
    rows = pd.Index(key).get_indexer(prices["key"])
    cols = wk_idx.get_indexer(prices["wm_yr_wk"])
    pw[rows, cols] = prices["sell_price"].to_numpy(np.float32)
    price = pw[:, wk_idx.get_indexer(cal["wm_yr_wk"])]

    snap_cols = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    snap_by_state = {s: cal[c].to_numpy(np.int8) for s, c in snap_cols.items()}
    snap = np.vstack([snap_by_state[s] for s in meta["state_id"]])

    for c in ["event_name_1", "event_type_1"]:
        cal[c] = cal[c].fillna("none")
    codes = {c: pd.factorize(meta[c], sort=True)[0].astype(np.int32) for c in ID_COLS}
    assert price.shape[1] == n_days
    return Panel(meta, Y, price, cal, snap, codes)


def _shifted_rolling_mean(Ym: np.ndarray, window: int, shift: int) -> np.ndarray:
    """mean(Ym[:, d-shift-window+1 : d-shift+1]) for every d; NaN where undefined."""
    n, T = Ym.shape
    filled = np.nan_to_num(Ym)
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(filled, axis=1, dtype=np.float64)], axis=1)
    out = np.full((n, T + HORIZON), np.nan, np.float32)
    d = np.arange(T + HORIZON)
    end = d - shift + 1  # exclusive end index into Ym
    start = end - window
    ok = (start >= 0) & (end <= T)
    out[:, ok] = ((cs[:, end[ok]] - cs[:, start[ok]]) / window).astype(np.float32)
    return out


def build_features(panel: Panel, origin: int, train_days: int = 728) -> tuple[np.ndarray, list[str], np.ndarray, pd.DataFrame]:
    """Feature table for days [origin - train_days, origin + 28).

    Returns (X, names, y, rows): ``rows`` holds series index and day index for each
    row; ``y`` is NaN on the 28 forecast days. X is one float32 block filled column
    by column, so peak memory stays near the size of X itself (~2.6 GB at full
    scale). Rows are day-major (sorted by day), so any day range — e.g. the train,
    validation and test windows — is a contiguous slice, i.e. a view, not a copy.
    """
    Yt = panel.Y[:, :origin].astype(np.float32)  # truncated: nothing at or after the origin exists here
    n, T = Yt.shape
    days = np.arange(origin - train_days, origin + HORIZON)
    n_days = len(days)

    first_sale = np.where((Yt != 0).any(axis=1), (Yt != 0).argmax(axis=1), T)
    # Leading zeros before the first sale are "not stocked", not zero demand.
    Ym = Yt.copy()
    Ym[np.arange(T)[None, :] < first_sale[:, None]] = np.nan
    price = panel.price[:, : origin + HORIZON]

    # Keep rows after the first sale with a valid price (otherwise the item is not on shelf).
    # Day-major flattening: element (i, j) of an (n, n_days) array -> row j * n + i.
    series_idx = np.tile(np.arange(n), n_days)
    day_idx = np.repeat(days, n)
    keep = (day_idx >= first_sale[series_idx]) & np.isfinite(price[:, days].T.reshape(-1))
    n_rows = int(keep.sum())

    cal = panel.cal.iloc[days]
    names = ([f"lag_{k}" for k in SALES_LAGS] + [f"rmean_{w}" for w in ROLL_WINDOWS]
             + ["zero_share_28", "rstd_28", "sell_price", "price_rel_max", "price_change_wk",
                "days_since_first_sale", "snap", "wday", "month", "year", "mday",
                "event_name_1", "event_type_1"] + ID_COLS)
    X = np.empty((n_rows, len(names)), np.float32)
    col = iter(range(len(names)))

    def put(arr2d):  # (n, n_days) -> next column
        X[:, next(col)] = arr2d.T.reshape(-1)[keep]

    def put_day(arr1d):  # per-day value, same for every series
        put(np.broadcast_to(np.asarray(arr1d, np.float32)[None, :], (n, n_days)))

    def put_series(arr1d):  # per-series value, same for every day
        put(np.broadcast_to(np.asarray(arr1d, np.float32)[:, None], (n, n_days)))

    Yext = np.concatenate([Ym, np.full((n, HORIZON), np.nan, np.float32)], axis=1)
    for k in SALES_LAGS:
        lag = np.full((n, n_days), np.nan, np.float32)
        src = days - k
        ok = src >= 0
        lag[:, ok] = Yext[:, src[ok]]
        put(lag)
    del Yext
    rmean_28 = None
    for w in ROLL_WINDOWS:
        rm = _shifted_rolling_mean(Ym, w, HORIZON)[:, days]
        put(rm)
        if w == 28:
            rmean_28 = rm
    # Recent intermittency: share of zero days in the 28 days ending at lag 28.
    zero = np.where(np.isnan(Ym), np.nan, (Ym == 0).astype(np.float32))
    put(_shifted_rolling_mean(zero, 28, HORIZON)[:, days])
    del zero
    sq = _shifted_rolling_mean(Ym ** 2, 28, HORIZON)[:, days]
    put(np.sqrt(np.clip(sq - rmean_28 ** 2, 0, None)))
    del sq, rmean_28

    run_max = pd.DataFrame(price.T).cummax().to_numpy(np.float32).T  # causal running max
    prev_week = np.concatenate([np.full((n, 7), np.nan, np.float32), price[:, :-7]], axis=1)
    put(price[:, days])
    put((price / run_max)[:, days])
    put((price / prev_week)[:, days])
    del run_max, prev_week
    put((days[None, :] - first_sale[:, None]).astype(np.float32))
    put(panel.snap[:, days])

    put_day(cal["wday"].to_numpy())
    put_day(cal["month"].to_numpy())
    put_day(cal["year"].to_numpy())
    put_day(cal["date"].dt.day.to_numpy())
    put_day(pd.Categorical(panel.cal["event_name_1"]).codes[days])
    put_day(pd.Categorical(panel.cal["event_type_1"]).codes[days])
    for c in ID_COLS:
        put_series(panel.codes[c])
    assert next(col, None) is None

    target = np.full((n, n_days), np.nan, np.float32)
    train_part = days < origin
    target[:, train_part] = Yt[:, days[train_part]]
    y = target.T.reshape(-1)[keep]
    rows = pd.DataFrame({"series": series_idx[keep].astype(np.int32), "day": day_idx[keep].astype(np.int32)})
    return X, names, y, rows
