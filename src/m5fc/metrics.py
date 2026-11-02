"""Phase 2 metrics: MASE and WAPE (and why not MAPE).

All functions work on arrays shaped (n_series, horizon); ``y_train`` is
(n_series, n_train_days).

MASE
    MAE of the forecast divided by the in-sample MAE of the seasonal naive
    (lag 7) on the training data. Below 1 means the forecast's out-of-sample
    error is smaller than the naive method's typical in-sample one-step-ahead
    error. Scale-free, so it can be averaged across series of any volume.
    The scale is computed from each series' first sale onward (leading zeros
    mean "not stocked yet"). Because the denominator is in-sample, the seasonal
    naive itself does not score exactly 1 on the test window.

WAPE
    sum |error| / sum |actual| over all series and days of a level, i.e. a
    volume-weighted error: high sellers dominate, as they do in revenue terms.

MAPE
    Undefined whenever the actual is zero, which is 68% of bottom-level days.
    ``mape`` is kept only to make that point measurable (``zero_actual_share``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SEASON, first_nonzero


def mase_scale(y_train: np.ndarray, m: int = SEASON) -> np.ndarray:
    """In-sample MAE of the lag-m seasonal naive, from each series' first sale onward.

    Returns NaN for series with fewer than m+1 observations after the first sale
    and for series whose scale is exactly zero (constant series).
    """
    y_train = np.asarray(y_train, dtype=np.float64)
    T = y_train.shape[1]
    start = first_nonzero(y_train)
    diffs = np.abs(y_train[:, m:] - y_train[:, :-m])  # diffs[:, j] compares t=j+m with t=j
    valid = np.arange(m, T)[None, :] >= (start[:, None] + m)  # both points after first sale
    n = valid.sum(axis=1)
    total = np.where(valid, diffs, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = total / n
    scale[(n == 0) | (scale == 0)] = np.nan
    return scale


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)).mean(axis=1)


def mase(y_true: np.ndarray, y_pred: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Per-series MASE. NaN where the scale is undefined."""
    return mae(y_true, y_pred) / scale


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Volume-weighted error across every series and day passed in."""
    y_true = np.asarray(y_true, float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - np.asarray(y_pred, float)).sum() / denom) if denom > 0 else np.nan


def series_wape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, float)
    denom = np.abs(y_true).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.abs(y_true - np.asarray(y_pred, float)).sum(axis=1) / denom
    out[denom == 0] = np.nan
    return out


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-series MAPE — infinite/NaN as soon as any actual is zero. Do not use on M5."""
    y_true = np.asarray(y_true, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.abs(y_true - np.asarray(y_pred, float)) / np.abs(y_true)).mean(axis=1)


def zero_actual_share(y_true: np.ndarray) -> float:
    return float((np.asarray(y_true) == 0).mean())


def evaluate_level(
    y_train: np.ndarray, y_true: np.ndarray, forecasts: dict[str, np.ndarray], level: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every model on one level.

    Returns (summary, per_series): summary has one row per model; per_series
    holds per-series MASE for later analysis.
    """
    scale = mase_scale(y_train)
    ref = forecasts.get("seasonal_naive")
    ref_mase = mase(y_true, ref, scale) if ref is not None else None
    rows, per_series = [], {}
    for model, fc in forecasts.items():
        s_mase = mase(y_true, fc, scale)
        ok = np.isfinite(s_mase)
        per_series[model] = s_mase
        row = {
            "level": level,
            "model": model,
            "n_series": len(s_mase),
            "n_scored": int(ok.sum()),
            "mase_mean": float(np.mean(s_mase[ok])),
            "mase_median": float(np.median(s_mase[ok])),
            "wape": wape(y_true, fc),
        }
        if ref_mase is not None:
            both = ok & np.isfinite(ref_mase)
            row["share_beats_snaive"] = float((s_mase[both] < ref_mase[both]).mean())
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_series)


def overall(summary: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average across levels (M5 convention: each level counts the same)."""
    cols = ["mase_mean", "mase_median", "wape"] + (
        ["share_beats_snaive"] if "share_beats_snaive" in summary else [])
    out = summary.groupby("model", sort=False)[cols].mean().reset_index()
    out.insert(0, "level", "avg(1-12)")
    return out
