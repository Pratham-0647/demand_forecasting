"""Phase 1 baselines: seasonal naive, ETS and ARIMA.

Every forecaster takes a training ``Hierarchy`` (days before the forecast origin)
and returns an array of shape (n_series, horizon), rows aligned with ``h.ids``.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from .data import HORIZON, SEASON, Hierarchy, to_long


def seasonal_naive(h: Hierarchy, horizon: int = HORIZON, season: int = SEASON) -> np.ndarray:
    """Forecast = value from ``season`` days ago, i.e. repeat the last observed week."""
    last_season = h.Y[:, -season:]
    reps = int(np.ceil(horizon / season))
    return np.tile(last_season, reps)[:, :horizon].astype(np.float32)


def _statsforecast(
    h: Hierarchy,
    model,
    alias: str,
    horizon: int,
    max_train_days: int | None,
    n_jobs: int,
) -> np.ndarray:
    from statsforecast import StatsForecast

    df = to_long(h, max_days=max_train_days)
    sf = StatsForecast(models=[model], freq="D", n_jobs=n_jobs, fallback_model=None)
    fc = sf.forecast(df=df, h=horizon)
    fc["unique_id"] = fc["unique_id"].astype(str)
    wide = fc.pivot(index="unique_id", columns="ds", values=alias)
    wide = wide.reindex(h.ids["unique_id"].to_numpy())
    out = wide.to_numpy(dtype=np.float32)
    # Demand cannot be negative; clipping is a no-op for valid forecasts.
    return np.clip(np.nan_to_num(out, nan=0.0), 0.0, None)


def ets(h: Hierarchy, horizon: int = HORIZON, max_train_days: int | None = None, n_jobs: int = -1) -> np.ndarray:
    """AutoETS with weekly seasonality (model selected per series by AICc)."""
    from statsforecast.models import AutoETS

    return _statsforecast(h, AutoETS(season_length=SEASON), "AutoETS", horizon, max_train_days, n_jobs)


# Restricted AutoARIMA search. The full default search cost ~1.2 s/series (14 h for
# all 42,840 series on 4 cores); capping orders and using the CSS approximation
# during order selection brings it to ~0.13 s/series. Daily retail demand rarely
# needs p, q > 2 or seasonal P, Q > 1 once weekly seasonality is modelled.
ARIMA_KWARGS = dict(max_p=2, max_q=2, max_P=1, max_Q=1, approximation=True)


def arima(h: Hierarchy, horizon: int = HORIZON, max_train_days: int | None = None, n_jobs: int = -1) -> np.ndarray:
    """AutoARIMA with weekly seasonality (orders selected per series by AICc)."""
    from statsforecast.models import AutoARIMA

    model = AutoARIMA(season_length=SEASON, **ARIMA_KWARGS)
    return _statsforecast(h, model, "AutoARIMA", horizon, max_train_days, n_jobs)


MODELS = {
    "seasonal_naive": seasonal_naive,
    "ets": ets,
    "arima": arima,
}


def all_models() -> dict:
    """Phase 1 baselines plus the Phase 6 intermittent-demand methods."""
    from .intermittent import MODELS as INTERMITTENT

    return {**MODELS, **INTERMITTENT}


def forecast_frame(h: Hierarchy, fc: np.ndarray, model: str, origin_date: pd.Timestamp) -> pd.DataFrame:
    """Wide forecast table: one row per series, columns F1..F{horizon}."""
    cols = {f"F{i + 1}": fc[:, i] for i in range(fc.shape[1])}
    return pd.DataFrame(
        {"unique_id": h.ids["unique_id"].to_numpy(), "level": h.ids["level"].to_numpy(),
         "model": model, "origin": origin_date, **cols}
    )


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def default_jobs() -> int:
    return os.cpu_count() or 1
