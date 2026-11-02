"""Phase 4/5: global LightGBM models trained across all 30,490 bottom series.

One model per objective: Tweedie for the point forecast (count data with many
zeros), and quantile regression for P10/P50/P90. Bottom-level forecasts are
summed up the hierarchy for upper levels (bottom-up) — see m5fc.reconcile.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import numpy as np

from .data import HORIZON
from .features import CAT_FEATURES, Panel, build_features

# Tuned for the 4-core / 14 GB container: a full-scale round with 255 leaves at
# lr 0.05 took ~7 s, so use a higher learning rate, fewer leaves and 50% row
# bagging. Early stopping on the last 28 training days picks the round count.
BASE_PARAMS = dict(
    learning_rate=0.1,
    num_leaves=127,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.5,
    bagging_freq=1,
    lambda_l2=0.1,
    max_bin=255,
    verbose=-1,
    num_threads=0,
    seed=42,
)

DEFAULT_ROUNDS = 400

OBJECTIVES = {
    "point": dict(objective="tweedie", tweedie_variance_power=1.1),
    "q10": dict(objective="quantile", alpha=0.1),
    "q50": dict(objective="quantile", alpha=0.5),
    "q90": dict(objective="quantile", alpha=0.9),
}


def fit_predict(
    panel: Panel,
    origin: int,
    targets: tuple[str, ...] = ("point", "q10", "q50", "q90"),
    train_days: int = 728,
    num_rounds: int = DEFAULT_ROUNDS,
    valid_days: int = 28,
    log=print,
) -> tuple[dict[str, np.ndarray], dict[str, lgb.Booster]]:
    """Train one model per target and return bottom-level forecasts (n_series, 28).

    The last ``valid_days`` of training are held out for early stopping only;
    that window is strictly before the origin, so the test period is never seen.
    """
    t0 = time.perf_counter()
    X, names, y, rows = build_features(panel, origin, train_days=train_days)
    log(f"  features: {X.shape[0]:,} rows x {X.shape[1]} cols in {time.perf_counter() - t0:.0f}s")

    # Rows are sorted by day, so the three windows are contiguous slices (views).
    day = rows["day"].to_numpy()
    a = int(np.searchsorted(day, origin - valid_days))
    b = int(np.searchsorted(day, origin))
    cat = [names.index(c) for c in CAT_FEATURES if c in names]
    dtrain = lgb.Dataset(X[:a], y[:a], feature_name=names, categorical_feature=cat)
    dvalid = lgb.Dataset(X[a:b], y[a:b], feature_name=names, categorical_feature=cat, reference=dtrain)
    dtrain.construct()
    dvalid.construct()
    X_test = X[b:].copy()
    del X  # LightGBM keeps its own binned copy
    test_series = rows["series"].to_numpy()[b:]
    test_h = day[b:] - origin

    out, models = {}, {}
    for tgt in targets:
        t1 = time.perf_counter()
        params = {**BASE_PARAMS, **OBJECTIVES[tgt]}
        model = lgb.train(
            params, dtrain, num_boost_round=num_rounds, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        pred = np.clip(model.predict(X_test, num_iteration=model.best_iteration), 0, None)
        fc = np.zeros((panel.n_series, HORIZON), np.float32)  # not on shelf -> 0
        fc[test_series, test_h] = pred
        out[tgt] = fc
        models[tgt] = model
        log(f"  {tgt:5s}: best_iter={model.best_iteration} in {time.perf_counter() - t1:.0f}s")
    return out, models
