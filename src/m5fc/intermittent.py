"""Phase 6: intermittent-demand methods and demand classification.

Croston's method splits demand into non-zero size and the interval between
non-zero sales, smooths each with SES, and forecasts size / interval. SBA
(Syntetos-Boylan) corrects Croston's upward bias by (1 - alpha/2). TSB smooths
the probability of a sale instead of the interval, so it decays towards zero
for items that have stopped selling (obsolescence), which Croston cannot do.

Series are classified with the Syntetos-Boylan-Croston scheme on the training
window (from first sale): ADI = mean interval between sales, CV2 = squared
coefficient of variation of non-zero sizes; cut-offs ADI 1.32, CV2 0.49.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import HORIZON, Hierarchy, first_nonzero

ADI_CUT, CV2_CUT = 1.32, 0.49


def classify(y_train: np.ndarray) -> pd.DataFrame:
    """ADI, CV2 and SBC class per series ('smooth', 'erratic', 'intermittent', 'lumpy')."""
    y_train = np.asarray(y_train, float)
    start = first_nonzero(y_train)
    n, T = y_train.shape
    after = np.arange(T)[None, :] >= start[:, None]
    nz = (y_train != 0) & after
    n_nz = nz.sum(axis=1)
    n_days = (T - start).clip(min=1)
    adi = np.where(n_nz > 0, n_days / np.maximum(n_nz, 1), np.inf)
    sizes = np.where(nz, y_train, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nanmean(sizes, axis=1)
        cv2 = np.nanvar(sizes, axis=1) / mean ** 2
    cv2 = np.nan_to_num(cv2, nan=0.0)
    cls = np.select(
        [(adi < ADI_CUT) & (cv2 < CV2_CUT), (adi < ADI_CUT), (cv2 < CV2_CUT)],
        ["smooth", "erratic", "intermittent"], default="lumpy")
    return pd.DataFrame({"adi": adi, "cv2": cv2, "sbc_class": cls,
                         "zero_share": 1 - n_nz / n_days})


def _sf(h: Hierarchy, model, alias: str, max_train_days: int | None, n_jobs: int) -> np.ndarray:
    from .baselines import _statsforecast

    return _statsforecast(h, model, alias, HORIZON, max_train_days, n_jobs)


def croston(h: Hierarchy, horizon: int = HORIZON, max_train_days: int | None = None, n_jobs: int = -1):
    from statsforecast.models import CrostonClassic

    return _sf(h, CrostonClassic(), "CrostonClassic", max_train_days, n_jobs)


def croston_sba(h: Hierarchy, horizon: int = HORIZON, max_train_days: int | None = None, n_jobs: int = -1):
    from statsforecast.models import CrostonSBA

    return _sf(h, CrostonSBA(), "CrostonSBA", max_train_days, n_jobs)


def tsb(h: Hierarchy, horizon: int = HORIZON, max_train_days: int | None = None, n_jobs: int = -1):
    from statsforecast.models import TSB

    return _sf(h, TSB(alpha_d=0.1, alpha_p=0.1), "TSB", max_train_days, n_jobs)


MODELS = {"croston": croston, "croston_sba": croston_sba, "tsb": tsb}
