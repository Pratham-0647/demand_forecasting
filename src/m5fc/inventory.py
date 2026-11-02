"""Phase 8: turn quantile forecasts into reorder points and price the trade-off.

Policy (per item x store): continuous review with a 7-day replenishment lead
time. When stock falls to the reorder point (ROP) an order is placed, so the
ROP must cover demand over the lead time with the target probability:

    ROP(z) = mu_L + z * sigma_L

mu_L  = sum of daily point forecasts over the lead time
sigma_L = sqrt(sum of daily variances), daily sigma read off the forecast's
          own quantile spread: (P90 - P10) / (2 * 1.2816)  (assumes independent days)

z = 1.2816 is the "P90 policy" (90% cycle service level); sweeping z traces the
service-level vs. inventory-cost curve.

Evaluated on realised demand in the four non-overlapping 7-day windows of the
test period:
  cycle service level  share of windows with lead-time demand <= ROP
  fill rate            1 - units short / units demanded
  safety stock         ROP - mu_L (units held beyond expected demand), valued at
                       sell price x annual holding rate / 52 per week
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

Z90 = norm.ppf(0.9)


def lead_time_windows(daily: np.ndarray, lead: int = 7) -> np.ndarray:
    """(n, h) daily values -> (n, h // lead) sums over consecutive non-overlapping windows."""
    n, h = daily.shape
    k = h // lead
    return daily[:, : k * lead].reshape(n, k, lead).sum(axis=2)


def lead_time_sigma(q10: np.ndarray, q90: np.ndarray, lead: int = 7) -> np.ndarray:
    sigma_d = np.clip(q90 - q10, 0, None) / (2 * Z90)
    return np.sqrt(lead_time_windows(sigma_d ** 2, lead))


def evaluate_policy(
    mu_L: np.ndarray, sigma_L: np.ndarray, demand_L: np.ndarray, unit_holding: np.ndarray, z: float
) -> dict:
    """Score one service-level target. All arrays are (n_series, n_windows); unit_holding is per unit-window."""
    rop = np.ceil(np.clip(mu_L + z * sigma_L, 0, None))  # whole units
    short = np.clip(demand_L - rop, 0, None)
    safety = np.clip(rop - mu_L, 0, None)
    return {
        "z": z,
        "target_csl": float(norm.cdf(z)),
        "cycle_service_level": float((demand_L <= rop).mean()),
        "fill_rate": float(1 - short.sum() / max(demand_L.sum(), 1e-9)),
        "safety_stock_units": float(safety.sum() / demand_L.shape[1]),  # per week, all series
        "holding_cost_per_week": float((safety * unit_holding).sum() / demand_L.shape[1]),
        "rop_units_per_week": float(rop.sum() / demand_L.shape[1]),
        "units_short_per_week": float(short.sum() / demand_L.shape[1]),
    }


def sweep(mu_L, sigma_L, demand_L, unit_holding, z_values=None) -> list[dict]:
    """Score a grid of safety factors z (target CSL = Phi(z)); default z = 0, 0.25, ..., 5."""
    z_values = z_values if z_values is not None else np.arange(0, 5.01, 0.25)
    return [evaluate_policy(mu_L, sigma_L, demand_L, unit_holding, float(z)) for z in z_values]
