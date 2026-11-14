import numpy as np
from scipy.stats import norm

from m5fc import inventory as inv


def test_lead_time_windows_and_sigma():
    daily = np.arange(28, dtype=float)[None, :]
    np.testing.assert_array_equal(inv.lead_time_windows(daily), [[21, 70, 119, 168]])
    q10, q90 = np.zeros((1, 28)), np.full((1, 28), 2 * inv.Z90)  # daily sigma = 1
    np.testing.assert_allclose(inv.lead_time_sigma(q10, q90), np.sqrt(7))


def test_policy_hits_target_service_level_when_model_is_right():
    rng = np.random.default_rng(0)
    mu, sigma = np.full((20000, 4), 50.0), np.full((20000, 4), 10.0)
    demand = rng.normal(mu, sigma)
    res = inv.evaluate_policy(mu, sigma, demand, np.ones_like(mu), z=norm.ppf(0.9))
    assert abs(res["cycle_service_level"] - 0.9) < 0.02  # ceil() adds a little
    curve = inv.sweep(mu, sigma, demand, np.ones_like(mu))
    costs = [r["holding_cost_per_week"] for r in curve]
    fills = [r["fill_rate"] for r in curve]
    assert costs == sorted(costs) and fills == sorted(fills)  # more stock -> more service
