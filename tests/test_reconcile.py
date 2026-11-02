import numpy as np
import pandas as pd
import pytest

from m5fc import reconcile as rc
from m5fc.data import summing_matrix


@pytest.fixture
def hier():
    # 2 states x 2 stores x 3 items, store-major like the sales file.
    meta = pd.DataFrame([
        {"item_id": f"I{i}", "dept_id": "D", "cat_id": "C", "store_id": f"{st}_{s}", "state_id": st}
        for st in ("A", "B") for s in (1, 2) for i in range(3)])
    ids, S = summing_matrix(meta)
    levels = ids["level"].to_numpy()
    return rc.bottom_first_S(S, levels), levels


def _dense_G(S, W):
    Wi = np.linalg.inv(W)
    return np.linalg.solve(S.T @ Wi @ S, S.T @ Wi)


def test_bottom_first_identity(hier):
    S, levels = hier
    np.testing.assert_array_equal(S[levels == 12].toarray(), np.eye(S.shape[1]))


@pytest.mark.parametrize("method", ["ols", "wls_struct", "wls_var", "mint_shrink"])
def test_matches_dense_formula_and_is_coherent(hier, method):
    S, levels = hier
    rng = np.random.default_rng(0)
    n, m = S.shape
    y_hat = S.toarray() @ rng.uniform(5, 10, (m, 4)) + rng.normal(0, 2, (n, 4))
    E = rng.normal(0, 1, (n, 30)) * np.sqrt(np.asarray(S.sum(axis=1)))
    y_tilde, info = rc.reconcile(y_hat, S, method, levels, errors=E, nonnegative=False)

    Sd = S.toarray()
    var = (E ** 2).mean(axis=1)
    W = {"ols": np.eye(n), "wls_struct": np.diag(Sd.sum(axis=1)), "wls_var": np.diag(var)}.get(method)
    if method == "mint_shrink":
        lam = info["lambda"]
        W = lam * np.diag(var) + (1 - lam) * (E @ E.T) / E.shape[1]
    expected = Sd @ _dense_G(Sd, W) @ y_hat
    np.testing.assert_allclose(y_tilde, expected, rtol=1e-6, atol=1e-6)
    assert rc.incoherence(y_tilde, S, levels).max() < 1e-6 * np.abs(y_tilde).max()


def test_reconciliation_leaves_coherent_forecasts_unchanged(hier):
    S, levels = hier
    y = S @ np.random.default_rng(1).uniform(0, 5, (S.shape[1], 3))
    for method in ["bu", "ols", "wls_struct"]:
        np.testing.assert_allclose(rc.reconcile(y, S, method, levels)[0], y, atol=1e-6)


def test_shrinkage_lambda_matches_bruteforce():
    rng = np.random.default_rng(2)
    E = rng.normal(size=(6, 20))
    E[1] += 0.8 * E[0]
    k = E.shape[1]
    X = E / np.sqrt((E ** 2).mean(axis=1, keepdims=True))
    W = X[:, None, :] * X[None, :, :]  # (n, n, k)
    r = W.mean(axis=2)
    var_r = ((W - r[..., None]) ** 2).sum(axis=2) / (k * (k - 1))
    off = ~np.eye(len(E), dtype=bool)
    expected = np.clip(var_r[off].sum() / (r[off] ** 2).sum(), 1e-3, 1)
    assert rc.shrinkage_lambda(E) == pytest.approx(expected)


def test_set_negative_to_zero_keeps_coherence_and_nonnegativity(hier):
    S, levels = hier
    rng = np.random.default_rng(5)
    n, m = S.shape
    y_hat = S.toarray() @ rng.uniform(0, 1, (m, 3)) + rng.normal(0, 3, (n, 3))  # noisy -> negatives
    plain, _ = rc.reconcile(y_hat, S, "ols", levels, nonnegative=False)
    assert (plain[levels == 12] < 0).any()
    y_nn, info = rc.reconcile(y_hat, S, "ols", levels, nonnegative=True)
    assert (y_nn >= -1e-9).all()
    assert rc.incoherence(y_nn, S, levels).max() < 1e-6 * np.abs(y_nn).max()
    assert info["negative_share_fixed"] > 0
