"""Phase 7: forecast reconciliation across the 12-level hierarchy.

Base forecasts made independently per level do not add up: the store forecasts
do not sum to the state forecast, and so on. Reconciliation maps the stacked
base forecasts y_hat (n = 42,840) to coherent ones y_tilde = S G y_hat, where
S (n x m, m = 30,490 bottom series) is the summing matrix.

Methods
-------
bu           bottom-up: y_tilde = S y_hat_bottom (ignores upper-level forecasts)
ols          G = (S'S)^-1 S'                        (W = I)
wls_struct   G = (S'L S)^-1 S'L, L = diag(1 / S1)   (W ∝ number of bottom series)
wls_var      L = diag(1 / sigma^2) from past forecast errors
mint_shrink  L = W^-1, W = shrinkage covariance of past forecast errors
             (Schäfer-Strimmer target: diagonal)

At m = 30,490 the m x m matrix S'LS is dense, so nothing is formed explicitly:
S'LS v is computed matrix-free from the sparse S, and the system is solved
with preconditioned conjugate gradients for all 28 horizons at once. For MinT,
W is "diagonal + rank-k" (k = number of past error vectors), so W^-1 v comes
from the Woodbury identity in O(n k).

The error matrix for wls_var / mint_shrink must come from forecast origins
strictly before the one being reconciled (point-in-time).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse

VAR_FLOOR = 1e-3


def bottom_first_S(S: sparse.csr_matrix, levels: np.ndarray, bottom_level: int = 12) -> sparse.csr_matrix:
    """Reorder S's columns so the bottom-level block of rows is the identity.

    Then the bottom forecast vector is ordered like the bottom-level rows of the
    hierarchy and ``S @ b`` is directly comparable with stacked forecasts.
    """
    bottom_rows = S[np.flatnonzero(levels == bottom_level)]
    assert (np.diff(bottom_rows.indptr) == 1).all(), "each bottom row must have exactly one 1"
    perm = bottom_rows.indices  # column of the single 1 in each bottom row, in row order
    assert len(perm) == S.shape[1] and len(np.unique(perm)) == S.shape[1]
    return S[:, perm].tocsr()


@dataclass
class Precision:
    """L = W^-1 as an operator: diag part plus optional Woodbury low-rank correction."""

    inv_diag: np.ndarray  # (n,)
    V: np.ndarray | None = None  # (n, k): W = diag(1/inv_diag) + V V'
    M_inv: np.ndarray | None = None  # (k, k): (I + V' D^-1 V)^-1

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = self.inv_diag[:, None] * x
        if self.V is None:
            return y
        DV = self.inv_diag[:, None] * self.V
        return y - DV @ (self.M_inv @ (DV.T @ x))

    def diag(self) -> np.ndarray:
        if self.V is None:
            return self.inv_diag
        DV = self.inv_diag[:, None] * self.V
        return self.inv_diag - np.einsum("ij,jk,ik->i", DV, self.M_inv, DV)


def shrinkage_lambda(E: np.ndarray) -> float:
    """Schäfer-Strimmer shrinkage intensity for the correlation matrix of E (n x k).

    Uses only k x k quantities, so it scales to n = 42,840 without forming n x n.
    """
    n, k = E.shape
    sd = np.sqrt(np.maximum((E ** 2).mean(axis=1), VAR_FLOOR))
    X = E / sd[:, None]
    c = (X ** 2).sum(axis=0)  # (k,)
    sum_w2 = (c ** 2).sum() - (X ** 4).sum()  # off-diagonal sum over t of (x_it x_jt)^2
    G = X.T @ X  # (k, k)
    sum_r2 = (G ** 2).sum() / k ** 2 - (((X ** 2).sum(axis=1) / k) ** 2).sum()  # off-diagonal r_ij^2
    var_sum = (sum_w2 - k * sum_r2) / (k * (k - 1))
    return float(np.clip(var_sum / max(sum_r2, 1e-12), 1e-3, 1.0))


def precision(method: str, S: sparse.csr_matrix, errors: np.ndarray | None = None) -> tuple[Precision, dict]:
    n = S.shape[0]
    if method == "ols":
        return Precision(np.ones(n)), {}
    if method == "wls_struct":
        return Precision(1.0 / np.asarray(S.sum(axis=1)).ravel()), {}
    if errors is None:
        raise ValueError(f"{method} needs past forecast errors")
    E = np.asarray(errors, float)
    var = np.maximum((E ** 2).mean(axis=1), VAR_FLOOR)
    if method == "wls_var":
        return Precision(1.0 / var), {}
    if method == "mint_shrink":
        k = E.shape[1]
        lam = shrinkage_lambda(E)
        # W = lam*D + (1-lam)*E E'/k ; diag(W) = D up to the floor.
        inv_diag = 1.0 / (lam * var)
        V = np.sqrt((1 - lam) / k) * E
        M = np.eye(k) + V.T @ (inv_diag[:, None] * V)
        return Precision(inv_diag, V, np.linalg.inv(M)), {"lambda": lam}
    raise ValueError(method)


def _pcg(apply_A: Callable, b: np.ndarray, diag: np.ndarray, tol=1e-8, max_iter=500) -> tuple[np.ndarray, int]:
    """Jacobi-preconditioned CG, vectorised over the columns of b."""
    x = np.zeros_like(b)
    r = b.copy()
    z = r / diag[:, None]
    p = z.copy()
    rz = (r * z).sum(axis=0)
    bnorm = np.linalg.norm(b, axis=0) + 1e-30
    for it in range(1, max_iter + 1):
        Ap = apply_A(p)
        alpha = rz / np.maximum((p * Ap).sum(axis=0), 1e-300)
        x += alpha * p
        r -= alpha * Ap
        if (np.linalg.norm(r, axis=0) / bnorm).max() < tol:
            return x, it
        z = r / diag[:, None]
        rz_new = (r * z).sum(axis=0)
        p = z + (rz_new / np.maximum(rz, 1e-300)) * p
        rz = rz_new
    return x, max_iter


def reconcile(
    y_hat: np.ndarray,
    S: sparse.csr_matrix,
    method: str,
    levels: np.ndarray | None = None,
    errors: np.ndarray | None = None,
    nonnegative: bool = True,
) -> tuple[np.ndarray, dict]:
    """Reconcile stacked base forecasts y_hat (n x h). Returns (y_tilde, info).

    S must be bottom-first (see ``bottom_first_S``); ``levels`` is required for bu.
    With ``nonnegative`` the reconciled bottom forecasts are clipped at zero and
    re-aggregated, which keeps the result coherent.
    """
    y_hat = np.asarray(y_hat, float)
    info: dict = {"method": method}
    if method == "bu":
        b = y_hat[np.flatnonzero(levels == levels.max())]
    else:
        L, extra = precision(method, S, errors)
        info.update(extra)
        St = S.T.tocsr()
        S2 = S.multiply(S).T.tocsr()  # for the preconditioner diagonal

        def apply_A(v):
            return St @ L.apply(S @ v)

        diag = np.maximum(S2 @ L.diag(), 1e-12)
        b, iters = _pcg(apply_A, St @ L.apply(y_hat), diag)
        info["cg_iterations"] = iters
    if nonnegative:
        b = np.clip(b, 0, None)
    return S @ b, info


def incoherence(y: np.ndarray, S: sparse.csr_matrix, levels: np.ndarray) -> np.ndarray:
    """Per-series |y - S y_bottom| summed over the horizon, relative to |y| — 0 means coherent."""
    b = y[np.flatnonzero(levels == levels.max())]
    gap = np.abs(y - S @ b).sum(axis=1)
    return gap / np.maximum(np.abs(y).sum(axis=1), 1e-9)
