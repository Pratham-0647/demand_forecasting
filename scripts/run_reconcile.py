"""Phase 7: reconcile base forecasts across the 12 levels and measure the effect.

Base forecast sets
  ets      ETS at every level (independent per level, so incoherent)
  ets_lgb  ETS at levels 1-11, global LightGBM at the bottom level

Methods: bu, ols, wls_struct, wls_var, mint_shrink (see m5fc.reconcile).
wls_var / mint_shrink estimate the error covariance from forecast errors at
*earlier* origins only, so they are skipped at the first origin.

Also builds coherent P10/P50/P90 at every level: reconciled point forecast plus
per-series empirical quantiles of the same method's errors at earlier origins.

    python scripts/run_reconcile.py
Writes results/reconciliation.md, results/reconciliation_by_origin.csv,
results/hierarchical_quantiles.md.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import find_origins, load_forecasts, to_markdown  # noqa: E402
from m5fc import metrics  # noqa: E402
from m5fc import reconcile as rc  # noqa: E402
from m5fc.data import HORIZON, LEVEL_NAMES, load_sales, summing_matrix  # noqa: E402

RES_DIR = ROOT / "results"
METHODS = ["base", "bu", "ols", "wls_struct", "wls_var", "mint_shrink"]
QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def stacked_base(origin: int, ids: pd.DataFrame, base_set: str) -> np.ndarray | None:
    out = np.empty((len(ids), HORIZON))
    for lvl in range(1, 13):
        mask = (ids["level"] == lvl).to_numpy()
        model = "lightgbm" if (base_set == "ets_lgb" and lvl == 12) else "ets"
        fc = load_forecasts(origin, lvl, ids.loc[mask, "unique_id"].to_numpy(), models=[model])
        if model not in fc:
            return None
        out[mask] = fc[model]
    return out


def main() -> None:
    meta, bottom = load_sales()
    ids, S_raw = summing_matrix(meta)
    levels = ids["level"].to_numpy()
    S = rc.bottom_first_S(S_raw, levels)
    Y = np.asarray(S_raw @ bottom.astype(np.float64))
    origins = find_origins()

    rows, qrows = [], []
    errors: dict[tuple[str, str], list[np.ndarray]] = {}
    for origin in origins:
        y_train, y_true = Y[:, :origin], Y[:, origin: origin + HORIZON]
        scale = metrics.mase_scale(y_train)
        # Point-in-time: freeze the error history *before* this origin. Errors made at
        # this origin are only added once every method here has run (otherwise
        # wls_var / mint_shrink would see the very test window they are scored on).
        past = {k: list(v) for k, v in errors.items()}
        new_errors: dict[tuple[str, str], np.ndarray] = {}
        for base_set in ["ets", "ets_lgb"]:
            y_hat = stacked_base(origin, ids, base_set)
            if y_hat is None:
                continue
            for method in METHODS:
                prior = past.get((base_set, "base"), [])
                if method in ("wls_var", "mint_shrink") and not prior:
                    continue
                t0 = time.perf_counter()
                if method == "base":
                    y_rec, info = y_hat, {}
                else:
                    E = np.hstack(prior) if prior else None
                    y_rec, info = rc.reconcile(y_hat, S, method, levels, errors=E)
                secs = time.perf_counter() - t0
                gap = rc.incoherence(y_rec, S, levels)
                vol = np.abs(y_rec).sum(axis=1)
                s_mase = metrics.mase(y_true, y_rec, scale)
                for lvl in range(1, 13):
                    m = levels == lvl
                    ok = m & np.isfinite(s_mase)
                    rows.append({"origin": origin, "base_set": base_set, "method": method, "level": lvl,
                                 "mase_mean": float(s_mase[ok].mean()), "wape": metrics.wape(y_true[m], y_rec[m]),
                                 "incoherence": float(gap[m].sum() / max(vol[m].sum(), 1e-9)), "seconds": secs,
                                 "lambda": info.get("lambda"), "cg_iterations": info.get("cg_iterations"),
                                 "negative_share_fixed": info.get("negative_share_fixed")})
                # Coherent quantiles from this method's own errors at earlier origins.
                own_prior = past.get((base_set, method), [])
                if own_prior:
                    E_own = np.hstack(own_prior)
                    for lvl in range(1, 13):
                        m = levels == lvl
                        ok = m & np.isfinite(scale)
                        rec = {"origin": origin, "base_set": base_set, "method": method, "level": lvl,
                               "n_prior_origins": len(own_prior)}
                        for q, a in QUANTILES.items():
                            # error e = y - y_hat, so the alpha-quantile of y is y_hat + q_alpha(e).
                            qf = np.clip(y_rec[m] + np.quantile(E_own[m], a, axis=1)[:, None], 0, None)
                            rec[f"spl_{q}"] = float(metrics.scaled_pinball(
                                y_true[ok], qf[ok[m]], a, scale[ok]).mean())
                            rec[f"le_{q}"] = metrics.coverage(y_true[m], qf)
                        qrows.append(rec)
                new_errors[(base_set, method)] = y_true - y_rec
                print(f"d_{origin} {base_set:8s} {method:12s} {secs:6.1f}s "
                      f"total-WAPE={rows[-12]['wape']:.4f} bottom-MASE={rows[-1]['mase_mean']:.4f} "
                      f"{info.get('lambda', '')}", flush=True)

        for k, e in new_errors.items():
            errors.setdefault(k, []).append(e)

    df = pd.DataFrame(rows)
    # Leakage guard: covariance-based methods need errors from an *earlier* origin.
    first = df[df["origin"] == origins[0]]["method"].unique()
    assert not {"wls_var", "mint_shrink"} & set(first), "error history leaked into the first origin"
    RES_DIR.mkdir(exist_ok=True)
    df.to_csv(RES_DIR / "reconciliation_by_origin.csv", index=False, float_format="%.6f")

    # Compare methods on origins where every method ran (those with prior errors).
    full = df.groupby("origin")["method"].nunique()
    eval_origins = full[full == len(METHODS)].index.tolist()
    sub = df[df["origin"].isin(eval_origins)]
    md = ["# Reconciliation across the 12 levels\n",
          f"Methods compared on origins {', '.join(f'd_{o}' for o in eval_origins)} (each uses only "
          "forecast errors from earlier origins for wls_var / mint_shrink).\n"]
    for base_set in ["ets", "ets_lgb"]:
        b = sub[sub["base_set"] == base_set]
        if b.empty:
            continue
        avg = b.groupby(["level", "method"])[["mase_mean", "wape", "incoherence"]].mean().reset_index()
        title = {"ets": "ETS at every level", "ets_lgb": "ETS levels 1-11 + LightGBM bottom level"}[base_set]
        for col, label in [("incoherence", "Incoherence: Σ|y - S·y_bottom| / Σ|y| per level (0 = coherent)"),
                           ("mase_mean", "Mean MASE"), ("wape", "WAPE")]:
            t = avg.pivot(index="level", columns="method", values=col)[METHODS].reset_index()
            t.insert(1, "level_name", t["level"].map(LEVEL_NAMES))
            md += [f"\n## {title} — {label}\n", to_markdown(t, 4)]
        cross = avg.groupby("method")[["mase_mean", "wape"]].mean().reindex(METHODS).reset_index()
        md += [f"\n### {title} — average over the 12 levels\n", to_markdown(cross, 4)]
    lam = df.dropna(subset=["lambda"]).groupby(["origin", "base_set"])["lambda"].first().reset_index()
    md += ["\n## MinT shrinkage intensity (lambda; 1 = diagonal only)\n", to_markdown(lam, 3), ""]
    (RES_DIR / "reconciliation.md").write_text("\n".join(md))
    print("\n".join(md))

    if qrows:
        q = pd.DataFrame(qrows)
        last = q["origin"].max()
        ql = q[q["origin"] == last]
        cols = ["spl_q10", "spl_q50", "spl_q90", "le_q10", "le_q50", "le_q90"]
        qmd = ["# Coherent P10/P50/P90 at every level\n",
               f"Origin d_{last}. Quantile = reconciled point forecast + empirical quantile of that method's "
               "own errors at earlier origins (per series, pooled over horizons). Because every quantile is "
               "built on the reconciled point forecast, the P50 forecasts are coherent across levels by "
               "construction; the P10/P90 offsets are per series.\n"]
        for base_set in ql["base_set"].unique():
            for method in ["mint_shrink", "wls_struct", "bu"]:
                t = ql[(ql["base_set"] == base_set) & (ql["method"] == method)]
                if t.empty:
                    continue
                t = t[["level"] + cols].copy()
                t.insert(1, "level_name", t["level"].map(LEVEL_NAMES))
                qmd += [f"\n## {base_set} / {method} (prior origins: {int(ql['n_prior_origins'].max())})\n",
                        to_markdown(t, 3)]
        q.to_csv(RES_DIR / "hierarchical_quantiles_by_origin.csv", index=False, float_format="%.5f")
        (RES_DIR / "hierarchical_quantiles.md").write_text("\n".join(qmd) + "\n")


if __name__ == "__main__":
    main()
