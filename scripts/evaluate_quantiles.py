"""Phase 5: pinball loss and calibration of P10/P50/P90 forecasts (bottom level).

Compares LightGBM quantile regression with a simple baseline: the empirical
quantile of the same weekday over the last 8 weeks before the origin.

Calibration: for count data with many zeros, P(Y <= q_alpha) >= alpha and
P(Y < q_alpha) <= alpha both hold for a correct quantile, so both are reported.
A well-calibrated P90 has "at or below" >= 0.9 and "strictly below" <= 0.9.

    python scripts/evaluate_quantiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import find_origins, load_forecasts, to_markdown  # noqa: E402
from m5fc import metrics  # noqa: E402
from m5fc.data import HORIZON, build_hierarchy  # noqa: E402
from m5fc.intermittent import classify  # noqa: E402

RES_DIR = ROOT / "results"
ALPHAS = {"q10": 0.1, "q50": 0.5, "q90": 0.9}


def weekday_empirical_quantiles(y_train: np.ndarray, alpha: float, weeks: int = 8) -> np.ndarray:
    """Quantile of the same weekday over the last ``weeks`` weeks, repeated over the horizon."""
    last = y_train[:, -7 * weeks:].reshape(len(y_train), weeks, 7)  # (n, weeks, weekday-position)
    q = np.quantile(last, alpha, axis=1, method="inverted_cdf")  # an observed value, as counts are integers
    return np.tile(q, HORIZON // 7)


def score(y_true, qs: dict[str, np.ndarray], scale, mask=None) -> dict:
    m = np.ones(len(y_true), bool) if mask is None else mask
    ok = m & np.isfinite(scale)
    out = {}
    for name, alpha in ALPHAS.items():
        out[f"pinball_{name}"] = float(metrics.pinball(y_true[m], qs[name][m], alpha).mean())
        out[f"spl_{name}"] = float(metrics.scaled_pinball(y_true[ok], qs[name][ok], alpha, scale[ok]).mean())
        out[f"le_{name}"] = metrics.coverage(y_true[m], qs[name][m])
        out[f"lt_{name}"] = float((y_true[m] < qs[name][m] - 1e-9).mean())
    out["coverage_p10_p90"] = metrics.interval_coverage(y_true[m], qs["q10"][m], qs["q90"][m])
    out["crossing_rate"] = float(((qs["q10"][m] > qs["q50"][m] + 1e-9) | (qs["q50"][m] > qs["q90"][m] + 1e-9)).mean())
    return out


def main() -> None:
    h = build_hierarchy(levels=[12])
    ids = h.ids["unique_id"].to_numpy()
    rows = []
    for origin in find_origins():
        lgb = load_forecasts(origin, 12, ids, models=[f"lightgbm_{q}" for q in ALPHAS])
        if len(lgb) < 3:
            continue
        y_train, y_true = h.Y[:, :origin], h.Y[:, origin: origin + HORIZON]
        scale = metrics.mase_scale(y_train)
        cls = classify(y_train)["sbc_class"].to_numpy()
        candidates = {
            "lightgbm_quantile": {q: lgb[f"lightgbm_{q}"] for q in ALPHAS},
            "weekday_empirical_8w": {q: weekday_empirical_quantiles(y_train, a) for q, a in ALPHAS.items()},
        }
        for model, qs in candidates.items():
            for group, mask in [("all", None)] + [(c, cls == c) for c in ["smooth", "erratic", "intermittent", "lumpy"]]:
                rows.append({"origin": origin, "model": model, "group": group, **score(y_true, qs, scale, mask)})
    if not rows:
        raise SystemExit("no LightGBM quantile forecasts found — run scripts/run_gbm.py")
    df = pd.DataFrame(rows)
    RES_DIR.mkdir(exist_ok=True)
    df.to_csv(RES_DIR / "quantiles_by_origin.csv", index=False, float_format="%.5f")
    avg = df.drop(columns="origin").groupby(["model", "group"], sort=False).mean().reset_index()
    allg = avg[avg["group"] == "all"]

    loss = allg[["model"] + [f"pinball_{q}" for q in ALPHAS] + [f"spl_{q}" for q in ALPHAS]]
    calib = pd.DataFrame([
        {"model": r["model"], "quantile": q, "nominal": a, "share_y_le_q": r[f"le_{q}"], "share_y_lt_q": r[f"lt_{q}"]}
        for _, r in allg.iterrows() for q, a in ALPHAS.items()])
    by_class = avg[avg["model"] == "lightgbm_quantile"][
        ["group", "spl_q50", "spl_q90", "le_q90", "lt_q90", "coverage_p10_p90", "crossing_rate"]]
    md = [
        "# Probabilistic forecasts — P10 / P50 / P90 at the bottom level\n",
        f"Origins: {', '.join(f'd_{o}' for o in sorted(df['origin'].unique()))}.\n",
        "## Pinball loss (raw units) and scaled pinball loss (÷ MASE scale)\n",
        to_markdown(loss, 4),
        "\n## Calibration (a correct alpha-quantile has share_y_lt_q <= alpha <= share_y_le_q)\n",
        to_markdown(calib),
        "\n## 80% interval coverage and quantile crossing\n",
        to_markdown(allg[["model", "coverage_p10_p90", "crossing_rate"]]),
        "\n## LightGBM by demand class\n",
        to_markdown(by_class),
        "",
    ]
    (RES_DIR / "quantiles.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
