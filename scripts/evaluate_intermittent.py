"""Phase 6: compare Croston-family methods with the other models on intermittent series.

Bottom level (30,490 item x store series). Each series is classified on its
training window at each origin (Syntetos-Boylan-Croston: ADI 1.32, CV2 0.49).

    python scripts/evaluate_intermittent.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate import POINT_MODELS, find_origins, load_forecasts, to_markdown  # noqa: E402
from m5fc import metrics  # noqa: E402
from m5fc.data import HORIZON, build_hierarchy  # noqa: E402
from m5fc.intermittent import classify  # noqa: E402

RES_DIR = ROOT / "results"
CLASS_ORDER = ["smooth", "erratic", "intermittent", "lumpy"]


def main() -> None:
    h = build_hierarchy(levels=[12])
    ids = h.ids["unique_id"].to_numpy()
    rows, class_rows = [], []
    for origin in find_origins():
        fcs = load_forecasts(origin, 12, ids)
        if "croston" not in fcs:
            continue
        y_train, y_true = h.Y[:, :origin], h.Y[:, origin: origin + HORIZON]
        cls = classify(y_train)
        scale = metrics.mase_scale(y_train)
        groups = {c: (cls["sbc_class"] == c).to_numpy() for c in CLASS_ORDER}
        groups["intermittent+lumpy (ADI>=1.32)"] = (cls["adi"] >= 1.32).to_numpy()
        groups["all"] = np.ones(len(cls), bool)
        for c in CLASS_ORDER:
            class_rows.append({"origin": origin, "class": c, "share": groups[c].mean(),
                               "zero_share": cls.loc[groups[c], "zero_share"].mean()})
        for model, fc in fcs.items():
            s_mase = metrics.mase(y_true, fc, scale)
            for g, m in groups.items():
                ok = m & np.isfinite(s_mase)
                rows.append({"origin": origin, "group": g, "model": model, "n_series": int(m.sum()),
                             "mase_mean": float(s_mase[ok].mean()), "wape": metrics.wape(y_true[m], fc[m]),
                             "bias": float((fc[m].sum() - y_true[m].sum()) / max(y_true[m].sum(), 1))})
    if not rows:
        raise SystemExit("no Croston forecasts found — run run_baselines.py --models croston croston_sba tsb --levels 12")
    df = pd.DataFrame(rows)
    origins = sorted(df["origin"].unique())
    avg = df.groupby(["group", "model"], sort=False)[["n_series", "mase_mean", "wape", "bias"]].mean().reset_index()
    models = [m for m in POINT_MODELS if m in set(avg["model"])]
    groups = list(dict.fromkeys(avg["group"]))

    # Models run at fewer origins are averaged over different windows: say so in the header.
    n_orig = df.groupby("model")["origin"].nunique()
    label = {m: m if n_orig[m] == len(origins) else f"{m} ({n_orig[m]} origin{'s' if n_orig[m] > 1 else ''})"
             for m in models}

    def pivot(col):
        t = avg.pivot(index="group", columns="model", values=col).reindex(groups)[models]
        return t.rename(columns=label).reset_index()

    classes = pd.DataFrame(class_rows).groupby("class")[["share", "zero_share"]].mean().reindex(CLASS_ORDER).reset_index()
    RES_DIR.mkdir(exist_ok=True)
    df.to_csv(RES_DIR / "intermittent_by_origin.csv", index=False, float_format="%.5f")
    md = [
        "# Intermittent demand — bottom level (30,490 series)\n",
        f"Origins: {', '.join(f'd_{o}' for o in origins)}; numbers averaged over origins. Columns marked "
        "with an origin count cover fewer windows and are not directly comparable.\n",
        "## Demand classes (Syntetos-Boylan-Croston, on each origin's training window)\n",
        to_markdown(classes),
        "\n## Mean MASE by demand class\n",
        to_markdown(pivot("mase_mean")),
        "\n## WAPE by demand class\n",
        to_markdown(pivot("wape")),
        "\n## Bias: (sum forecast - sum actual) / sum actual\n",
        to_markdown(pivot("bias")),
        "",
    ]
    (RES_DIR / "intermittent.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
