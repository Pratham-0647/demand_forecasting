"""Phase 2: score saved forecasts with MASE and WAPE at every hierarchy level.

    python scripts/evaluate.py                 # origin 1913 (d_1914-d_1941)

Writes results/metrics_origin_<origin>.csv and .md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5fc import metrics  # noqa: E402
from m5fc.data import HORIZON, LEVEL_NAMES, build_hierarchy  # noqa: E402

FC_DIR = ROOT / "outputs" / "forecasts"
RES_DIR = ROOT / "results"
MODEL_ORDER = ["seasonal_naive", "ets", "arima"]


def load_forecasts(origin: int, level: int, ids: pd.Series) -> dict[str, np.ndarray]:
    out = {}
    for model_dir in sorted((FC_DIR / f"origin_{origin}").glob("*/")):
        path = model_dir / f"level_{level:02d}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).set_index("unique_id").reindex(ids.to_numpy())
        if df.isna().any().any():
            raise ValueError(f"{path} does not cover every series of level {level}")
        out[model_dir.name] = df[[f"F{i + 1}" for i in range(HORIZON)]].to_numpy(float)
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    return dict(sorted(out.items(), key=lambda kv: order.get(kv[0], 99)))


def to_markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--origin", type=int, default=1913)
    args = p.parse_args()

    hier = build_hierarchy()
    y_train_all = hier.Y[:, : args.origin]
    y_true_all = hier.Y[:, args.origin : args.origin + HORIZON]

    summaries, zero_share = [], {}
    for lvl in sorted(hier.ids["level"].unique()):
        mask = (hier.ids["level"] == lvl).to_numpy()
        fcs = load_forecasts(args.origin, lvl, hier.ids.loc[mask, "unique_id"])
        if not fcs:
            continue
        zero_share[lvl] = metrics.zero_actual_share(y_true_all[mask])
        s, _ = metrics.evaluate_level(y_train_all[mask], y_true_all[mask], fcs, lvl)
        summaries.append(s)
    summary = pd.concat(summaries, ignore_index=True)

    # Average across levels only over levels that every model has been scored on.
    complete = summary.groupby("level")["model"].nunique() == summary["model"].nunique()
    full = summary[summary["level"].isin(complete[complete].index)]
    ov = metrics.overall(full)

    summary.insert(1, "level_name", summary["level"].map(LEVEL_NAMES))
    RES_DIR.mkdir(exist_ok=True)
    stem = RES_DIR / f"metrics_origin_{args.origin}"
    summary.to_csv(stem.with_suffix(".csv"), index=False)

    start = hier.dates[args.origin].date()
    end = hier.dates[args.origin + HORIZON - 1].date()
    pivot = lambda col: (summary.pivot(index=["level", "level_name"], columns="model", values=col)  # noqa: E731
                         [[m for m in MODEL_ORDER if m in summary["model"].unique()]].reset_index())
    md = [
        f"# Baseline metrics — origin d_{args.origin}, forecasting {start} to {end} (28 days)\n",
        f"Levels in the cross-level average: {len(complete[complete])} of 12.\n",
        "## Mean MASE by level (lower is better; scale = in-sample seasonal naive, lag 7)\n",
        to_markdown(pivot("mase_mean")),
        "\n## WAPE by level (volume-weighted)\n",
        to_markdown(pivot("wape")),
        "\n## Share of series where the model beats seasonal naive (MASE)\n",
        to_markdown(pivot("share_beats_snaive")),
        "\n## Cross-level average (each level weighted equally)\n",
        to_markdown(ov.drop(columns="level")),
        "\n## Zero actuals in the test window (why MAPE is unusable)\n",
        to_markdown(pd.DataFrame({"level": list(zero_share), "zero_share": list(zero_share.values())})),
        "",
    ]
    stem.with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
