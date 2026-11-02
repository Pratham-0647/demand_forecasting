"""Phases 2-3: score saved point forecasts with MASE and WAPE at every hierarchy
level and every rolling origin, then average across origins.

    python scripts/evaluate.py                                  # every origin found
    python scripts/evaluate.py --origins 1913

Writes results/metrics_by_origin.csv (one row per origin x level x model) and
results/metrics.md (tables averaged over origins).
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
POINT_MODELS = ["seasonal_naive", "ets", "arima", "croston", "croston_sba", "tsb", "lightgbm", "lightgbm_q50"]


def load_forecasts(origin: int, level: int, ids: np.ndarray, models=POINT_MODELS) -> dict[str, np.ndarray]:
    out = {}
    for model in models:
        path = FC_DIR / f"origin_{origin}" / model / f"level_{level:02d}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path).set_index("unique_id").reindex(ids)
        cols = [f"F{i + 1}" for i in range(HORIZON)]
        if df[cols].isna().any().any():
            raise ValueError(f"{path} does not cover every series of level {level}")
        out[model] = df[cols].to_numpy(float)
    return out


def to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def find_origins() -> list[int]:
    return sorted(int(p.name.split("_")[1]) for p in FC_DIR.glob("origin_*") if p.is_dir())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--origins", nargs="+", type=int, default=None)
    args = p.parse_args()
    origins = args.origins or find_origins()

    hier = build_hierarchy()
    rows = []
    for origin in origins:
        y_train_all = hier.Y[:, :origin]
        y_true_all = hier.Y[:, origin: origin + HORIZON]
        for lvl in range(1, 13):
            mask = (hier.ids["level"] == lvl).to_numpy()
            fcs = load_forecasts(origin, lvl, hier.ids.loc[mask, "unique_id"].to_numpy())
            if not fcs:
                continue
            s, _ = metrics.evaluate_level(y_train_all[mask], y_true_all[mask], fcs, lvl)
            s.insert(0, "origin", origin)
            rows.append(s)
    by_origin = pd.concat(rows, ignore_index=True)
    by_origin.insert(2, "level_name", by_origin["level"].map(LEVEL_NAMES))
    RES_DIR.mkdir(exist_ok=True)
    by_origin.to_csv(RES_DIR / "metrics_by_origin.csv", index=False, float_format="%.5f")

    # A model's numbers are averaged only over origins where it exists at that level;
    # n_origins in the table makes the coverage explicit.
    avg = (by_origin.groupby(["level", "level_name", "model"], sort=False)
           .agg(n_origins=("origin", "nunique"), mase_mean=("mase_mean", "mean"),
                wape=("wape", "mean"), share_beats_snaive=("share_beats_snaive", "mean"))
           .reset_index())
    models = [m for m in POINT_MODELS if m in set(avg["model"])]
    full_models = [m for m in models
                   if avg.loc[avg["model"] == m, "level"].nunique() == 12
                   and (avg.loc[avg["model"] == m, "n_origins"] == len(origins)).all()]

    def pivot(col):
        t = avg.pivot(index=["level", "level_name"], columns="model", values=col)
        return t[[m for m in models if m in t.columns]].reset_index()

    cross = (avg[avg["model"].isin(full_models)].groupby("model", sort=False)[["mase_mean", "wape"]]
             .mean().reindex(full_models).reset_index())
    cov = (by_origin.groupby("model")["origin"].apply(lambda s: ", ".join(map(str, sorted(s.unique()))))
           .reindex(models).reset_index().rename(columns={"origin": "origins"}))
    cov["levels"] = [", ".join(map(str, sorted(avg.loc[avg["model"] == m, "level"].unique()))) for m in models]

    dates = hier.dates
    windows = ", ".join(f"d_{o + 1}-d_{o + HORIZON} ({dates[o].date()} to {dates[o + HORIZON - 1].date()})"
                        for o in origins)
    md = [
        "# Point-forecast accuracy — rolling origin, expanding window\n",
        f"Origins: {len(origins)}; 28-day test windows: {windows}.",
        "Each origin trains on every day before it (expanding window); numbers are means over origins.\n",
        "## Mean MASE by level (scale: in-sample seasonal naive, lag 7; lower is better)\n",
        to_markdown(pivot("mase_mean")),
        "\n## WAPE by level (volume-weighted)\n",
        to_markdown(pivot("wape")),
        "\n## Share of series where the model beats seasonal naive on MASE\n",
        to_markdown(pivot("share_beats_snaive")),
        "\n## Cross-level average (12 levels weighted equally)\n",
        to_markdown(cross),
        "\n## Coverage (which origins and levels each model was run on)\n",
        to_markdown(cov),
        "",
    ]
    (RES_DIR / "metrics.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
