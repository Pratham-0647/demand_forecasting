"""Phases 4-5: global LightGBM point + quantile models at one forecast origin.

    python scripts/run_gbm.py --origin 1913
    python scripts/run_gbm.py --origin 1885 --targets point

Writes, under outputs/forecasts/origin_<origin>/:
  lightgbm/level_XX.parquet       point forecast, bottom level summed to all 12 levels
  lightgbm_q10|q50|q90/level_12   quantile forecasts (bottom level; quantiles do not add up)
and results/gbm_feature_importance_origin_<origin>.csv.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5fc import gbm  # noqa: E402
from m5fc.baselines import forecast_frame  # noqa: E402
from m5fc.data import Hierarchy, summing_matrix  # noqa: E402
from m5fc.features import load_panel  # noqa: E402

OUT_DIR = ROOT / "outputs" / "forecasts"
RES_DIR = ROOT / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--origin", type=int, default=1913)
    p.add_argument("--targets", nargs="+", default=["point", "q10", "q50", "q90"], choices=list(gbm.OBJECTIVES))
    p.add_argument("--train-days", type=int, default=728)
    p.add_argument("--num-rounds", type=int, default=gbm.DEFAULT_ROUNDS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    base = OUT_DIR / f"origin_{args.origin}"
    names = {"point": "lightgbm", "q10": "lightgbm_q10", "q50": "lightgbm_q50", "q90": "lightgbm_q90"}
    todo = [t for t in args.targets if args.force or not (base / names[t] / "level_12.parquet").exists()]
    if not todo:
        print("nothing to do")
        return

    panel = load_panel()
    ids, S = summing_matrix(panel.meta)
    cal_dates = panel.cal["date"]
    origin_date = cal_dates.iloc[args.origin - 1]
    print(f"origin d_{args.origin}: training LightGBM for {todo}", flush=True)
    t0 = time.perf_counter()
    fcs, models = gbm.fit_predict(panel, args.origin, targets=tuple(todo), train_days=args.train_days,
                                  num_rounds=args.num_rounds, log=lambda m: print(m, flush=True))

    levels = ids["level"].to_numpy()
    for tgt, fc in fcs.items():
        stacked = np.asarray(S @ fc.astype(np.float64), np.float32)  # bottom-up to every level
        h = Hierarchy(ids, stacked, pd.DatetimeIndex([]))
        out_levels = sorted(set(levels)) if tgt == "point" else [12]
        for lvl in out_levels:
            hl = h.level(lvl)
            path = base / names[tgt] / f"level_{lvl:02d}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            forecast_frame(hl, hl.Y, names[tgt], origin_date).to_parquet(path, index=False)

    RES_DIR.mkdir(exist_ok=True)
    imp = pd.DataFrame({
        tgt: pd.Series(m.feature_importance("gain"), index=m.feature_name()) for tgt, m in models.items()})
    imp = (imp / imp.sum()).sort_values(imp.columns[0], ascending=False)
    imp.to_csv(RES_DIR / f"gbm_feature_importance_origin_{args.origin}.csv", float_format="%.4f")
    with (base / "timings.jsonl").open("a") as f:
        f.write(json.dumps({"model": "lightgbm", "targets": todo, "seconds": round(time.perf_counter() - t0, 1),
                            "best_iterations": {t: m.best_iteration for t, m in models.items()}}) + "\n")
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
