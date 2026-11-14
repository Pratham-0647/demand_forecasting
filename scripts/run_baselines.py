"""Phase 1: fit seasonal naive, ETS and ARIMA on every series of the 12-level hierarchy.

Forecasts are written per (origin, model, level) so an interrupted run resumes
where it stopped. Example:

    python scripts/run_baselines.py                       # all models, all levels
    python scripts/run_baselines.py --models seasonal_naive ets --levels 1 2 3
"""
from __future__ import annotations

import os

# One BLAS thread per worker; statsforecast parallelises across series, and
# oversubscribed BLAS threads made ETS ~10x slower in benchmarks.
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m5fc import baselines  # noqa: E402
from m5fc.data import HORIZON, LEVELS, build_hierarchy  # noqa: E402

OUT_DIR = ROOT / "outputs" / "forecasts"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--origin", type=int, default=1913,
                   help="Number of training days; forecasts cover the next 28 (default: d_1914-d_1941).")
    p.add_argument("--models", nargs="+", default=list(baselines.MODELS), choices=list(baselines.all_models()))
    p.add_argument("--levels", nargs="+", type=int, default=[lvl for lvl, _, _ in LEVELS])
    p.add_argument("--max-train-days", type=int, default=730,
                   help="ETS/ARIMA fit on at most this many recent days (seasonal naive needs only 7).")
    p.add_argument("--n-jobs", type=int, default=baselines.default_jobs())
    p.add_argument("--force", action="store_true", help="Recompute even if the output file exists.")
    args = p.parse_args()

    warnings.filterwarnings("ignore")
    hier = build_hierarchy(levels=args.levels)
    if args.origin + HORIZON > hier.Y.shape[1]:
        raise SystemExit(f"origin {args.origin} + horizon {HORIZON} exceeds {hier.Y.shape[1]} days")
    train = hier.slice_days(args.origin)
    origin_date = hier.dates[args.origin - 1]

    log_path = OUT_DIR / f"origin_{args.origin}" / "timings.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        fn = baselines.all_models()[model]
        for lvl in args.levels:
            path = OUT_DIR / f"origin_{args.origin}" / model / f"level_{lvl:02d}.parquet"
            if path.exists() and not args.force:
                print(f"skip  {model:15s} L{lvl:<2d} (exists)", flush=True)
                continue
            h = train.level(lvl)
            kwargs = {} if model == "seasonal_naive" else {
                "max_train_days": args.max_train_days, "n_jobs": args.n_jobs}
            t0 = time.perf_counter()
            fc = fn(h, horizon=HORIZON, **kwargs)
            secs = time.perf_counter() - t0
            path.parent.mkdir(parents=True, exist_ok=True)
            baselines.forecast_frame(h, fc, model, origin_date).to_parquet(path, index=False)
            rec = {"model": model, "level": lvl, "n_series": len(h.ids), "seconds": round(secs, 2),
                   "max_train_days": kwargs.get("max_train_days"), "n_jobs": kwargs.get("n_jobs")}
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"done  {model:15s} L{lvl:<2d} {len(h.ids):6d} series {secs:8.1f}s", flush=True)


if __name__ == "__main__":
    main()
