#!/usr/bin/env bash
# Full compute queue, run sequentially (one job at a time keeps memory under the
# container limit). Every step skips outputs that already exist, so the script
# can be re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
FINAL=1913
EARLIER="1885 1857 1829"
INTERMITTENT="croston croston_sba tsb"

# 1. Final origin: baselines on all levels, LightGBM point + quantiles, Croston family.
python scripts/run_baselines.py --origin $FINAL
OMP_NUM_THREADS=4 python scripts/run_gbm.py --origin $FINAL
python scripts/run_baselines.py --origin $FINAL --models $INTERMITTENT --levels 12

# 2. Earlier origins (rolling origin, expanding window): fast models first.
for o in $EARLIER; do
  python scripts/run_baselines.py --origin $o --models seasonal_naive ets
  python scripts/run_baselines.py --origin $o --models $INTERMITTENT --levels 12
  OMP_NUM_THREADS=4 python scripts/run_gbm.py --origin $o --targets point
done

# 3. ARIMA at the earlier origins (the slowest model, ~1.3 h per origin).
for o in $EARLIER; do
  python scripts/run_baselines.py --origin $o --models arima
done
echo "ALL DONE"
