# Hierarchical Demand Forecasting (M5)

Forecasting daily unit sales for Walmart's M5 dataset across the full 12-level
hierarchy, starting from honest baselines.

## Data

`data/` holds the M5 files as zstd parquet (converted from CSV by `data/shrink.py`):

| File | Shape | Contents |
| --- | --- | --- |
| `sales_train_evaluation.parquet` | 30,490 × 1,947 | Daily unit sales d_1–d_1941 (2011-01-29 → 2016-05-22) |
| `calendar.parquet` | 1,969 × 14 | Dates, weekdays, events, SNAP flags |
| `sell_prices.parquet` | 6,841,121 × 4 | Weekly price per store × item |

Verified: 30,490 bottom series (3,049 items × 10 stores), 1,941 days, **68.0% zero-sales days**.

### Hierarchy

`m5fc.data.build_hierarchy()` aggregates the bottom series into the 12 M5 levels
with a sparse summing matrix — **42,840 series** in total:

| Level | Grouping | Series |
| ---: | --- | ---: |
| 1 | Total | 1 |
| 2 | State | 3 |
| 3 | Store | 10 |
| 4 | Category | 3 |
| 5 | Department | 7 |
| 6 | State × Category | 9 |
| 7 | State × Department | 21 |
| 8 | Store × Category | 30 |
| 9 | Store × Department | 70 |
| 10 | Item | 3,049 |
| 11 | State × Item | 9,147 |
| 12 | Item × Store | 30,490 |

## Phase 1 — Baselines

`m5fc/baselines.py`, run with `scripts/run_baselines.py`. Default forecast origin
is day 1913: train on d_1–d_1913, forecast the 28 days d_1914–d_1941 (the M5
evaluation window). Every model is fit on every one of the 42,840 series.

| Model | Implementation | Notes |
| --- | --- | --- |
| Seasonal naive | numpy | Forecast = value 7 days earlier (repeat the last week). The MASE reference. |
| ETS | `statsforecast.AutoETS(season_length=7)` | Error/trend/season form chosen per series by AICc. |
| ARIMA | `statsforecast.AutoARIMA(season_length=7)` | Orders capped at p,q ≤ 2, P,Q ≤ 1; CSS approximation during order search. |

Design choices:

- **Leading zeros are trimmed** before fitting. In M5 they mostly mean the item
  was not yet stocked, not zero demand.
- **ETS and ARIMA see the last 730 days** (`--max-train-days`). Two years covers
  the weekly and annual pattern and bounds cost.
- **Restricted ARIMA search.** The default AutoARIMA search measured ~1.2 s/series
  on 4 cores (~14 h for all series); the restricted search takes ~0.13 s/series.
- **One BLAS thread per worker.** With default threading, the statsforecast
  workers oversubscribed the CPU and ETS ran ~10× slower.
- Forecasts are clipped at zero and saved per `(origin, model, level)` under
  `outputs/forecasts/` (git-ignored), so an interrupted run resumes.

```bash
pip install -r requirements.txt
python scripts/run_baselines.py            # all models, all 12 levels
python -m pytest
```
