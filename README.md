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

Full run at origin d_1913 (4 cores): seasonal naive < 1 s, ETS 37 min,
ARIMA 104 min for all 42,840 series.

## Phase 2 — Metrics

`m5fc/metrics.py`, run with `scripts/evaluate.py`.

- **MASE** = MAE of the forecast ÷ in-sample MAE of the lag-7 seasonal naive,
  computed per series from its first sale. Below 1 beats the naive's typical
  in-sample error. Scale-free, so it averages across series of any volume.
  The denominator is in-sample, so the seasonal naive itself does not score
  exactly 1 on the test window (it scores 1.18 at the bottom level here).
- **WAPE** = Σ|error| ÷ Σ|actual| over all series and days of a level: a
  volume-weighted view in which high sellers dominate.
- **No MAPE.** 68% of bottom-level days are zero, so MAPE divides by zero
  on most observations. `metrics.mape` is kept only to demonstrate this.
- Levels are averaged with equal weight (the M5 convention), so the 30,490
  bottom series do not swamp the single total series.

### Baseline results — origin d_1913 (forecasting 2016-04-25 → 2016-05-22)

| Level | Series | MASE sNaive | MASE ETS | MASE ARIMA | WAPE sNaive | WAPE ETS | WAPE ARIMA |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 Total | 1 | 1.177 | 1.065 | **0.987** | 0.086 | 0.078 | **0.072** |
| 2 State | 3 | 1.044 | **0.942** | 1.022 | 0.088 | **0.080** | 0.086 |
| 3 Store | 10 | 1.031 | **0.964** | **0.964** | 0.097 | 0.090 | **0.089** |
| 4 Category | 3 | 1.015 | 0.895 | **0.855** | 0.089 | **0.083** | 0.086 |
| 5 Department | 7 | 1.300 | 1.168 | **1.112** | 0.094 | 0.088 | **0.085** |
| 6 State × Cat | 9 | 0.955 | 0.866 | **0.850** | 0.096 | **0.089** | 0.090 |
| 7 State × Dept | 21 | 1.183 | 1.054 | **1.042** | 0.106 | **0.100** | 0.101 |
| 8 Store × Cat | 30 | 0.994 | **0.863** | 0.932 | 0.114 | **0.103** | 0.107 |
| 9 Store × Dept | 70 | 1.150 | **0.995** | 1.030 | 0.132 | **0.118** | 0.126 |
| 10 Item | 3,049 | 1.104 | **0.879** | 0.914 | 0.364 | **0.300** | 0.317 |
| 11 State × Item | 9,147 | 1.132 | **0.920** | 0.933 | 0.555 | **0.451** | 0.463 |
| 12 Item × Store | 30,490 | 1.175 | **1.034** | 1.036 | 0.862 | **0.721** | 0.731 |
| **Avg of 12 levels** | | 1.105 | **0.970** | 0.973 | 0.224 | **0.192** | 0.196 |

Readings:

- ETS and ARIMA beat the seasonal naive at **every** level, on both metrics.
  At the bottom level ETS beats it on 64% of individual series.
- The test window is harder than the training history: the seasonal naive
  scores MASE > 1 at most levels. The models also only just beat their own
  in-sample naive error at the bottom (MASE ≈ 1.03).
- WAPE climbs from 7–9% at aggregate levels to 72% at item × store: noise
  that cancels in aggregates dominates the bottom. This is the motivation
  for reconciliation (Phase 7).
