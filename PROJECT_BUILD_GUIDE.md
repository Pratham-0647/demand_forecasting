# Project Build Guide — Pratham Manchanda

**Purpose:** complete build instructions for every project on the three CVs. Written so it can be pasted into a fresh Claude chat as context.

**Status of each project:**

| \# | Project | On CVs | Status |
| :---- | :---- | :---- | :---- |
| 1 | Real-Time Fraud Detection Service | CV1, CV2 | **To build** |
| 2 | Equity Factor Research & Systematic Backtest | CV1 (ML-framed), CV2 (finance-framed) | **To build** |
| 3 | Hierarchical Demand Forecasting | CV1, CV2, CV3 | **To build** |
| 4 | Inventory Optimization (SQL) | CV1, CV2, CV3 | Built |
| 5 | Revenue Leakage — Hospitality | CV3 | Built |
| 6 | BI Dashboard for Sales Analytics | CV3 | Built |

**Deadline:** interviews December 2026\. Build order: fraud → equity factors → forecasting.

---

# Environment setup

pip install pandas numpy scikit-learn lightgbm matplotlib seaborn

pip install fastapi uvicorn pydantic mlflow pytest        \# project 1

pip install yfinance statsmodels                          \# project 2

pip install statsforecast                                 \# project 3

Data already downloaded, in `F:\projects\data\`:

- `fraud/` — `train_transaction.csv` (683 MB), `train_identity.csv` (26 MB)  
- `m5/` — `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`  
- `online_retail/` — `online_retail_II.csv`

---

---

# PROJECT 1 — Real-Time Fraud Detection Service

**Stack:** Python, LightGBM, FastAPI, Docker, MLflow **CV dates:** Jun 2026 – Aug 2026 **Time to build:** 3–4 weeks

## What the CV claims — everything here must become true

- End-to-end pipeline on **590,540** transactions, **434** features, **3.5%** fraud rate (1:27.6 imbalance)  
- Velocity and entity-aggregate features across **13,553** card identifiers under point-in-time constraints  
- Evaluated on **PR-AUC** and **precision@k** against logistic and majority-class baselines  
- Threshold set from a **cost matrix**  
- Deployed **Dockerized FastAPI** endpoint with **PSI drift monitoring** and unit tests in **GitHub Actions CI**

## Dataset

**IEEE-CIS Fraud Detection** (Kaggle competition). Already at `F:\projects\data\fraud\`.

- `train_transaction.csv` — 590,540 rows, 394 columns  
- `train_identity.csv` — 144,233 rows, 41 columns (only 24.4% of transactions have identity data)  
- Join key: `TransactionID`  
- Target: `isFraud` (20,663 positives \= 3.50%)  
- `TransactionDT` — seconds from an arbitrary reference point, spanning 182 days. **This is your time axis.**  
- `card1`–`card6`, `addr1`, `addr2`, `P_emaildomain` — entity identifiers, used for aggregate features  
- `V1`–`V339` — anonymised engineered features  
- `C1`–`C14` — counting features  
- `D1`–`D15` — timedelta features

## Build phases

### Phase 1 — Baseline (2–3 days)

Get a number before doing anything clever.

1. Load and merge on `TransactionID`, left join.  
2. **Sort by `TransactionDT`, split 80/20.** Train on the past, validate on the future. Never use `train_test_split` with shuffling — the same card appears repeatedly, and a random split leaks future behaviour of a card into training.  
3. Label-encode object columns; LightGBM handles integer-coded categoricals natively.  
4. Train LightGBM, default-ish params.  
5. Report PR-AUC, ROC-AUC, precision@100/1000/5000.  
6. **Also train a logistic regression and a majority-class baseline.** Your CV claims you benchmarked against both.

Expected range: ROC-AUC 0.90–0.93 for a competent baseline. If you see 0.98+, you have leakage.

### Phase 2 — Feature engineering (1 week)

This is where the CV claim about *velocity and entity-aggregate features under point-in-time constraints* is earned.

**Velocity features** — for each card, how many transactions in the last 1h / 24h / 7d, and the sum and mean of `TransactionAmt` over those windows.

**Entity aggregates** — for each `card1`, `addr1`, `P_emaildomain`: transaction count, mean amount, std of amount, time since last transaction.

**The point-in-time rule:** every feature for a transaction at time *t* must use only data from before *t*. The trap is computing `df.groupby('card1')['TransactionAmt'].mean()` over the whole dataset — that includes future transactions and inflates your score. Use expanding windows (`groupby().expanding()`) or explicit rolling joins on `TransactionDT`.

**How to verify you haven't leaked:** if a feature causes a large jump in validation score, be suspicious. Drop it and see whether performance falls to a plausible level.

### Phase 3 — Threshold from a cost matrix (2 days)

A fraud model outputs a probability; the business needs a decision. Choose the cutoff by expected cost, not 0.5.

cost \= (false\_negatives × average\_fraud\_loss) \+ (false\_positives × cost\_of\_blocking\_a\_good\_customer)

Assume something defensible — e.g. fraud loss \= transaction amount, false-positive cost \= ₹200 of friction — sweep the threshold from 0 to 1, plot cost against threshold, pick the minimum. **Put that plot in the README.**

### Phase 4 — FastAPI service (3–4 days)

- `POST /score` — accepts a transaction JSON, returns `{fraud_probability, decision, model_version}`  
- `GET /health` — returns 200  
- Validate inputs with a `pydantic` model  
- Load the model once at startup, not per request  
- Measure and record p95 latency under repeated calls

### Phase 5 — Docker (2 days)

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install \--no-cache-dir \-r requirements.txt

COPY . .

CMD \["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"\]

`docker build -t fraud-api .` then `docker run -p 8000:8000 fraud-api`. Deploy free on Render, Railway or Hugging Face Spaces.

### Phase 6 — Drift monitoring (3 days)

**PSI (Population Stability Index)** measures how much a feature's distribution has shifted:

PSI \= Σ (actual% − expected%) × ln(actual% / expected%)

across bins of the feature. Convention: below 0.1 \= stable, 0.1–0.25 \= moderate shift, above 0.25 \= significant. Build a small Streamlit page showing PSI per feature, the score distribution, and the alert rate over time.

### Phase 7 — Tests and CI (2 days)

At least 8 tests: feature functions return expected values on toy inputs, no NaNs where they shouldn't be, the API returns 200 with valid input and 422 with invalid, the model file loads. Wire into `.github/workflows/test.yml`.

## Concepts you must be able to explain

**Why PR-AUC over ROC-AUC?** At a 3.5% positive rate, ROC-AUC is dominated by the vast negative class and looks flattering. PR-AUC focuses on the positive class, which is what a review team experiences.

**What is precision@k?** Of the *k* transactions your model ranks riskiest, what fraction were actually fraud. It matches how fraud teams work — they can only review so many per day.

**Why time-based splitting?** Fraud patterns drift, and the same card recurs. Random splitting lets the model learn a card's future behaviour.

**What is PSI and why monitor it?** True labels arrive with a lag — a chargeback may take 60 days. So you can't measure accuracy live. PSI on inputs is a proxy that tells you the world has changed before your metrics do.

## Definition of done

- [ ] Live public URL accepting POST requests  
- [ ] PR-AUC, precision@k vs. logistic and majority-class baselines, in a table  
- [ ] Cost-vs-threshold plot with the chosen operating point  
- [ ] ≥8 tests passing in GitHub Actions  
- [ ] PSI dashboard live  
- [ ] p95 latency measured  
- [ ] README with architecture diagram and a limitations section

---

---

# PROJECT 2 — Equity Factor Research & Systematic Backtest

**Stack:** Python, SQL, LightGBM, Power BI **CV dates:** Jan 2026 – Mar 2026 **Time to build:** 4–5 weeks. **This is the hardest project and it goes to the sharpest audience.**

## What the CV claims

- **7 factors** — 12-1 momentum, short-term reversal, idiosyncratic volatility, liquidity, beta (plus two more)  
- **200** NSE-listed equities over **11 years** (2,862 trading days)  
- Winsorization and sector-neutral cross-sectional standardization  
- **Liquidity-defined point-in-time universe in SQL** to eliminate survivorship bias  
- Signal quality via **rank IC, IC t-statistics, decay curves, decile-spread monotonicity**  
- LightGBM ranking model under **purged, embargoed walk-forward cross-validation**  
- Long-short decile backtest **net of transaction costs and turnover**  
- **Sharpe, Sortino, maximum drawdown** and parameter-sensitivity analysis

## Dataset

`yfinance`, free, verified working. Nifty 200 constituents with `.NS` suffix (e.g. `RELIANCE.NS`).

Verified: 47 of 50 large-caps sampled have full history; 2,862 trading days from 2015-01-01 to 2026-07-31.

Note: `TATAMOTORS.NS` returns "possibly delisted" — a live example of survivorship bias to cite in interviews.

## The core idea — read this twice

This is a **cross-sectional** problem, not a time-series one. You are not asking "will Reliance go up?" You are asking: *"across 200 stocks on this date, which will outperform the others?"*

That reframing is what makes the project serious. It gives you portfolio construction, information coefficients, factor decay and multiple-testing problems — none of which exist in a price-prediction project.

## Build phases

### Phase 1 — Data layer (4–5 days)

1. Pull daily OHLCV for your ticker list, `auto_adjust=True`.  
2. Store in SQLite: `prices(date, ticker, open, high, low, close, volume)` and `sectors(ticker, sector)`.  
3. Write the universe query in **SQL**, not pandas — this is where the SQL claim is earned.

**Universe construction (survivorship bias fix):** at each monthly rebalance date, select the top 200 tickers by trailing 60-day median traded value (close × volume). This is causal, reproducible, and needs no historical index membership. Explaining *why* you did this is a strong interview answer.

### Phase 2 — Factors (1 week)

All price/volume derived, no fundamentals — free fundamentals aren't point-in-time and would introduce look-ahead bias you can't fix.

1. **12-1 momentum** — return from t-252 to t-21. Skip the last month because short-term reversal works against momentum there.  
2. **Short-term reversal** — negative of the last 21-day return.  
3. **Low volatility** — inverse of trailing 60-day realised volatility.  
4. **Idiosyncratic volatility** — std of residuals from a rolling regression of stock returns on Nifty returns.  
5. **Liquidity** — trailing 60-day median traded value.  
6. **52-week high proximity** — close ÷ 252-day rolling max.  
7. **Beta** — rolling 252-day regression slope against the index.

### Phase 3 — Cross-sectional processing (3 days)

At **each** rebalance date, independently:

1. Winsorize at the 1st and 99th percentile  
2. Z-score across the cross-section  
3. Demean within sector

Every step uses only that date's data. Never compute statistics over the full sample.

### Phase 4 — Signal evaluation (1 week) — the heart of the project

**Rank Information Coefficient:** Spearman correlation between factor rank and forward-return rank, computed at each rebalance date. You get a time series of ICs.

- **Mean IC** — 0.02–0.05 monthly is a respectable, real signal. **Above 0.10 sustained almost always means a bug or look-ahead bias.**  
- **IC t-statistic** \= mean(IC) ÷ std(IC) × √n. Above 2 suggests the signal isn't noise.  
- **Decay curve** — IC at 5, 10, 21, 63-day horizons. Tells you the natural holding period.  
- **Decile spread** — return of top decile minus bottom decile. Check **monotonicity**: does decile 1 \< 2 \< 3 … \< 10? A factor that only works at the extremes is fragile.  
- **Factor correlation matrix** — are your seven factors seven signals, or three?

### Phase 5 — ML ranking model (1 week)

LightGBM predicting forward cross-sectional return rank from the factor set.

**Purged, embargoed walk-forward CV — the critical piece.** With a 21-day forward return label, consecutive samples overlap in time. A sample from day 100 has a label extending to day 121; if day 110 is in your validation set, training on day 100 leaks. So:

- **Purge:** drop training samples whose label window overlaps the validation window  
- **Embargo:** additionally drop a buffer (say 5 days) after the validation window, because of serial correlation

**Baselines you must beat:** best single factor, and equal-weight composite. If LightGBM doesn't beat the equal-weight average — which happens often at this signal-to-noise ratio — **that is your finding, and it's a good one.**

### Phase 6 — Portfolio and backtest (1 week)

- Long top decile, short bottom decile; also a long-only variant vs Nifty 50  
- Equal-weight vs volatility-scaled position sizing  
- Monthly vs weekly rebalancing, compared on turnover  
- Transaction costs in basis points plus slippage  
- **Headline chart:** performance vs assumed cost, with the breakeven marked

### Phase 7 — Risk and robustness (4 days)

- CAGR, **Sharpe** (excess return ÷ volatility), **Sortino** (÷ downside deviation only), **maximum drawdown**, **Calmar** (return ÷ max DD), hit rate, turnover  
- Rolling Sharpe, drawdown chart, monthly return heatmap  
- Residual market beta — is your long-short actually market-neutral?  
- **Parameter-sensitivity heatmap** — does the strategy work across the lookback grid, or only at one lucky setting?  
- **Count every variant you tested and write the number down.**

### Phase 8 — Output

Power BI dashboard (equity curves, drawdown, monthly heatmap, cost sensitivity) plus a **research report PDF** written like a quant research note: methodology, results, limitations, next steps. That document is a better interview artifact than the code.

## Concepts you must be able to explain

- **Rank IC vs Pearson** — returns are fat-tailed; rank correlation is robust to outliers and captures monotone non-linear relationships.  
- **Why skip a month in momentum** — short-term reversal dominates at the 1-month horizon.  
- **Sector neutralisation** — otherwise your "momentum factor" is partly a sector bet.  
- **Purging and embargo** — see Phase 5\. This is the single best technical talking point in the project.  
- **Sharpe vs Sortino** — Sharpe penalises all volatility including upside; Sortino only penalises downside. If Sortino ≫ Sharpe, your losses are small and gains are lumpy.  
- **Survivorship bias** — see Phase 1\.  
- **Multiple testing** — test 50 variants and one looks great by chance. Report how many you tried.

## Definition of done

- [ ] IC table: mean IC, t-stat, decay curve per factor  
- [ ] Decile spread bars with monotonicity visible  
- [ ] Purged/embargoed CV implemented and documented  
- [ ] Backtest net of costs, with the cost-sensitivity chart  
- [ ] Sharpe, Sortino, max drawdown, Calmar reported  
- [ ] Parameter-sensitivity heatmap  
- [ ] Variant count stated  
- [ ] Research report PDF

---

---

# PROJECT 3 — Hierarchical Demand Forecasting

**Stack:** Python, LightGBM, statsforecast **CV dates:** Oct 2025 – Dec 2025 **Time to build:** 2–3 weeks

## What the CV claims

- **30,490** retail series, **12** aggregation levels, **5.3 years** of history  
- Benchmarked against **seasonal naive, ETS and ARIMA** using **MASE** under **rolling-origin backtesting with expanding windows**  
- Probabilistic **P10/P50/P90** forecasts evaluated by **pinball loss**, reconciled across levels  
- Intermittent demand across **68%** zero-sales days via **Croston-style methods**  
- (CV3 variant) Converted quantile forecasts into **inventory reorder points**

## Dataset

**M5 Forecasting Accuracy**, at `F:\projects\data\m5\`.

Verified figures: 30,490 bottom-level series (3,049 items × 10 stores), 1,941 days (2011-01-29 → 2016-05-22), 3 states, 3 categories, 7 departments, **59,181,090** observations, **68% zero-sales**, 162 calendar events.

The 12 aggregation levels are the M5 standard: total, state, store, category, department, item, and their combinations.

## Build phases

### Phase 1 — Baselines first (3 days)

Most students skip this and never learn their fancy model lost.

- **Seasonal naive** — forecast \= value from 7 days ago. This is the benchmark MASE is defined against.  
- **ETS** and **ARIMA** via `statsforecast` (fast enough at this scale).

### Phase 2 — Metrics (1 day)

**MASE** \= mean absolute error ÷ mean absolute error of the seasonal naive. Below 1 means you beat naive. Scale-free, so it aggregates across series.

**Never use MAPE.** 68% of series-days are zero, so it's undefined or explodes. Being able to explain that is a strong interview moment.

Also report **WAPE** (weighted absolute percentage error) for a volume-weighted view.

### Phase 3 — Rolling-origin backtesting (3 days)

Multiple forecast origins with an expanding window: train on days 1–1000, predict 1001–1028; train 1–1028, predict 1029–1056; and so on. Never a single split, never shuffled.

### Phase 4 — Global model (1 week)

One LightGBM trained across **all** series with lag features (7, 14, 28 days), rolling means, calendar features (day of week, month, SNAP days, events) and price features from `sell_prices.csv`.

Compare against per-series local models. The global approach learns cross-series patterns and handles new items with no history — this is how it's done in industry now.

### Phase 5 — Probabilistic forecasts (4 days)

Inventory decisions need quantiles, not point forecasts. Train LightGBM with `objective='quantile'` at alpha \= 0.1, 0.5, 0.9. Evaluate with **pinball loss**.

**Check calibration:** is your P90 actually exceeded 10% of the time? Most quantile forecasts are badly calibrated and nobody checks.

### Phase 6 — Intermittent demand (3 days)

68% of series-days are zero. Croston's method decomposes demand into size and inter-arrival interval and forecasts each separately. `statsforecast` has `CrostonClassic`.

### Phase 7 — Reconciliation (3 days)

SKU-level forecasts won't sum to category totals. Apply bottom-up or MinT reconciliation and show the coherence gain.

### Phase 8 — Decision layer (CV3 version)

Convert the P90 forecast into a reorder point given a target service level and holding cost. Plot service level against inventory cost and mark the recommended operating point.

## Definition of done

- [ ] Results table: your model vs seasonal naive, ETS, ARIMA — in MASE and WAPE, across multiple origins  
- [ ] Quantile forecasts with pinball loss and a calibration check  
- [ ] Croston comparison on intermittent series  
- [ ] Reconciliation before/after coherence check  
- [ ] Service-level vs holding-cost curve (CV3)  
- [ ] README documenting leakage risks (future promo flags, full-series scaling statistics)

---

---

# PROJECTS 4–6 — Already built: what to be able to defend

These are on your CVs and complete. You won't rebuild them, but you must be able to answer questions.

## Inventory Optimization Using Advanced SQL Analytics

**Claims:** 109K+ rows, advanced SQL, 14.86% stockout risk, 16.07% overstock, 11.28 unit average gap, category concentration in Electronics & Toys vs Clothing & Groceries.

**Expect:** *"How did you define stockout risk?"* — have the exact SQL logic ready. *"Write me a query for SKU-level stock health across stores."* — practise until automatic. *"Window functions or subqueries?"* — know which you used and why.

## Revenue Leakage & Profitability — Hospitality

**Claims:** 14.88% cancellation-driven leakage, ADR/RevPAR/Occupancy, 57.9% occupancy, 11pp drops in low-demand periods, 9–13% RevPAR uplift and 6–10% occupancy improvement projected.

**Expect:** *"Define ADR and RevPAR."* — ADR \= room revenue ÷ rooms sold; RevPAR \= room revenue ÷ rooms available (= ADR × occupancy). *"How did you model the 9–13% uplift?"* — this is a projection, so be clear about the assumptions behind it and don't present it as a realised result.

## BI Dashboard for Sales Analytics

**Claims:** 10K+ records, 5 regions, custom DAX, 35% performance improvement, 40% reduction in manual effort.

**Expect:** *"How did you measure the 35%?"* — **prepare this answer.** Report refresh time before and after is the defensible version. *"What DAX did you write?"* — have two or three measures you can recite. *"What did normalization change?"* — star schema, fewer duplicate rows, smaller model size.

---

---

# Using this document in a new Claude chat

Paste this in and say something like:

> "I'm building Project 1 (fraud detection). I'm at Phase 2, feature engineering. Here's my current code — help me build the velocity features with point-in-time correctness."

Everything Claude needs is here: the dataset paths, the exact CV claims that must become true, the phase structure, and the concepts.

## Priority if time runs short

1. **Fraud detection** — appears on two CVs, has a deployment story nothing else gives you  
2. **Equity factor research** — CV 2's hero, and CV 2 goes to the audience that will interrogate hardest  
3. **Forecasting** — appears on all three but is the most compressible

If only one gets built properly, make it fraud. If CV 2's factor project won't get built, retire CV 2 rather than send it.  
