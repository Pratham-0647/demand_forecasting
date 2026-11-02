# Probabilistic forecasts — P10 / P50 / P90 at the bottom level

Origins: d_1913.

## Pinball loss (raw units) and scaled pinball loss (÷ MASE scale)

| model | pinball_q10 | pinball_q50 | pinball_q90 | spl_q10 | spl_q50 | spl_q90 |
|---|---|---|---|---|---|---|
| lightgbm_quantile | 0.1415 | 0.4941 | 0.3240 | 0.1026 | 0.4357 | 0.3508 |
| weekday_empirical_8w | 0.1408 | 0.5073 | 0.3832 | 0.1048 | 0.4467 | 0.4102 |

## Calibration (a correct alpha-quantile has share_y_lt_q <= alpha <= share_y_le_q)

| model | quantile | nominal | share_y_le_q | share_y_lt_q |
|---|---|---|---|---|
| lightgbm_quantile | q10 | 0.100 | 0.545 | 0.008 |
| lightgbm_quantile | q50 | 0.500 | 0.625 | 0.274 |
| lightgbm_quantile | q90 | 0.900 | 0.892 | 0.890 |
| weekday_empirical_8w | q10 | 0.100 | 0.561 | 0.013 |
| weekday_empirical_8w | q50 | 0.500 | 0.671 | 0.146 |
| weekday_empirical_8w | q90 | 0.900 | 0.919 | 0.719 |

## 80% interval coverage and quantile crossing

| model | coverage_p10_p90 | crossing_rate |
|---|---|---|
| lightgbm_quantile | 0.883 | 0.000 |
| weekday_empirical_8w | 0.906 | 0.000 |

## LightGBM by demand class

| group | spl_q50 | spl_q90 | le_q90 | lt_q90 | coverage_p10_p90 | crossing_rate |
|---|---|---|---|---|---|---|
| all | 0.436 | 0.351 | 0.892 | 0.890 | 0.883 | 0.000 |
| smooth | 0.384 | 0.204 | 0.905 | 0.905 | 0.864 | 0.000 |
| erratic | 0.381 | 0.236 | 0.881 | 0.881 | 0.818 | 0.006 |
| intermittent | 0.439 | 0.371 | 0.893 | 0.890 | 0.890 | 0.000 |
| lumpy | 0.450 | 0.339 | 0.884 | 0.884 | 0.871 | 0.000 |
