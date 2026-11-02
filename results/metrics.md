# Point-forecast accuracy — rolling origin, expanding window

Origins: 1; 28-day test windows: d_1914-d_1941 (2016-04-25 to 2016-05-22).
Each origin trains on every day before it (expanding window); numbers are means over origins.

## Mean MASE by level (scale: in-sample seasonal naive, lag 7; lower is better)

| level | level_name | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | total | 1.177 | 1.065 | 0.987 |  |  |  | 0.494 |  |
| 2 | state | 1.044 | 0.942 | 1.022 |  |  |  | 0.543 |  |
| 3 | store | 1.031 | 0.964 | 0.964 |  |  |  | 0.738 |  |
| 4 | category | 1.015 | 0.895 | 0.855 |  |  |  | 0.593 |  |
| 5 | department | 1.300 | 1.168 | 1.112 |  |  |  | 0.877 |  |
| 6 | state_category | 0.955 | 0.866 | 0.850 |  |  |  | 0.663 |  |
| 7 | state_department | 1.183 | 1.054 | 1.042 |  |  |  | 0.830 |  |
| 8 | store_category | 0.994 | 0.863 | 0.932 |  |  |  | 0.773 |  |
| 9 | store_department | 1.150 | 0.995 | 1.030 |  |  |  | 0.868 |  |
| 10 | item | 1.104 | 0.879 | 0.914 |  |  |  | 0.895 |  |
| 11 | state_item | 1.132 | 0.920 | 0.933 |  |  |  | 0.931 |  |
| 12 | item_store | 1.175 | 1.034 | 1.036 | 1.057 | 1.043 | 1.044 | 1.052 | 0.871 |

## WAPE by level (volume-weighted)

| level | level_name | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | total | 0.086 | 0.078 | 0.072 |  |  |  | 0.036 |  |
| 2 | state | 0.088 | 0.080 | 0.086 |  |  |  | 0.046 |  |
| 3 | store | 0.097 | 0.090 | 0.089 |  |  |  | 0.063 |  |
| 4 | category | 0.089 | 0.083 | 0.086 |  |  |  | 0.044 |  |
| 5 | department | 0.094 | 0.088 | 0.085 |  |  |  | 0.051 |  |
| 6 | state_category | 0.096 | 0.089 | 0.090 |  |  |  | 0.057 |  |
| 7 | state_department | 0.106 | 0.100 | 0.101 |  |  |  | 0.068 |  |
| 8 | store_category | 0.114 | 0.103 | 0.107 |  |  |  | 0.079 |  |
| 9 | store_department | 0.132 | 0.118 | 0.126 |  |  |  | 0.096 |  |
| 10 | item | 0.364 | 0.300 | 0.317 |  |  |  | 0.311 |  |
| 11 | state_item | 0.555 | 0.451 | 0.463 |  |  |  | 0.460 |  |
| 12 | item_store | 0.862 | 0.721 | 0.731 | 0.755 | 0.746 | 0.734 | 0.735 | 0.685 |

## Share of series where the model beats seasonal naive on MASE

| level | level_name | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | total | 0.000 | 1.000 | 1.000 |  |  |  | 1.000 |  |
| 2 | state | 0.000 | 1.000 | 0.667 |  |  |  | 1.000 |  |
| 3 | store | 0.000 | 0.900 | 0.700 |  |  |  | 0.800 |  |
| 4 | category | 0.000 | 1.000 | 0.667 |  |  |  | 1.000 |  |
| 5 | department | 0.000 | 1.000 | 0.857 |  |  |  | 0.857 |  |
| 6 | state_category | 0.000 | 0.778 | 0.667 |  |  |  | 1.000 |  |
| 7 | state_department | 0.000 | 0.714 | 0.667 |  |  |  | 0.905 |  |
| 8 | store_category | 0.000 | 0.833 | 0.733 |  |  |  | 0.867 |  |
| 9 | store_department | 0.000 | 0.814 | 0.729 |  |  |  | 0.900 |  |
| 10 | item | 0.000 | 0.901 | 0.839 |  |  |  | 0.849 |  |
| 11 | state_item | 0.000 | 0.826 | 0.801 |  |  |  | 0.796 |  |
| 12 | item_store | 0.000 | 0.644 | 0.631 | 0.629 | 0.640 | 0.613 | 0.626 | 0.724 |

## Cross-level average (12 levels weighted equally)

| model | mase_mean | wape |
|---|---|---|
| seasonal_naive | 1.105 | 0.224 |
| ets | 0.970 | 0.192 |
| arima | 0.973 | 0.196 |
| lightgbm | 0.771 | 0.171 |

## Coverage (which origins and levels each model was run on)

| model | origins | levels |
|---|---|---|
| seasonal_naive | 1913 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| ets | 1913 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| arima | 1913 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| croston | 1913 | 12 |
| croston_sba | 1913 | 12 |
| tsb | 1913 | 12 |
| lightgbm | 1913 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| lightgbm_q50 | 1913 | 12 |
