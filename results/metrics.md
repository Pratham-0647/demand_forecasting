# Point-forecast accuracy — rolling origin, expanding window

Origins: 4; 28-day test windows: d_1830-d_1857 (2016-02-01 to 2016-02-28), d_1858-d_1885 (2016-02-29 to 2016-03-27), d_1886-d_1913 (2016-03-28 to 2016-04-24), d_1914-d_1941 (2016-04-25 to 2016-05-22).
Each origin trains on every day before it (expanding window); numbers are means over origins.

## Mean MASE by level (scale: in-sample seasonal naive, lag 7; lower is better)

| level | level_name | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | total | 1.149 | 0.853 | 0.961 |  |  |  | 0.642 |  |
| 2 | state | 1.072 | 0.875 | 0.893 |  |  |  | 0.670 |  |
| 3 | store | 1.091 | 0.898 | 0.912 |  |  |  | 0.780 |  |
| 4 | category | 1.090 | 0.844 | 0.841 |  |  |  | 0.670 |  |
| 5 | department | 1.257 | 1.122 | 0.990 |  |  |  | 0.823 |  |
| 6 | state_category | 1.051 | 0.854 | 0.818 |  |  |  | 0.732 |  |
| 7 | state_department | 1.178 | 1.040 | 0.976 |  |  |  | 0.841 |  |
| 8 | store_category | 1.070 | 0.866 | 0.918 |  |  |  | 0.811 |  |
| 9 | store_department | 1.179 | 1.000 | 0.997 |  |  |  | 0.891 |  |
| 10 | item | 1.102 | 0.883 | 0.912 |  |  |  | 0.905 |  |
| 11 | state_item | 1.115 | 0.912 | 0.933 |  |  |  | 0.932 |  |
| 12 | item_store | 1.153 | 1.032 | 1.036 | 1.091 | 1.075 | 1.027 | 1.051 | 0.871 |

## WAPE by level (volume-weighted)

| level | level_name | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | total | 0.087 | 0.064 | 0.071 |  |  |  | 0.049 |  |
| 2 | state | 0.095 | 0.078 | 0.078 |  |  |  | 0.060 |  |
| 3 | store | 0.107 | 0.088 | 0.088 |  |  |  | 0.073 |  |
| 4 | category | 0.096 | 0.074 | 0.082 |  |  |  | 0.057 |  |
| 5 | department | 0.108 | 0.091 | 0.084 |  |  |  | 0.067 |  |
| 6 | state_category | 0.110 | 0.091 | 0.088 |  |  |  | 0.070 |  |
| 7 | state_department | 0.124 | 0.108 | 0.100 |  |  |  | 0.083 |  |
| 8 | store_category | 0.128 | 0.106 | 0.108 |  |  |  | 0.089 |  |
| 9 | store_department | 0.149 | 0.127 | 0.126 |  |  |  | 0.107 |  |
| 10 | item | 0.387 | 0.325 | 0.326 |  |  |  | 0.337 |  |
| 11 | state_item | 0.576 | 0.474 | 0.463 |  |  |  | 0.487 |  |
| 12 | item_store | 0.875 | 0.741 | 0.731 | 0.788 | 0.778 | 0.747 | 0.761 | 0.685 |

## Share of series where the model beats seasonal naive on MASE

| level | level_name | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | total | 0.000 | 1.000 | 1.000 |  |  |  | 1.000 |  |
| 2 | state | 0.000 | 1.000 | 0.833 |  |  |  | 1.000 |  |
| 3 | store | 0.000 | 0.900 | 0.750 |  |  |  | 0.800 |  |
| 4 | category | 0.000 | 0.917 | 0.667 |  |  |  | 1.000 |  |
| 5 | department | 0.000 | 0.786 | 0.857 |  |  |  | 0.929 |  |
| 6 | state_category | 0.000 | 0.889 | 0.833 |  |  |  | 0.944 |  |
| 7 | state_department | 0.000 | 0.738 | 0.786 |  |  |  | 0.881 |  |
| 8 | store_category | 0.000 | 0.883 | 0.767 |  |  |  | 0.833 |  |
| 9 | store_department | 0.000 | 0.832 | 0.779 |  |  |  | 0.843 |  |
| 10 | item | 0.000 | 0.878 | 0.838 |  |  |  | 0.838 |  |
| 11 | state_item | 0.000 | 0.802 | 0.801 |  |  |  | 0.770 |  |
| 12 | item_store | 0.000 | 0.615 | 0.631 | 0.600 | 0.611 | 0.588 | 0.596 | 0.724 |

## Cross-level average (12 levels weighted equally)

| model | mase_mean | wape |
|---|---|---|
| seasonal_naive | 1.126 | 0.237 |
| ets | 0.932 | 0.197 |
| lightgbm | 0.812 | 0.187 |

## Coverage (which origins and levels each model was run on)

| model | origins | levels |
|---|---|---|
| seasonal_naive | 1829, 1857, 1885, 1913 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| ets | 1829, 1857, 1885, 1913 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| arima | 1885, 1913 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| croston | 1829, 1857, 1885, 1913 | 12 |
| croston_sba | 1829, 1857, 1885, 1913 | 12 |
| tsb | 1829, 1857, 1885, 1913 | 12 |
| lightgbm | 1829, 1857, 1885, 1913 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 |
| lightgbm_q50 | 1913 | 12 |
