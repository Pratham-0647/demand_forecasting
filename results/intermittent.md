# Intermittent demand — bottom level (30,490 series)

Origins: d_1829, d_1857, d_1885, d_1913; numbers averaged over origins. Columns marked with an origin count cover fewer windows and are not directly comparable.

## Demand classes (Syntetos-Boylan-Croston, on each origin's training window)

| class | share | zero_share |
|---|---|---|
| smooth | 0.062 | 0.141 |
| erratic | 0.028 | 0.168 |
| intermittent | 0.728 | 0.673 |
| lumpy | 0.182 | 0.543 |

## Mean MASE by demand class

| group | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 (1 origin) |
|---|---|---|---|---|---|---|---|---|
| smooth | 0.975 | 0.773 | 0.792 | 0.812 | 0.804 | 0.795 | 0.781 | 0.768 |
| erratic | 0.891 | 0.738 | 0.745 | 0.790 | 0.777 | 0.739 | 0.764 | 0.763 |
| intermittent | 1.188 | 1.086 | 1.077 | 1.147 | 1.130 | 1.077 | 1.104 | 0.878 |
| lumpy | 1.115 | 0.952 | 0.957 | 1.006 | 0.992 | 0.954 | 0.975 | 0.899 |
| intermittent+lumpy (ADI>=1.32) | 1.174 | 1.059 | 1.053 | 1.119 | 1.103 | 1.052 | 1.078 | 0.882 |
| all | 1.153 | 1.032 | 1.028 | 1.091 | 1.075 | 1.027 | 1.051 | 0.871 |

## WAPE by demand class

| group | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 (1 origin) |
|---|---|---|---|---|---|---|---|---|
| smooth | 0.563 | 0.455 | 0.464 | 0.479 | 0.474 | 0.467 | 0.457 | 0.449 |
| erratic | 0.748 | 0.629 | 0.635 | 0.673 | 0.661 | 0.626 | 0.650 | 0.603 |
| intermittent | 1.041 | 0.897 | 0.905 | 0.962 | 0.950 | 0.904 | 0.927 | 0.804 |
| lumpy | 1.004 | 0.847 | 0.855 | 0.895 | 0.882 | 0.849 | 0.872 | 0.774 |
| intermittent+lumpy (ADI>=1.32) | 1.027 | 0.879 | 0.887 | 0.938 | 0.925 | 0.884 | 0.907 | 0.793 |
| all | 0.875 | 0.741 | 0.749 | 0.788 | 0.778 | 0.747 | 0.761 | 0.685 |

## Bias: (sum forecast - sum actual) / sum actual

| group | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 (1 origin) |
|---|---|---|---|---|---|---|---|---|
| smooth | -0.045 | -0.020 | -0.013 | 0.007 | -0.043 | -0.005 | -0.020 | -0.103 |
| erratic | -0.036 | 0.003 | 0.024 | 0.066 | 0.012 | 0.012 | 0.045 | -0.238 |
| intermittent | -0.072 | -0.048 | -0.053 | 0.016 | -0.035 | -0.035 | -0.057 | -0.495 |
| lumpy | -0.083 | -0.044 | -0.036 | 0.004 | -0.046 | -0.034 | -0.041 | -0.404 |
| intermittent+lumpy (ADI>=1.32) | -0.076 | -0.047 | -0.047 | 0.012 | -0.039 | -0.035 | -0.051 | -0.462 |
| all | -0.064 | -0.035 | -0.030 | 0.016 | -0.035 | -0.022 | -0.033 | -0.346 |
