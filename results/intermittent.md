# Intermittent demand — bottom level (30,490 series)

Origins: d_1913; numbers averaged over origins.

## Demand classes (Syntetos-Boylan-Croston, on each origin's training window)

| class | share | zero_share |
|---|---|---|
| smooth | 0.062 | 0.141 |
| erratic | 0.028 | 0.169 |
| intermittent | 0.728 | 0.672 |
| lumpy | 0.182 | 0.541 |

## Mean MASE by demand class

| group | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|
| smooth | 0.958 | 0.759 | 0.778 | 0.793 | 0.784 | 0.786 | 0.767 | 0.768 |
| erratic | 0.917 | 0.762 | 0.766 | 0.791 | 0.781 | 0.770 | 0.772 | 0.763 |
| intermittent | 1.208 | 1.082 | 1.082 | 1.101 | 1.087 | 1.090 | 1.102 | 0.878 |
| lumpy | 1.156 | 0.977 | 0.982 | 1.011 | 0.999 | 0.987 | 0.995 | 0.899 |
| intermittent+lumpy (ADI>=1.32) | 1.198 | 1.061 | 1.062 | 1.083 | 1.069 | 1.070 | 1.081 | 0.882 |
| all | 1.175 | 1.034 | 1.036 | 1.057 | 1.043 | 1.044 | 1.052 | 0.871 |

## WAPE by demand class

| group | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|
| smooth | 0.549 | 0.440 | 0.452 | 0.466 | 0.459 | 0.459 | 0.445 | 0.449 |
| erratic | 0.712 | 0.600 | 0.604 | 0.635 | 0.627 | 0.605 | 0.616 | 0.603 |
| intermittent | 1.036 | 0.884 | 0.893 | 0.920 | 0.909 | 0.896 | 0.900 | 0.804 |
| lumpy | 0.974 | 0.804 | 0.814 | 0.842 | 0.831 | 0.814 | 0.824 | 0.774 |
| intermittent+lumpy (ADI>=1.32) | 1.013 | 0.855 | 0.864 | 0.891 | 0.880 | 0.866 | 0.872 | 0.793 |
| all | 0.862 | 0.721 | 0.731 | 0.755 | 0.746 | 0.734 | 0.735 | 0.685 |

## Bias: (sum forecast - sum actual) / sum actual

| group | seasonal_naive | ets | arima | croston | croston_sba | tsb | lightgbm | lightgbm_q50 |
|---|---|---|---|---|---|---|---|---|
| smooth | -0.030 | -0.008 | 0.004 | 0.029 | -0.023 | 0.015 | 0.023 | -0.103 |
| erratic | -0.122 | -0.082 | -0.055 | -0.004 | -0.053 | -0.071 | -0.015 | -0.238 |
| intermittent | -0.072 | -0.055 | -0.057 | -0.009 | -0.058 | -0.035 | -0.056 | -0.495 |
| lumpy | -0.102 | -0.077 | -0.068 | -0.022 | -0.071 | -0.051 | -0.059 | -0.404 |
| intermittent+lumpy (ADI>=1.32) | -0.083 | -0.063 | -0.061 | -0.014 | -0.063 | -0.041 | -0.057 | -0.462 |
| all | -0.074 | -0.051 | -0.044 | -0.002 | -0.052 | -0.030 | -0.032 | -0.346 |
