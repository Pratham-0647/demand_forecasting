# Coherent P10/P50/P90 at every level

Origin d_1913. Quantile = reconciled point forecast + empirical quantile of that method's own errors at earlier origins (per series, pooled over horizons). The centre of every distribution is the coherent reconciled forecast; the quantiles themselves are not additive and should not be (a quantile of a sum is not the sum of quantiles). Caveat: at aggregate levels the 28 errors of one origin are strongly autocorrelated, so 3 prior origins are effectively ~3 independent draws and the intervals under-cover when a window shifts level.


## ets / mint_shrink (prior origins: 3)

| level | level_name | spl_q10 | spl_q50 | spl_q90 | le_q10 | le_q50 | le_q90 |
|---|---|---|---|---|---|---|---|
| 1 | total | 0.190 | 0.468 | 0.205 | 0.036 | 0.179 | 0.821 |
| 2 | state | 0.191 | 0.492 | 0.266 | 0.024 | 0.262 | 0.595 |
| 3 | store | 0.191 | 0.479 | 0.236 | 0.046 | 0.268 | 0.700 |
| 4 | category | 0.190 | 0.416 | 0.211 | 0.024 | 0.226 | 0.798 |
| 5 | department | 0.235 | 0.646 | 0.391 | 0.061 | 0.240 | 0.668 |
| 6 | state_category | 0.189 | 0.440 | 0.219 | 0.048 | 0.317 | 0.778 |
| 7 | state_department | 0.218 | 0.584 | 0.347 | 0.065 | 0.308 | 0.714 |
| 8 | store_category | 0.186 | 0.439 | 0.213 | 0.070 | 0.349 | 0.800 |
| 9 | store_department | 0.213 | 0.553 | 0.276 | 0.073 | 0.345 | 0.764 |
| 10 | item | 0.189 | 0.466 | 0.276 | 0.159 | 0.483 | 0.860 |
| 11 | state_item | 0.165 | 0.467 | 0.290 | 0.301 | 0.521 | 0.867 |
| 12 | item_store | 0.153 | 0.476 | 0.369 | 0.562 | 0.646 | 0.878 |

## ets / wls_struct (prior origins: 3)

| level | level_name | spl_q10 | spl_q50 | spl_q90 | le_q10 | le_q50 | le_q90 |
|---|---|---|---|---|---|---|---|
| 1 | total | 0.166 | 0.442 | 0.179 | 0.071 | 0.250 | 0.893 |
| 2 | state | 0.173 | 0.464 | 0.172 | 0.060 | 0.357 | 0.821 |
| 3 | store | 0.182 | 0.458 | 0.203 | 0.054 | 0.336 | 0.793 |
| 4 | category | 0.173 | 0.386 | 0.195 | 0.060 | 0.321 | 0.869 |
| 5 | department | 0.231 | 0.591 | 0.336 | 0.041 | 0.311 | 0.730 |
| 6 | state_category | 0.178 | 0.417 | 0.201 | 0.052 | 0.393 | 0.825 |
| 7 | state_department | 0.218 | 0.552 | 0.293 | 0.061 | 0.364 | 0.776 |
| 8 | store_category | 0.178 | 0.420 | 0.202 | 0.068 | 0.404 | 0.856 |
| 9 | store_department | 0.208 | 0.519 | 0.257 | 0.070 | 0.373 | 0.816 |
| 10 | item | 0.183 | 0.449 | 0.267 | 0.157 | 0.498 | 0.878 |
| 11 | state_item | 0.160 | 0.455 | 0.283 | 0.298 | 0.529 | 0.878 |
| 12 | item_store | 0.145 | 0.461 | 0.363 | 0.559 | 0.646 | 0.884 |

## ets / bu (prior origins: 3)

| level | level_name | spl_q10 | spl_q50 | spl_q90 | le_q10 | le_q50 | le_q90 |
|---|---|---|---|---|---|---|---|
| 1 | total | 0.168 | 0.464 | 0.181 | 0.107 | 0.250 | 0.857 |
| 2 | state | 0.174 | 0.469 | 0.190 | 0.071 | 0.381 | 0.857 |
| 3 | store | 0.182 | 0.476 | 0.224 | 0.050 | 0.357 | 0.821 |
| 4 | category | 0.185 | 0.455 | 0.232 | 0.071 | 0.369 | 0.857 |
| 5 | department | 0.234 | 0.653 | 0.409 | 0.056 | 0.316 | 0.694 |
| 6 | state_category | 0.186 | 0.448 | 0.221 | 0.060 | 0.421 | 0.849 |
| 7 | state_department | 0.216 | 0.587 | 0.329 | 0.080 | 0.372 | 0.772 |
| 8 | store_category | 0.181 | 0.444 | 0.222 | 0.079 | 0.418 | 0.850 |
| 9 | store_department | 0.208 | 0.547 | 0.274 | 0.082 | 0.371 | 0.815 |
| 10 | item | 0.184 | 0.454 | 0.272 | 0.158 | 0.500 | 0.875 |
| 11 | state_item | 0.160 | 0.456 | 0.286 | 0.298 | 0.531 | 0.876 |
| 12 | item_store | 0.145 | 0.457 | 0.363 | 0.559 | 0.647 | 0.883 |

## ets_lgb / mint_shrink (prior origins: 3)

| level | level_name | spl_q10 | spl_q50 | spl_q90 | le_q10 | le_q50 | le_q90 |
|---|---|---|---|---|---|---|---|
| 1 | total | 0.183 | 0.491 | 0.198 | 0.000 | 0.036 | 0.464 |
| 2 | state | 0.178 | 0.475 | 0.213 | 0.000 | 0.119 | 0.619 |
| 3 | store | 0.188 | 0.481 | 0.202 | 0.007 | 0.157 | 0.693 |
| 4 | category | 0.173 | 0.395 | 0.161 | 0.012 | 0.131 | 0.786 |
| 5 | department | 0.237 | 0.660 | 0.346 | 0.015 | 0.224 | 0.617 |
| 6 | state_category | 0.175 | 0.397 | 0.208 | 0.016 | 0.258 | 0.770 |
| 7 | state_department | 0.221 | 0.577 | 0.304 | 0.037 | 0.267 | 0.704 |
| 8 | store_category | 0.189 | 0.445 | 0.199 | 0.044 | 0.283 | 0.807 |
| 9 | store_department | 0.219 | 0.551 | 0.271 | 0.047 | 0.312 | 0.751 |
| 10 | item | 0.192 | 0.485 | 0.278 | 0.155 | 0.461 | 0.850 |
| 11 | state_item | 0.171 | 0.486 | 0.295 | 0.303 | 0.518 | 0.861 |
| 12 | item_store | 0.167 | 0.504 | 0.376 | 0.562 | 0.648 | 0.876 |

## ets_lgb / wls_struct (prior origins: 3)

| level | level_name | spl_q10 | spl_q50 | spl_q90 | le_q10 | le_q50 | le_q90 |
|---|---|---|---|---|---|---|---|
| 1 | total | 0.157 | 0.411 | 0.171 | 0.071 | 0.250 | 0.893 |
| 2 | state | 0.164 | 0.435 | 0.165 | 0.060 | 0.357 | 0.833 |
| 3 | store | 0.175 | 0.443 | 0.200 | 0.054 | 0.329 | 0.807 |
| 4 | category | 0.165 | 0.373 | 0.187 | 0.060 | 0.321 | 0.869 |
| 5 | department | 0.224 | 0.578 | 0.328 | 0.041 | 0.301 | 0.730 |
| 6 | state_category | 0.172 | 0.407 | 0.194 | 0.044 | 0.385 | 0.841 |
| 7 | state_department | 0.212 | 0.539 | 0.281 | 0.065 | 0.364 | 0.770 |
| 8 | store_category | 0.173 | 0.415 | 0.201 | 0.073 | 0.396 | 0.860 |
| 9 | store_department | 0.203 | 0.507 | 0.255 | 0.066 | 0.375 | 0.814 |
| 10 | item | 0.174 | 0.446 | 0.264 | 0.150 | 0.489 | 0.874 |
| 11 | state_item | 0.151 | 0.449 | 0.281 | 0.292 | 0.524 | 0.876 |
| 12 | item_store | 0.132 | 0.459 | 0.364 | 0.556 | 0.642 | 0.882 |

## ets_lgb / bu (prior origins: 3)

| level | level_name | spl_q10 | spl_q50 | spl_q90 | le_q10 | le_q50 | le_q90 |
|---|---|---|---|---|---|---|---|
| 1 | total | 0.091 | 0.191 | 0.144 | 0.000 | 0.429 | 0.964 |
| 2 | state | 0.107 | 0.250 | 0.143 | 0.024 | 0.440 | 0.905 |
| 3 | store | 0.137 | 0.355 | 0.177 | 0.086 | 0.461 | 0.879 |
| 4 | category | 0.120 | 0.319 | 0.176 | 0.000 | 0.333 | 0.881 |
| 5 | department | 0.164 | 0.497 | 0.278 | 0.051 | 0.362 | 0.735 |
| 6 | state_category | 0.133 | 0.358 | 0.185 | 0.044 | 0.440 | 0.873 |
| 7 | state_department | 0.172 | 0.467 | 0.253 | 0.068 | 0.391 | 0.801 |
| 8 | store_category | 0.151 | 0.388 | 0.195 | 0.073 | 0.425 | 0.857 |
| 9 | store_department | 0.180 | 0.463 | 0.242 | 0.077 | 0.401 | 0.822 |
| 10 | item | 0.172 | 0.464 | 0.269 | 0.153 | 0.495 | 0.870 |
| 11 | state_item | 0.149 | 0.458 | 0.285 | 0.293 | 0.527 | 0.873 |
| 12 | item_store | 0.129 | 0.455 | 0.366 | 0.557 | 0.644 | 0.881 |
