# RMTClean report

Seed 20260101.

## 1. Theory checks

**Pure noise** (400 series, 1000 observations, q = 0.40): every true eigenvalue is 1, yet the sample eigenvalues range from 0.14 to 2.62; Marchenko-Pastur predicts 0.14 to 2.66. Kolmogorov-Smirnov distance to the predicted density: 0.006.

**Known truth** (market mode + 8 sectors + heavy-tailed idiosyncratic variance, N = 300, T = 600, 5 draws):

|  | error of the inverse matrix vs truth | true risk of the min-variance portfolio / best possible |
|---|---|---|
| Sample covariance | 2.215 | 1.443 |
| Linear shrinkage (Ledoit-Wolf) | 1.752 | 1.364 |
| Eigenvalue clipping (Marchenko-Pastur) | 0.196 | 1.145 |
| Rotationally invariant estimator | 0.224 | 1.099 |
| Cross-validated eigenvalues | 0.214 | 1.110 |
| oracle (uses the truth) | 0.213 | 1.103 |

## 2. Spectrum of real stock returns

Random universe of 450 Nasdaq stocks, last 500 days.

|  | value |
|---|---|
| assets | 450 |
| days | 500 |
| q | 0.90 |
| Marchenko-Pastur upper edge | 3.80 |
| eigenvalues above the edge | 8 |
| share of eigenvalues inside the noise band % | 98.22 |
| largest eigenvalue (market mode) | 109.49 |
| variance explained by the market mode % | 24.33 |
| smallest eigenvalue | 0.00 |
| participation of the market mode (1 = all stocks equally) | 0.74 |

## 3. Out-of-sample minimum-variance portfolios

Every 21 trading days: estimate on the last T days, build the minimum-variance portfolio (weights sum to 1, shorting allowed), hold for 21 days. Out of sample 2019-12-30 to 2023-05-31 (861 days), averaged over random universes of N stocks.

### Realised annualised volatility, %

| estimator | N = 100, T = 500 (q = 0.2) | N = 200, T = 500 (q = 0.4) | N = 300, T = 500 (q = 0.6) | N = 450, T = 500 (q = 0.9) | N = 300, T = 250 (q = 1.2) | N = 750, T = 500 (q = 1.5) |
|---|---|---|---|---|---|---|
| Sample covariance | 20.9 | 21.2 | 21.3 | 30.9 | 29.2 | 19.9 |
| Linear shrinkage (Ledoit-Wolf) | 20.5 | 20.0 | 19.5 | 17.7 | 18.4 | 16.0 |
| Eigenvalue clipping (Marchenko-Pastur) | 20.6 | 19.6 | 19.5 | 16.5 | 17.6 | 15.4 |
| Rotationally invariant estimator | 20.3 | 19.2 | 18.9 | 16.3 | n/a | n/a |
| Cross-validated eigenvalues | 20.2 | 19.1 | 19.1 | 16.3 | 17.5 | n/a |
| Equal weight (no covariance) | 29.1 | 29.8 | 30.1 | 29.2 | 29.1 | 29.4 |

### Headline case: N = 450, T = 500 (q = 0.9)

| estimator | realised vol % | vol vs sample % | predicted vol % | realised / predicted | gross leverage |
|---|---|---|---|---|---|
| Sample covariance | 30.87 | 0.00 | 2.29 | 13.48 | 13.20 |
| Linear shrinkage (Ledoit-Wolf) | 17.66 | -42.79 | 4.86 | 3.63 | 4.66 |
| Eigenvalue clipping (Marchenko-Pastur) | 16.49 | -46.56 | 6.56 | 2.51 | 2.86 |
| Rotationally invariant estimator | 16.35 | -47.04 | 6.84 | 2.39 | 2.97 |
| Cross-validated eigenvalues | 16.33 | -47.09 | 6.96 | 2.35 | 2.72 |
| Equal weight (no covariance) | 29.20 | -5.41 | n/a | n/a | 1.00 |

Variance of the sample-covariance portfolio vs the rotationally invariant estimator portfolio, HAC t-statistics by universe: 4.4, 4.6, 6.7.
