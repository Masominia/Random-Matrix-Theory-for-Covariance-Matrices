# Validation checks

| area | check | value | threshold | status |
|---|---|---|---|---|
| Theory | pure noise: distance between the eigenvalue distribution and Marchenko-Pastur (KS) | 5.95e-03 | < 0.03 | PASS |
| Theory | pure noise: largest eigenvalue vs the Marchenko-Pastur edge (ratio) | 0.9817 | 0.97 to 1.05 | PASS |
| Theory | known truth: RIE portfolio risk within 3% of the oracle that sees the true matrix | 0.9964 | < 1.03 | PASS |
| Theory | known truth: RIE portfolio risk vs the sample matrix (ratio) | 0.7615 | < 0.85 | PASS |
| Portfolio | headline: Rotationally invariant estimator cuts realised volatility vs the sample covariance (%) | -47.0408 | < -20 | PASS |
| Portfolio | headline: variance reduction is significant in every universe (largest p-value) | 5.89e-06 | < 0.01 | PASS |
| Portfolio | cleaning reduces risk underestimation (realised / predicted, cleaned vs sample) | 0.1775 | < 0.5 | PASS |
| Portfolio | no cleaner hurts when data are plentiful (q = 0.2, worst change vs sample, %) | -1.2697 | < 3 | PASS |

8 PASS, 0 WARN, 0 FAIL.
