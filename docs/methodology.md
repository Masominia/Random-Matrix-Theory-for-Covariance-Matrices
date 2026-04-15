# Methodology

## 1. The problem

A covariance matrix of N assets has N(N+1)/2 parameters; T days of returns provide N x T numbers. Risk models, portfolio optimisers and capital allocation all use the matrix, usually through its inverse, and the inverse is dominated by the smallest eigenvalues, which are exactly the ones estimated worst. The relevant quantity is q = N / T: sample estimates are only reliable when q is close to zero, and in practice q is between 0.1 and several units.

## 2. What random matrix theory says

**Marchenko-Pastur (1967).** For T observations of N independent unit-variance series, as N and T grow with q fixed, the eigenvalues of the sample correlation matrix E fill the interval [(1 - sqrt(q))^2, (1 + sqrt(q))^2] with density

    rho(x) = sqrt((x+ - x)(x - x-)) / (2 pi q x).

Every true eigenvalue is 1; the spread is pure estimation noise. For q = 0.4 the sample eigenvalues run from 0.14 to 2.66.

**Consequence for portfolios.** A minimum-variance optimiser loads on the directions with the smallest sample eigenvalues, which are noise. In the classical setting the true risk of the optimised portfolio is 1 / (1 - q) times the best achievable, and its *predicted* risk is (1 - q) times the best, so risk is underestimated by a factor 1 / (1 - q) in variance. At q = 0.9 that is a factor 10. A unit test verifies the 1 / (1 - q) law by simulation.

**Rotational invariance.** With no prior view on the eigenvectors of the true matrix, the best estimator keeps the sample eigenvectors v_k and only changes the eigenvalues: Xi = sum_k xi_k v_k v_k'. The optimal ("oracle") choice is xi_k = v_k' C v_k, which needs the unknown truth C. The remarkable result (Ledoit and Peche 2011; Bun, Bouchaud and Potters 2017) is that in the large-N limit the oracle eigenvalues can be computed from the sample spectrum alone.

## 3. Estimators

All work on the correlation matrix (volatilities are estimated separately on the same window) and are rescaled to unit diagonal.

* **Sample**: E itself; pseudo-inverse when q >= 1.
* **Linear shrinkage** (Ledoit and Wolf 2004): (1 - a) E + a I, with the intensity a that minimises expected Frobenius loss, estimated from the data. The implementation is checked against scikit-learn.
* **Eigenvalue clipping** (Laloux, Cizeau, Bouchaud and Potters 1999): eigenvalues below the Marchenko-Pastur upper edge (1 + sqrt(q))^2 are replaced by their average, preserving the trace; those above are kept.
* **Rotationally invariant estimator (RIE)**, q < 1:

      z_k = lambda_k - i N^(-1/2),   s_k = (1/N) sum_{j != k} 1 / (z_k - lambda_j),
      xi_k = lambda_k / |1 - q + q z_k s_k|^2,

  followed by the Bun-Bouchaud-Potters correction of the small eigenvalues, xi_k -> Gamma_k xi_k when Gamma_k > 1, where Gamma_k compares the same expression evaluated with the Marchenko-Pastur Stieltjes transform (variance fitted to the smallest eigenvalue) to lambda_k. The small imaginary part is the physicist's regularisation: it smooths the discrete spectrum at the scale of the eigenvalue spacing.
* **Cross-validated eigenvalues** (Bartz 2016; Bun, Bouchaud and Potters 2018): split the T days into 5 blocks; for each block compute eigenvectors without it and measure the variance of the held-out days along each of them; average, then enforce monotonicity by isotonic regression. Works for any q and makes no distributional assumption.

## 4. Experiments

**Pure noise.** Kolmogorov-Smirnov distance between the eigenvalues of a simulated N = 400, T = 1,000 noise matrix and the Marchenko-Pastur law; position of the largest eigenvalue against the edge.

**Known truth.** A correlation matrix with a market mode, eight sector factors and heavy-tailed idiosyncratic variances; Gaussian samples with q = 0.5. For each estimator: relative error of the inverse matrix, and the *true* risk of the minimum-variance portfolio built from the estimate, relative to the best possible. The oracle that uses the true matrix to set the eigenvalues is the ceiling for any rotationally invariant estimator.

**Real data.** 1,455 Nasdaq stocks, 2018-2023. Every 21 trading days: draw nothing new (universes are fixed per run), estimate on the last T days, form the global minimum-variance portfolio w = Sigma^-1 1 / (1' Sigma^-1 1) (weights sum to one, shorting allowed), hold 21 days, record daily returns. Reported: realised annualised volatility, the volatility the estimator predicted for its own portfolio, their ratio, gross leverage. Configurations span q = 0.2 to 1.5; each is averaged over one to three random universes. Significance: HAC t-test on the difference of squared demeaned daily returns between the sample-covariance portfolio and the cleaned one, per universe.

The minimum-variance portfolio is used because it depends only on the covariance matrix: any difference in realised risk is attributable to the estimator, with no expected-return forecast in the way.

## 5. Reading the results

* The gain grows with q, as theory predicts: about 3% lower volatility at q = 0.2, 10% at q = 0.4 to 0.6 and 47% at q = 0.9, where the sample-covariance portfolio is riskier than holding all stocks equally.
* The non-linear cleaners (RIE, cross-validation, clipping) beat linear shrinkage, most clearly at high q: one shrinkage intensity cannot both leave the market mode alone and lift the noise eigenvalues enough.
* Risk forecasts: the sample covariance predicted 2.3% volatility for a portfolio that realised 31% (a factor 13); the cleaned estimators were off by a factor 2.4. The remaining gap is not estimation noise: the test period opens with the COVID crash, volatilities are estimated on a calm window, and returns are fat-tailed. Cleaning correlations does not fix a stale volatility level.
* At q > 1 the pseudo-inverse sample portfolio looks better than at q = 0.9. This is the known non-monotonic ("double descent") behaviour of minimum-norm solutions; cleaning still improves on it by 20 to 40%.

## 6. Limitations

* Survivorship-biased universe and a 3.4-year test period that includes one crisis.
* Long-short minimum variance without transaction costs or leverage limits; cleaned portfolios need 3x gross leverage, the sample one 13x, so costs would widen the gap.
* Static equal-weight windows; no exponential weighting, no separate volatility model, no factor structure imposed. Each would improve all estimators.
* The RIE formula assumes q < 1 and independent, identically distributed observations.

## References

* Marchenko, V. and Pastur, L. (1967), "Distribution of eigenvalues for some sets of random matrices", *Mathematics of the USSR-Sbornik*.
* Laloux, L., Cizeau, P., Bouchaud, J.-P. and Potters, M. (1999), "Noise dressing of financial correlation matrices", *Physical Review Letters*.
* Plerou, V. et al. (1999), "Universal and nonuniversal properties of cross correlations in financial time series", *Physical Review Letters*.
* Ledoit, O. and Wolf, M. (2004), "A well-conditioned estimator for large-dimensional covariance matrices", *Journal of Multivariate Analysis*.
* Ledoit, O. and Peche, S. (2011), "Eigenvectors of some large sample covariance matrix ensembles", *Probability Theory and Related Fields*.
* Bun, J., Bouchaud, J.-P. and Potters, M. (2017), "Cleaning large correlation matrices: tools from random matrix theory", *Physics Reports*.
* Bartz, D. (2016), "Cross-validation based nonlinear shrinkage", arXiv:1611.00798.
* Pafka, S. and Kondor, I. (2003), "Noisy covariance matrices and portfolio optimization II", *Physica A*.
