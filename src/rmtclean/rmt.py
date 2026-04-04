"""Random matrix theory for correlation matrices.

With N assets and T observations, q = N / T, the sample correlation matrix E
of pure noise has eigenvalues spread over the Marchenko-Pastur interval
[(1 - sqrt(q))^2, (1 + sqrt(q))^2] although every true eigenvalue is 1. The
eigenvalues of E are a distorted image of the truth (small ones too small,
large ones too large); the eigenvectors are the only thing a rotation-
invariant estimator may keep. Each cleaner below replaces the eigenvalues
lambda_k of E by xi_k and returns sum_k xi_k v_k v_k', rescaled to unit
diagonal.

* clipping (Laloux et al. 1999): eigenvalues below the Marchenko-Pastur edge
  are noise and are replaced by their average;
* linear shrinkage (Ledoit and Wolf 2004): xi = (1 - a) lambda + a, with the
  optimal intensity a estimated from the data;
* rotationally invariant estimator (Ledoit and Peche 2011; Bun, Bouchaud and
  Potters 2017): the optimal non-linear shrinkage, from the Stieltjes
  transform of the sample spectrum evaluated just off the real axis;
* cross-validated eigenvalues (Bartz 2016; Bun et al. 2018): xi_k is the
  out-of-fold variance along the in-fold eigenvector k, made monotone.
"""

from __future__ import annotations

import numpy as np


def mp_edges(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def mp_pdf(x, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density (continuous part; for q > 1 there is also a mass 1 - 1/q at zero)."""
    lo, hi = mp_edges(q, sigma2)
    x = np.asarray(x, float)
    out = np.zeros_like(x)
    m = (x > lo) & (x < hi)
    out[m] = np.sqrt((hi - x[m]) * (x[m] - lo)) / (2 * np.pi * q * sigma2 * x[m])
    return out


def mp_cdf(x, q: float, n_grid: int = 20001) -> np.ndarray:
    lo, hi = mp_edges(q)
    g = np.linspace(lo, hi, n_grid)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (mp_pdf(g[1:], q) + mp_pdf(g[:-1], q)) * np.diff(g))])
    return np.interp(x, g, c / c[-1])


def correlation(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample correlation matrix and volatilities of a (T, N) return panel."""
    Xc = X - X.mean(0)
    sd = Xc.std(0)
    Z = Xc / sd
    return Z.T @ Z / len(X), sd


def _rebuild(xi: np.ndarray, V: np.ndarray) -> np.ndarray:
    C = (V * xi) @ V.T
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def clip(lam: np.ndarray, V: np.ndarray, q: float) -> np.ndarray:
    """Eigenvalue clipping at the Marchenko-Pastur upper edge (trace preserved)."""
    xi = lam.copy()
    noise = lam <= mp_edges(q)[1]
    if noise.any():
        xi[noise] = lam[noise].mean()
    return _rebuild(xi, V)


def ledoit_wolf_intensity(Z: np.ndarray) -> float:
    """Optimal linear shrinkage intensity towards the identity for standardised data Z (T, N)."""
    T, N = Z.shape
    E = Z.T @ Z / T
    d2 = np.sum((E - np.eye(N)) ** 2) / N
    # b2 = (1 / T^2) sum_t ||z_t z_t' - E||_F^2 / N, computed without forming N x N matrices per t
    sq = np.sum(Z**2, axis=1)
    b2 = (np.sum(sq**2) / T - np.sum(E**2) * 1.0) / (T * N)
    return float(np.clip(b2 / d2, 0.0, 1.0))


def linear_shrinkage(E: np.ndarray, a: float) -> np.ndarray:
    return (1 - a) * E + a * np.eye(len(E))


def rie(lam: np.ndarray, V: np.ndarray, q: float) -> np.ndarray:
    """Debiased rotationally invariant estimator of Bun, Bouchaud and Potters (2017), valid for q < 1."""
    N = len(lam)
    z = lam - 1j / np.sqrt(N)
    diff = z[:, None] - lam[None, :]
    np.fill_diagonal(diff, np.inf)
    s = np.sum(1.0 / diff, axis=1) / N
    xi = lam / np.abs(1 - q + q * z * s) ** 2
    # Debiasing of the small eigenvalues with the Marchenko-Pastur Stieltjes transform
    sigma2 = lam.min() / (1 - np.sqrt(q)) ** 2
    lo, hi = mp_edges(q, sigma2)
    g = (z + sigma2 * (q - 1) - np.sqrt(z - lo + 0j) * np.sqrt(z - hi + 0j)) / (2 * q * z * sigma2)
    gamma = sigma2 * np.abs(1 - q + q * z * g) ** 2 / lam
    xi = np.where(gamma > 1, gamma * xi, xi)
    xi *= lam.sum() / xi.sum()
    return _rebuild(xi, V)


def _isotonic(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators: closest non-decreasing sequence."""
    vals, wts = [], []
    for v in y:
        vals.append(float(v))
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            v2 = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            vals[-2:], wts[-2:] = [v2], [w]
    return np.repeat(vals, np.array(wts, int))


def cv_eigenvalues(Z: np.ndarray, V: np.ndarray, lam: np.ndarray, folds: int = 5) -> np.ndarray:
    """Cross-validated eigenvalues: variance of held-out data along eigenvectors estimated without it."""
    T, N = Z.shape
    xi = np.zeros(N)
    for idx in np.array_split(np.arange(T), folds):
        mask = np.ones(T, bool)
        mask[idx] = False
        _, Vf = np.linalg.eigh(Z[mask].T @ Z[mask] / mask.sum())
        xi += np.sum((Z[idx] @ Vf) ** 2, axis=0)
    xi = _isotonic(xi / T)  # eigh returns ascending order
    xi = np.maximum(xi, 1e-8)
    xi *= lam.sum() / xi.sum()
    return _rebuild(xi, V)


def clean_all(X: np.ndarray, methods, folds: int = 5) -> tuple[dict, np.ndarray, dict]:
    """All requested estimators of the correlation matrix of X (T, N); also returns vols and diagnostics."""
    T, N = X.shape
    q = N / T
    Xc = X - X.mean(0)
    sd = Xc.std(0)
    Z = Xc / sd
    E = Z.T @ Z / T
    lam, V = np.linalg.eigh(E)
    out, info = {}, {"q": q, "eigenvalues": lam}
    for m in methods:
        if m == "sample":
            out[m] = E
        elif m == "clipping":
            out[m] = clip(lam, V, q)
        elif m == "linear shrinkage":
            info["lw_intensity"] = a = ledoit_wolf_intensity(Z)
            out[m] = linear_shrinkage(E, a)
        elif m == "RIE" and q < 0.95:
            out[m] = rie(lam, V, q)
        elif m == "cross-validated":
            out[m] = cv_eigenvalues(Z, V, lam, folds)
    return out, sd, info


def gmv_weights(C: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights for covariance D C D (sum to 1, long-short); pseudo-inverse if singular."""
    try:
        x = np.linalg.solve(C, 1.0 / sd)
        if not np.all(np.isfinite(x)) or np.linalg.cond(C) > 1e12:
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        x = np.linalg.pinv(C, rcond=1e-10) @ (1.0 / sd)
    w = x / sd
    return w / w.sum()
