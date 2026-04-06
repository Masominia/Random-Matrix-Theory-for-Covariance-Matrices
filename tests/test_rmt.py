import numpy as np
import pytest
from scipy.integrate import quad

from rmtclean import rmt


def test_marchenko_pastur_density():
    for q in (0.2, 0.5, 0.9):
        lo, hi = rmt.mp_edges(q)
        assert quad(lambda x, q=q: rmt.mp_pdf(np.array([x]), q)[0], lo, hi)[0] == pytest.approx(1.0, abs=1e-6)
        assert quad(lambda x, q=q: x * rmt.mp_pdf(np.array([x]), q)[0], lo, hi)[0] == pytest.approx(1.0, abs=1e-6)
    assert rmt.mp_edges(0.25) == pytest.approx((0.25, 2.25))


def test_noise_spectrum_follows_marchenko_pastur():
    rng = np.random.default_rng(0)
    N, T = 300, 900
    E, _ = rmt.correlation(rng.standard_normal((T, N)))
    lam = np.linalg.eigvalsh(E)
    ks = np.abs(rmt.mp_cdf(lam, N / T) - (np.arange(1, N + 1) - 0.5) / N).max()
    assert ks < 0.03 and lam.max() == pytest.approx(rmt.mp_edges(N / T)[1], rel=0.05)


def test_cleaners_return_valid_correlation_matrices():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 80)) + 0.5 * rng.standard_normal((200, 1))
    est, sd, info = rmt.clean_all(X, ["sample", "clipping", "linear shrinkage", "RIE", "cross-validated"])
    assert set(est) == {"sample", "clipping", "linear shrinkage", "RIE", "cross-validated"}
    for name, C in est.items():
        assert np.allclose(np.diag(C), 1), name
        assert np.allclose(C, C.T), name
        assert np.linalg.eigvalsh(C).min() > 0, name
    assert 0 <= info["lw_intensity"] <= 1 and sd.shape == (80,)
    assert "RIE" not in rmt.clean_all(rng.standard_normal((50, 60)), ["RIE", "clipping"])[0]  # RIE needs q < 1


def test_clipping_removes_noise_but_keeps_signal():
    rng = np.random.default_rng(2)
    N, T = 200, 600
    X = rng.standard_normal((T, N)) + 0.6 * rng.standard_normal((T, 1))  # one strong common factor
    E, _ = rmt.correlation(X)
    lam, V = np.linalg.eigh(E)
    C = rmt.clip(lam, V, N / T)
    xi = np.linalg.eigvalsh(C)
    assert xi[-1] == pytest.approx(lam[-1], rel=0.02)  # market mode untouched
    assert xi[:-1].std() < 0.1 * lam[:-1].std()  # bulk flattened


def test_ledoit_wolf_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.covariance")
    rng = np.random.default_rng(3)
    X = rng.standard_normal((120, 40)) @ np.diag(np.linspace(0.5, 2, 40))
    Z = (X - X.mean(0)) / X.std(0)
    ours = rmt.ledoit_wolf_intensity(Z)
    ref = sk.LedoitWolf(assume_centered=True).fit(Z).shrinkage_
    assert ours == pytest.approx(ref, rel=0.02)


def test_isotonic_and_cv_trace():
    assert rmt._isotonic(np.array([1.0, 3.0, 2.0, 4.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    rng = np.random.default_rng(4)
    X = rng.standard_normal((150, 60))
    Z = (X - X.mean(0)) / X.std(0)
    lam, V = np.linalg.eigh(Z.T @ Z / 150)
    C = rmt.cv_eigenvalues(Z, V, lam)
    assert np.all(np.diff(np.linalg.eigvalsh(C)) >= -1e-10)
    assert np.linalg.eigvalsh(C).max() < lam.max()  # pure noise: the spectrum is pulled towards 1


def test_rie_close_to_oracle_and_better_than_sample():
    rng = np.random.default_rng(5)
    N, T = 200, 400
    B = np.column_stack([0.6 * np.ones(N)] + [0.5 * (rng.integers(0, 5, N) == j) for j in range(5)])
    Ct = B @ B.T + np.diag(0.3 + rng.random(N))
    d = np.sqrt(np.diag(Ct))
    Ct = Ct / np.outer(d, d)
    X = rng.standard_normal((T, N)) @ np.linalg.cholesky(Ct).T
    est, _, _ = rmt.clean_all(X, ["sample", "RIE"])
    risk = {}
    for k, C in est.items():
        w = rmt.gmv_weights(C, np.ones(N))
        risk[k] = w @ Ct @ w
    wopt = rmt.gmv_weights(Ct, np.ones(N))
    best = wopt @ Ct @ wopt
    assert risk["RIE"] < 0.75 * risk["sample"]
    assert risk["RIE"] < 1.35 * best
    assert risk["sample"] / best == pytest.approx(1 / (1 - N / T), rel=0.25)  # classical result: 1 / (1 - q)


def test_gmv_weights():
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    w = rmt.gmv_weights(C, np.array([0.1, 0.2]))
    assert w.sum() == pytest.approx(1) and w[0] / w[1] == pytest.approx(4.0)  # inverse-variance weights
    sing = np.ones((3, 3))
    assert np.isfinite(rmt.gmv_weights(sing, np.ones(3))).all()


def test_fast_pipeline(tmp_path):
    from rmtclean.pipeline import run

    m = run(out=tmp_path, fast=True)
    assert not [c for c in m["checks"] if c["status"] == "FAIL"]
    for f in [
        "report.md",
        "validation.md",
        "metrics.json",
        "oos_results.csv",
        "figures/spectrum.png",
        "figures/eigenvalue_cleaning.png",
        "figures/out_of_sample.png",
    ]:
        assert (tmp_path / f).exists(), f
