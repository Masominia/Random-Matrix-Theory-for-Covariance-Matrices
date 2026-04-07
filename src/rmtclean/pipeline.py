"""End-to-end run: Marchenko-Pastur check on pure noise -> estimators against a known truth -> spectrum of real
returns -> out-of-sample minimum-variance portfolios for a range of q = N / T -> report and checks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import t as student_t

from . import plotting, rmt
from .report import md_table

ROOT = Path(__file__).resolve().parents[2]
LABEL = {
    "sample": "Sample covariance",
    "linear shrinkage": "Linear shrinkage (Ledoit-Wolf)",
    "clipping": "Eigenvalue clipping (Marchenko-Pastur)",
    "RIE": "Rotationally invariant estimator",
    "cross-validated": "Cross-validated eigenvalues",
}


def load_returns(raw: Path = ROOT / "data" / "raw") -> pd.DataFrame:
    p = pd.read_csv(raw / "nasdaq_prices_daily.csv.gz", index_col=0, parse_dates=True)
    return np.log(p).diff().dropna()


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if hasattr(o, "item"):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def planted_truth(N: int, rng) -> np.ndarray:
    """A correlation matrix with a market mode, sector blocks and a power-law bulk."""
    k = 8
    B = np.column_stack([0.5 + 0.2 * rng.random(N)] + [0.45 * (rng.integers(0, k, N) == j) for j in range(k)])
    C = B @ B.T + np.diag(0.3 + rng.pareto(3.0, N))
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def backtest(R: np.ndarray, cols: np.ndarray, T: int, hold: int, methods, folds: int, start: int) -> dict:
    """Monthly minimum-variance portfolios from each estimator; out-of-sample daily returns."""
    out = {m: {"ret": [], "pred": [], "gross": []} for m in methods}
    ew, lw = [], []
    for t in range(start, len(R) - hold + 1, hold):
        X, F = R[t - T : t][:, cols], R[t : t + hold][:, cols]
        C, sd, info = rmt.clean_all(X, methods, folds)
        lw.append(info.get("lw_intensity", np.nan))
        for m, c in C.items():
            w = rmt.gmv_weights(c, sd)
            out[m]["ret"].append(F @ w)
            out[m]["pred"].append(np.sqrt(252 * w @ ((c * np.outer(sd, sd)) @ w)))
            out[m]["gross"].append(np.abs(w).sum())
        ew.append(F.mean(1))
    res = {
        m: {"ret": np.concatenate(v["ret"]), "pred": float(np.mean(v["pred"])), "gross": float(np.mean(v["gross"]))}
        for m, v in out.items()
        if v["ret"]
    }
    res["equal weight"] = {"ret": np.concatenate(ew), "pred": np.nan, "gross": 1.0}
    res["_lw"] = float(np.nanmean(lw))
    return res


def variance_test(a: np.ndarray, b: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Is the variance of a larger than that of b? HAC t-test on the difference of squared demeaned returns."""
    d = (a - a.mean()) ** 2 - (b - b.mean()) ** 2
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lags + 1):
        v += 2 * (1 - k / (lags + 1)) * (dc[k:] @ dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    return float(stat), float(student_t.sf(stat, n - 1))


def run(cfg_path=ROOT / "configs" / "default.yaml", out=ROOT / "reports", fast: bool = False) -> dict:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if fast:
        cfg["configs"] = [{"N": 60, "T": 250, "universes": 1}, {"N": 200, "T": 250, "universes": 1}]
        cfg["headline"] = {"N": 200, "T": 250}
        cfg["theory"] = {"noise": {"N": 150, "T": 400}, "planted": {"N": 100, "T": 200, "replications": 2}}
    out = Path(out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    methods, folds = cfg["methods"], cfg["cv_folds"]
    md, metrics, checks = [], {}, []

    def check(area, name, value, threshold, ok, warn_only=False):
        checks.append(
            {
                "area": area,
                "check": name,
                "value": value,
                "threshold": threshold,
                "status": "PASS" if ok else ("WARN" if warn_only else "FAIL"),
            }
        )

    # 1. Theory: pure noise
    nz = cfg["theory"]["noise"]
    q0 = nz["N"] / nz["T"]
    E0, _ = rmt.correlation(rng.standard_normal((nz["T"], nz["N"])))
    lam0 = np.linalg.eigvalsh(E0)
    ks = float(np.abs(rmt.mp_cdf(lam0, q0) - (np.arange(1, len(lam0) + 1) - 0.5) / len(lam0)).max())
    lo, hi = rmt.mp_edges(q0)
    check(
        "Theory",
        "pure noise: distance between the eigenvalue distribution and Marchenko-Pastur (KS)",
        ks,
        "< 0.03",
        ks < 0.03,
    )
    check(
        "Theory",
        "pure noise: largest eigenvalue vs the Marchenko-Pastur edge (ratio)",
        float(lam0.max() / hi),
        "0.97 to 1.05",
        0.97 < lam0.max() / hi < 1.05,
    )
    metrics["noise"] = {"q": q0, "ks": ks, "edges": [lo, hi], "min": float(lam0.min()), "max": float(lam0.max())}

    # 2. Theory: known truth
    pl = cfg["theory"]["planted"]
    errs, gm = [], []
    for _ in range(pl["replications"]):
        Ctrue = planted_truth(pl["N"], rng)
        L = np.linalg.cholesky(Ctrue)
        X = rng.standard_normal((pl["T"], pl["N"])) @ L.T
        est, sd, info = rmt.clean_all(X, methods, folds)
        lam, V = np.linalg.eigh(est["sample"])
        est["oracle (uses the truth)"] = rmt._rebuild(np.einsum("ik,ij,jk->k", V, Ctrue, V), V)
        Pinv = np.linalg.inv(Ctrue)
        errs.append({m: np.linalg.norm(np.linalg.inv(c) - Pinv) / np.linalg.norm(Pinv) for m, c in est.items()})
        row = {}
        for m, c in est.items():
            w = rmt.gmv_weights(c, np.ones(pl["N"]))
            row[m] = np.sqrt(w @ Ctrue @ w)
        wopt = rmt.gmv_weights(Ctrue, np.ones(pl["N"]))
        gm.append({m: v / np.sqrt(wopt @ Ctrue @ wopt) for m, v in row.items()})
    planted = pd.DataFrame(
        {
            "error of the inverse matrix vs truth": pd.DataFrame(errs).mean(),
            "true risk of the min-variance portfolio / best possible": pd.DataFrame(gm).mean(),
        }
    )
    planted.index = [LABEL.get(i, i) for i in planted.index]
    metrics["planted"] = planted.to_dict()
    risk = planted.iloc[:, 1]
    check(
        "Theory",
        "known truth: RIE portfolio risk within 3% of the oracle that sees the true matrix",
        float(risk[LABEL["RIE"]] / risk["oracle (uses the truth)"]),
        "< 1.03",
        risk[LABEL["RIE"]] < 1.03 * risk["oracle (uses the truth)"],
    )
    check(
        "Theory",
        "known truth: RIE portfolio risk vs the sample matrix (ratio)",
        float(risk[LABEL["RIE"]] / risk[LABEL["sample"]]),
        "< 0.85",
        risk[LABEL["RIE"]] < 0.85 * risk[LABEL["sample"]],
    )

    # 3. Real spectrum
    Rdf = load_returns()
    R = Rdf.to_numpy()
    hd = cfg["headline"]
    cols = rng.choice(R.shape[1], hd["N"], replace=False)
    X = R[-hd["T"] :][:, cols]
    E, _ = rmt.correlation(X)
    lam, V = np.linalg.eigh(E)
    qh = hd["N"] / hd["T"]
    edge = rmt.mp_edges(qh)[1]
    ipr = np.sum(V**4, axis=0)
    spec = {
        "assets": hd["N"],
        "days": hd["T"],
        "q": qh,
        "Marchenko-Pastur upper edge": edge,
        "eigenvalues above the edge": int((lam > edge).sum()),
        "share of eigenvalues inside the noise band %": float(100 * (lam <= edge).mean()),
        "largest eigenvalue (market mode)": float(lam[-1]),
        "variance explained by the market mode %": float(100 * lam[-1] / hd["N"]),
        "smallest eigenvalue": float(lam[0]),
        "participation of the market mode (1 = all stocks equally)": float(1 / (hd["N"] * ipr[-1])),
    }
    metrics["spectrum"] = spec
    plotting.spectra(lam0, q0, lam, qh, figs / "spectrum.png")
    est_h, _, _ = rmt.clean_all(X, [m for m in methods if m != "sample"], folds)
    xi = {LABEL[m]: np.sort(np.linalg.eigvalsh(c)) for m, c in est_h.items()}
    plotting.shrinkage(lam, xi, figs / "eigenvalue_cleaning.png")

    md += [
        "# RMTClean report",
        "",
        f"Seed {seed}.",
        "",
        "## 1. Theory checks",
        "",
        f"**Pure noise** ({nz['N']} series, {nz['T']} observations, q = {q0:.2f}): every true eigenvalue is 1, yet "
        f"the sample eigenvalues range from {lam0.min():.2f} to {lam0.max():.2f}; Marchenko-Pastur predicts "
        f"{lo:.2f} to {hi:.2f}. Kolmogorov-Smirnov distance to the predicted density: {ks:.3f}.",
        "",
        f"**Known truth** (market mode + 8 sectors + heavy-tailed idiosyncratic variance, N = {pl['N']}, "
        f"T = {pl['T']}, {pl['replications']} draws):",
        "",
        md_table(planted, 3),
        "",
        "## 2. Spectrum of real stock returns",
        "",
        f"Random universe of {hd['N']} Nasdaq stocks, last {hd['T']} days.",
        "",
        md_table(
            pd.Series(
                {k: (f"{v:,d}" if isinstance(v, int) else f"{v:.2f}") for k, v in spec.items()}, name="value"
            ).to_frame()
        ),
        "",
    ]

    # 4. Out-of-sample portfolios
    hold = cfg["holding_days"]
    start = max(c["T"] for c in cfg["configs"])
    rows, detail = [], {}
    for c in cfg["configs"]:
        ms = [m for m in methods if m not in c.get("skip", [])]
        runs = [
            backtest(R, rng.choice(R.shape[1], c["N"], replace=False), c["T"], hold, ms, folds, start)
            for _ in range(c["universes"])
        ]
        key = f"N = {c['N']}, T = {c['T']}"
        detail[key] = runs
        for m in ms + ["equal weight"]:
            ok = [r for r in runs if m in r]
            if not ok:
                continue
            volr = np.mean([r[m]["ret"].std() * np.sqrt(252) for r in ok])
            rows.append(
                {
                    "config": key,
                    "q": c["N"] / c["T"],
                    "estimator": LABEL.get(m, "Equal weight (no covariance)"),
                    "realised vol %": 100 * volr,
                    "predicted vol %": 100 * np.mean([r[m]["pred"] for r in ok]),
                    "gross leverage": np.mean([r[m]["gross"] for r in ok]),
                }
            )
    res = pd.DataFrame(rows)
    res["realised / predicted"] = res["realised vol %"] / res["predicted vol %"]
    base = res[res["estimator"] == LABEL["sample"]].set_index("config")["realised vol %"]
    res["vol vs sample %"] = 100 * (res["realised vol %"] / res["config"].map(base) - 1)
    vol_tab = res.pivot(index="estimator", columns="config", values="realised vol %")
    order_c = [f"N = {c['N']}, T = {c['T']}" for c in cfg["configs"]]
    order_e = [LABEL[m] for m in methods] + ["Equal weight (no covariance)"]
    vol_tab = vol_tab.loc[order_e, order_c]
    vol_tab.columns = [f"{k} (q = {int(k.split(',')[0][4:]) / int(k.split('= ')[-1]):.1f})" for k in order_c]
    hk = f"N = {hd['N']}, T = {hd['T']}"
    head = (
        res[res["config"] == hk]
        .set_index("estimator")
        .loc[[e for e in order_e if e in set(res[res["config"] == hk]["estimator"])]]
    )
    head = head[["realised vol %", "vol vs sample %", "predicted vol %", "realised / predicted", "gross leverage"]]
    best = "RIE" if "RIE" in detail[hk][0] else "clipping"
    tests = [variance_test(r["sample"]["ret"], r[best]["ret"]) for r in detail[hk]]
    n_oos = len(detail[hk][0]["sample"]["ret"])
    metrics["oos"] = {
        "days": n_oos,
        "first_day": str(Rdf.index[start].date()),
        "last_day": str(Rdf.index[-1].date()),
        "vol_table": vol_tab.to_dict(),
        "headline": head.to_dict(orient="index"),
        "variance_test_sample_vs_best": tests,
        "best": best,
        "lw_intensity": float(np.mean([r["_lw"] for r in detail[hk]])),
    }
    red = float(head.loc[LABEL[best], "vol vs sample %"])
    check(
        "Portfolio",
        f"headline: {LABEL[best]} cuts realised volatility vs the sample covariance (%)",
        red,
        "< -20",
        red < -20,
    )
    check(
        "Portfolio",
        "headline: variance reduction is significant in every universe (largest p-value)",
        float(max(p for _, p in tests)),
        "< 0.01",
        max(p for _, p in tests) < 0.01,
    )
    under_s = float(head.loc[LABEL["sample"], "realised / predicted"])
    under_c = float(head.loc[LABEL[best], "realised / predicted"])
    check(
        "Portfolio",
        "cleaning reduces risk underestimation (realised / predicted, cleaned vs sample)",
        under_c / under_s,
        "< 0.5",
        under_c < 0.5 * under_s,
    )
    low_q = res[(res["q"] < 0.3)]
    worst_low = float(
        low_q[~low_q["estimator"].isin(["Equal weight (no covariance)", LABEL["sample"]])]["vol vs sample %"].max()
    )
    check(
        "Portfolio",
        "no cleaner hurts when data are plentiful (q = 0.2, worst change vs sample, %)",
        worst_low,
        "< 3",
        worst_low < 3,
        warn_only=True,
    )
    plotting.oos(res, order_c, LABEL, detail[hk][0], Rdf.index[start : start + n_oos], figs / "out_of_sample.png")
    md += [
        "## 3. Out-of-sample minimum-variance portfolios",
        "",
        f"Every {hold} trading days: estimate on the last T days, build the minimum-variance portfolio (weights sum "
        f"to 1, shorting allowed), hold for {hold} days. Out of sample {metrics['oos']['first_day']} to "
        f"{metrics['oos']['last_day']} ({n_oos} days), averaged over random universes of N stocks.",
        "",
        "### Realised annualised volatility, %",
        "",
        md_table(vol_tab, 1),
        "",
        f"### Headline case: {hk} (q = {hd['N'] / hd['T']:.1f})",
        "",
        md_table(head, 2),
        "",
        f"Variance of the sample-covariance portfolio vs the {LABEL[best].lower()} portfolio, HAC t-statistics by "
        f"universe: {', '.join(f'{s:.1f}' for s, _ in tests)}.",
        "",
    ]

    ch = pd.DataFrame(checks)
    val = [
        "# Validation checks",
        "",
        md_table(ch, 4, index=False),
        "",
        f"{(ch.status == 'PASS').sum()} PASS, {(ch.status == 'WARN').sum()} WARN, {(ch.status == 'FAIL').sum()} FAIL.",
        "",
    ]
    metrics["checks"] = checks
    (out / "report.md").write_text("\n".join(md))
    (out / "validation.md").write_text("\n".join(val))
    (out / "metrics.json").write_text(json.dumps(_clean(metrics), indent=2))
    res.round(4).to_csv(out / "oos_results.csv", index=False)
    return metrics
