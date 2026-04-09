"""Static figures for the report (matplotlib, PNG for the README)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NEUTRAL = "#8a8985"
CRITICAL = "#c4271a"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.dpi": 150,
        "axes.axisbelow": True,
    }
)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def spectra(lam0, q0, lam, q, path):
    from .rmt import mp_edges, mp_pdf

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.0))
    x = np.linspace(0, mp_edges(q0)[1] * 1.15, 400)
    a1.hist(lam0, bins=50, density=True, color=SERIES[0], alpha=0.6, label="eigenvalues, pure noise")
    a1.plot(x, mp_pdf(x, q0), color=INK, lw=1.8, label="Marchenko-Pastur law")
    a1.axvline(1, color=CRITICAL, lw=1.2, ls="--", label="truth: all eigenvalues equal 1")
    a1.set_title(f"Noise alone spreads the spectrum (q = {q0:.2f})", loc="left")
    a1.set_xlabel("eigenvalue")
    a1.legend(fontsize=8)
    edge = mp_edges(q)[1]
    x = np.linspace(0, edge * 1.6, 400)
    a2.hist(lam[lam < edge * 1.6], bins=60, density=True, color=SERIES[1], alpha=0.6, label="stock correlations (bulk)")
    a2.plot(x, mp_pdf(x, q), color=INK, lw=1.8, label="Marchenko-Pastur law (pure noise)")
    a2.axvline(edge, color=NEUTRAL, lw=1, ls=":")
    n_out = int((lam > edge).sum())
    a2.set_title(f"Real returns: {n_out} eigenvalues above the noise edge, largest {lam[-1]:.0f}", loc="left")
    a2.set_xlabel("eigenvalue (signal eigenvalues beyond the axis)")
    a2.legend(fontsize=8)
    _save(fig, path)


def shrinkage(lam, xi: dict, path):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.plot(lam, lam, color=INK, lw=1, label="no cleaning")
    for i, (name, v) in enumerate(xi.items()):
        ax.plot(lam, v, color=SERIES[i], lw=1.6, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sample eigenvalue")
    ax.set_ylabel("cleaned eigenvalue")
    ax.set_title("What each cleaner does to the spectrum", loc="left")
    ax.legend(fontsize=8)
    _save(fig, path)


def oos(res: pd.DataFrame, order_c, label, run, dates, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.2))
    sub = res[res["config"].str.endswith("T = 500")]
    for i, m in enumerate(["sample", "linear shrinkage", "clipping", "RIE", "cross-validated"]):
        d = sub[sub["estimator"] == label[m]].sort_values("q")
        if len(d):
            a1.plot(d["q"], d["realised vol %"], "o-", color=SERIES[i], ms=4, label=label[m])
    d = sub[sub["estimator"].str.startswith("Equal")].sort_values("q")
    a1.plot(d["q"], d["realised vol %"], color=NEUTRAL, ls="--", label="equal weight")
    a1.set_xlabel("q = number of stocks / days of history")
    a1.set_ylabel("realised volatility out of sample, %")
    a1.set_title("Minimum-variance portfolios, T = 500", loc="left")
    a1.legend(fontsize=7.5, loc="center left")
    for m in ("sample", "RIE"):
        if m in run:
            r = pd.Series(run[m]["ret"], index=dates)
            a2.plot(
                r.index,
                100 * r.rolling(63).std() * np.sqrt(252),
                color=SERIES[0 if m == "sample" else 3],
                label=label[m],
            )
    r = pd.Series(run["equal weight"]["ret"], index=dates)
    a2.plot(r.index, 100 * r.rolling(63).std() * np.sqrt(252), color=NEUTRAL, ls="--", label="equal weight")
    a2.set_ylabel("rolling 3-month volatility, %")
    a2.set_title("Headline case through time", loc="left")
    a2.legend(fontsize=8)
    _save(fig, path)
