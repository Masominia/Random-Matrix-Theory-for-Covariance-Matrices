# Data

The pipeline reads only this folder.

| File | Content | Coverage | Source |
|---|---|---|---|
| `nasdaq_prices_daily.csv.gz` | Adjusted daily closing prices of 1,455 Nasdaq-listed stocks | 2018-01-02 to 2023-05-31 (1,362 days, no missing values) | skfolio datasets (`skfolio/skfolio-datasets` on GitHub), originally from Yahoo Finance |

Returns are daily log returns (1,361 days). Universes of N stocks are drawn at random with a fixed seed.

**Caveats.**

* **Survivorship.** The panel contains only stocks with a complete history over the period, so delisted names are absent. This inflates average returns; it matters much less for a study of risk, and affects all estimators alike.
* **Short history.** With 500 days needed for estimation, the out-of-sample period is 2020-2023 (861 days) and starts just before the COVID crash, which is why realised volatilities exceed predicted ones for every estimator.
* **No filters.** No winsorisation and no liquidity screen: some small caps have daily moves beyond 50%. Results are for random Nasdaq universes, not an investable index.

Refresh with `python scripts/fetch_data.py`.
