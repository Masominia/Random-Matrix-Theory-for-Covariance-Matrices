"""Download the Nasdaq price panel used by skfolio (1,455 stocks, daily, 2018-2023) into data/raw.

Usage:  python scripts/fetch_data.py [--out data/raw]
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/skfolio/skfolio-datasets/main/datasets/nasdaq_dataset.csv.gz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "raw"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=120) as r:  # noqa: S310 (fixed https URL)
        (out / "nasdaq_prices_daily.csv.gz").write_bytes(r.read())
    print("nasdaq_prices_daily.csv.gz written")


if __name__ == "__main__":
    main()
