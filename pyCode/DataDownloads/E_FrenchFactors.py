# ABOUTME: Downloads Ken French's official monthly FF3 factors (benchmark and risk-free rate).
"""
Run:     python DataDownloads/E_FrenchFactors.py   (from pyCode/)
Inputs:  Ken French data library, F-F_Research_Data_Factors_CSV.zip
Outputs: pyData/Intermediate/french_ff3.parquet
         date (month-end), mkt_rf, smb, hml, rf (decimals, not percent)

We need this file twice: as the benchmark to validate our replication, and for RF
(the one-month T-bill rate) to turn the market return into MKT-RF.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_ff3.parquet"
FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rf"]


def download():
    lines = download_french_csv("F-F_Research_Data_Factors_CSV.zip")

    # The CSV has a text header, then monthly rows (date = YYYYMM), then a block of
    # annual rows (date = YYYY). Keeping rows whose date has 6 digits keeps the monthly block.
    rows = []
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 5 and len(fields[0]) == 6 and fields[0].isdigit():
            rows.append(fields)

    french = pd.DataFrame(rows, columns=["yyyymm"] + FACTOR_COLUMNS)
    french[FACTOR_COLUMNS] = french[FACTOR_COLUMNS].astype(float) / 100  # percent -> decimal
    # Month-end dates, to match our CRSP dates.
    french["date"] = pd.to_datetime(french["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return french[["date"] + FACTOR_COLUMNS]


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Ken French's FF3 factors...", flush=True)
        french = download()
        require_unique(french, ["date"], "French FF3")
        french.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        french = pd.read_parquet(OUTPUT_FILE)

    print(f"months: {len(french):,}   dates: {french['date'].min().date()} to "
          f"{french['date'].max().date()}")
    print("Last 3 months:")
    print(french.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
