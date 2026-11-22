# ABOUTME: Downloads Ken French's official daily FF3 factors (daily benchmark and daily risk-free rate).
"""
Run:     python DataDownloads/M_FrenchFactorsDaily.py   (from pyCode/)
Inputs:  Ken French data library, F-F_Research_Data_Factors_daily_CSV.zip
Outputs: pyData/Intermediate/french_ff3_daily.parquet
         date (trading day), mkt_rf, smb, hml, rf (decimals, not percent)

Same role as E_FrenchFactors.py, one frequency down: the benchmark for our daily factors,
and the daily risk-free rate to turn the daily market return into MKT-RF. French's daily
RF is the one-month T-bill rate spread over the trading days of the month.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_ff3_daily.parquet"
FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rf"]


def download():
    lines = download_french_csv("F-F_Research_Data_Factors_daily_CSV.zip")

    # The CSV has a text header, then one row per trading day (date = YYYYMMDD).
    # Keeping rows whose date has 8 digits skips the header and the copyright line.
    rows = []
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 5 and len(fields[0]) == 8 and fields[0].isdigit():
            rows.append(fields)

    french = pd.DataFrame(rows, columns=["yyyymmdd"] + FACTOR_COLUMNS)
    french[FACTOR_COLUMNS] = french[FACTOR_COLUMNS].astype(float) / 100  # percent -> decimal
    french["date"] = pd.to_datetime(french["yyyymmdd"], format="%Y%m%d")
    return french[["date"] + FACTOR_COLUMNS]


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Ken French's daily FF3 factors...", flush=True)
        french = download()
        require_unique(french, ["date"], "French FF3 daily")
        french.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        french = pd.read_parquet(OUTPUT_FILE)

    # --- Sanity checks ---------------------------------------------------------------------
    print(f"trading days: {len(french):,}   dates: {french['date'].min().date()} to "
          f"{french['date'].max().date()}")
    days_per_year = french.groupby(french["date"].dt.year).size()
    print(f"trading days per year: median {days_per_year.median():.0f}, "
          f"min {days_per_year.min()} ({days_per_year.idxmin()}), "
          f"max {days_per_year.max()} ({days_per_year.idxmax()})")
    print("Last 3 days:")
    print(french.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
