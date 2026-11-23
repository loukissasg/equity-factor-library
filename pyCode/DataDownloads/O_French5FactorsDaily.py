# ABOUTME: Downloads Ken French's daily Fama-French 5 factors (daily benchmark for RMW).
"""
Run:     python DataDownloads/O_French5FactorsDaily.py   (from pyCode/)
Inputs:  Ken French data library, F-F_Research_Data_5_Factors_2x3_daily_CSV.zip
Outputs: pyData/Intermediate/french_ff5_daily.parquet
         date (trading day), mkt_rf, smb, hml, rmw, cma, rf (decimals, not percent)

Daily counterpart of G_French5Factors.py. We use its RMW to validate our daily RMW.
(Its SMB is the average of the SMBs of three sorts, so it differs from the FF3 SMB.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_ff5_daily.parquet"
FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]


def download():
    lines = download_french_csv("F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

    # Text header, then one row per trading day: YYYYMMDD and six values.
    rows = []
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 7 and len(fields[0]) == 8 and fields[0].isdigit():
            rows.append(fields)

    french = pd.DataFrame(rows, columns=["yyyymmdd"] + FACTOR_COLUMNS)
    french[FACTOR_COLUMNS] = french[FACTOR_COLUMNS].astype(float) / 100  # percent -> decimal
    french["date"] = pd.to_datetime(french["yyyymmdd"], format="%Y%m%d")
    return french[["date"] + FACTOR_COLUMNS]


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Ken French's daily 5 factors...", flush=True)
        french = download()
        require_unique(french, ["date"], "French FF5 daily")
        french.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        french = pd.read_parquet(OUTPUT_FILE)

    print(f"trading days: {len(french):,}   dates: {french['date'].min().date()} to "
          f"{french['date'].max().date()}")
    print("Last 3 days:")
    print(french.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
