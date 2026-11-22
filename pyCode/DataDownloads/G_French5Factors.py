# ABOUTME: Downloads Ken French's monthly five factors (2x3), the benchmark for our RMW.
"""
Run:     python DataDownloads/G_French5Factors.py   (from pyCode/)
Inputs:  Ken French data library, F-F_Research_Data_5_Factors_2x3_CSV.zip
Outputs: pyData/Intermediate/french_ff5.parquet
         date (month-end), mkt_rf, smb, hml, rmw, cma, rf (decimals, not percent)

RMW (robust minus weak profitability) = 1/2 (SR + BR) - 1/2 (SW + BW), from six
portfolios on size x operating profitability. Note: the SMB in this file is *not* the
FF3 SMB; it averages the SMBs of the size x B/M, size x OP and size x investment sorts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_ff5.parquet"
EXPECTED_HEADER = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]


def download():
    lines = download_french_csv("F-F_Research_Data_5_Factors_2x3_CSV.zip")

    # Check the column order instead of assuming it: the header line is ",Mkt-RF,SMB,...".
    header = next([f.strip() for f in line.split(",")][1:] for line in lines
                  if line.strip().startswith(",Mkt-RF"))
    if header != EXPECTED_HEADER:
        raise SystemExit(f"unexpected columns in the 5-factor file: {header}")

    # Monthly rows have a 6-digit date (YYYYMM); the annual block below has 4 digits.
    rows = []
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 7 and len(fields[0]) == 6 and fields[0].isdigit():
            rows.append(fields)

    french = pd.DataFrame(rows, columns=["yyyymm"] + FACTOR_COLUMNS)
    french[FACTOR_COLUMNS] = french[FACTOR_COLUMNS].astype(float) / 100  # percent -> decimal
    french["date"] = pd.to_datetime(french["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return french[["date"] + FACTOR_COLUMNS]


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Ken French's 5 factors...", flush=True)
        french = download()
        require_unique(french, ["date"], "French FF5")
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
