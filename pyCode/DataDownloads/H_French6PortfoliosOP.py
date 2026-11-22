# ABOUTME: Downloads Ken French's six 2x3 size x operating-profitability portfolios (the building blocks of RMW).
"""
Run:     python DataDownloads/H_French6PortfoliosOP.py   (from pyCode/)
Inputs:  Ken French data library, 6_Portfolios_ME_OP_2x3_CSV.zip
Outputs: pyData/Intermediate/french_6_portfolios_op.parquet
         long table: table (vw_return / n_firms / avg_me), date, SL, SN, SH, BL, BN, BH
         returns in decimals, avg_me in $ millions

French labels the columns SMALL LoOP, ME1 OP2, SMALL HiOP, BIG LoOP, ME2 OP2, BIG HiOP.
We rename them to our generic labels (L = low signal = weak profitability, H = high =
robust), so they line up with pyData/Portfolios/FF2x3_OperProf.parquet. Comparing the
six portfolios one by one shows which leg an RMW gap comes from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, PORTFOLIO_NAMES, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_6_portfolios_op.parquet"
FRENCH_COLUMNS = ["SMALL LoOP", "ME1 OP2", "SMALL HiOP", "BIG LoOP", "ME2 OP2", "BIG HiOP"]

# The file holds several tables one after the other; we keep three monthly ones.
TABLES = {
    "Average Value Weighted Returns -- Monthly": "vw_return",
    "Number of Firms in Portfolios": "n_firms",
    "Average Market Cap": "avg_me",
}


def download():
    lines = download_french_csv("6_Portfolios_ME_OP_2x3_CSV.zip")

    # Check the column order once, instead of assuming it matches SL..BH.
    header = next([f.strip() for f in line.split(",")][1:] for line in lines
                  if line.strip().startswith(",SMALL"))
    if header != FRENCH_COLUMNS:
        raise SystemExit(f"unexpected columns in the ME x OP file: {header}")

    # A title line starts a table; the monthly rows below it (date = YYYYMM, then six
    # values) belong to that table until the next title.
    rows = []
    current_table = None
    for line in lines:
        title = line.strip()
        if title in TABLES:
            current_table = TABLES[title]
            continue
        fields = [f.strip() for f in line.split(",")]
        is_monthly_row = len(fields) == 7 and len(fields[0]) == 6 and fields[0].isdigit()
        if is_monthly_row and current_table is not None:
            rows.append([current_table] + fields)
        elif title and not is_monthly_row and not title.startswith(","):
            current_table = None  # any other title ends the table we were reading

    french_6 = pd.DataFrame(rows, columns=["table", "yyyymm"] + PORTFOLIO_NAMES)
    french_6["date"] = pd.to_datetime(french_6["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    french_6[PORTFOLIO_NAMES] = french_6[PORTFOLIO_NAMES].astype(float)
    # The file header says missing data are coded -99.99 or -999; make them real missing values.
    french_6[PORTFOLIO_NAMES] = french_6[PORTFOLIO_NAMES].mask(
        french_6[PORTFOLIO_NAMES].isin([-99.99, -999]))
    is_return = french_6["table"] == "vw_return"
    french_6.loc[is_return, PORTFOLIO_NAMES] = french_6.loc[is_return, PORTFOLIO_NAMES] / 100
    return french_6[["table", "date"] + PORTFOLIO_NAMES]


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Ken French's size x OP portfolios...", flush=True)
        french_6 = download()
        require_unique(french_6, ["table", "date"], "French 6 portfolios ME x OP")
        french_6.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        french_6 = pd.read_parquet(OUTPUT_FILE)

    print(french_6.groupby("table")["date"].agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()
