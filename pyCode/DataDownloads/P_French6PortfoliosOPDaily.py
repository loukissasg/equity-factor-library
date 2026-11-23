# ABOUTME: Downloads Ken French's six 2x3 size x operating-profitability portfolios at daily frequency.
"""
Run:     python DataDownloads/P_French6PortfoliosOPDaily.py   (from pyCode/)
Inputs:  Ken French data library, 6_Portfolios_ME_OP_2x3_daily_CSV.zip
Outputs: pyData/Intermediate/french_6_portfolios_op_daily.parquet
         long table: table (vw_return / n_firms), date, SL, SN, SH, BL, BN, BH
         returns in decimals; L = weak, H = robust profitability

Daily counterpart of H_French6PortfoliosOP.py, the building blocks of French's RMW.
Comparing them with ours (pyData/Portfolios/FF2x3Daily_OperProf.parquet) shows which
portfolio a daily RMW gap comes from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, PORTFOLIO_NAMES, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_6_portfolios_op_daily.parquet"
FRENCH_COLUMNS = ["SMALL LoOP", "ME1 OP2", "SMALL HiOP", "BIG LoOP", "ME2 OP2", "BIG HiOP"]

# The file holds several tables one after the other; we keep two. Note: in this daily
# file French titles the daily returns table "-- Monthly" (checked 2026-11-22: the rows
# below it are daily, YYYYMMDD). Both spellings are accepted in case he fixes it.
TABLES = {
    "Average Value Weighted Returns -- Daily": "vw_return",
    "Average Value Weighted Returns -- Monthly": "vw_return",
    "Number of Firms in Portfolios": "n_firms",
}


def download():
    lines = download_french_csv("6_Portfolios_ME_OP_2x3_daily_CSV.zip")

    # Check the column order once, instead of assuming it matches SL..BH.
    header = next([f.strip() for f in line.split(",")][1:] for line in lines
                  if line.strip().startswith(",SMALL"))
    if header != FRENCH_COLUMNS:
        raise SystemExit(f"unexpected columns in the daily ME x OP file: {header}")

    # A title line starts a table; the daily rows below it (date = YYYYMMDD, then six
    # values) belong to that table until the next title.
    rows = []
    current_table = None
    for line in lines:
        title = line.strip()
        fields = [f.strip() for f in line.split(",")]
        is_daily_row = len(fields) == 7 and len(fields[0]) == 8 and fields[0].isdigit()
        if is_daily_row:
            if current_table is not None:
                rows.append([current_table] + fields)
        elif title and not title.startswith(","):
            current_table = TABLES.get(title)  # a table we keep, or None for any other title

    french_6 = pd.DataFrame(rows, columns=["table", "yyyymmdd"] + PORTFOLIO_NAMES)
    french_6["date"] = pd.to_datetime(french_6["yyyymmdd"], format="%Y%m%d")
    french_6[PORTFOLIO_NAMES] = french_6[PORTFOLIO_NAMES].astype(float)
    is_return = french_6["table"] == "vw_return"
    french_6.loc[is_return, PORTFOLIO_NAMES] = french_6.loc[is_return, PORTFOLIO_NAMES] / 100
    return french_6[["table", "date"] + PORTFOLIO_NAMES]


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Ken French's daily size x OP portfolios...", flush=True)
        french_6 = download()
        # Stop if a table was not found (e.g. French renamed a title): better than
        # silently saving a file without returns.
        missing = {"vw_return", "n_firms"} - set(french_6["table"])
        if missing:
            raise SystemExit(f"tables not found in the daily ME x OP file: {missing}")
        require_unique(french_6, ["table", "date"], "French 6 portfolios OP daily")
        french_6.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        french_6 = pd.read_parquet(OUTPUT_FILE)

    print(french_6.groupby("table")["date"].agg(["count", "min", "max"]).to_string())
    # French marks missing values with -99.99 or -999; there should be none in our period.
    returns = french_6[(french_6["table"] == "vw_return") & (french_6["date"] >= "1963-07-01")]
    print(f"missing-value codes in returns since July 1963: "
          f"{(returns[PORTFOLIO_NAMES] <= -0.9999).sum().sum()}")


if __name__ == "__main__":
    main()
