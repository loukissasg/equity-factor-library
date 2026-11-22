# ABOUTME: Downloads Ken French's six 2x3 size-B/M portfolios at daily frequency (returns and number of firms).
"""
Run:     python DataDownloads/N_French6PortfoliosDaily.py   (from pyCode/)
Inputs:  Ken French data library, 6_Portfolios_2x3_daily_CSV.zip
Outputs: pyData/Intermediate/french_6_portfolios_daily.parquet
         long table: table (vw_return / n_firms), date, SL, SN, SH, BL, BN, BH
         returns in decimals

Daily counterpart of F_French6Portfolios.py. Comparing our six daily portfolios with
these shows which portfolio a daily factor gap comes from (Validation/CompareFrenchDaily.py),
and compounding them to months shows how big the daily-vs-monthly gap is in French's
own data (Validation/CompareDailyMonthly.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, PORTFOLIO_NAMES, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_6_portfolios_daily.parquet"

# The file holds four tables one after the other (VW returns, EW returns, number of firms,
# average firm size); we keep two. The columns are always SL SN SH BL BN BH, in that order.
TABLES = {
    "Average Value Weighted Returns -- Daily": "vw_return",
    "Number of Firms in Portfolios": "n_firms",
}


def download():
    lines = download_french_csv("6_Portfolios_2x3_daily_CSV.zip")

    # Walk through the file: a title line starts a table, and the daily rows below it
    # (date = YYYYMMDD, then six values) belong to that table until the next title.
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
        print("Downloading Ken French's daily 2x3 portfolios...", flush=True)
        french_6 = download()
        require_unique(french_6, ["table", "date"], "French 6 portfolios daily")
        french_6.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        french_6 = pd.read_parquet(OUTPUT_FILE)

    # --- Sanity checks ---------------------------------------------------------------------
    print(french_6.groupby("table")["date"].agg(["count", "min", "max"]).to_string())
    # French marks missing values with -99.99 or -999; there should be none in our period.
    returns = french_6[(french_6["table"] == "vw_return") & (french_6["date"] >= "1963-07-01")]
    print(f"missing-value codes in returns since July 1963: "
          f"{(returns[PORTFOLIO_NAMES] <= -0.9999).sum().sum()}")


if __name__ == "__main__":
    main()
