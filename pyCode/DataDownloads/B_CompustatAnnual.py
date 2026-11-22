# ABOUTME: Downloads the Compustat annual items needed for book equity and profitability.
"""
Run:     python DataDownloads/B_CompustatAnnual.py   (from pyCode/)
Inputs:  WRDS comp.funda
Outputs: pyData/Intermediate/compustat_annual.parquet
         one row per firm (gvkey) and fiscal year-end: gvkey, datadate, fyear and the
         items below (all in $ millions)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import COMPUSTAT_START, INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.wrds_connection import get_wrds_connection  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "compustat_annual.parquet"

# Book equity (Predictors/BMdec.py)
BE_ITEMS = ["seq", "ceq", "pstk", "at", "lt", "pstkrv", "pstkl", "txditc"]
# Profitability (Predictors/OperProf.py, Predictors/GP.py):
#   revt  total revenue          cogs  cost of goods sold
#   xsga  selling, general and administrative expense
#   xint  interest expense       mib   minority (noncontrolling) interest, balance sheet
PROFIT_ITEMS = ["revt", "cogs", "xsga", "xint", "mib"]
ITEMS = BE_ITEMS + PROFIT_ITEMS


def download():
    # The four standard filters keep one clean record per firm-year:
    #   indfmt='INDL'  industrial format (not the separate financial-services format)
    #   datafmt='STD'  standardized data (not restated)
    #   popsrc='D'     domestic (North American) firms
    #   consol='C'     consolidated statements
    # The signals themselves are built in Predictors/.
    db = get_wrds_connection()

    # Check that every item exists in comp.funda before querying, so a renamed field
    # gives a clear message instead of an SQL error.
    # (A plain information_schema query; wrds' describe_table() made the download below hang.)
    funda_columns = set(db.raw_sql("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'comp' AND table_name = 'funda'
    """)["column_name"].str.lower())
    missing = [item for item in ITEMS if item not in funda_columns]
    if missing:
        db.close()
        raise SystemExit(f"comp.funda has no column(s) {missing}")
    print(f"all {len(ITEMS)} items found in comp.funda: {', '.join(ITEMS)}")

    comp = db.raw_sql(f"""
        SELECT gvkey, datadate, fyear, {", ".join(ITEMS)}
        FROM comp.funda
        WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
          AND datadate >= '{COMPUSTAT_START}'
    """, date_cols=["datadate"])
    db.close()
    return comp


def main():
    make_folders()
    # Download again if asked, if there is no cached file, or if the cached file was made
    # before an item was added to ITEMS (otherwise the new signal would silently fail).
    cached_items_missing = []
    if OUTPUT_FILE.exists():
        cached_columns = pd.read_parquet(OUTPUT_FILE).columns
        cached_items_missing = [item for item in ITEMS if item not in cached_columns]

    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists() or cached_items_missing:
        if cached_items_missing:
            print(f"cached file lacks {cached_items_missing}: downloading again", flush=True)
        print("Downloading Compustat annual...", flush=True)
        comp = download()
        comp.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        comp = pd.read_parquet(OUTPUT_FILE)

    print(f"rows: {len(comp):,}   firms (gvkey): {comp['gvkey'].nunique():,}")
    print(f"dates: {comp['datadate'].min().date()} to {comp['datadate'].max().date()}")
    firms_per_year = comp.groupby(comp["datadate"].dt.year)["gvkey"].nunique()
    print("Firms per fiscal-year-end calendar year, every 5th year:")
    print(firms_per_year[firms_per_year.index % 5 == 0].to_string())
    print("Share missing, by item (seq is the main equity item; the others are fallbacks):")
    print(comp[ITEMS].isna().mean().map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
