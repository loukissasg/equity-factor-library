# ABOUTME: Downloads CRSP daily stock data (CIZ format) for the Fama-French universe, one file per year.
"""
Run:     python DataDownloads/A_CRSPDaily.py   (from pyCode/)
Inputs:  WRDS crsp.dsf_v2
Outputs: pyData/Intermediate/crsp_daily/crsp_daily_YYYY.parquet, one file per year
         one row per security (PERMNO) and trading day: permno, permco, dlycaldt,
         dlyprevdt, dlyret, dlyprc, shrout, dlyprevcap, primaryexch, dlydelflg

The daily table has ~75M rows for our universe since 1960, so we download one year at a
time. Each year is saved as soon as it arrives: if the connection drops, a rerun only
fetches the missing years (unless REFRESH_DOWNLOADS = True).
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import (CRSP_DAILY_DIR, CRSP_END_YEAR, CRSP_START, REFRESH_DOWNLOADS,  # noqa: E402
                    make_folders)
from utils.checks import require_unique  # noqa: E402
from utils.wrds_connection import get_wrds_connection  # noqa: E402

LAST_YEAR = CRSP_END_YEAR
MAX_ATTEMPTS = 3  # per year, if WRDS drops the connection

# The same universe as the monthly data: ordinary common shares of US firms on NYSE,
# AMEX or NASDAQ (old shrcd 10/11, exchcd 1/2/3). {t} is the table alias, because the
# filter is applied to two rows below. See DataDownloads/CheckCIZSchemaDaily.py.
FF_FILTER = """
        {t}.sharetype        = 'NS'
    AND {t}.securitytype     = 'EQTY'
    AND {t}.securitysubtype  = 'COM'
    AND {t}.usincflg         = 'Y'
    AND {t}.issuertype      IN ('ACOR', 'CORP')
    AND {t}.primaryexch     IN ('N', 'A', 'Q')
    AND {t}.conditionaltype  = 'RW'
    AND {t}.tradingstatusflg = 'A'
"""


def year_file(year):
    return CRSP_DAILY_DIR / f"crsp_daily_{year}.parquet"


def download_year(db, year):
    # Two kinds of rows:
    #  1. Ordinary trading days that pass the common-stock filter.
    #  2. Delisting rows (dlydelflg = 'Y'). In the daily table the delisting return is an
    #     extra row, dated the trading day after the last trade, and its classification
    #     fields are blank ('N/A', primaryexch 'X'), so the filter would drop it and we
    #     would silently lose every delisting loss. We keep it if the stock passed the
    #     filter on its last trading day (dlyprevdt), i.e. it was in our universe.
    crsp = db.raw_sql(f"""
        SELECT d.permno, d.permco, d.dlycaldt, d.dlyprevdt, d.dlyret, d.dlyprc, d.shrout,
               d.dlyprevcap, d.primaryexch, d.dlydelflg
        FROM crsp.dsf_v2 d
        WHERE d.dlycaldt BETWEEN '{year}-01-01' AND '{year}-12-31'
          AND (
                ({FF_FILTER.format(t='d')})
             OR (d.dlydelflg = 'Y'
                 AND EXISTS (SELECT 1 FROM crsp.dsf_v2 p
                             WHERE p.permno = d.permno AND p.dlycaldt = d.dlyprevdt
                               AND {FF_FILTER.format(t='p')}))
          )
    """, date_cols=["dlycaldt", "dlyprevdt"])
    # Smaller types: the full sample is large, and these ids fit in 32-bit integers.
    crsp["permno"] = crsp["permno"].astype("int32")
    crsp["permco"] = crsp["permco"].astype("int32")
    for col in ["dlyret", "dlyprc", "shrout", "dlyprevcap"]:
        crsp[col] = crsp[col].astype("float64")
    return crsp


def main():
    make_folders()
    first_year = int(CRSP_START[:4])
    years = list(range(first_year, LAST_YEAR + 1))
    to_fetch = [y for y in years if REFRESH_DOWNLOADS or not year_file(y).exists()]
    print(f"CRSP daily: {len(years) - len(to_fetch)} of {len(years)} years cached, "
          f"downloading {len(to_fetch)}", flush=True)

    if to_fetch:
        db = get_wrds_connection()
        for year in to_fetch:
            start = time.time()
            # WRDS sometimes drops long connections ("server closed the connection
            # unexpectedly", seen in 1991 on the first run). Reconnect and retry the year.
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    crsp = download_year(db, year)
                    break
                except Exception as error:  # noqa: BLE001  (any database error)
                    if attempt == MAX_ATTEMPTS:
                        raise
                    print(f"  {year}: attempt {attempt} failed ({type(error).__name__}), "
                          f"reconnecting...", flush=True)
                    try:
                        db.close()
                    except Exception:  # noqa: BLE001  (connection is already dead)
                        pass
                    time.sleep(30)
                    db = get_wrds_connection()
            require_unique(crsp, ["permno", "dlycaldt"], f"CRSP daily {year}")
            crsp.to_parquet(year_file(year), index=False)
            print(f"  {year}: {len(crsp):>9,} rows, {crsp['permno'].nunique():>5,} stocks, "
                  f"{(crsp['dlydelflg'] == 'Y').sum():>4} delisting rows  "
                  f"({time.time() - start:.0f}s)", flush=True)
        db.close()

    # --- Sanity checks (one year at a time, so memory stays small) -------------------------
    print("\nPer year: rows, stocks, trading days, delisting rows, missing returns", flush=True)
    summary = []
    for year in years:
        crsp = pd.read_parquet(year_file(year), columns=["permno", "dlycaldt", "dlyret", "dlydelflg"])
        trading = crsp[crsp["dlydelflg"] != "Y"]
        summary.append({
            "year": year,
            "rows": len(crsp),
            "stocks": crsp["permno"].nunique(),
            "trading_days": trading["dlycaldt"].nunique(),
            "delisting_rows": int((crsp["dlydelflg"] == "Y").sum()),
            "missing_ret_%": round(100 * crsp["dlyret"].isna().mean(), 2),
        })
    summary = pd.DataFrame(summary)
    print(summary[(summary["year"] % 5 == 0) | (summary["year"] == LAST_YEAR)].to_string(index=False))
    print(f"total: {summary['rows'].sum():,} rows, {summary['delisting_rows'].sum():,} delisting rows")
    # About 252 trading days a year; far fewer would mean a year was cut off. (1968 is
    # legitimately short: the NYSE closed on Wednesdays in the second half of 1968 to
    # clear its paperwork backlog.)
    short = summary[summary["trading_days"] < 220]
    if not short.empty:
        raise SystemExit(f"Years with too few trading days (incomplete download?):\n{short}")


if __name__ == "__main__":
    main()
