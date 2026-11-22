# ABOUTME: Downloads CRSP monthly stock data (CIZ format) for the Fama-French universe.
"""
Run:     python DataDownloads/A_CRSPMonthly.py   (from pyCode/)
Inputs:  WRDS crsp.msf_v2
Outputs: pyData/Intermediate/crsp_monthly.parquet
         one row per security (PERMNO) and month: permno, permco, mthcaldt, mthret,
         mthprc, shrout, primaryexch
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import CRSP_START, INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.wrds_connection import get_wrds_connection  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "crsp_monthly.parquet"


def download():
    # Universe: ordinary common shares of US firms on NYSE, AMEX or NASDAQ. This is the
    # CIZ version of the classic filter shrcd in (10, 11) and exchcd in (1, 2, 3):
    #   sharetype='NS'                ordinary shares (drops ADRs, units, SBIs)
    #   securitytype='EQTY', securitysubtype='COM'   common equity (drops ETFs, CEFs)
    #   usincflg='Y'                  incorporated in the US (old shrcd 12 = foreign)
    #   issuertype in ACOR/CORP       operating companies (drops REITs, old shrcd 18)
    #   primaryexch in N/A/Q          NYSE / AMEX / NASDAQ
    #   conditionaltype='RW', tradingstatusflg='A'   regular-way trading, active
    # CheckCIZSchema.py confirmed that delisting months pass this filter, so the
    # delisting returns that CIZ puts into mthret are kept.
    db = get_wrds_connection()
    crsp = db.raw_sql(f"""
        SELECT permno, permco, mthcaldt, mthret, mthprc, shrout, primaryexch
        FROM crsp.msf_v2
        WHERE mthcaldt >= '{CRSP_START}'
          AND sharetype        = 'NS'
          AND securitytype     = 'EQTY'
          AND securitysubtype  = 'COM'
          AND usincflg         = 'Y'
          AND issuertype      IN ('ACOR', 'CORP')
          AND primaryexch     IN ('N', 'A', 'Q')
          AND conditionaltype  = 'RW'
          AND tradingstatusflg = 'A'
    """, date_cols=["mthcaldt"])
    db.close()
    crsp["permno"] = crsp["permno"].astype(int)
    crsp["permco"] = crsp["permco"].astype(int)
    return crsp


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading CRSP monthly (a few minutes)...", flush=True)
        crsp = download()
        require_unique(crsp, ["permno", "mthcaldt"], "CRSP monthly")
        crsp.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        crsp = pd.read_parquet(OUTPUT_FILE)

    print(f"rows: {len(crsp):,}   stocks (permno): {crsp['permno'].nunique():,}   "
          f"firms (permco): {crsp['permco'].nunique():,}")
    print(f"dates: {crsp['mthcaldt'].min().date()} to {crsp['mthcaldt'].max().date()}")
    print(f"missing returns: {crsp['mthret'].isna().mean():.2%}   "
          f"missing prices: {crsp['mthprc'].isna().mean():.2%}")
    december = crsp[crsp["mthcaldt"].dt.month == 12]
    per_year = december.groupby([december["mthcaldt"].dt.year, "primaryexch"]).size().unstack()
    print("Stocks in December, by exchange (N = NYSE, A = AMEX, Q = NASDAQ), every 5th year:")
    print(per_year[per_year.index % 5 == 0].to_string())


if __name__ == "__main__":
    main()
