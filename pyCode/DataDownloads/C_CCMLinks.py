# ABOUTME: Downloads the CRSP-Compustat (CCM) link table.
"""
Run:     python DataDownloads/C_CCMLinks.py   (from pyCode/)
Inputs:  WRDS crsp.ccmxpf_lnkhist
Outputs: pyData/Intermediate/ccm_links.parquet
         gvkey, permno, linktype, linkprim, linkdt, linkenddt
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.wrds_connection import get_wrds_connection  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "ccm_links.parquet"


def download():
    # Compustat identifies firms by gvkey, CRSP by permno. The CCM table says which gvkey
    # maps to which permno, and over which dates.
    #   linktype LU / LC   links CRSP has researched (C = confirmed); other types are
    #                      weaker or duplicate links
    #   linkprim P / C     the firm's primary security, so one gvkey maps to one permno
    #                      at a time
    # linkenddt is missing when the link is still active; we fill it with today's date.
    db = get_wrds_connection()
    ccm = db.raw_sql("""
        SELECT gvkey, lpermno AS permno, linktype, linkprim, linkdt, linkenddt
        FROM crsp.ccmxpf_lnkhist
        WHERE linktype IN ('LU', 'LC')
          AND linkprim IN ('P', 'C')
    """, date_cols=["linkdt", "linkenddt"])
    db.close()
    ccm["permno"] = ccm["permno"].astype(int)
    ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp.today().normalize())
    return ccm


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading CCM links...", flush=True)
        ccm = download()
        ccm.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        ccm = pd.read_parquet(OUTPUT_FILE)

    print(f"links: {len(ccm):,}   gvkeys: {ccm['gvkey'].nunique():,}   "
          f"permnos: {ccm['permno'].nunique():,}")
    print(ccm.groupby(["linktype", "linkprim"]).size().rename("n").to_string())


if __name__ == "__main__":
    main()
