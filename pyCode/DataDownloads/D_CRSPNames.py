# ABOUTME: Downloads each CRSP security's latest company name and ticker.
"""
Run:     python DataDownloads/D_CRSPNames.py   (from pyCode/)
Inputs:  WRDS crsp.stocknames_v2
Outputs: pyData/Intermediate/crsp_names.parquet
         permno, ticker, issuernm

Only used to make tables readable (e.g. "BERKSHIRE HATHAWAY" instead of a PERMNO in
Validation/HMLGapDiagnostics.py). Not needed for the factors themselves.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.wrds_connection import get_wrds_connection  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "crsp_names.parquet"


def download():
    db = get_wrds_connection()
    names = db.raw_sql("""
        SELECT DISTINCT ON (permno) permno, ticker, issuernm
        FROM crsp.stocknames_v2
        ORDER BY permno, nameenddt DESC
    """)
    db.close()
    names["permno"] = names["permno"].astype(int)
    return names


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading CRSP company names...", flush=True)
        names = download()
        require_unique(names, ["permno"], "CRSP names")
        names.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        names = pd.read_parquet(OUTPUT_FILE)

    print(f"securities: {len(names):,}")


if __name__ == "__main__":
    main()
