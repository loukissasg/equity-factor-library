# ABOUTME: Downloads each Compustat firm's industry code (SIC) from comp.names, used to drop financial firms from GP.
"""
Run:     python DataDownloads/I_CompustatNames.py   (from pyCode/)
Inputs:  WRDS comp.names
Outputs: pyData/Intermediate/compustat_names.parquet
         one row per firm (gvkey): gvkey, sic (4-digit SIC code, as text)

Why comp.names: OpenAP's GP takes `sic` from comp.names (their DataDownloads/
CompustatAnnual.py joins it in by gvkey), and we copy their filter exactly. This is
the *header* SIC: the firm's current industry, one value per firm for its whole history.
(Compustat also has a historical SIC, funda.sich, but it is often missing before the
late 1980s; OpenAP does not use it for GP.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.wrds_connection import get_wrds_connection  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "compustat_names.parquet"


def download():
    db = get_wrds_connection()
    # Check the column exists first (information_schema, not wrds' describe_table(),
    # which made a later query hang in B_CompustatAnnual.py).
    columns = set(db.raw_sql("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'comp' AND table_name = 'names'
    """)["column_name"].str.lower())
    if not {"gvkey", "sic"} <= columns:
        db.close()
        raise SystemExit(f"comp.names lacks gvkey or sic; columns found: {sorted(columns)}")

    names = db.raw_sql("SELECT gvkey, sic FROM comp.names")
    db.close()
    return names


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Compustat names (SIC codes)...", flush=True)
        names = download()
        require_unique(names, ["gvkey"], "Compustat names")
        names.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        names = pd.read_parquet(OUTPUT_FILE)

    sic = pd.to_numeric(names["sic"], errors="coerce")
    print(f"firms (gvkey): {len(names):,}   missing or non-numeric SIC: {sic.isna().mean():.1%}   "
          f"financials (SIC 6000-6999): {sic.between(6000, 6999).mean():.1%}")


if __name__ == "__main__":
    main()
