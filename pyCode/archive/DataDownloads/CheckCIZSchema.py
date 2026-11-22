# ABOUTME: Checks the CRSP v2 (CIZ) schema on WRDS before we rely on its field names.
"""
Run:     python DataDownloads/CheckCIZSchema.py   (from pyCode/; master.py runs it only
         if RUN_SCHEMA_CHECK = True in config.py)
Inputs:  WRDS information_schema, crsp.msf_v2
Outputs: printed report only

CRSP replaced its legacy format (msf, msenames, shrcd, exchcd) with the CIZ format.
Most online FF tutorials still use the old fields, so before writing any filters we
confirm that the CIZ columns we rely on exist and see which values they take.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

from config import CRSP_START  # noqa: E402
from utils.wrds_connection import get_wrds_connection  # noqa: E402

NEEDED_COLUMNS = {
    ("crsp", "msf_v2"): [
        "permno", "permco", "mthcaldt", "mthret", "mthprc", "shrout", "mthcap",
        "mthprevcap", "sharetype", "securitytype", "securitysubtype", "usincflg",
        "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg",
    ],
    ("comp", "funda"): [
        "gvkey", "datadate", "fyear", "indfmt", "datafmt", "popsrc", "consol",
        "seq", "ceq", "pstk", "at", "lt", "pstkrv", "pstkl", "txditc",
    ],
    ("crsp", "ccmxpf_lnkhist"): [
        "gvkey", "lpermno", "lpermco", "linktype", "linkprim", "linkdt", "linkenddt",
    ],
}

CLASSIFICATION_FIELDS = [
    "sharetype", "securitytype", "securitysubtype", "usincflg",
    "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg",
]

# The proposed filter: ordinary common shares (old shrcd 10/11) of US firms on NYSE,
# AMEX or NASDAQ (old exchcd 1/2/3).
FF_FILTER = """
        sharetype        = 'NS'
    AND securitytype     = 'EQTY'
    AND securitysubtype  = 'COM'
    AND usincflg         = 'Y'
    AND issuertype      IN ('ACOR', 'CORP')
    AND primaryexch     IN ('N', 'A', 'Q')
    AND conditionaltype  = 'RW'
    AND tradingstatusflg = 'A'
"""


def main():
    print("Checking the CIZ schema on WRDS...", flush=True)
    db = get_wrds_connection()

    # --- Do the columns we need exist? -------------------------------------------------
    print("\n--- Required columns ---")
    for (schema, table), columns in NEEDED_COLUMNS.items():
        found = db.raw_sql(f"""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = '{schema}' AND table_name = '{table}'
        """)
        missing = [c for c in columns if c not in set(found["column_name"])]
        status = "OK" if not missing else f"MISSING: {missing}"
        print(f"{schema}.{table:<16} {len(columns)} columns checked -> {status}")

    # --- Date range ----------------------------------------------------------------------
    print("\n--- CRSP monthly coverage ---")
    coverage = db.raw_sql("""
        SELECT MIN(mthcaldt) AS first_month, MAX(mthcaldt) AS last_month, COUNT(*) AS n_rows
        FROM crsp.msf_v2
    """)
    print(coverage.to_string(index=False))

    # --- Which values do the classification fields take? ---------------------------------
    print(f"\n--- Values of the CIZ classification fields (stock-months since {CRSP_START[:4]}) ---")
    for field in CLASSIFICATION_FIELDS:
        counts = db.raw_sql(f"""
            SELECT {field} AS value, COUNT(*) AS n
            FROM crsp.msf_v2
            WHERE mthcaldt >= '{CRSP_START}'
            GROUP BY {field}
            ORDER BY n DESC
        """)
        summary = ", ".join(f"{v}: {n:,}" for v, n in zip(counts["value"], counts["n"]))
        print(f"{field:<17} {summary}")

    # --- How much does the FF-style filter keep? -----------------------------------------
    print("\n--- Effect of the common-stock filter (December of every 10th year) ---")
    kept = db.raw_sql(f"""
        SELECT EXTRACT(YEAR FROM mthcaldt)::int AS year,
               COUNT(*) AS all_rows,
               SUM(CASE WHEN {FF_FILTER} THEN 1 ELSE 0 END) AS kept_rows
        FROM crsp.msf_v2
        WHERE mthcaldt >= '{CRSP_START}' AND EXTRACT(MONTH FROM mthcaldt) = 12
        GROUP BY 1
        ORDER BY 1
    """)
    kept["share_kept"] = (kept["kept_rows"] / kept["all_rows"]).round(2)
    print(kept[kept["year"] % 10 == 0].to_string(index=False))

    # --- Is market cap really |price| x shares? ---------------------------------------------
    # Also tells us the units of shrout (thousands).
    print("\n--- ME check: |mthprc| * shrout vs mthcap (December 2024 sample) ---")
    me_check = db.raw_sql("""
        SELECT permno, mthprc, shrout, mthcap
        FROM crsp.msf_v2
        WHERE mthcaldt = '2024-12-31' AND mthcap IS NOT NULL
        LIMIT 5
    """)
    me_check["abs_prc_x_shrout"] = me_check["mthprc"].abs() * me_check["shrout"]
    print(me_check.to_string(index=False))

    # --- Do delisting months survive the filter? ---------------------------------------------
    # CIZ folds the delisting return into mthret of a stock's last month. If that row
    # failed our filter, we would silently lose delisting losses and overstate returns.
    print("\n--- Delisting months (mthdelflg <> 'N'): kept by the filter? ---")
    delist = db.raw_sql(f"""
        SELECT CASE WHEN {FF_FILTER} THEN 'kept' ELSE 'dropped' END AS by_filter,
               COUNT(*) AS delisting_months
        FROM crsp.msf_v2
        WHERE mthcaldt >= '{CRSP_START}' AND mthdelflg <> 'N'
          AND securitytype = 'EQTY' AND sharetype = 'NS' AND usincflg = 'Y'
        GROUP BY 1
    """)
    print(delist.to_string(index=False))

    db.close()


if __name__ == "__main__":
    main()
