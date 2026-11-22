# ABOUTME: Checks the CRSP v2 (CIZ) *daily* schema on WRDS before we build the daily pipeline on it.
"""
Run:     python DataDownloads/CheckCIZSchemaDaily.py   (from pyCode/)
Inputs:  WRDS information_schema, crsp.dsf_v2, crsp.msf_v2 (for cross-checks)
Outputs: printed report only

The daily pipeline replaces crsp.msf_v2 with crsp.dsf_v2. Before writing any download we
confirm (1) which columns the daily table has, (2) that the monthly common-stock filter
works on it, (3) that |price| x shares = CRSP's market cap, and (4) how delisting
returns are stored: compounding a month of daily returns should reproduce mthret, also
in delisting months. If (4) failed we would silently lose delisting losses.

Finding (2026-11-19): daily delisting returns sit on a separate row (dlydelflg='Y') that
the common-stock filter drops, so the daily download must add those rows back. With
them, compounded daily returns match mthret in 100% of complete stock-months.

Every query is limited to a few dates or years: the full daily table has ~100M rows.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

from utils.wrds_connection import get_wrds_connection  # noqa: E402

# Columns we expect, by analogy with msf_v2 (mthret -> dlyret, mthcaldt -> dlycaldt, ...).
EXPECTED_COLUMNS = [
    "permno", "permco", "dlycaldt", "dlyret", "dlyprc", "shrout", "dlycap", "dlyprevcap",
    "dlydelflg", "dlyprcflg", "sharetype", "securitytype", "securitysubtype", "usincflg",
    "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg",
]

CLASSIFICATION_FIELDS = [
    "sharetype", "securitytype", "securitysubtype", "usincflg",
    "issuertype", "primaryexch", "conditionaltype", "tradingstatusflg",
]

# Same filter as DataDownloads/A_CRSPMonthly.py (old shrcd 10/11, exchcd 1/2/3).
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
    print("Checking the CIZ daily schema on WRDS...", flush=True)
    db = get_wrds_connection()

    # --- Which columns does dsf_v2 have? ------------------------------------------------
    print("\n--- crsp.dsf_v2 columns ---")
    columns = db.raw_sql("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'crsp' AND table_name = 'dsf_v2'
        ORDER BY ordinal_position
    """)
    if columns.empty:
        raise SystemExit("crsp.dsf_v2 not found in information_schema: check the table name.")
    print(", ".join(f"{c} ({t})" for c, t in zip(columns["column_name"], columns["data_type"])))
    missing = [c for c in EXPECTED_COLUMNS if c not in set(columns["column_name"])]
    print(f"\nexpected columns missing: {missing if missing else 'none'}")
    if missing:
        db.close()
        raise SystemExit("Stop: adapt EXPECTED_COLUMNS / the queries below to the real names.")

    # --- Coverage -----------------------------------------------------------------------
    # Uses the date index; a COUNT(*) over the whole table would be slow.
    print("\n--- Coverage ---")
    coverage = db.raw_sql("SELECT MIN(dlycaldt) AS first_day, MAX(dlycaldt) AS last_day FROM crsp.dsf_v2")
    print(coverage.to_string(index=False))

    # --- Classification values (one recent year) -------------------------------------------
    print("\n--- Values of the classification fields (stock-days in 2024) ---")
    for field in CLASSIFICATION_FIELDS:
        counts = db.raw_sql(f"""
            SELECT {field} AS value, COUNT(*) AS n
            FROM crsp.dsf_v2
            WHERE dlycaldt BETWEEN '2024-01-01' AND '2024-12-31'
            GROUP BY {field} ORDER BY n DESC
        """)
        print(f"{field:<17} " + ", ".join(f"{v}: {n:,}" for v, n in zip(counts["value"], counts["n"])))

    # --- Rows per year after the filter (size of the download) ------------------------------
    print("\n--- Filtered stock-days and stocks, sample years ---")
    for year in [1963, 1980, 2000, 2024]:
        size = db.raw_sql(f"""
            SELECT COUNT(*) AS stock_days, COUNT(DISTINCT permno) AS stocks,
                   COUNT(DISTINCT dlycaldt) AS trading_days
            FROM crsp.dsf_v2
            WHERE dlycaldt BETWEEN '{year}-01-01' AND '{year}-12-31' AND {FF_FILTER}
        """)
        print(f"{year}: " + size.to_string(index=False, header=(year == 1963)))

    # --- ME check: |dlyprc| x shrout vs dlycap ---------------------------------------------
    print("\n--- ME check, 2024-12-31: |dlyprc| * shrout vs dlycap ---")
    me = db.raw_sql(f"""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN ABS(ABS(dlyprc) * shrout - dlycap) < 1 THEN 1 ELSE 0 END) AS n_equal,
               SUM(CASE WHEN dlyprc < 0 THEN 1 ELSE 0 END) AS n_negative_price
        FROM crsp.dsf_v2
        WHERE dlycaldt = '2024-12-31' AND dlycap IS NOT NULL AND {FF_FILTER}
    """)
    print(me.to_string(index=False))
    print("price flags (dlyprcflg), 2024:")
    print(db.raw_sql(f"""
        SELECT dlyprcflg, COUNT(*) AS n FROM crsp.dsf_v2
        WHERE dlycaldt BETWEEN '2024-01-01' AND '2024-12-31' AND {FF_FILTER}
        GROUP BY 1 ORDER BY 2 DESC
    """).to_string(index=False))

    # --- Delisting rows: how does CIZ daily store them? -------------------------------------
    # Unlike msf_v2 (delisting return folded into the last mthret), dsf_v2 puts it on an
    # extra row with dlydelflg='Y', dated the trading day *after* the last trade (it can
    # fall in the next month). On that row the classification fields are 'N/A' /
    # primaryexch='X' / tradingstatusflg='D', so the common-stock filter DROPS it. The
    # download must therefore add these rows back for stocks that pass the filter.
    print("\n--- Delisting rows (dlydelflg='Y'), 2015-2024: classification fields ---")
    print(db.raw_sql(f"""
        SELECT primaryexch, tradingstatusflg,
               CASE WHEN {FF_FILTER} THEN 'kept' ELSE 'dropped' END AS by_filter,
               COUNT(*) AS n, COUNT(dlyret) AS with_return
        FROM crsp.dsf_v2
        WHERE dlycaldt BETWEEN '2015-01-01' AND '2024-12-31' AND dlydelflg = 'Y'
        GROUP BY 1, 2, 3 ORDER BY 4 DESC
    """).to_string(index=False))
    print("delisting row return vs crsp.stkdelists.delret (2015-2024):")
    print(db.raw_sql("""
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN ABS(d.dlyret - s.delret) < 1e-6 THEN 1 ELSE 0 END) AS equal,
               SUM(CASE WHEN d.dlycaldt = s.deldlydt THEN 1 ELSE 0 END) AS dated_deldlydt
        FROM crsp.dsf_v2 d JOIN crsp.stkdelists s ON s.permno = d.permno
        WHERE d.dlycaldt BETWEEN '2015-01-01' AND '2024-12-31' AND d.dlydelflg = 'Y'
          AND d.dlyret IS NOT NULL
    """).to_string(index=False))

    # --- Do compounded daily returns reproduce mthret? ---------------------------------------
    # Each daily row counts in its own month, except the delisting row, which counts in the
    # month of the last trade (dlyprevdt), as in msf_v2. GREATEST(...) avoids ln(0) for a
    # -100% day; the CASE is needed because Postgres' GREATEST ignores NULLs (a missing
    # return would otherwise become ln(1e-12)). Missing daily returns are a new stock's
    # first day (dlyretmissflg='NS') or a gap whose return CRSP books on the next day as a
    # multi-day return (dlyretdurflg='P1'...), so they should still compound to mthret.
    compound_sql = """
        WITH d AS (
            SELECT permno,
                   DATE_TRUNC('month', CASE WHEN dlydelflg = 'Y' THEN COALESCE(dlyprevdt, dlycaldt)
                                            ELSE dlycaldt END) AS month,
                   EXP(SUM(CASE WHEN dlyret IS NOT NULL
                                THEN LN(GREATEST(1 + dlyret, 1e-12)) END)) - 1 AS ret_from_daily,
                   COUNT(*) - COUNT(dlyret) AS missing_days,
                   MAX(CASE WHEN dlydelflg = 'Y' THEN 1 ELSE 0 END) AS has_delisting_row
            FROM crsp.dsf_v2
            WHERE dlycaldt BETWEEN '{start}' AND '{end_plus}'
            GROUP BY 1, 2
        )
        SELECT {label} AS sample,
               CASE WHEN d.missing_days > 0 THEN 'some daily ret missing' ELSE 'complete' END AS days,
               COUNT(*) AS stock_months,
               SUM(CASE WHEN ABS(d.ret_from_daily - m.mthret) < 1e-4 THEN 1 ELSE 0 END) AS match_1bp,
               ROUND(MAX(ABS(d.ret_from_daily - m.mthret)), 4) AS max_abs_diff
        FROM crsp.msf_v2 m
        JOIN d ON d.permno = m.permno AND d.month = DATE_TRUNC('month', m.mthcaldt)
        WHERE m.mthcaldt BETWEEN '{start}' AND '{end}' AND m.mthret IS NOT NULL {extra}
        GROUP BY 1, 2 ORDER BY 2
    """
    print("\n--- Compounded dlyret (incl. delisting rows) vs mthret ---")
    print(db.raw_sql(compound_sql.format(start="2024-01-01", end="2024-12-31", end_plus="2025-01-15",
                                         label="'all 2024'", extra="")).to_string(index=False))
    print(db.raw_sql(compound_sql.format(start="2015-01-01", end="2024-12-31", end_plus="2025-01-15",
                                         label="'delisting months 2015-24'",
                                         extra="AND m.mthdelflg <> 'N'")).to_string(index=False))

    db.close()


if __name__ == "__main__":
    main()
