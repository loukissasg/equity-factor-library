# ABOUTME: Builds the firm-month master table (returns, market equity, last month's ME) used by every later script.
"""
Run:     python SignalMasterTable.py   (from pyCode/)
Inputs:  pyData/Intermediate/crsp_monthly_from_daily.parquet   (DailyMasterTable.py)
Outputs: pyData/Intermediate/SignalMasterTable.parquet
         one row per firm (PERMCO) and month: permno, permco, date, yyyymm, ret, me,
         me_lag, primaryexch, is_nyse (ME in $ thousands)

The monthly data comes from daily CRSP: month-end price and shares from the last trading
day, monthly returns compounded from daily returns. Validation/CheckDailyVsMonthlyCRSP.py
showed that month-end ME is identical to monthly CRSP (msf_v2) in every row, so the
annual signals (June size, December B/M) are unchanged by the switch to daily data.
"""

import pandas as pd

from config import INTERMEDIATE, make_folders
from utils.checks import require_unique

INPUT_FILE = INTERMEDIATE / "crsp_monthly_from_daily.parquet"
OUTPUT_FILE = INTERMEDIATE / "SignalMasterTable.parquet"


def main():
    make_folders()
    print("Building the master table...", flush=True)
    panel = pd.read_parquet(INPUT_FILE)

    # --- Dates --------------------------------------------------------------------------
    # CRSP dates are the last *trading* day of the month (e.g. 2024-11-29), French's the
    # last *calendar* day. Moving both to calendar month-end lets them merge cleanly.
    panel["date"] = panel["mthcaldt"] + pd.offsets.MonthEnd(0)
    require_unique(panel, ["permno", "date"], "CRSP security-months")

    # --- Market equity of each security --------------------------------------------------
    # ME = price x shares outstanding. In CIZ, prices are always positive; months without a
    # trade use the bid-ask average, flagged by mthprcflg='BA' (the old SIZ format used a
    # negative price instead). abs() is kept as a safeguard.
    # shrout is in thousands, so ME is in $ thousands (checked against mthcap).
    panel["me_security"] = panel["mthprc"].abs() * panel["shrout"]
    panel.loc[panel["me_security"] <= 0, "me_security"] = None

    # --- Aggregate ME to the firm (PERMCO) ------------------------------------------------
    # Some firms have several common share classes, each with its own PERMNO (e.g.
    # Alphabet class A and class C). Size is a property of the *firm*, so Fama-French add
    # up the ME of all classes. We keep one row per firm-month: the class with the
    # largest ME, which carries the firm's total ME. Its return stands in for the firm's.
    panel["me"] = panel.groupby(["permco", "date"])["me_security"].transform("sum", min_count=1)

    classes_per_firm = panel.groupby(["permco", "date"])["permno"].transform("count")
    n_multi = panel.loc[classes_per_firm > 1, ["permco", "date"]].drop_duplicates().shape[0]
    print(f"firm-months with more than one share class: {n_multi:,}")

    last_month = panel["date"].max()
    multi = panel[(panel["date"] == last_month) & (classes_per_firm > 1)]
    example = (multi.groupby("permco")
               .agg(classes=("permno", "count"), me_firm_bn=("me", "first"))
               .sort_values("me_firm_bn", ascending=False).head(3))
    example["me_firm_bn"] = (example["me_firm_bn"] / 1e6).round(1)  # $ thousands -> $ billions
    print(f"largest multi-class firms in {last_month.date()} (ME in $bn):")
    print(example.to_string())

    # Keep the largest class. Sorting puts it first in each firm-month (missing ME last, so
    # a firm whose only class has no price this month still keeps its row; a delisting
    # month can have a return but no price).
    rows_before = len(panel)
    panel = panel.sort_values(["permco", "date", "me_security"], ascending=[True, True, False],
                              na_position="last")
    panel = panel.drop_duplicates(["permco", "date"], keep="first")
    print(f"rows: {rows_before:,} security-months -> {len(panel):,} firm-months")

    # --- Last month's ME (the portfolio weight) ---------------------------------------------
    # A value-weighted return for month t uses ME at the end of month t-1: what an investor
    # holds at the start of the month. ME at the end of month t would be look-ahead,
    # because it already contains the month-t return. The lag is only accepted if the
    # previous row is exactly one month earlier (a data gap would give a stale weight).
    panel = panel.sort_values(["permco", "date"])
    previous = panel.groupby("permco")[["date", "me"]].shift(1)
    months_apart = ((panel["date"].dt.year - previous["date"].dt.year) * 12
                    + panel["date"].dt.month - previous["date"].dt.month)
    panel["me_lag"] = previous["me"].where(months_apart == 1)

    # --- NYSE flag ------------------------------------------------------------------------------
    # Breakpoints use NYSE stocks only (see Portfolios/FF2x3.py).
    panel["is_nyse"] = panel["primaryexch"] == "N"

    # --- yyyymm key ---------------------------------------------------------------------------------
    # Signals in pyData/Predictors/ are keyed by (permno, yyyymm), as in OpenAP. An integer
    # month key avoids any ambiguity about which day of the month a date refers to.
    panel["yyyymm"] = panel["date"].dt.year * 100 + panel["date"].dt.month

    panel = panel.rename(columns={"mthret": "ret"})
    panel = panel[["permno", "permco", "date", "yyyymm", "ret", "me", "me_lag",
                   "primaryexch", "is_nyse"]]
    panel = panel.reset_index(drop=True)
    require_unique(panel, ["permco", "date"], "SignalMasterTable")
    panel.to_parquet(OUTPUT_FILE, index=False)

    # --- Sanity checks ----------------------------------------------------------------------------
    print(f"rows: {len(panel):,}   firms: {panel['permco'].nunique():,}   "
          f"dates: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"missing ME: {panel['me'].isna().mean():.2%}   "
          f"missing ME lag: {panel['me_lag'].isna().mean():.2%} "
          f"(each firm's first month has no lag)")
    print(f"rows with a return but no ME lag (cannot be weighted): "
          f"{(panel['ret'].notna() & panel['me_lag'].isna()).mean():.2%}")

    december = panel[panel["date"].dt.month == 12]
    yearly = december.groupby(december["date"].dt.year).agg(
        firms=("permco", "count"),
        nyse_firms=("is_nyse", "sum"),
        total_me_bn=("me", "sum"),
    )
    yearly["total_me_bn"] = (yearly["total_me_bn"] / 1e6).round(0)  # $ thousands -> $ billions
    print("December snapshot, every 5th year (total ME in $bn):")
    print(yearly[yearly.index % 5 == 0].to_string())


if __name__ == "__main__":
    main()
