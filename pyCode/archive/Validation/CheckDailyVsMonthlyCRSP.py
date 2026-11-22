# ABOUTME: Cross-checks the monthly file built from daily CRSP against the monthly CRSP download (msf_v2).
"""
Run:     python Validation/CheckDailyVsMonthlyCRSP.py   (from pyCode/)
Inputs:  pyData/Intermediate/crsp_monthly_from_daily.parquet   (DailyMasterTable.py)
         pyData/Intermediate/crsp_monthly.parquet              (old monthly download)
Outputs: printed report only

The daily pipeline gets month-end ME (for the June size and December B/M signals) and
monthly returns from daily data instead of a separate monthly download. If our daily
download and compounding are right, both sources should agree row by row: same stocks,
same months, same returns (delistings included), same prices and shares.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE  # noqa: E402

TOLERANCE = 1e-4  # 1 basis point for returns; relative 1e-4 for prices and shares


def main():
    print("Comparing monthly CRSP from daily data with the monthly download...", flush=True)
    daily = pd.read_parquet(INTERMEDIATE / "crsp_monthly_from_daily.parquet")
    monthly = pd.read_parquet(INTERMEDIATE / "crsp_monthly.parquet")

    # Join on the calendar month, not the date: a stock that stops trading mid-month has
    # its last trading day as the daily month-end, while msf_v2 uses the month's last
    # trading day.
    for df in (daily, monthly):
        df["yyyymm"] = df["mthcaldt"].dt.year * 100 + df["mthcaldt"].dt.month
    both = daily.merge(monthly, on=["permno", "yyyymm"], how="outer",
                       suffixes=("_d", "_m"), indicator=True)

    print("\n--- Coverage (security-months) ---")
    print(both["_merge"].map({"both": "in both", "left_only": "only from daily",
                              "right_only": "only in monthly"}).value_counts().to_string())
    only_daily = both[both["_merge"] == "left_only"]
    only_monthly = both[both["_merge"] == "right_only"]
    if len(only_daily):
        print(f"only from daily, by year (first 10): "
              f"{(only_daily['yyyymm'] // 100).value_counts().sort_index().head(10).to_dict()}")
    if len(only_monthly):
        print(f"only in monthly, by year (first 10): "
              f"{(only_monthly['yyyymm'] // 100).value_counts().sort_index().head(10).to_dict()}")

    # These are months in which a stock left our universe; DailyMasterTable.py gives them no
    # month-end price, so they cannot enter a June or December sort.
    print(f"only-from-daily months with a month-end price (should be 0): "
          f"{only_daily['mthprc_d'].notna().sum():,}")

    matched = both[both["_merge"] == "both"].copy()

    # --- Returns ----------------------------------------------------------------------------
    has_both = matched["mthret_d"].notna() & matched["mthret_m"].notna()
    diff = (matched["mthret_d"] - matched["mthret_m"]).abs()
    matched["ret_ok"] = diff < TOLERANCE
    print("\n--- Monthly returns: compounded daily vs mthret ---")
    print(f"both returns present: {has_both.sum():,}   within 1bp: "
          f"{matched.loc[has_both, 'ret_ok'].mean():.4%}")
    print(f"return only from daily: {(matched['mthret_d'].notna() & matched['mthret_m'].isna()).sum():,}   "
          f"only in monthly: {(matched['mthret_d'].isna() & matched['mthret_m'].notna()).sum():,}")
    matched["decade"] = matched["yyyymm"] // 1000 * 10
    by_decade = matched[has_both].groupby("decade")["ret_ok"].agg(["count", "mean"])
    by_decade["mean"] = (100 * by_decade["mean"]).round(3)
    print("within 1bp, by decade (%):")
    print(by_decade.rename(columns={"count": "months", "mean": "within_1bp_%"}).T.to_string())

    worst = matched[has_both].assign(diff=diff).sort_values("diff", ascending=False).head(8)
    print("largest return differences:")
    print(worst[["permno", "yyyymm", "mthret_d", "mthret_m", "diff"]].to_string(index=False))

    # --- Month-end price, shares and ME (what the size and B/M signals use) ------------------
    print("\n--- Month-end price, shares and ME ---")
    me_d = matched["mthprc_d"].abs() * matched["shrout_d"]
    me_m = matched["mthprc_m"].abs() * matched["shrout_m"]
    for name, a, b in [("price", matched["mthprc_d"], matched["mthprc_m"]),
                       ("shares", matched["shrout_d"], matched["shrout_m"]),
                       ("ME", me_d, me_m)]:
        present = a.notna() & b.notna()
        ok = ((a - b).abs() <= TOLERANCE * b.abs())[present]
        print(f"{name:<7} both present: {present.sum():,}   equal: {ok.mean():.4%}")
    exch_ok = (matched["primaryexch_d"] == matched["primaryexch_m"]).mean()
    print(f"exchange equal: {exch_ok:.4%}")

    # Total ME in June and December, the two dates the signals use.
    for month in (6, 12):
        sel = matched["yyyymm"] % 100 == month
        totals = pd.DataFrame({"me_daily": me_d[sel], "me_monthly": me_m[sel],
                               "year": matched.loc[sel, "yyyymm"] // 100}).groupby("year").sum()
        gap = (totals["me_daily"] / totals["me_monthly"] - 1).abs().max()
        print(f"total ME in month {month:>2}: largest gap between the two sources in any year: {gap:.6%}")


if __name__ == "__main__":
    main()
