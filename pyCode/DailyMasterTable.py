# ABOUTME: Builds the firm-day master table (returns, ME, ME at the start of each return) and a monthly CRSP file derived from daily data.
"""
Run:     python DailyMasterTable.py   (from pyCode/)
Inputs:  pyData/Intermediate/crsp_daily/crsp_daily_YYYY.parquet
Outputs: pyData/Intermediate/DailyMasterTable/DailyMasterTable_YYYY.parquet, one per year
             one row per firm (PERMCO) and trading day: permno, permco, date, ret, me,
             me_lag, weight, primaryexch, is_nyse (ME and weight in $ thousands)
               me_lag = ME at the start of the return (yesterday's price x shares)
               weight = yesterday's price x shares at the end of last month: the weight
                        the daily value-weighted portfolios use (see add_weights)
         pyData/Intermediate/crsp_monthly_from_daily.parquet
             one row per security and month, same columns as the old monthly download
             (permno, permco, mthcaldt, mthret, mthprc, shrout, primaryexch), so the
             annual signals (June ME, December ME) can be built from daily data alone

Two jobs, both done one year at a time because the full daily panel (~75M rows) is too
big to hold in memory comfortably:
  1. Firm-day panel: what the daily portfolios are built from.
  2. Month-end file: the last trading day of each month gives month-end price and
     shares; daily returns compounded within the month give the monthly return.
"""

import numpy as np
import pandas as pd

from config import CRSP_DAILY_DIR, CRSP_END_YEAR, CRSP_START, DAILY_MASTER_DIR, INTERMEDIATE, make_folders
from utils.checks import require_unique

LAST_YEAR = CRSP_END_YEAR

MONTHLY_FILE = INTERMEDIATE / "crsp_monthly_from_daily.parquet"


def add_weights(crsp, previous_tail):
    """
    Add 'weight_security': yesterday's price x shares outstanding at the end of last month.

    Why not yesterday's full ME (dlyprevcap)? Because ME also jumps when a firm issues
    shares. QXO Inc. (Jul 2024) went from 0.66m to 342m shares overnight; with
    yesterday's ME it instantly became 4-7% of the small-growth portfolio and over half
    of a micro-cap SMBQ cell, moving SMBQ by -6.6 points in one month. With this weight
    prices still update every day, while share issues and buybacks count from the next
    month-end, as in the monthly Fama-French method. On the first trading day of a month
    the weight equals last month's ME exactly. (Decision 2026-11-22, see TEAM_NOTES.)

    Computed from what CRSP gives per security: dlyprevcap = price x shares on dlyprevdt,
    so  weight = dlyprevcap x (shares at last month-end) / (shares on dlyprevdt).

    Splits: the two share counts must be in the same units. After Citigroup's 1-for-10
    reverse split on 9 May 2011, yesterday's price is ~$44 but April's share count is the
    pre-split 29.2bn, so without an adjustment Citigroup's weight was 10x too big (7% of
    the whole market). We compare split-adjusted shares, shrout x dlycumfacshr (CRSP's
    cumulative share factor, which changes with splits but not with share issues).
    previous_tail: each security's last trading-day row of the previous year (permno,
    dlycaldt, shrout, dlycumfacshr), so January can look back into December.
    Returns crsp with the new column, and the tail to pass to the next year.
    """
    trading = crsp.loc[crsp["dlydelflg"] != "Y", ["permno", "dlycaldt", "shrout", "dlycumfacshr"]].copy()
    # Split-adjusted shares. If CRSP has no factor (rare), take shares as they are.
    trading["shares_adj"] = trading["shrout"] * trading["dlycumfacshr"].fillna(1.0)
    history = pd.concat([previous_tail, trading], ignore_index=True)

    # Shares on the day the return starts from (dlyprevdt), to take dlyprevcap back to a price.
    shares_prev = (history[["permno", "dlycaldt", "shares_adj"]]
                   .rename(columns={"dlycaldt": "dlyprevdt", "shares_adj": "shares_prev"}))
    crsp = crsp.merge(shares_prev, on=["permno", "dlyprevdt"], how="left")

    # Shares on the last trading day of the previous month (month-end shares, as monthly
    # CRSP reports them). A delisting row counts in the month of its own date.
    history["month"] = history["dlycaldt"].dt.to_period("M")
    month_end_shares = (history.sort_values("dlycaldt").groupby(["permno", "month"])["shares_adj"]
                        .last().rename("shares_month_start").reset_index())
    month_end_shares["month"] = month_end_shares["month"] + 1   # end of m-1 -> used in m
    crsp["month"] = crsp["dlycaldt"].dt.to_period("M")
    crsp = crsp.merge(month_end_shares, on=["permno", "month"], how="left")

    # Both counts are split-adjusted, so the ratio is 1 unless shares were issued or
    # bought back since the month-end.
    ratio = crsp["shares_month_start"] / crsp["shares_prev"]
    crsp["weight_security"] = crsp["dlyprevcap"] * ratio
    # No month-start shares (the stock's first month in CRSP, e.g. an IPO) or no shares on
    # dlyprevdt: fall back to yesterday's ME. Such stocks are never in the June-sorted
    # portfolios in their first month; it only matters for the market return.
    no_ratio = ratio.isna() | ~np.isfinite(ratio) | (ratio <= 0)
    crsp.loc[no_ratio, "weight_security"] = crsp.loc[no_ratio, "dlyprevcap"]

    tail = trading.sort_values("dlycaldt").groupby("permno").tail(1)
    return crsp.drop(columns=["shares_prev", "shares_month_start", "month"]), tail


def firm_days(crsp):
    """One row per firm and day, with the firm's total ME, its ME at the start of the
    return, and its portfolio weight."""
    panel = crsp.rename(columns={"dlycaldt": "date", "dlyret": "ret"})

    # --- Market equity of each security ----------------------------------------------------
    # ME = price x shares outstanding (shrout in thousands -> ME in $ thousands). Checked
    # in CheckCIZSchemaDaily.py: identical to CRSP's dlycap. abs() is only a safeguard,
    # CIZ prices are never negative.
    panel["me_security"] = panel["dlyprc"].abs() * panel["shrout"]
    panel.loc[panel["me_security"] <= 0, "me_security"] = np.nan

    # --- ME at the start of the return interval, and the portfolio weight --------------------
    # A value-weighted return for day t must use ME *before* day t's return, otherwise it is
    # look-ahead. CRSP gives it directly: dlyprevcap is ME on dlyprevdt, the date the
    # return starts from. Usually that is the previous trading day, but it also does the
    # right thing in the two special cases:
    #   - a gap in trading: CRSP books the return over the gap on the next day (a
    #     multi-day return), and dlyprevcap is ME at the start of that gap;
    #   - the delisting row: dated the day after the last trade, dlyprevcap is ME on the
    #     last trading day.
    # The weight is the same price but with last month-end's shares (add_weights), so a
    # share issue does not change a stock's weight until the next month-end.
    panel["me_lag_security"] = panel["dlyprevcap"]
    panel.loc[panel["me_lag_security"] <= 0, "me_lag_security"] = np.nan
    panel.loc[~(panel["weight_security"] > 0), "weight_security"] = np.nan

    # --- Aggregate ME to the firm (PERMCO) -------------------------------------------------------
    # As in the monthly data: size is a property of the firm, so all share classes
    # (e.g. Alphabet A and C) are added up. min_count=1 keeps "missing" missing instead of 0.
    by_firm_day = panel.groupby(["permco", "date"])
    panel["me"] = by_firm_day["me_security"].transform("sum", min_count=1)
    panel["me_lag"] = by_firm_day["me_lag_security"].transform("sum", min_count=1)
    panel["weight"] = by_firm_day["weight_security"].transform("sum", min_count=1)

    # Keep one row per firm-day: the class with the largest ME, carrying the firm's total.
    # Its return stands in for the firm's. Missing ME sorts last, so a firm whose only row
    # today is its delisting row (no price) still keeps that row and its delisting return.
    panel = panel.sort_values(["permco", "date", "me_security"], ascending=[True, True, False],
                              na_position="last")
    panel = panel.drop_duplicates(["permco", "date"], keep="first")

    # NYSE flag, for the June breakpoints.
    panel["is_nyse"] = panel["primaryexch"] == "N"
    return panel[["permno", "permco", "date", "ret", "me", "me_lag", "weight", "primaryexch",
                  "is_nyse"]]


def month_pieces(crsp):
    """
    Two pieces of the security-month file for one year:
      returns: sum of log(1 + daily return) per security and month (summed again later,
               because a delisting row in early January belongs to the previous December)
      month_end: price, shares and exchange on the last trading day of each month
    """
    crsp = crsp.copy()
    # A delisting row is dated the day after the last trade, often in the next month.
    # Monthly CRSP counts the delisting return in the month of the last trade, so we do too.
    is_delisting = crsp["dlydelflg"] == "Y"
    month_date = crsp["dlycaldt"].where(~is_delisting, crsp["dlyprevdt"])
    crsp["yyyymm"] = month_date.dt.year * 100 + month_date.dt.month

    # Compounding: (1 + r1)(1 + r2)... - 1 = exp(sum log(1 + r)) - 1. A -100% day gives
    # log(0) = -inf and correctly turns the month into -100%.
    with np.errstate(divide="ignore"):
        crsp["log_gross"] = np.log1p(crsp["dlyret"])
    crsp["is_delisting"] = is_delisting
    returns = crsp.groupby(["permno", "yyyymm"]).agg(
        log_gross=("log_gross", "sum"), n_returns=("dlyret", "count"),
        delisted=("is_delisting", "max")).reset_index()

    # Month-end date, shares and exchange: the last trading day in the month.
    trading = crsp[~is_delisting].sort_values(["permno", "dlycaldt"])
    month_end = trading.groupby(["permno", "yyyymm"]).tail(1)
    month_end = month_end[["permno", "permco", "yyyymm", "dlycaldt", "shrout", "primaryexch"]]

    # Was the stock still in our universe at the end of the month? Our download only has
    # days that pass the common-stock filter, so a stock whose last row is before the
    # market's last trading day of the month left the universe mid-month (or delisted).
    market_last_day = trading.groupby("yyyymm")["dlycaldt"].max().rename("market_last_day")
    month_end = month_end.merge(market_last_day, on="yyyymm", how="left")
    month_end["at_month_end"] = month_end["dlycaldt"] == month_end["market_last_day"]

    # Month-end price: the last day in the month *with a price*. Some days, mostly NASDAQ
    # in the 1970s, have neither a trade nor a bid-ask quote; monthly CRSP then reports
    # the last valid price of the month (with month-end shares), and so do we (checked in
    # Validation/CheckDailyVsMonthlyCRSP.py).
    priced = trading[trading["dlyprc"].notna()].groupby(["permno", "yyyymm"]).tail(1)
    month_end = month_end.merge(priced[["permno", "yyyymm", "dlyprc"]],
                                on=["permno", "yyyymm"], how="left")
    month_end = month_end.rename(columns={"dlycaldt": "mthcaldt", "dlyprc": "mthprc"})
    month_end = month_end[["permno", "permco", "yyyymm", "mthcaldt", "mthprc", "shrout",
                           "primaryexch", "at_month_end"]]
    return returns, month_end


def check_prevcap(crsp):
    """Share of rows where dlyprevcap equals |price| x shares on dlyprevdt (same year only)."""
    me = crsp[["permno", "dlycaldt", "dlyprc", "shrout"]].copy()
    me["me_on_prevdt"] = me["dlyprc"].abs() * me["shrout"]
    merged = crsp[["permno", "dlyprevdt", "dlyprevcap"]].merge(
        me[["permno", "dlycaldt", "me_on_prevdt"]],
        left_on=["permno", "dlyprevdt"], right_on=["permno", "dlycaldt"], how="inner")
    merged = merged[merged["dlyprevcap"].notna() & merged["me_on_prevdt"].notna()]
    equal = (merged["dlyprevcap"] - merged["me_on_prevdt"]).abs() < 1
    return equal.mean(), len(merged)


def main():
    make_folders()
    print("Building the daily master table, one year at a time...", flush=True)
    years = range(int(CRSP_START[:4]), LAST_YEAR + 1)

    yearly = []
    all_returns, all_month_ends = [], []
    # Each security's last trading day of the previous year, for January's look-back.
    previous_tail = pd.DataFrame({"permno": pd.Series(dtype="int32"),
                                  "dlycaldt": pd.Series(dtype="datetime64[ns]"),
                                  "shrout": pd.Series(dtype="float64"),
                                  "dlycumfacshr": pd.Series(dtype="float64"),
                                  "shares_adj": pd.Series(dtype="float64")})
    for year in years:
        crsp = pd.read_parquet(CRSP_DAILY_DIR / f"crsp_daily_{year}.parquet")
        crsp, previous_tail = add_weights(crsp, previous_tail)

        panel = firm_days(crsp)
        require_unique(panel, ["permco", "date"], f"DailyMasterTable {year}")
        panel.to_parquet(DAILY_MASTER_DIR / f"DailyMasterTable_{year}.parquet", index=False)

        returns, month_end = month_pieces(crsp)
        all_returns.append(returns)
        all_month_ends.append(month_end)

        usable = panel["ret"].notna() & panel["weight"].notna()
        # How often does the weight differ from yesterday's ME by more than 1%? Only on
        # days after a share-count change within the month, so this should be small.
        ratio = panel["weight"] / panel["me_lag"]
        differs = (ratio - 1).abs() > 0.01
        # A factor of 2 or more within a month should only come from a big share issue or
        # buyback (QXO), never from a split (the weights are split-adjusted).
        far_off = (ratio > 2) | (ratio < 0.5)
        row = {
            "year": year,
            "security_days": len(crsp),
            "firm_days": len(panel),
            "firms": panel["permco"].nunique(),
            "usable_%": round(100 * usable.mean(), 2),   # has a return and a weight
            "weight_vs_me_lag_>1%_%": round(100 * differs.mean(), 2),
            "factor_2_off_days": int(far_off.sum()),
        }
        if year % 10 == 0 or year == LAST_YEAR:
            share, n = check_prevcap(crsp)
            row["prevcap_ok_%"] = round(100 * share, 3)
        yearly.append(row)
        print(f"  {year}: {len(crsp):>9,} security-days -> {len(panel):>9,} firm-days", flush=True)

    # --- Security-month file from daily data ---------------------------------------------------
    returns = pd.concat(all_returns).groupby(["permno", "yyyymm"], as_index=False).sum()
    month_end = pd.concat(all_month_ends)
    monthly = month_end.merge(returns, on=["permno", "yyyymm"], how="left")
    # A month with no daily return at all (e.g. a stock's first and only day) has no
    # monthly return either.
    monthly["mthret"] = np.expm1(monthly["log_gross"]).where(monthly["n_returns"] > 0)

    # Month-end ME only for stocks still in our universe at the end of the month. The
    # signals use "ME at the end of June / December"; a stock that moved off NYSE/AMEX/
    # NASDAQ mid-month has no such ME (monthly CRSP has no row for that month at all). A
    # stock that *delisted* during the month keeps its last price, as in monthly CRSP. The
    # partial month's return is kept either way.
    left_universe = ~monthly["at_month_end"] & ~monthly["delisted"].fillna(False).astype(bool)
    monthly.loc[left_universe, "mthprc"] = np.nan
    print(f"\nsecurity-months where the stock left the universe mid-month (no month-end ME): "
          f"{left_universe.sum():,}")

    monthly = monthly[["permno", "permco", "mthcaldt", "mthret", "mthprc", "shrout", "primaryexch"]]
    monthly = monthly.sort_values(["permno", "mthcaldt"]).reset_index(drop=True)
    require_unique(monthly, ["permno", "mthcaldt"], "crsp_monthly_from_daily")
    monthly.to_parquet(MONTHLY_FILE, index=False)

    # --- Sanity checks ----------------------------------------------------------------------------
    yearly = pd.DataFrame(yearly)
    print("\nEvery 5th year: security-days, firm-days (after PERMCO aggregation), firms, "
          "share of firm-days with return and weight, share where the weight differs from "
          "yesterday's ME by > 1%; dlyprevcap check every 10th year")
    show = yearly[(yearly["year"] % 5 == 0) | (yearly["year"] == LAST_YEAR)]
    print(show.to_string(index=False, na_rep=""))
    print(f"total firm-days: {yearly['firm_days'].sum():,}")

    print(f"\nMonthly file from daily data: {len(monthly):,} security-months, "
          f"{monthly['permno'].nunique():,} stocks, "
          f"{monthly['mthcaldt'].min().date()} to {monthly['mthcaldt'].max().date()}")
    print(f"missing monthly returns: {monthly['mthret'].isna().mean():.2%}")


if __name__ == "__main__":
    main()
