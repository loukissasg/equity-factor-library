# ABOUTME: Daily version of the Fama-French 2x3 sort: same June portfolios as FF2x3.py, value-weighted returns every trading day.
"""
Use:     from Portfolios.FF2x3Daily import ff2x3_daily
         p = ff2x3_daily("BMdec", +1)          (called by Factors/FF3Daily.py)
Run:     python Portfolios/FF2x3Daily.py BMdec [+1|-1]   (from pyCode/, to inspect one sort)
Inputs:  pyData/Predictors/Size.parquet, <signal>.parquet   (via the June sort in FF2x3.py)
         pyData/Intermediate/SignalMasterTable.parquet      (PERMCO and NYSE flag in June)
         pyData/Intermediate/DailyMasterTable/DailyMasterTable_YYYY.parquet   (daily returns)
Outputs: pyData/Portfolios/FF2x3Daily_<signal>.parquet   daily: SL SN SH BL BN BH smb spread
         pyData/Portfolios/FF2x3Daily_<signal>_assignments.parquet   one row per firm and June
         pyData/Portfolios/FF2x3Daily_<signal>_holdings.parquet      per day and portfolio:
             number of stocks and total weight ($ thousands)

What changes from the monthly version, and what doesn't:
  - The sort is *identical*. The portfolios are formed once a year in June, from annual
    signals, so we reuse eligible_stocks(), nyse_breakpoints() and assign_portfolios()
    from FF2x3.py unchanged. Same NYSE breakpoints, same six portfolios, same stocks.
  - Only the return step is new: instead of one value-weighted return per month, we
    compute one per trading day, held July t to June t+1 as before.
  - Weights: yesterday's price x shares outstanding at the end of last month (the
    'weight' column of the DailyMasterTable). Prices update daily; share issues count
    from the next month-end, so a firm cannot jump to a huge weight mid-month (QXO,
    July 2024). On the first day of each month the weight equals last month's ME, the
    monthly Fama-French weight.
  - The daily panel (~73M firm-days) is too big for memory, so returns are computed one
    year at a time from the yearly DailyMasterTable files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (CRSP_END_YEAR, DAILY_MASTER_DIR, FIRST_SORT_YEAR,  # noqa: E402
                    MONTHLY_METHOD_REF, MONTHLY_PIPELINE_REF, PORTFOLIO_NAMES, PORTFOLIOS,
                    PREDICTORS, make_folders)
from Portfolios.FF2x3 import assign_portfolios, eligible_stocks, nyse_breakpoints  # noqa: E402


def daily_returns_one_year(year, assignments):
    """Value-weighted daily returns of the six portfolios for the trading days of one calendar year."""
    days = pd.read_parquet(DAILY_MASTER_DIR / f"DailyMasterTable_{year}.parquet",
                           columns=["permco", "date", "ret", "weight", "primaryexch"])

    # Each day belongs to the most recent June sort: July-December to this year's sort,
    # January-June to last year's. Same rule as the monthly version.
    days["sort_year"] = year
    days.loc[days["date"].dt.month <= 6, "sort_year"] = year - 1
    days = days.merge(assignments[["permco", "sort_year", "portfolio"]],
                      on=["permco", "sort_year"], how="inner")

    # A stock counts on a day only if it has a return and a weight. Two special days:
    #   - a day with no trade and no quote has no return; CRSP books the move over the gap
    #     on the next trading day, so the stock skips the gap day and nothing is lost;
    #   - the delisting row (primaryexch 'X', dated the day after the last trade) carries
    #     the delisting return, weighted with the price of the last trading day. Keeping it
    #     is what stops delisting losses from silently disappearing.
    days = days[days["ret"].notna() & days["weight"].notna() & (days["weight"] > 0)]

    # Weight = price at the start of the return (yesterday) x shares at last month-end.
    # Today's price already contains today's return, so using it would be look-ahead.
    days["ret_x_weight"] = days["ret"] * days["weight"]
    grouped = days.groupby(["date", "portfolio"])
    returns = (grouped["ret_x_weight"].sum() / grouped["weight"].sum()).unstack()

    holdings = grouped.agg(n_stocks=("permco", "count"), total_weight=("weight", "sum")).reset_index()
    n_delisting_rows = int((days["primaryexch"] == "X").sum())
    return returns, holdings, n_delisting_rows


def compound_to_months(daily):
    """
    Monthly returns from daily returns: (1 + r1)(1 + r2)... - 1 within each month.
    A month with a missing daily return (e.g. an empty 5x10 cell) is missing as a whole:
    pandas' sum would otherwise skip the gap, i.e. silently count that day as 0%.
    """
    months = daily.index.to_period("M")
    compounded = np.expm1(np.log1p(daily).groupby(months).sum())
    days_with_return = daily.notna().groupby(months).sum()
    trading_days = daily.groupby(months).size()
    return compounded.where(days_with_return.eq(trading_days, axis=0))


def monthly_portfolios(p_daily, sign):
    """
    Monthly version of ff2x3_daily's output: each of the six portfolios compounded to a
    monthly return, then smb and spread formed from the *monthly* portfolios. (A
    long-short factor is a difference of portfolios, and compounding a difference is not
    the same as the difference of the compounded portfolios.) Month-end dates.
    """
    p = compound_to_months(p_daily[PORTFOLIO_NAMES])
    p["smb"] = (p["SL"] + p["SN"] + p["SH"]) / 3 - (p["BL"] + p["BN"] + p["BH"]) / 3
    p["spread"] = sign * ((p["SH"] + p["BH"]) / 2 - (p["SL"] + p["BL"]) / 2)
    p.index = p.index.to_timestamp(how="end").normalize()
    p.index.name = "date"
    return p


def frozen_monthly_file(file_name):
    """The monthly-method version of a portfolio file, if one was frozen (else None)."""
    for folder in [MONTHLY_PIPELINE_REF, MONTHLY_METHOD_REF]:
        if (folder / file_name).exists():
            return folder / file_name
    return None


def ff2x3_daily(signal_name, sign):
    """
    Same 2x3 sort as ff2x3(signal_name, sign), but returns *daily* portfolio returns:
        smb    = 1/3 (SL + SN + SH) - 1/3 (BL + BN + BH)
        spread = sign * [1/2 (SH + BH) - 1/2 (SL + BL)]
    computed day by day from the daily portfolio returns.
    """
    if sign not in (1, -1):
        raise ValueError(f"ff2x3_daily: sign must be +1 or -1, got {sign}")

    make_folders()
    print(f"Forming the 2x3 portfolios on size x {signal_name} (sign {sign:+d}), daily returns...",
          flush=True)

    # --- The June sort: unchanged from the monthly version --------------------------------------
    eligible = eligible_stocks(signal_name)
    breakpoints = nyse_breakpoints(eligible)
    assignments = assign_portfolios(eligible, breakpoints)
    counts = assignments.groupby(["sort_year", "portfolio"]).size().unstack()[PORTFOLIO_NAMES]
    print(f"sort years: {assignments['sort_year'].min()} to {assignments['sort_year'].max()}   "
          f"firm-years in the sort: {len(assignments):,}")
    print("Stocks per portfolio, every 5th June (full table in FF2x3.py output):")
    print(counts[(counts.index % 5 == 0) | (counts.index == counts.index.max())].to_string())

    # --- Daily returns, one calendar year at a time ------------------------------------------------
    # The first holding period starts in July of the first sort year.
    all_returns, all_holdings = [], []
    delisting_rows = 0
    for year in range(FIRST_SORT_YEAR, CRSP_END_YEAR + 1):
        returns, holdings, n_delisting = daily_returns_one_year(year, assignments)
        all_returns.append(returns)
        all_holdings.append(holdings)
        delisting_rows += n_delisting
        if year % 10 == 0 or year == CRSP_END_YEAR:
            print(f"  {year}: {len(returns)} trading days", flush=True)

    p = pd.concat(all_returns)[PORTFOLIO_NAMES].sort_index()
    holdings = pd.concat(all_holdings, ignore_index=True)

    # SMB and the spread, exactly as in the monthly version, but formed every day.
    p["smb"] = (p["SL"] + p["SN"] + p["SH"]) / 3 - (p["BL"] + p["BN"] + p["BH"]) / 3
    p["spread"] = sign * ((p["SH"] + p["BH"]) / 2 - (p["SL"] + p["BL"]) / 2)

    p.index.name = "date"
    p.to_parquet(PORTFOLIOS / f"FF2x3Daily_{signal_name}.parquet")
    assignments.to_parquet(PORTFOLIOS / f"FF2x3Daily_{signal_name}_assignments.parquet", index=False)
    holdings.to_parquet(PORTFOLIOS / f"FF2x3Daily_{signal_name}_holdings.parquet", index=False)

    # --- Sanity checks ------------------------------------------------------------------------------
    print(f"\ntrading days: {len(p):,}   dates: {p.index.min().date()} to {p.index.max().date()}   "
          f"missing portfolio returns: {p[PORTFOLIO_NAMES].isna().sum().sum()}")
    days_per_year = p.groupby(p.index.year).size()
    print(f"trading days per year: median {days_per_year.median():.0f}, "
          f"min {days_per_year.min()} ({days_per_year.idxmin()}, first half-year or 1968), "
          f"max {days_per_year.max()}")
    print(f"delisting returns included in the portfolios: {delisting_rows:,}")

    # Smallest number of stocks a portfolio held on any day (a near-empty portfolio would
    # give a noisy return), and the average per day.
    n_stocks = holdings.pivot(index="date", columns="portfolio", values="n_stocks")[PORTFOLIO_NAMES]
    print("Stocks per portfolio per day: minimum and average over all days")
    print(pd.DataFrame({"min": n_stocks.min(), "mean": n_stocks.mean().round(0)}).T.to_string())

    # Signals with values but no June size (e.g. the 55 BMdec firm-years of firms that left
    # the universe during their June) must not enter the sort: no size, no portfolio.
    size = pd.read_parquet(PREDICTORS / "Size.parquet")
    signal = pd.read_parquet(PREDICTORS / f"{signal_name}.parquet")
    signal_june = signal[signal["yyyymm"] % 100 == 6]
    no_size = signal_june.merge(size, on=["permno", "yyyymm"], how="left", indicator=True)
    no_size = no_size[no_size["_merge"] == "left_only"].copy()
    no_size["sort_year"] = no_size["yyyymm"] // 100
    in_sort = no_size[["permno", "sort_year"]].merge(assignments, on=["permno", "sort_year"],
                                                     how="inner")
    print(f"{signal_name} June values without Size: {len(no_size):,}   of these in the sort: "
          f"{len(in_sort)}")
    if len(in_sort) > 0:
        raise ValueError("ff2x3_daily: firm-years without June size entered the sort")

    stats = pd.DataFrame({
        "mean_daily_bp": (p.mean() * 10000).round(2),
        "std_annual_%": (p.std() * (252 ** 0.5) * 100).round(1),
    })
    print("Mean daily return (basis points) and annualized std (%, 252 days), full sample:")
    print(stats.T.to_string())

    # Same stocks in the same portfolios as the old monthly sort (frozen copy, if one
    # exists for this signal)?
    monthly_assignments_file = frozen_monthly_file(f"FF2x3_{signal_name}_assignments.parquet")
    if monthly_assignments_file is not None:
        old = pd.read_parquet(monthly_assignments_file, columns=["permco", "sort_year", "portfolio"])
        both = assignments.merge(old, on=["permco", "sort_year"], how="outer",
                                 suffixes=("", "_monthly"), indicator=True)
        same = (both["_merge"] == "both") & (both["portfolio"] == both["portfolio_monthly"])
        print(f"assignments identical to the monthly FF2x3_{signal_name} sort: "
              f"{same.sum():,} of {len(both):,}")

    # Quick check against the monthly pipeline (full comparison in Stage 5): compound each
    # daily portfolio to months and compare with the monthly portfolio returns.
    monthly_file = frozen_monthly_file(f"FF2x3_{signal_name}.parquet")
    if monthly_file is not None:
        monthly = pd.read_parquet(monthly_file)[PORTFOLIO_NAMES]
        monthly.index = monthly.index.to_period("M")
        from_daily = compound_to_months(p[PORTFOLIO_NAMES])
        both = from_daily.join(monthly, how="inner", rsuffix="_monthly")
        rows = {}
        for name in PORTFOLIO_NAMES:
            gap = both[name] - both[f"{name}_monthly"]
            rows[name] = {"corr": round(both[name].corr(both[f"{name}_monthly"]), 5),
                          "mean_abs_gap_bp": round(gap.abs().mean() * 10000, 2),
                          "months_gap_>1bp_%": round(100 * (gap.abs() > 0.0001).mean(), 1)}
        print(f"Daily compounded to months vs monthly FF2x3_{signal_name} ({len(both)} months):")
        print(pd.DataFrame(rows).to_string())
    return p


if __name__ == "__main__":
    # e.g. python Portfolios/FF2x3Daily.py BMdec +1
    if len(sys.argv) < 2:
        raise SystemExit("usage: python Portfolios/FF2x3Daily.py <signal_name> [+1|-1]")
    ff2x3_daily(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
