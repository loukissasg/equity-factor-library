# ABOUTME: Builds the profitability factor RMW (robust minus weak) from the size x OperProf 2x3 sort, daily and monthly.
"""
Run:     python Factors/RMW.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, OperProf.parquet   (via ff2x3_daily)
         pyData/Intermediate/DailyMasterTable/              (daily returns, via ff2x3_daily)
Outputs: pyData/Portfolios/FF2x3Daily_OperProf*.parquet     (written by ff2x3_daily)
         pyData/Portfolios/FF2x3Daily_OperProf_monthly.parquet   the six portfolios compounded
                                                             to months, plus smb and spread
         pyData/Factors/RMWDaily.parquet   daily rmw, smb_op (decimals)
         pyData/Factors/RMW.parquet        monthly rmw, smb_op, compounded from daily
         output/rmw_replicated.csv         the monthly file, for Excel

RMW = 1/2 (S_robust + B_robust) - 1/2 (S_weak + B_weak), Ken French's definition, where
"robust" = top 30% and "weak" = bottom 30% of OperProf (NYSE breakpoints).

Same construction as FF3 since the move to daily data: the June sort is unchanged, the
portfolios earn value-weighted returns every trading day (weights = yesterday's price x
shares at last month-end),
and the monthly factor is built from the six portfolios compounded to months.

smb_op is the SMB of this same sort (small minus big, holding profitability constant).
It is not used yet; Fama-French (2015) average it with the SMBs of other sorts.

Adding this factor does not touch FF3: it only reads the shared Size signal and writes
its own files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, OUTPUT, PORTFOLIOS, make_folders  # noqa: E402
from Portfolios.FF2x3Daily import ff2x3_daily, monthly_portfolios  # noqa: E402


def summary(factors, periods_per_year):
    """Mean and volatility in % per year, and annualized Sharpe ratio."""
    return pd.DataFrame({
        "mean_annual_%": factors.mean() * periods_per_year * 100,
        "std_annual_%": factors.std() * periods_per_year ** 0.5 * 100,
        "sharpe": factors.mean() / factors.std() * periods_per_year ** 0.5,
    }).round(2)


def main():
    make_folders()

    # sign +1: high operating profitability ("robust") is the long side.
    p = ff2x3_daily("OperProf", +1)

    # --- Daily factor ------------------------------------------------------------------------------
    daily = pd.DataFrame(index=p.index)
    daily["rmw"] = p["spread"]  # 1/2 (SH + BH) - 1/2 (SL + BL), H = robust, L = weak
    daily["smb_op"] = p["smb"]
    daily.index.name = "date"
    daily.to_parquet(FACTORS / "RMWDaily.parquet")

    # --- Monthly factor, from the six portfolios compounded to months ----------------------------
    pm = monthly_portfolios(p, +1)
    pm.to_parquet(PORTFOLIOS / "FF2x3Daily_OperProf_monthly.parquet")
    monthly = pd.DataFrame(index=pm.index)
    monthly["rmw"] = pm["spread"]
    monthly["smb_op"] = pm["smb"]
    monthly.to_parquet(FACTORS / "RMW.parquet")
    monthly.to_csv(OUTPUT / "rmw_replicated.csv", float_format="%.6f")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"\nRMW daily: {len(daily):,} days, {daily.index.min().date()} to {daily.index.max().date()}"
          f"   monthly: {len(monthly)} months, missing values: {monthly.isna().sum().sum()}")
    print("Daily (annualized with 252 days):")
    print(summary(daily, 252).to_string())
    print("Monthly, compounded from daily (annualized with 12 months):")
    print(summary(monthly, 12).to_string())
    by_decade = monthly[["rmw"]].groupby((monthly.index.year // 10) * 10).mean() * 12 * 100
    by_decade.index = [f"{d}s" for d in by_decade.index]
    print("Average annual RMW premium by decade (%, monthly):")
    print(by_decade.round(1).T.to_string())


if __name__ == "__main__":
    main()
