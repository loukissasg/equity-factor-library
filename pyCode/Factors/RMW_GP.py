# ABOUTME: Robustness version of the profitability factor: RMW_GP from the size x GP (gross profitability) 2x3 sort, daily and monthly.
"""
Run:     python Factors/RMW_GP.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, GP.parquet   (via ff2x3_daily)
         pyData/Intermediate/DailyMasterTable/        (daily returns, via ff2x3_daily)
Outputs: pyData/Portfolios/FF2x3Daily_GP*.parquet     (written by ff2x3_daily)
         pyData/Factors/RMW_GPDaily.parquet   daily rmw_gp, smb_gp (decimals)
         pyData/Factors/RMW_GP.parquet        monthly rmw_gp, smb_gp, compounded from daily

RMW_GP = 1/2 (S_high GP + B_high GP) - 1/2 (S_low GP + B_low GP).

Same recipe as RMW, only the profitability measure changes (gross profits / assets
instead of operating profits / book equity). If both factors behave alike, the
profitability premium does not hinge on one accounting definition.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, make_folders  # noqa: E402
from Factors.RMW import summary  # noqa: E402
from Portfolios.FF2x3Daily import ff2x3_daily, monthly_portfolios  # noqa: E402


def main():
    make_folders()

    # sign +1: high gross profitability is the long side (Novy-Marx 2013).
    p = ff2x3_daily("GP", +1)

    daily = pd.DataFrame(index=p.index)
    daily["rmw_gp"] = p["spread"]
    daily["smb_gp"] = p["smb"]
    daily.index.name = "date"
    daily.to_parquet(FACTORS / "RMW_GPDaily.parquet")

    # Monthly: compound the six portfolios first, then form the factor (see FF2x3Daily.py).
    pm = monthly_portfolios(p, +1)
    monthly = pd.DataFrame(index=pm.index)
    monthly["rmw_gp"] = pm["spread"]
    monthly["smb_gp"] = pm["smb"]
    monthly.to_parquet(FACTORS / "RMW_GP.parquet")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"\nRMW_GP daily: {len(daily):,} days   monthly: {len(monthly)} months, "
          f"{monthly.index.min().date()} to {monthly.index.max().date()}, "
          f"missing values: {monthly.isna().sum().sum()}")
    print("Daily (annualized with 252 days):")
    print(summary(daily, 252).to_string())
    print("Monthly, compounded from daily (annualized with 12 months):")
    print(summary(monthly, 12).to_string())


if __name__ == "__main__":
    main()
