# ABOUTME: Builds SMBQ, the size factor controlling for quality (Asness et al. 2018), from the size x OperProf 5x5 sort, daily and monthly.
"""
Run:     python Factors/SMBQ.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, BMdec.parquet, OperProf.parquet  (via size_quality_5x5_daily)
         pyData/Intermediate/DailyMasterTable/                         (daily returns)
Outputs: pyData/Portfolios/SizeQuality5x5Daily_OperProf*.parquet   (written by size_quality_5x5_daily)
         pyData/Factors/SMBQDaily.parquet   daily smbq and the five within-quintile spreads
         pyData/Factors/SMBQ.parquet        the same, monthly, from the 25 portfolios
                                            compounded to months
         output/smbq_replicated.csv         the monthly file, for Excel

SMBQ = 1/5 * sum over quality quintiles q of (S1Qq - S5Qq)
     = average, across the five quality quintiles, of smallest minus largest size quintile.

Why: within a quality quintile, small and big stocks have similar quality, so the
small-minus-big spread is not a hidden bet against quality (Asness et al. 2018). Equal
weights across quintiles give each quality level the same say.
Note: unlike SMB (median split), SMBQ compares the extreme size quintiles, so it is a
more aggressive size bet and should be more volatile than SMB.

Daily and monthly, as for FF3 and RMW: the 25 portfolios earn daily value-weighted
returns (weights = yesterday's price x shares at last month-end); the monthly factor is
built from the 25 portfolios
compounded to months, not by compounding the daily factor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, OUTPUT, make_folders  # noqa: E402
from Portfolios.SizeQuality5x5Daily import size_quality_5x5_daily  # noqa: E402

SIGNAL = "OperProf"   # main quality signal (validated against French's RMW in Stage 2)
FACTOR = "smbq"
OUTPUT_NAME = "SMBQ"


def smbq_from_5x5(p, factor_name):
    """SMBQ and the five small-minus-big spreads, one per quality quintile (any frequency)."""
    factors = pd.DataFrame(index=p.index)
    for q in range(1, 6):
        factors[f"{factor_name}_q{q}"] = p[f"S1Q{q}"] - p[f"S5Q{q}"]
    spread_columns = [f"{factor_name}_q{q}" for q in range(1, 6)]
    # mean(axis=1) with skipna=False: if a cell is empty on some day (month), SMBQ is
    # missing then rather than silently becoming an average of four quintiles.
    factors[factor_name] = factors[spread_columns].mean(axis=1, skipna=False)
    return factors[[factor_name] + spread_columns]


def summary(factors, periods_per_year):
    """Mean and volatility in % per year, and annualized Sharpe ratio."""
    return pd.DataFrame({
        "mean_annual_%": factors.mean() * periods_per_year * 100,
        "std_annual_%": factors.std() * periods_per_year ** 0.5 * 100,
        "sharpe": factors.mean() / factors.std() * periods_per_year ** 0.5,
    }).round(2)


def main():
    make_folders()
    p_daily, p_monthly = size_quality_5x5_daily(SIGNAL)

    daily = smbq_from_5x5(p_daily, FACTOR)
    monthly = smbq_from_5x5(p_monthly, FACTOR)
    daily.to_parquet(FACTORS / f"{OUTPUT_NAME}Daily.parquet")
    monthly.to_parquet(FACTORS / f"{OUTPUT_NAME}.parquet")
    monthly.to_csv(OUTPUT / f"{OUTPUT_NAME.lower()}_replicated.csv", float_format="%.6f")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"\n{OUTPUT_NAME}: {len(daily):,} days, {len(monthly)} months, "
          f"{monthly.index.min().date()} to {monthly.index.max().date()}   "
          f"missing: {daily[FACTOR].isna().sum()} days, {monthly[FACTOR].isna().sum()} months")
    print("SMBQ and small-minus-big within each quality quintile (q1 = junk, q5 = quality)")
    print("daily (252 days):")
    print(summary(daily, 252).to_string())
    print("monthly (12 months):")
    print(summary(monthly, 12).to_string())


if __name__ == "__main__":
    main()
