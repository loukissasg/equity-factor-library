# ABOUTME: Robustness version of SMBQ: size factor controlling for quality measured by gross profitability (GP), daily and monthly.
"""
Run:     python Factors/SMBQ_GP.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, BMdec.parquet, GP.parquet  (via size_quality_5x5_daily)
Outputs: pyData/Portfolios/SizeQuality5x5Daily_GP*.parquet   (written by size_quality_5x5_daily)
         pyData/Factors/SMBQ_GPDaily.parquet   daily smbq_gp and the five within-quintile spreads
         pyData/Factors/SMBQ_GP.parquet        the same, monthly (25 portfolios compounded)

Same recipe as Factors/SMBQ.py, with GP (Novy-Marx, financials excluded) as the quality
signal. Because GP drops financial firms, this sort has fewer stocks than SMBQ's.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

from config import FACTORS, make_folders  # noqa: E402
from Factors.SMBQ import smbq_from_5x5, summary  # noqa: E402  (same formula, defined once)
from Portfolios.SizeQuality5x5Daily import size_quality_5x5_daily  # noqa: E402


def main():
    make_folders()
    p_daily, p_monthly = size_quality_5x5_daily("GP")
    daily = smbq_from_5x5(p_daily, "smbq_gp")
    monthly = smbq_from_5x5(p_monthly, "smbq_gp")
    daily.to_parquet(FACTORS / "SMBQ_GPDaily.parquet")
    monthly.to_parquet(FACTORS / "SMBQ_GP.parquet")

    print(f"\nSMBQ_GP: {len(daily):,} days, {len(monthly)} months   missing: "
          f"{daily['smbq_gp'].isna().sum()} days, {monthly['smbq_gp'].isna().sum()} months")
    print("daily (252 days):")
    print(summary(daily, 252).to_string())
    print("monthly (12 months):")
    print(summary(monthly, 12).to_string())


if __name__ == "__main__":
    main()
