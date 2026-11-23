# ABOUTME: Robustness version of SMBQ: size factor controlling for quality measured by gross profitability (GP).
"""
Run:     python Factors/SMBQ_GP.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, BMdec.parquet, GP.parquet  (via size_quality_5x5)
Outputs: pyData/Portfolios/SizeQuality5x5_GP*.parquet   (written by size_quality_5x5)
         pyData/Factors/SMBQ_GP.parquet   monthly smbq_gp and the five within-quintile spreads

Same recipe as Factors/SMBQ.py, with GP (Novy-Marx, financials excluded) as the quality
signal. Because GP drops financial firms, this sort has fewer stocks than SMBQ's.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, make_folders  # noqa: E402
from Factors.SMBQ import smbq_from_5x5  # noqa: E402  (same formula, defined once)
from Portfolios.SizeQuality5x5 import size_quality_5x5  # noqa: E402


def main():
    make_folders()
    p = size_quality_5x5("GP")
    factors = smbq_from_5x5(p, "smbq_gp")
    factors.index.name = "date"
    factors.to_parquet(FACTORS / "SMBQ_GP.parquet")

    print(f"\nSMBQ_GP months: {len(factors)}   dates: {factors.index.min().date()} to "
          f"{factors.index.max().date()}   missing months: {factors['smbq_gp'].isna().sum()}")
    stats = pd.DataFrame({
        "mean_annual_%": factors.mean() * 12 * 100,
        "std_annual_%": factors.std() * 12 ** 0.5 * 100,
        "sharpe": factors.mean() / factors.std() * 12 ** 0.5,
    }).round(2)
    print(stats.to_string())


if __name__ == "__main__":
    main()
