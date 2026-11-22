# ABOUTME: Three more junk-neutral versions of SMBQ (quality deciles, dropping the junk quintile, composite quality score).
"""
Run:     python Factors/SMBQ_JunkNeutral.py   (from pyCode/, after Factors/SMBQ.py and
         Predictors/QualityComp.py)
Inputs:  pyData/Portfolios/SizeQuality5x5_OperProf.parquet   (written by Factors/SMBQ.py)
         pyData/Predictors/Size.parquet, BMdec.parquet, OperProf.parquet, QualityComp.parquet
Outputs: pyData/Portfolios/SizeQuality5x10_OperProf*.parquet, SizeQuality5x5_QualityComp*.parquet
         pyData/Factors/SMBQ_JunkNeutral.parquet   monthly smbq_d10, smbq_exq1, smbq_comp

Why: the main SMBQ (5 size x 5 OperProf quintiles) stayed short quality. Two reasons
(see TEAM_NOTES Section 10): (1) the junk quintile Q1 has no lower limit, and small
stocks in it are far junkier than big ones; half of all small stocks sit there.
(2) OperProf measures profitability only, while QMJ also measures safety. Each
version below attacks one of these:
  smbq_d10   5 size quintiles x 10 OperProf deciles, average of S1 - S5 across the 10.
             Finer groups: small and big stocks in a cell are closer in quality.
  smbq_exq1  the 5x5 OperProf sort, average of S1 - S5 across Q2..Q5 only.
             Drops the cell where the junk mismatch sits (but also drops junk stocks).
  smbq_comp  5x5 sort on QualityComp (profitability + low leverage + low volatility).
             Controls for safety as well as profits.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, PORTFOLIOS, make_folders  # noqa: E402
from Portfolios.SizeQuality5x5 import size_quality_5x5  # noqa: E402


def small_minus_big(p, quality_groups):
    """Equal-weighted average across the given quality groups of S1Qq - S5Qq."""
    spreads = pd.DataFrame({q: p[f"S1Q{q}"] - p[f"S5Q{q}"] for q in quality_groups})
    # skipna=False: an empty cell makes the month missing instead of silently averaging fewer.
    return spreads.mean(axis=1, skipna=False)


def main():
    make_folders()
    factors = pd.DataFrame()

    # --- 1. Quality deciles -------------------------------------------------------------------
    p10 = size_quality_5x5("OperProf", quality_groups=10)
    factors["smbq_d10"] = small_minus_big(p10, range(1, 11))

    # --- 2. Drop the junk quintile --------------------------------------------------------------
    # Same 25 portfolios as the main SMBQ (read from disk, not rebuilt).
    p5 = pd.read_parquet(PORTFOLIOS / "SizeQuality5x5_OperProf.parquet")
    factors["smbq_exq1"] = small_minus_big(p5, range(2, 6))

    # --- 3. Composite quality score ------------------------------------------------------------------
    pc = size_quality_5x5("QualityComp")
    factors["smbq_comp"] = small_minus_big(pc, range(1, 6))

    factors.index.name = "date"
    factors.to_parquet(FACTORS / "SMBQ_JunkNeutral.parquet")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"\nmonths: {len(factors)}   dates: {factors.index.min().date()} to "
          f"{factors.index.max().date()}   missing months: {factors.isna().sum().to_dict()}")
    stats = pd.DataFrame({
        "mean_annual_%": factors.mean() * 12 * 100,
        "std_annual_%": factors.std() * 12 ** 0.5 * 100,
        "sharpe": factors.mean() / factors.std() * 12 ** 0.5,
    }).round(2)
    print(stats.to_string())


if __name__ == "__main__":
    main()
