# ABOUTME: Three more junk-neutral versions of SMBQ (quality deciles, dropping the junk quintile, composite quality score), daily and monthly.
"""
Run:     python Factors/SMBQ_JunkNeutral.py   (from pyCode/, after Factors/SMBQ.py and
         Predictors/QualityComp.py)
Inputs:  pyData/Portfolios/SizeQuality5x5Daily_OperProf.parquet, ..._monthly.parquet
             (written by Factors/SMBQ.py)
         pyData/Predictors/Size.parquet, BMdec.parquet, OperProf.parquet, QualityComp.parquet
Outputs: pyData/Portfolios/SizeQuality5x10Daily_OperProf*.parquet, SizeQuality5x5Daily_QualityComp*.parquet
         pyData/Factors/SMBQ_JunkNeutralDaily.parquet   daily smbq_d10, smbq_exq1, smbq_comp
         pyData/Factors/SMBQ_JunkNeutral.parquet        the same, monthly (portfolios compounded)

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
from Factors.SMBQ import summary  # noqa: E402
from Portfolios.SizeQuality5x5Daily import size_quality_5x5_daily  # noqa: E402


def small_minus_big(p, quality_groups):
    """Equal-weighted average across the given quality groups of S1Qq - S5Qq (any frequency)."""
    spreads = pd.DataFrame({q: p[f"S1Q{q}"] - p[f"S5Q{q}"] for q in quality_groups})
    # skipna=False: an empty cell makes the day (month) missing instead of silently
    # averaging fewer groups.
    return spreads.mean(axis=1, skipna=False)


def main():
    make_folders()
    daily, monthly = pd.DataFrame(), pd.DataFrame()

    # --- 1. Quality deciles -------------------------------------------------------------------
    p10_daily, p10_monthly = size_quality_5x5_daily("OperProf", quality_groups=10)
    daily["smbq_d10"] = small_minus_big(p10_daily, range(1, 11))
    monthly["smbq_d10"] = small_minus_big(p10_monthly, range(1, 11))

    # --- 2. Drop the junk quintile --------------------------------------------------------------
    # Same 25 portfolios as the main SMBQ (read from disk, not rebuilt).
    p5_daily = pd.read_parquet(PORTFOLIOS / "SizeQuality5x5Daily_OperProf.parquet")
    p5_monthly = pd.read_parquet(PORTFOLIOS / "SizeQuality5x5Daily_OperProf_monthly.parquet")
    daily["smbq_exq1"] = small_minus_big(p5_daily, range(2, 6))
    monthly["smbq_exq1"] = small_minus_big(p5_monthly, range(2, 6))

    # --- 3. Composite quality score ------------------------------------------------------------------
    pc_daily, pc_monthly = size_quality_5x5_daily("QualityComp")
    daily["smbq_comp"] = small_minus_big(pc_daily, range(1, 6))
    monthly["smbq_comp"] = small_minus_big(pc_monthly, range(1, 6))

    daily.index.name = monthly.index.name = "date"
    daily.to_parquet(FACTORS / "SMBQ_JunkNeutralDaily.parquet")
    monthly.to_parquet(FACTORS / "SMBQ_JunkNeutral.parquet")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"\n{len(daily):,} days, {len(monthly)} months, {monthly.index.min().date()} to "
          f"{monthly.index.max().date()}   missing days: {daily.isna().sum().to_dict()}   "
          f"missing months: {monthly.isna().sum().to_dict()}")
    print("daily (252 days):")
    print(summary(daily, 252).to_string())
    print("monthly (12 months):")
    print(summary(monthly, 12).to_string())


if __name__ == "__main__":
    main()
