# ABOUTME: Builds SMBQ, the size factor controlling for quality (Asness et al. 2018), from the size x OperProf 5x5 sort.
"""
Run:     python Factors/SMBQ.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, BMdec.parquet, OperProf.parquet  (via size_quality_5x5)
Outputs: pyData/Portfolios/SizeQuality5x5_OperProf*.parquet   (written by size_quality_5x5)
         pyData/Factors/SMBQ.parquet   monthly smbq and the five within-quintile spreads
         output/smbq_replicated.csv    the same, for Excel

SMBQ = 1/5 * sum over quality quintiles q of (S1Qq - S5Qq)
     = average, across the five quality quintiles, of smallest minus largest size quintile.

Why: within a quality quintile, small and big stocks have similar quality, so the
small-minus-big spread is not a hidden bet against quality (Asness et al. 2018). Equal
weights across quintiles give each quality level the same say.
Note: unlike SMB (median split), SMBQ compares the extreme size quintiles, so it is a
more aggressive size bet and should be more volatile than SMB.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, OUTPUT, make_folders  # noqa: E402
from Portfolios.SizeQuality5x5 import size_quality_5x5  # noqa: E402

SIGNAL = "OperProf"   # main quality signal (validated against French's RMW in Stage 2)
FACTOR = "smbq"
OUTPUT_NAME = "SMBQ"


def smbq_from_5x5(p, factor_name):
    """SMBQ and the five small-minus-big spreads, one per quality quintile."""
    factors = pd.DataFrame(index=p.index)
    for q in range(1, 6):
        factors[f"{factor_name}_q{q}"] = p[f"S1Q{q}"] - p[f"S5Q{q}"]
    spread_columns = [f"{factor_name}_q{q}" for q in range(1, 6)]
    # mean(axis=1) with skipna=False: if a cell is empty in some month, SMBQ is missing that
    # month rather than silently becoming an average of four quintiles.
    factors[factor_name] = factors[spread_columns].mean(axis=1, skipna=False)
    return factors[[factor_name] + spread_columns]


def main():
    make_folders()
    p = size_quality_5x5(SIGNAL)
    factors = smbq_from_5x5(p, FACTOR)
    factors.index.name = "date"
    factors.to_parquet(FACTORS / f"{OUTPUT_NAME}.parquet")
    factors.to_csv(OUTPUT / f"{OUTPUT_NAME.lower()}_replicated.csv", float_format="%.6f")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"\n{OUTPUT_NAME} months: {len(factors)}   dates: {factors.index.min().date()} to "
          f"{factors.index.max().date()}   missing months: {factors[FACTOR].isna().sum()}")
    stats = pd.DataFrame({
        "mean_annual_%": factors.mean() * 12 * 100,
        "std_annual_%": factors.std() * 12 ** 0.5 * 100,
        "sharpe": factors.mean() / factors.std() * 12 ** 0.5,
    }).round(2)
    print("SMBQ and small-minus-big within each quality quintile (q1 = junk, q5 = quality):")
    print(stats.to_string())


if __name__ == "__main__":
    main()
