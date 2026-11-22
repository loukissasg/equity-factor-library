# ABOUTME: Builds the profitability factor RMW (robust minus weak) from the size x OperProf 2x3 sort.
"""
Run:     python Factors/RMW.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, OperProf.parquet   (via ff2x3)
Outputs: pyData/Portfolios/FF2x3_OperProf*.parquet  (written by ff2x3)
         pyData/Factors/RMW.parquet   monthly rmw, smb_op (decimals)
         output/rmw_replicated.csv    the same, for Excel

RMW = 1/2 (S_robust + B_robust) - 1/2 (S_weak + B_weak), Ken French's definition, where
"robust" = top 30% and "weak" = bottom 30% of OperProf (NYSE breakpoints).

smb_op is the SMB of this same sort (small minus big, holding profitability constant).
It is not used yet; Fama-French (2015) average it with the SMBs of other sorts.

Adding this factor does not touch FF3: it only reads the shared Size signal and writes
its own files.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, OUTPUT, make_folders  # noqa: E402
from Portfolios.FF2x3 import ff2x3  # noqa: E402


def main():
    make_folders()

    # sign +1: high operating profitability ("robust") is the long side.
    p = ff2x3("OperProf", +1)

    factors = pd.DataFrame(index=p.index)
    factors["rmw"] = p["spread"]  # 1/2 (SH + BH) - 1/2 (SL + BL), H = robust, L = weak
    factors["smb_op"] = p["smb"]
    factors.index.name = "date"
    factors.to_parquet(FACTORS / "RMW.parquet")
    factors.to_csv(OUTPUT / "rmw_replicated.csv", float_format="%.6f")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"\nRMW months: {len(factors)}   dates: {factors.index.min().date()} to "
          f"{factors.index.max().date()}")
    stats = pd.DataFrame({
        "mean_annual_%": factors.mean() * 12 * 100,
        "std_annual_%": factors.std() * 12 ** 0.5 * 100,
        "sharpe": factors.mean() / factors.std() * 12 ** 0.5,
    }).round(2)
    print(stats.to_string())
    by_decade = factors[["rmw"]].groupby((factors.index.year // 10) * 10).mean() * 12 * 100
    by_decade.index = [f"{d}s" for d in by_decade.index]
    print("Average annual RMW premium by decade (%):")
    print(by_decade.round(1).T.to_string())


if __name__ == "__main__":
    main()
