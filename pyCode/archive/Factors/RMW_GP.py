# ABOUTME: Robustness version of the profitability factor: RMW_GP from the size x GP (gross profitability) 2x3 sort.
"""
Run:     python Factors/RMW_GP.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, GP.parquet   (via ff2x3)
Outputs: pyData/Portfolios/FF2x3_GP*.parquet  (written by ff2x3)
         pyData/Factors/RMW_GP.parquet   monthly rmw_gp, smb_gp (decimals)

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
from Portfolios.FF2x3 import ff2x3  # noqa: E402


def main():
    make_folders()

    # sign +1: high gross profitability is the long side (Novy-Marx 2013).
    p = ff2x3("GP", +1)

    factors = pd.DataFrame(index=p.index)
    factors["rmw_gp"] = p["spread"]
    factors["smb_gp"] = p["smb"]
    factors.index.name = "date"
    factors.to_parquet(FACTORS / "RMW_GP.parquet")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"\nRMW_GP months: {len(factors)}   dates: {factors.index.min().date()} to "
          f"{factors.index.max().date()}")
    stats = pd.DataFrame({
        "mean_annual_%": factors.mean() * 12 * 100,
        "std_annual_%": factors.std() * 12 ** 0.5 * 100,
        "sharpe": factors.mean() / factors.std() * 12 ** 0.5,
    }).round(2)
    print(stats.to_string())


if __name__ == "__main__":
    main()
