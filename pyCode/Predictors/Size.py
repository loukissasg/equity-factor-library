# ABOUTME: Size signal: firm market equity at the end of June of each sort year.
"""
Run:     python Predictors/Size.py   (from pyCode/)
Inputs:  pyData/Intermediate/SignalMasterTable.parquet
Outputs: pyData/Predictors/Size.parquet   [permno, yyyymm, Size]
         yyyymm = YYYY06, Size = firm ME (all share classes) at end of June, $ thousands

Portfolios are formed at the end of June of year t. Size is measured at that same
date, so it is known when the portfolios are formed.

Note: OpenAP's "Size" is log(ME), every month. Ours is the level of June ME, which is
what the Fama-French sort uses (the median split is the same for ME and log ME).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FIRST_SORT_YEAR, INTERMEDIATE, make_folders  # noqa: E402
from utils.save_predictor import save_predictor  # noqa: E402

INPUT_FILE = INTERMEDIATE / "SignalMasterTable.parquet"


def main():
    make_folders()
    print("Building the size signal (June ME)...", flush=True)
    panel = pd.read_parquet(INPUT_FILE)

    # The master table has one row per firm (PERMCO) and month, under the PERMNO of the
    # firm's largest share class, and `me` is already the firm total (see
    # SignalMasterTable.py). So the June row is the firm's size.
    june = panel[(panel["date"].dt.month == 6) & (panel["date"].dt.year >= FIRST_SORT_YEAR)].copy()
    june["Size"] = june["me"]

    size = save_predictor(june, "Size")

    # --- Sanity checks ---------------------------------------------------------------------------
    per_year = size.groupby(size["yyyymm"] // 100).agg(
        firms=("permno", "count"),
        median_me_june_m=("Size", "median"),
    )
    per_year["median_me_june_m"] = (per_year["median_me_june_m"] / 1000).round(0)  # $ millions
    print(f"June rows in the master table: {len(june):,}   missing June ME: "
          f"{june['me'].isna().mean():.2%} (not saved)")
    print("Firms each June, every 5th year (median June ME of all firms, $ millions):")
    print(per_year[(per_year.index % 5 == 0) | (per_year.index == per_year.index.max())]
          .to_string())


if __name__ == "__main__":
    main()
