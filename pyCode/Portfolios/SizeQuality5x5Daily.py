# ABOUTME: Daily version of the 5x5 size x quality sort: same June portfolios as SizeQuality5x5.py, value-weighted returns every trading day.
"""
Use:     from Portfolios.SizeQuality5x5Daily import size_quality_5x5_daily
         p = size_quality_5x5_daily("OperProf")      (called by the SMBQ scripts in Factors/)
Run:     python Portfolios/SizeQuality5x5Daily.py OperProf [quality_groups]   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, BMdec.parquet (universe), <signal>.parquet
         pyData/Intermediate/SignalMasterTable.parquet      (PERMCO and NYSE flag in June)
         pyData/Intermediate/DailyMasterTable/DailyMasterTable_YYYY.parquet   (daily returns)
Outputs: pyData/Portfolios/SizeQuality<tag>Daily_<signal>.parquet          daily S1Q1 ... S5Q<n>
         pyData/Portfolios/SizeQuality<tag>Daily_<signal>_monthly.parquet  the same, compounded
                                                                          to months
         pyData/Portfolios/SizeQuality<tag>Daily_<signal>_assignments.parquet
         pyData/Portfolios/SizeQuality<tag>Daily_<signal>_holdings.parquet  per day and cell:
             number of stocks and total weight (ME, $ thousands)
         (<tag> = 5x5, or 5x10 for quality deciles)

Exactly the FF2x3Daily.py idea applied to the 5x5 grid:
  - The June sort is *identical* to the monthly one: eligible_stocks() and
    assign_portfolios() are imported from SizeQuality5x5.py unchanged.
  - The return step is the generic daily one from FF2x3Daily.py (any June assignments,
    weights = yesterday's price x shares at last month-end, one calendar year at a time).
Why a 5x5 sort, and the universe: see SizeQuality5x5.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import CRSP_END_YEAR, FIRST_SORT_YEAR, PORTFOLIOS, make_folders  # noqa: E402
from Portfolios.FF2x3Daily import (compound_to_months, daily_returns_one_year,  # noqa: E402
                                   frozen_monthly_file)
from Portfolios.SizeQuality5x5 import (MIN_STOCKS, assign_portfolios, eligible_stocks,  # noqa: E402
                                       portfolio_names)


def size_quality_5x5_daily(signal_name, universe_signal="BMdec", quality_groups=5):
    """
    Size quintiles x signal_name groups, daily returns of S1Q1 .. S5Q<n>. Also saves the
    portfolios compounded to months (month-end dates), which the factor scripts use for
    the monthly factors.
    """
    make_folders()
    names = portfolio_names(quality_groups)
    tag = "5x5" if quality_groups == 5 else f"5x{quality_groups}"
    print(f"Forming the {tag} portfolios on size x {signal_name}, daily returns "
          f"(universe: stocks with {universe_signal})...", flush=True)

    # --- The June sort: unchanged from the monthly version --------------------------------------
    eligible = eligible_stocks(signal_name, universe_signal)
    assignments = assign_portfolios(eligible, quality_groups)
    counts = (assignments.groupby(["sort_year", "portfolio"]).size().unstack()
              .reindex(columns=names).fillna(0).astype(int))
    thin = counts.stack()
    thin = thin[thin < MIN_STOCKS]
    print(f"sort years {counts.index.min()}-{counts.index.max()}, firm-years {len(assignments):,}, "
          f"smallest cell in any June: {counts.min().min()} stocks, "
          f"cell-years with < {MIN_STOCKS} stocks: {len(thin)}")
    if len(thin) > 0:
        print(thin.rename("n_stocks").reset_index().groupby("portfolio")
              .agg(years=("sort_year", "count"), first=("sort_year", "min"),
                   last=("sort_year", "max"), fewest=("n_stocks", "min")).to_string())

    # --- Daily returns, one calendar year at a time --------------------------------------------
    all_returns, all_holdings = [], []
    delisting_rows = 0
    for year in range(FIRST_SORT_YEAR, CRSP_END_YEAR + 1):
        returns, holdings, n_delisting = daily_returns_one_year(year, assignments)
        all_returns.append(returns)
        all_holdings.append(holdings)
        delisting_rows += n_delisting
    p = pd.concat(all_returns).reindex(columns=names).sort_index()
    p.index.name = "date"
    holdings = pd.concat(all_holdings, ignore_index=True)

    # Monthly portfolios: compound each cell first (see compound_to_months for empty cells).
    p_monthly = compound_to_months(p)
    p_monthly.index = p_monthly.index.to_timestamp(how="end").normalize()
    p_monthly.index.name = "date"

    p.to_parquet(PORTFOLIOS / f"SizeQuality{tag}Daily_{signal_name}.parquet")
    p_monthly.to_parquet(PORTFOLIOS / f"SizeQuality{tag}Daily_{signal_name}_monthly.parquet")
    assignments.to_parquet(PORTFOLIOS / f"SizeQuality{tag}Daily_{signal_name}_assignments.parquet",
                           index=False)
    holdings.to_parquet(PORTFOLIOS / f"SizeQuality{tag}Daily_{signal_name}_holdings.parquet",
                        index=False)

    # --- Sanity checks ------------------------------------------------------------------------
    print(f"trading days: {len(p):,}   {p.index.min().date()} to {p.index.max().date()}   "
          f"missing cell-days (empty cell): {int(p.isna().sum().sum())}   "
          f"missing cell-months: {int(p_monthly.isna().sum().sum())}   "
          f"delisting returns included: {delisting_rows:,}")
    # The fewest stocks any cell held on any day: daily returns of a cell with a handful
    # of stocks are noisy.
    n_stocks = holdings.pivot(index="date", columns="portfolio", values="n_stocks").reindex(columns=names)
    fewest = n_stocks.min().sort_values().head(5)
    print("Fewest stocks held on any day, the 5 thinnest cells: "
          + ", ".join(f"{cell} {int(n)}" for cell, n in fewest.items()))

    # Same stocks as the monthly sort, and do the daily cells compound to the monthly ones?
    # (Only where a frozen monthly-method version of this sort exists.)
    frozen = frozen_monthly_file(f"SizeQuality{tag}_{signal_name}_assignments.parquet")
    if frozen is not None:
        old = pd.read_parquet(frozen, columns=["permco", "sort_year", "portfolio"])
        both = assignments.merge(old, on=["permco", "sort_year"], how="outer",
                                 suffixes=("", "_monthly"), indicator=True)
        same = (both["_merge"] == "both") & (both["portfolio"] == both["portfolio_monthly"])
        print(f"assignments identical to the frozen monthly sort: {same.sum():,} of {len(both):,}")
        old_returns = pd.read_parquet(frozen_monthly_file(f"SizeQuality{tag}_{signal_name}.parquet"))
        gap = (p_monthly - old_returns.reindex(columns=names)).stack()
        print(f"daily compounded vs frozen monthly cells: mean |gap| {gap.abs().mean() * 1e4:.1f} bp, "
              f"cell-months with |gap| > 0.5%: {(gap.abs() > 0.005).sum()} of {len(gap):,}")
        if (gap.abs() > 0.005).any():
            print("   largest: " + ", ".join(f"{d:%Y-%m} {c} {v * 100:+.2f}%"
                                         for (d, c), v in gap[gap.abs().nlargest(4).index].items()))
    return p, p_monthly


if __name__ == "__main__":
    # e.g. python Portfolios/SizeQuality5x5Daily.py OperProf
    if len(sys.argv) < 2:
        raise SystemExit("usage: python Portfolios/SizeQuality5x5Daily.py <signal_name> [quality_groups]")
    size_quality_5x5_daily(sys.argv[1], quality_groups=int(sys.argv[2]) if len(sys.argv) > 2 else 5)
