# ABOUTME: Generic 5x5 sort: size quintiles x quality-signal quintiles (NYSE breakpoints), 25 value-weighted portfolios.
"""
Use:     from Portfolios.SizeQuality5x5 import size_quality_5x5
         p = size_quality_5x5("OperProf")      (called by the scripts in Factors/)
Run:     python Portfolios/SizeQuality5x5.py OperProf   (from pyCode/, to inspect one sort)
Inputs:  pyData/Predictors/Size.parquet, BMdec.parquet (universe), <signal>.parquet
         pyData/Intermediate/SignalMasterTable.parquet
Outputs: pyData/Portfolios/SizeQuality5x5_<signal>.parquet   monthly returns, columns
             S1Q1 ... S5Q5 (S1 = smallest size quintile, Q5 = highest quality)
         pyData/Portfolios/SizeQuality5x5_<signal>_assignments.parquet   one row per
             firm and June: size and quality quintile, signals and breakpoints
         pyData/Portfolios/SizeQuality5x5_<signal>_holdings.parquet      per month and
             portfolio: number of stocks and average weight (ME, $ thousands)

Why a 5x5 sort (Asness, Frazzini, Israel, Moskowitz and Pedersen 2018, "Size matters, if
you control your junk"): small stocks are on average lower quality ("junk") than big
stocks. SMB therefore mixes a size bet with a short-quality bet, and junk earns low
returns. Comparing small and big stocks *within the same quality quintile* removes the
quality difference between the two legs, so what is left is closer to a pure size effect.

Each June t, two *independent* sorts (as in FF2x3):
  size:    5 quintiles of June ME, NYSE breakpoints (20/40/60/80th percentiles)
  quality: 5 quintiles of the signal, NYSE breakpoints
Held July t to June t+1, value-weighted with last month's ME.

Universe: the same stocks as the FF3 sort (they have June size and a BMdec value, i.e.
positive BE, December ME and two years in Compustat), which also have the quality
signal. Keeping the FF3 universe makes SMBQ directly comparable with SMB. As in FF2x3,
this file contains no signal-specific logic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, PORTFOLIOS, PREDICTORS, make_folders  # noqa: E402
from utils.checks import require_columns, require_unique  # noqa: E402

QUINTILES = [0.2, 0.4, 0.6, 0.8]
PORTFOLIO_NAMES_5x5 = [f"S{s}Q{q}" for s in range(1, 6) for q in range(1, 6)]
MIN_STOCKS = 10  # a cell with fewer stocks gives a noisy, unreliable return


def portfolio_names(quality_groups):
    """S1Q1 ... S5Q<n>: 5 size quintiles x n quality groups."""
    return [f"S{s}Q{q}" for s in range(1, 6) for q in range(1, quality_groups + 1)]


def group_cutoffs(n_groups):
    """Percentiles that split a sample into n equal groups, e.g. 5 -> [0.2, 0.4, 0.6, 0.8]."""
    return [k / n_groups for k in range(1, n_groups)]


def eligible_stocks(signal_name, universe_signal):
    """Stocks in the June sort: June size, the universe signal (BMdec) and the quality signal."""
    size = pd.read_parquet(PREDICTORS / "Size.parquet")
    universe = pd.read_parquet(PREDICTORS / f"{universe_signal}.parquet")[["permno", "yyyymm"]]
    signal = pd.read_parquet(PREDICTORS / f"{signal_name}.parquet")
    require_columns(signal, ["permno", "yyyymm", signal_name], signal_name)
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet",
                            columns=["permno", "permco", "yyyymm", "is_nyse"])

    # Predictor files only contain non-missing values, so inner merges keep exactly the
    # stocks that have all three. Annual signals are stored at yyyymm = YYYY06.
    eligible = size.merge(universe, on=["permno", "yyyymm"], how="inner")
    eligible = eligible.merge(signal, on=["permno", "yyyymm"], how="inner")
    eligible = eligible[eligible["yyyymm"] % 100 == 6]
    eligible = eligible.merge(panel, on=["permno", "yyyymm"], how="inner")
    eligible["sort_year"] = eligible["yyyymm"] // 100
    return eligible.rename(columns={signal_name: "signal"})


def nyse_quintile_breakpoints(eligible, column, cutoffs=QUINTILES):
    """NYSE percentiles of `column` per sort year (default 20/40/60/80th: quintiles)."""
    # NYSE-only breakpoints for the same reason as in FF2x3: otherwise the many tiny
    # NASDAQ stocks would put four of the five size cut-offs among micro-caps.
    nyse = eligible[eligible["is_nyse"]]
    breakpoints = nyse.groupby("sort_year")[column].quantile(cutoffs).unstack()
    breakpoints.columns = [f"{column}_p{round(q * 100)}" for q in cutoffs]
    return breakpoints.reset_index()


def quintile(values, breakpoints):
    """1 + number of breakpoints the value is above (a value equal to a breakpoint goes
    to the lower group, as in FF2x3)."""
    group = pd.Series(1, index=values.index)
    for column in breakpoints.columns:
        group += (values > breakpoints[column]).astype(int)
    return group


def assign_portfolios(eligible, quality_groups=5):
    """Size quintile 1-5 and quality group 1-n for each stock; the portfolio is the pair."""
    size_bp = nyse_quintile_breakpoints(eligible, "Size")
    signal_bp = nyse_quintile_breakpoints(eligible, "signal", group_cutoffs(quality_groups))
    eligible = eligible.merge(size_bp, on="sort_year", how="inner")
    eligible = eligible.merge(signal_bp, on="sort_year", how="inner")

    size_columns = [c for c in size_bp.columns if c != "sort_year"]
    signal_columns = [c for c in signal_bp.columns if c != "sort_year"]
    eligible["size_q"] = quintile(eligible["Size"], eligible[size_columns])
    eligible["quality_q"] = quintile(eligible["signal"], eligible[signal_columns])
    eligible["portfolio"] = ("S" + eligible["size_q"].astype(str)
                             + "Q" + eligible["quality_q"].astype(str))

    assignments = eligible[["permco", "permno", "sort_year", "portfolio", "size_q",
                            "quality_q", "is_nyse", "Size", "signal"]
                           + size_columns + signal_columns].reset_index(drop=True)
    require_unique(assignments, ["permco", "sort_year"], "5x5 assignments")
    return assignments


def value_weighted_returns(assignments, names=PORTFOLIO_NAMES_5x5):
    """Monthly value-weighted returns of the portfolios, held July t to June t+1."""
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet",
                            columns=["permco", "date", "ret", "me_lag"])

    # July-December belong to this year's June sort, January-June to last year's.
    returns = panel.copy()
    returns["sort_year"] = returns["date"].dt.year
    returns.loc[returns["date"].dt.month <= 6, "sort_year"] -= 1
    returns = returns.merge(assignments[["permco", "sort_year", "portfolio"]],
                            on=["permco", "sort_year"], how="inner")

    # A stock counts in a month only with a return and last month's ME (its weight);
    # delisting returns are already in mthret.
    returns = returns[returns["ret"].notna() & returns["me_lag"].notna() & (returns["me_lag"] > 0)]

    # Weights = ME at the end of the previous month (no look-ahead, investable).
    returns["ret_x_weight"] = returns["ret"] * returns["me_lag"]
    grouped = returns.groupby(["date", "portfolio"])
    portfolio_returns = (grouped["ret_x_weight"].sum() / grouped["me_lag"].sum()).unstack()
    portfolio_returns = portfolio_returns.reindex(columns=names)

    holdings = grouped.agg(n_stocks=("permco", "count"), avg_me_lag=("me_lag", "mean"))
    return portfolio_returns, holdings.reset_index()


def size_quality_5x5(signal_name, universe_signal="BMdec", quality_groups=5):
    """
    Size quintiles x signal_name groups. Returns monthly returns of the portfolios
    S1Q1 .. S5Q<n>. quality_groups = 5 is the standard 5x5 sort; 10 (deciles) splits each
    quality quintile in two, so small and big stocks in a cell are closer in quality.
    """
    make_folders()
    names = portfolio_names(quality_groups)
    # Files keep their old names for the 5x5 sort; other grids get their own suffix.
    tag = "5x5" if quality_groups == 5 else f"5x{quality_groups}"
    print(f"Forming the {tag} portfolios on size x {signal_name} "
          f"(universe: stocks with {universe_signal})...", flush=True)

    eligible = eligible_stocks(signal_name, universe_signal)
    assignments = assign_portfolios(eligible, quality_groups)

    # --- Stocks per cell, every June ---------------------------------------------------------
    counts = (assignments.groupby(["sort_year", "portfolio"]).size().unstack()
              .reindex(columns=names).fillna(0).astype(int))
    print(f"Stocks per cell, every June (S1 = small ... S5 = big, "
          f"Q1 = junk ... Q{quality_groups} = quality):")
    with pd.option_context("display.width", 400, "display.max_columns", 60):
        print(counts.to_string())

    # Flag thin cells: they are mostly big-and-junky stocks in the early years. A thin cell
    # is not wrong, but its return is driven by a handful of firms.
    thin = counts.stack()
    thin = thin[thin < MIN_STOCKS]
    if len(thin) > 0:
        print(f"\nWARNING: {len(thin)} cell-years with fewer than {MIN_STOCKS} stocks:")
        print(thin.rename("n_stocks").reset_index().groupby("portfolio")
              .agg(years=("sort_year", "count"), first=("sort_year", "min"),
                   last=("sort_year", "max"), fewest=("n_stocks", "min")).to_string())
    else:
        print(f"\nno cell has fewer than {MIN_STOCKS} stocks in any June")
    print(f"smallest cell overall: {counts.min().min()} stocks")

    # --- Returns ------------------------------------------------------------------------------
    p, holdings = value_weighted_returns(assignments, names)
    p.index.name = "date"
    p.to_parquet(PORTFOLIOS / f"SizeQuality{tag}_{signal_name}.parquet")
    assignments.to_parquet(PORTFOLIOS / f"SizeQuality{tag}_{signal_name}_assignments.parquet",
                           index=False)
    holdings.to_parquet(PORTFOLIOS / f"SizeQuality{tag}_{signal_name}_holdings.parquet",
                        index=False)

    # --- Sanity checks ------------------------------------------------------------------------
    n_missing = int(p.isna().sum().sum())
    print(f"months: {len(p)}   dates: {p.index.min().date()} to {p.index.max().date()}   "
          f"missing portfolio-months (empty cell): {n_missing}")
    means = (p.mean() * 100).values.reshape(5, quality_groups)
    print(f"Mean monthly return (%), rows = size quintile S1..S5, "
          f"columns = quality Q1..Q{quality_groups}:")
    with pd.option_context("display.width", 250):
        print(pd.DataFrame(means, index=[f"S{s}" for s in range(1, 6)],
                           columns=[f"Q{q}" for q in range(1, quality_groups + 1)])
              .round(2).to_string())
    return p


if __name__ == "__main__":
    # e.g. python Portfolios/SizeQuality5x5.py OperProf
    if len(sys.argv) < 2:
        raise SystemExit("usage: python Portfolios/SizeQuality5x5.py <signal_name> [quality_groups]")
    size_quality_5x5(sys.argv[1], quality_groups=int(sys.argv[2]) if len(sys.argv) > 2 else 5)
