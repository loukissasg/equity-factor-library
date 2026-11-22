# ABOUTME: Generic Fama-French 2x3 sort: size x any signal, six value-weighted portfolios, SMB and the signal's long-short spread.
"""
Use:     from Portfolios.FF2x3 import ff2x3
         p = ff2x3("BMdec", +1)          (called by the scripts in Factors/)
Run:     python Portfolios/FF2x3.py BMdec [+1|-1]   (from pyCode/, to inspect one sort)
Inputs:  pyData/Predictors/Size.parquet
         pyData/Predictors/<signal>.parquet         [permno, yyyymm, <signal>]
         pyData/Intermediate/SignalMasterTable.parquet
Outputs: pyData/Portfolios/FF2x3_<signal>.parquet   monthly: SL SN SH BL BN BH smb spread
         pyData/Portfolios/FF2x3_<signal>_assignments.parquet   one row per firm and June:
             portfolio label, size, signal and the breakpoints used
         pyData/Portfolios/FF2x3_<signal>_holdings.parquet      per month and portfolio:
             number of stocks and average weight (ME, $ thousands)

Each June t every stock that has both signals gets two labels:
  size:   S (small) or B (big), split at the NYSE median of June ME
  signal: L (low), N (neutral), H (high), split at the NYSE 30th and 70th percentiles
The six portfolios SL, SN, SH, BL, BN, BH are held from July t to June t+1.

This file contains *no* signal-specific logic. Which stocks have a signal (e.g. positive
book equity, two years in Compustat) is decided in the Predictors/ script. That way a
new factor is a new signal file plus one call to ff2x3(), and existing factors cannot
change when one is added.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, PORTFOLIO_NAMES, PORTFOLIOS, PREDICTORS, make_folders  # noqa: E402
from utils.checks import require_columns, require_unique  # noqa: E402


def eligible_stocks(signal_name):
    """Stocks in the June sort: they have June size and the signal (both in standard format)."""
    size = pd.read_parquet(PREDICTORS / "Size.parquet")
    signal = pd.read_parquet(PREDICTORS / f"{signal_name}.parquet")
    require_columns(signal, ["permno", "yyyymm", signal_name], signal_name)
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet",
                            columns=["permno", "permco", "yyyymm", "is_nyse"])

    # Predictor files only contain non-missing values, so an inner merge keeps exactly the
    # stocks that have both. Annual signals are stored at yyyymm = YYYY06.
    eligible = size.merge(signal, on=["permno", "yyyymm"], how="inner")
    eligible = eligible[eligible["yyyymm"] % 100 == 6]

    # The firm id (PERMCO) is needed to follow the firm through the holding period, and the
    # exchange to pick NYSE stocks for the breakpoints. Both come from the master table at
    # the June sort date.
    eligible = eligible.merge(panel, on=["permno", "yyyymm"], how="inner")
    eligible["sort_year"] = eligible["yyyymm"] // 100
    eligible = eligible.rename(columns={signal_name: "signal"})
    return eligible


def nyse_breakpoints(eligible):
    """Size median and signal 30th/70th percentiles, from NYSE stocks only."""
    # NASDAQ and AMEX are full of very small firms. If they set the breakpoints, the size
    # median would be tiny and the breakpoints would jump when NASDAQ entered CRSP in 1972.
    # NYSE breakpoints keep the split economically stable; all stocks are then sorted on them.
    nyse = eligible[eligible["is_nyse"]]
    return nyse.groupby("sort_year").agg(
        size_median=("Size", "median"),
        signal_30=("signal", lambda s: s.quantile(0.3)),
        signal_70=("signal", lambda s: s.quantile(0.7)),
        nyse_stocks=("permco", "count"),
    ).reset_index()


def assign_portfolios(eligible, breakpoints):
    """Label each stock S/B on size and L/N/H on the signal; the portfolio is the pair."""
    eligible = eligible.merge(breakpoints[["sort_year", "size_median", "signal_30", "signal_70"]],
                              on="sort_year", how="inner")

    eligible["size_group"] = "B"
    eligible.loc[eligible["Size"] <= eligible["size_median"], "size_group"] = "S"

    eligible["signal_group"] = "N"
    eligible.loc[eligible["signal"] <= eligible["signal_30"], "signal_group"] = "L"
    eligible.loc[eligible["signal"] > eligible["signal_70"], "signal_group"] = "H"

    eligible["portfolio"] = eligible["size_group"] + eligible["signal_group"]
    assignments = eligible[["permco", "permno", "sort_year", "portfolio", "size_group",
                            "signal_group", "is_nyse", "Size", "signal", "size_median",
                            "signal_30", "signal_70"]].reset_index(drop=True)
    require_unique(assignments, ["permco", "sort_year"], "FF2x3 assignments")
    return assignments


def value_weighted_returns(assignments):
    """Monthly value-weighted returns of the six portfolios, held July t to June t+1."""
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet",
                            columns=["permco", "date", "ret", "me_lag"])

    # Each monthly return is matched to the most recent June sort: July-December belong to
    # this year's sort, January-June to last year's.
    returns = panel.copy()
    returns["sort_year"] = returns["date"].dt.year
    returns.loc[returns["date"].dt.month <= 6, "sort_year"] -= 1
    returns = returns.merge(assignments[["permco", "sort_year", "portfolio"]],
                            on=["permco", "sort_year"], how="inner")

    # A stock contributes in a month only if it has a return and last month's ME (its
    # weight). A stock that delists mid-year drops out from the next month on; its
    # delisting return is already in its last mthret.
    returns = returns[returns["ret"].notna() & returns["me_lag"].notna() & (returns["me_lag"] > 0)]

    # Weights = ME at the end of the previous month: what an investor could actually hold,
    # and it stops thousands of tiny, illiquid stocks from dominating the return.
    returns["ret_x_weight"] = returns["ret"] * returns["me_lag"]
    grouped = returns.groupby(["date", "portfolio"])
    portfolio_returns = (grouped["ret_x_weight"].sum() / grouped["me_lag"].sum()).unstack()
    portfolio_returns = portfolio_returns[PORTFOLIO_NAMES]

    holdings = grouped.agg(n_stocks=("permco", "count"), avg_me_lag=("me_lag", "mean"))
    return portfolio_returns, holdings.reset_index()


def ff2x3(signal_name, sign, rebalance="annual"):
    """
    2x3 sort on size x signal_name. Returns a monthly DataFrame with the six portfolio
    returns (SL..BH) and two long-short factors:
        smb    = 1/3 (SL + SN + SH) - 1/3 (BL + BN + BH)
        spread = sign * [1/2 (SH + BH) - 1/2 (SL + BL)]
    sign = +1 if high values of the signal predict high returns (B/M, profitability),
    -1 if low values do; the spread is then always "good minus bad". The labels L/N/H
    always refer to the raw signal (H = highest values), whatever the sign.
    """
    # Fama-French rebalance once a year, in June, because accounting data is annual.
    # Monthly rebalancing (for market-based signals) is not implemented yet.
    if rebalance != "annual":
        raise NotImplementedError(f"ff2x3: rebalance='{rebalance}' is not implemented")
    if sign not in (1, -1):
        raise ValueError(f"ff2x3: sign must be +1 or -1, got {sign}")

    make_folders()
    print(f"Forming the 2x3 portfolios on size x {signal_name} (sign {sign:+d})...", flush=True)

    # --- Eligible stocks, NYSE breakpoints, portfolio labels ----------------------------------
    eligible = eligible_stocks(signal_name)
    breakpoints = nyse_breakpoints(eligible)
    assignments = assign_portfolios(eligible, breakpoints)

    print(f"NYSE breakpoints, every 5th June (size median in $ millions, {signal_name} "
          f"30th/70th percentiles):")
    show = breakpoints.copy()
    show["size_median"] = (show["size_median"] / 1000).round(0)  # $ thousands -> $ millions
    show[["signal_30", "signal_70"]] = show[["signal_30", "signal_70"]].round(3)
    print(show[(show["sort_year"] % 5 == 0) | (show["sort_year"] == show["sort_year"].max())]
          .to_string(index=False))

    # Every June: a portfolio with very few stocks would give a noisy, unreliable return.
    counts = assignments.groupby(["sort_year", "portfolio"]).size().unstack()[PORTFOLIO_NAMES]
    counts["total"] = counts.sum(axis=1)
    counts["nyse"] = assignments.groupby("sort_year")["is_nyse"].sum()
    print(f"Stocks per portfolio, every June (size x {signal_name}):")
    print(counts.to_string())

    # --- Returns and factors ------------------------------------------------------------------------
    p, holdings = value_weighted_returns(assignments)

    # SMB: one small and one big portfolio *within each signal group*, so both legs have the
    # same mix of high and low signal stocks and SMB captures size, not the signal.
    p["smb"] = (p["SL"] + p["SN"] + p["SH"]) / 3 - (p["BL"] + p["BN"] + p["BH"]) / 3

    # Spread: one small and one big leg on each side, so it captures the signal, not size.
    # The neutral portfolios (SN, BN) are not used.
    p["spread"] = sign * ((p["SH"] + p["BH"]) / 2 - (p["SL"] + p["BL"]) / 2)

    p.index.name = "date"
    p.to_parquet(PORTFOLIOS / f"FF2x3_{signal_name}.parquet")
    assignments.to_parquet(PORTFOLIOS / f"FF2x3_{signal_name}_assignments.parquet", index=False)
    holdings.to_parquet(PORTFOLIOS / f"FF2x3_{signal_name}_holdings.parquet", index=False)

    # --- Sanity checks ------------------------------------------------------------------------------
    print(f"months: {len(p)}   dates: {p.index.min().date()} to {p.index.max().date()}   "
          f"missing values: {p[PORTFOLIO_NAMES].isna().sum().sum()}")
    stats = pd.DataFrame({
        "mean_monthly_%": (p.mean() * 100).round(2),
        "std_annual_%": (p.std() * (12 ** 0.5) * 100).round(1),
    })
    print("Mean monthly return (%) and annualized std (%), full sample:")
    print(stats.T.to_string())
    return p


if __name__ == "__main__":
    # e.g. python Portfolios/FF2x3.py BMdec +1
    if len(sys.argv) < 2:
        raise SystemExit("usage: python Portfolios/FF2x3.py <signal_name> [+1|-1]")
    ff2x3(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
