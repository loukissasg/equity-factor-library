# ABOUTME: Builds the three Fama-French factors (MKT-RF, SMB, HML) at daily frequency from the daily 2x3 portfolios and the daily market.
"""
Run:     python Factors/FF3Daily.py   (from pyCode/)
Inputs:  pyData/Portfolios/FF2x3Daily_BMdec*.parquet   (written by ff2x3_daily, called here)
         pyData/Intermediate/DailyMasterTable/DailyMasterTable_YYYY.parquet   (market return)
         pyData/Intermediate/french_ff3_daily.parquet  (daily RF)
Outputs: pyData/Factors/FF3Daily.parquet   daily mkt_rf, smb, hml, mkt, rf (decimals)
         output/ff3_daily_replicated.csv   the same, for Excel

Same recipe as Factors/FF3.py, one frequency down. Each factor is a zero-investment
long-short portfolio, rebuilt every trading day from that day's portfolio returns.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import (CRSP_END_YEAR, DAILY_MASTER_DIR, FACTORS, FIRST_SORT_YEAR,  # noqa: E402
                    INTERMEDIATE, MONTHLY_PIPELINE_REF, OUTPUT, PORTFOLIO_NAMES, make_folders)
from Portfolios.FF2x3Daily import compound_to_months, ff2x3_daily  # noqa: E402

TRADING_DAYS_PER_YEAR = 252


def summary_stats(df):
    """Mean and volatility (in % per year) and annualized Sharpe ratio, from daily returns."""
    return pd.DataFrame({
        "mean_annual_%": df.mean() * TRADING_DAYS_PER_YEAR * 100,
        "std_annual_%": df.std() * (TRADING_DAYS_PER_YEAR ** 0.5) * 100,
        "sharpe": df.mean() / df.std() * (TRADING_DAYS_PER_YEAR ** 0.5),
    }).round(2)


def market_return_one_year(year):
    """Value-weighted return of all stocks in our universe, every trading day of one year."""
    days = pd.read_parquet(DAILY_MASTER_DIR / f"DailyMasterTable_{year}.parquet",
                           columns=["date", "ret", "me_lag"])
    # Same rule as the portfolios: a return and a weight (ME on the previous trading day).
    days = days[days["ret"].notna() & days["me_lag"].notna() & (days["me_lag"] > 0)].copy()
    days["ret_x_weight"] = days["ret"] * days["me_lag"]
    grouped = days.groupby("date")
    return grouped["ret_x_weight"].sum() / grouped["me_lag"].sum()


def main():
    make_folders()

    # --- SMB and HML from the size x B/M sort, every trading day ----------------------------------
    # sign +1: high B/M ("value") is the long side, as in Fama-French.
    p = ff2x3_daily("BMdec", +1)

    print("\nBuilding daily SMB, HML and MKT-RF...", flush=True)
    factors = pd.DataFrame(index=p.index)

    # SMB = 1/3 (SL + SN + SH) - 1/3 (BL + BN + BH): small minus big, holding B/M constant.
    factors["smb"] = p["smb"]

    # HML = 1/2 (SH + BH) - 1/2 (SL + BL): value minus growth, holding size constant.
    factors["hml"] = p["spread"]

    # Market: value-weighted return of *all* stocks in our CRSP universe, not only those in
    # the 2x3 sort (a stock without Compustat data or with negative book equity is still
    # part of the market investors hold). One year of daily data at a time.
    market = pd.concat([market_return_one_year(year)
                        for year in range(FIRST_SORT_YEAR, CRSP_END_YEAR + 1)])
    factors = factors.join(market.rename("mkt"), how="left")

    # MKT-RF: subtract French's daily risk-free rate (the one-month T-bill rate, per day).
    # Every one of our trading days must have one; a missing RF would mean the calendars
    # disagree, and we want to know that rather than silently drop days.
    french = pd.read_parquet(INTERMEDIATE / "french_ff3_daily.parquet").set_index("date")
    factors = factors.join(french[["rf"]], how="left")
    if factors["rf"].isna().any():
        missing = factors.index[factors["rf"].isna()]
        raise ValueError(f"FF3Daily: no French RF for {len(missing)} of our trading days, "
                         f"e.g. {[d.date() for d in missing[:5]]}")
    factors["mkt_rf"] = factors["mkt"] - factors["rf"]

    factors = factors[["mkt_rf", "smb", "hml", "mkt", "rf"]]
    factors.index.name = "date"
    factors.to_parquet(FACTORS / "FF3Daily.parquet")
    factors.to_csv(OUTPUT / "ff3_daily_replicated.csv", float_format="%.6f")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"trading days: {len(factors):,}   dates: {factors.index.min().date()} to "
          f"{factors.index.max().date()}   missing values: {factors.isna().sum().sum()}")

    # Same trading calendar as French? Days in his file within our sample that we lack.
    in_range = french.loc[factors.index.min():factors.index.max()]
    print(f"French trading days in our sample period: {len(in_range):,}   "
          f"not in ours: {len(in_range.index.difference(factors.index))}")

    print("Full sample (annualized with 252 trading days):")
    print(summary_stats(factors[["mkt_rf", "smb", "hml"]]).to_string())

    # Average annual premium by decade, comparable with the monthly table in FF3.py.
    by_decade = (factors[["mkt_rf", "smb", "hml"]]
                 .groupby((factors.index.year // 10) * 10).mean() * TRADING_DAYS_PER_YEAR * 100)
    by_decade.index = [f"{d}s" for d in by_decade.index]
    print("Average annual premium by decade (%):")
    print(by_decade.round(1).to_string())

    # Largest daily moves: should be famous market days (e.g. 19 Oct 1987), not data errors.
    print("Five largest absolute daily market returns (%):")
    largest = factors["mkt"].abs().nlargest(5).index
    print((factors.loc[largest, ["mkt", "smb", "hml"]] * 100).round(2).to_string())

    # Quick check against the old monthly pipeline (frozen copy; full comparison in
    # Validation/CompareDailyMonthly.py). A long-short factor is a difference of
    # portfolios, so compounding daily SMB directly is not the same as monthly SMB;
    # compound the six portfolios and the market first, then build the factors.
    monthly_file = MONTHLY_PIPELINE_REF / "FF3.parquet"
    if monthly_file.exists():
        legs = compound_to_months(p[PORTFOLIO_NAMES])
        from_daily = pd.DataFrame(index=legs.index)
        from_daily["smb"] = legs[["SL", "SN", "SH"]].mean(axis=1) - legs[["BL", "BN", "BH"]].mean(axis=1)
        from_daily["hml"] = legs[["SH", "BH"]].mean(axis=1) - legs[["SL", "BL"]].mean(axis=1)
        from_daily["mkt"] = compound_to_months(factors["mkt"])
        monthly = pd.read_parquet(monthly_file)[["smb", "hml", "mkt"]]
        monthly.index = monthly.index.to_period("M")
        both = from_daily.join(monthly, how="inner", rsuffix="_monthly")
        print(f"Daily legs compounded to months vs the old monthly FF3 ({len(both)} months), "
              f"correlation:")
        print("   " + "   ".join(f"{c}: {both[c].corr(both[c + '_monthly']):.5f}"
                                 for c in ["mkt", "smb", "hml"]))


if __name__ == "__main__":
    main()
