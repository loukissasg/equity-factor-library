# ABOUTME: Tests which daily weighting matches Ken French's daily portfolios: yesterday's ME, or buy-and-hold from last month-end ME.
"""
Run:     python Validation/DailyWeightsTest.py   (from pyCode/)
Inputs:  pyData/Portfolios/FF2x3Daily_BMdec.parquet, FF2x3Daily_BMdec_assignments.parquet
         pyData/Intermediate/DailyMasterTable/DailyMasterTable_YYYY.parquet
         pyData/Intermediate/french_6_portfolios_daily.parquet, french_ff3_daily.parquet
         pyData/reference/monthly_pipeline/FF2x3_BMdec.parquet   (old monthly portfolios)

Not run by master.py: a one-off diagnostic. Decision (2026-11-22): keep yesterday's-ME
weights in the main pipeline; QXO-type share issues are a known, documented caveat.
Outputs: pyData/Portfolios/FF2x3Daily_BMdec_buyhold.parquet   daily portfolios, buy-and-hold weights
         output/daily_weights_test.csv                       both weightings vs French

Why this test: our daily portfolios weight each stock by its ME on the previous trading
day. That ME jumps on the day a firm issues shares. QXO Inc. (permno 16680) was a $48m
firm in June 2024, sorted into Small-Low; on 19 July 2024 a $4bn equity investment raised
its shares from 0.66m to 342m, and its weight in our SL jumped to 4-7% while it moved
+56%, -32%, -80% on a tiny free float. French's SL ignores those days, which suggests his
daily weights do not pick up share issues within the month.

The alternative tested here is buy-and-hold within the month:
    weight on day t = ME at the end of last month x (1 + r) on each earlier day this month
This is what an investor gets by buying the portfolio at month-end ME and not trading.
Share issues then enter the weights only at the next month-end, as in the monthly data,
and compounding the daily returns gives exactly the monthly value-weighted return.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (CRSP_END_YEAR, DAILY_MASTER_DIR, FIRST_SORT_YEAR, INTERMEDIATE,  # noqa: E402
                    MONTHLY_PIPELINE_REF, OUTPUT, PORTFOLIO_NAMES, PORTFOLIOS, make_folders)
from Portfolios.FF2x3Daily import compound_to_months  # noqa: E402

DAYS = 252


def buy_and_hold_weights(days, previous_month_end_me):
    """
    Add a column 'w_bh' to one year of firm-days: ME at the end of the previous month,
    grown by the firm's returns on the earlier days of this month.
    previous_month_end_me: Series indexed by permco, ME on the last trading day of the
    month before this year's January (from the previous year's file).
    """
    days = days.sort_values(["permco", "date"]).copy()
    days["month"] = days["date"].dt.to_period("M")

    # ME at the end of each month: the firm's last ME in that month. A firm's weight for
    # month m is its ME at the end of month m-1, as in the monthly portfolios.
    month_end = days[days["me"].notna()].groupby(["permco", "month"])["me"].last().reset_index()
    month_end["month"] = month_end["month"] + 1   # end of m-1 is the start weight of m
    month_end = month_end.rename(columns={"me": "me_start"})
    january = previous_month_end_me.rename("me_start").reset_index()
    january["month"] = pd.Period(f"{days['date'].dt.year.iloc[0]}-01", "M")
    month_start = pd.concat([january, month_end], ignore_index=True)
    days = days.merge(month_start, on=["permco", "month"], how="left")

    # Growth since the start of the month, *before* today's return: the product of
    # (1 + r) over the earlier days. A day without a return counts as 0%: CRSP books the
    # move on the next day, so nothing is lost.
    gross = 1 + days["ret"].fillna(0)
    growth_incl_today = gross.groupby([days["permco"], days["month"]]).cumprod()
    days["w_bh"] = days["me_start"] * growth_incl_today / gross

    # The ME on the last trading day of this year starts next January.
    last_me = days[days["me"].notna()].groupby("permco")["me"].last()
    last_month = days["month"].max()
    last_me = last_me[last_me.index.isin(days.loc[days["month"] == last_month, "permco"])]
    return days, last_me


def portfolio_returns(days, weight):
    """Value-weighted daily returns of the six portfolios with the given weight column."""
    days = days[days["ret"].notna() & days[weight].notna() & (days[weight] > 0)].copy()
    days["ret_x_weight"] = days["ret"] * days[weight]
    grouped = days.groupby(["date", "portfolio"])
    return (grouped["ret_x_weight"].sum() / grouped[weight].sum()).unstack()


def compare(name, ours, french, french_factors):
    """Correlation and tracking error with French: the six portfolios, SMB and HML."""
    ours = ours.copy()
    ours["smb"] = ours[["SL", "SN", "SH"]].mean(axis=1) - ours[["BL", "BN", "BH"]].mean(axis=1)
    ours["hml"] = ours[["SH", "BH"]].mean(axis=1) - ours[["SL", "BL"]].mean(axis=1)
    french = french.join(french_factors[["smb", "hml"]], how="inner")
    days = ours.index.intersection(french.index)
    rows = {}
    for col in PORTFOLIO_NAMES + ["smb", "hml"]:
        gap = ours.loc[days, col] - french.loc[days, col]
        rows[col] = {"weights": name,
                     "correlation": ours.loc[days, col].corr(french.loc[days, col]),
                     "tracking_error_%": gap.std() * DAYS ** 0.5 * 100,
                     "mean_gap_%": gap.mean() * DAYS * 100}
    return pd.DataFrame(rows).T


def main():
    make_folders()
    print("Rebuilding the daily portfolios with buy-and-hold weights...", flush=True)
    assignments = pd.read_parquet(PORTFOLIOS / "FF2x3Daily_BMdec_assignments.parquet",
                                  columns=["permco", "sort_year", "portfolio"])

    # The first year needs ME at the end of June 1963 (the start weights for July).
    previous = pd.read_parquet(DAILY_MASTER_DIR / f"DailyMasterTable_{FIRST_SORT_YEAR - 1}.parquet",
                               columns=["permco", "date", "ret", "me"])
    _, last_me = buy_and_hold_weights(previous, pd.Series(dtype=float, name="me"))

    all_returns = []
    for year in range(FIRST_SORT_YEAR, CRSP_END_YEAR + 1):
        days = pd.read_parquet(DAILY_MASTER_DIR / f"DailyMasterTable_{year}.parquet",
                               columns=["permco", "date", "ret", "me"])
        days, last_me = buy_and_hold_weights(days, last_me)

        # Same holding-period rule as FF2x3Daily.py.
        days["sort_year"] = year
        days.loc[days["date"].dt.month <= 6, "sort_year"] = year - 1
        days = days.merge(assignments, on=["permco", "sort_year"], how="inner")
        all_returns.append(portfolio_returns(days, "w_bh"))
        if year % 10 == 0:
            print(f"  {year}", flush=True)

    buyhold = pd.concat(all_returns)[PORTFOLIO_NAMES].sort_index()
    buyhold = buyhold[buyhold.index >= f"{FIRST_SORT_YEAR}-07-01"]
    buyhold.index.name = "date"
    buyhold.to_parquet(PORTFOLIOS / "FF2x3Daily_BMdec_buyhold.parquet")

    # --- Both weightings vs French ---------------------------------------------------------------
    current = pd.read_parquet(PORTFOLIOS / "FF2x3Daily_BMdec.parquet")[PORTFOLIO_NAMES]
    french_6 = pd.read_parquet(INTERMEDIATE / "french_6_portfolios_daily.parquet")
    french = french_6[french_6["table"] == "vw_return"].set_index("date")[PORTFOLIO_NAMES]
    french_factors = pd.read_parquet(INTERMEDIATE / "french_ff3_daily.parquet").set_index("date")

    print(f"\ntrading days: {len(buyhold):,}   missing: {buyhold.isna().sum().sum()}")
    result = pd.concat([compare("yesterday_me", current, french, french_factors),
                        compare("buy_and_hold", buyhold, french, french_factors)])
    result.index.name = "series"
    result.to_csv(OUTPUT / "daily_weights_test.csv", float_format="%.4f")
    table = result.reset_index().pivot(index="series", columns="weights")
    table = table.reindex(PORTFOLIO_NAMES + ["smb", "hml"]).astype(float).round(4)
    print("\nOurs vs French, daily, 1963-2025 (tracking error and mean gap in % per year):")
    print(table.to_string())

    # QXO days: does the buy-and-hold SL match French's?
    qxo = pd.DataFrame({"SL_yesterday_me": current["SL"], "SL_buy_and_hold": buyhold["SL"],
                        "SL_french": french["SL"]}).loc["2024-07-22":"2024-07-31"] * 100
    print("\nSL around QXO's share issue (daily returns in %):")
    print(qxo.round(2).to_string())

    # Buy-and-hold should compound exactly to the monthly portfolios (both use last
    # month's ME as weights); remaining gaps come from the return data, not the method.
    monthly = pd.read_parquet(MONTHLY_PIPELINE_REF / "FF2x3_BMdec.parquet")[PORTFOLIO_NAMES]
    monthly.index = monthly.index.to_period("M")
    rows = {}
    for name, daily in [("yesterday_me", current), ("buy_and_hold", buyhold)]:
        gap = compound_to_months(daily) - monthly
        rows[name] = (gap.abs().mean() * 10000).round(2)
    print("\nDaily compounded to months vs monthly FF2x3_BMdec, mean absolute gap (bp):")
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
