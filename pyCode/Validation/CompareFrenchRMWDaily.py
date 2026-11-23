# ABOUTME: Validates our daily profitability factors (RMW, RMW_GP) and the six daily size x OP portfolios against Ken French's daily files.
"""
Run:     python Validation/CompareFrenchRMWDaily.py   (from pyCode/)
Inputs:  pyData/Factors/RMWDaily.parquet, RMW_GPDaily.parquet, RMW.parquet
         pyData/Portfolios/FF2x3Daily_OperProf.parquet, FF2x3Daily_OperProf_holdings.parquet
         pyData/Intermediate/french_ff5_daily.parquet, french_6_portfolios_op_daily.parquet,
                             french_ff5.parquet (monthly)
         pyData/reference/monthly_pipeline/RMW.parquet   (old monthly pipeline, msf_v2)
Outputs: output/rmw_daily_summary.csv          daily RMW and RMW_GP vs French's daily RMW
         output/rmw_daily_portfolio_match.csv  the six daily size x OP portfolios vs French's
         output/rmw_daily_largest_gaps.csv     the 10 days where our RMW misses French's most
         output/rmw_daily_vs_monthly.csv       monthly RMW from daily vs old monthly vs French
         output/rmw_daily_growth_of_1.png      cumulative daily returns, ours vs French
         output/rmw_daily_rolling_correlation.png   one-year rolling correlation

Same questions as CompareFrenchDaily.py for FF3, applied to RMW:
  1. Day by day, does our RMW move with French's? (correlation, regression, tracking error)
  2. Which of the six portfolios does a gap come from?
  3. Compounded to months, does the daily RMW match the old monthly RMW and French's?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config import (FACTORS, INTERMEDIATE, MONTHLY_PIPELINE_REF, OUTPUT,  # noqa: E402
                    PORTFOLIO_NAMES, PORTFOLIOS, make_folders)
from utils.plot_style import COLOR_EXTRA, COLOR_FRENCH, COLOR_MINE, COLOR_TEXT, apply_style, plt  # noqa: E402

DAYS = 252
SERIES = {"rmw": "RMW (OperProf)", "rmw_gp": "RMW_GP (gross profitability)"}


def compare(mine, ff, label):
    """One row: how close is a daily series to French's daily RMW?"""
    # Newey-West (5 lags) standard errors, as in CompareFrenchDaily.py: daily gaps can be
    # slightly autocorrelated, which makes plain OLS t-stats too large.
    regression = sm.OLS(mine, sm.add_constant(ff)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    return {
        "factor": label,
        "correlation": mine.corr(ff),
        "mean_mine_%": mine.mean() * DAYS * 100,
        "mean_french_%": ff.mean() * DAYS * 100,
        "std_mine_%": mine.std() * DAYS ** 0.5 * 100,
        "std_french_%": ff.std() * DAYS ** 0.5 * 100,
        "sharpe_mine": mine.mean() / mine.std() * DAYS ** 0.5,
        "sharpe_french": ff.mean() / ff.std() * DAYS ** 0.5,
        "slope": regression.params.iloc[1],
        "intercept_annual_%": regression.params.iloc[0] * DAYS * 100,
        "intercept_tstat_hac": regression.tvalues.iloc[0],
        "tracking_error_%": (mine - ff).std() * DAYS ** 0.5 * 100,
    }


def main():
    make_folders()
    apply_style()
    print("Comparing our daily profitability factors with Ken French's daily RMW...", flush=True)
    rmw = pd.read_parquet(FACTORS / "RMWDaily.parquet")
    rmw_gp = pd.read_parquet(FACTORS / "RMW_GPDaily.parquet")
    ff5 = pd.read_parquet(INTERMEDIATE / "french_ff5_daily.parquet").set_index("date")

    both = (rmw[["rmw"]].join(rmw_gp[["rmw_gp"]], how="inner")
            .join(ff5[["rmw"]].rename(columns={"rmw": "rmw_ff"}), how="inner"))
    start, end = both.index.min(), both.index.max()
    print(f"overlap: {start.date()} to {end.date()} ({len(both):,} trading days; ours has {len(rmw):,})")

    # --- 1. Summary --------------------------------------------------------------------------
    summary = pd.DataFrame([compare(both[c], both["rmw_ff"], SERIES[c]) for c in SERIES]).set_index("factor")
    summary.round(3).to_csv(OUTPUT / "rmw_daily_summary.csv")
    print("\nDaily, vs French's daily RMW (means, std, tracking error in % per year, 252 days):")
    print(summary.round(3).T.to_string())
    periods = {"1963-1979": ("1963", "1979"), "1980-1999": ("1980", "1999"), "2000-2025": ("2000", "2025")}
    print("Correlation with French by sub-period:")
    print(pd.DataFrame({name: {SERIES[c]: both.loc[a:b, c].corr(both.loc[a:b, "rmw_ff"]) for c in SERIES}
                        for name, (a, b) in periods.items()}).round(3).to_string())

    # --- 2. The six portfolios ----------------------------------------------------------------
    ours = pd.read_parquet(PORTFOLIOS / "FF2x3Daily_OperProf.parquet")[PORTFOLIO_NAMES]
    french_6 = pd.read_parquet(INTERMEDIATE / "french_6_portfolios_op_daily.parquet")
    french = french_6[french_6["table"] == "vw_return"].set_index("date")[PORTFOLIO_NAMES]
    french_n = french_6[french_6["table"] == "n_firms"].set_index("date")[PORTFOLIO_NAMES]
    holdings = pd.read_parquet(PORTFOLIOS / "FF2x3Daily_OperProf_holdings.parquet")
    our_n = holdings.pivot(index="date", columns="portfolio", values="n_stocks")[PORTFOLIO_NAMES]
    days = ours.index.intersection(french.index)
    gap = ours.loc[days] - french.loc[days]
    # RMW = 1/2 (SH + BH) - 1/2 (SL + BL): each portfolio's gap moves RMW's gap by +-1/2 or 0.
    rmw_weight = {"SH": 0.5, "BH": 0.5, "SL": -0.5, "BL": -0.5}
    match = pd.DataFrame({
        "correlation": {p: ours.loc[days, p].corr(french.loc[days, p]) for p in PORTFOLIO_NAMES},
        "mean_gap_%": gap.mean() * DAYS * 100,
        "tracking_error_%": gap.std() * DAYS ** 0.5 * 100,
        "contribution_to_rmw_gap_%": {p: rmw_weight.get(p, 0.0) * gap[p].mean() * DAYS * 100
                                      for p in PORTFOLIO_NAMES},
        "avg_stocks_mine": our_n.loc[days].mean(),
        "avg_stocks_french": french_n.loc[days].mean(),
    })
    match.round(3).to_csv(OUTPUT / "rmw_daily_portfolio_match.csv")
    print(f"\nThe six size x OP portfolios, ours vs French ({len(days):,} days; L = weak, H = robust):")
    print(match.round(3).to_string())

    gaps = gap * 100
    gaps["rmw_gap"] = (both.loc[days, "rmw"] - both.loc[days, "rmw_ff"]) * 100
    worst = gaps.loc[gaps["rmw_gap"].abs().nlargest(10).index].sort_index()
    worst["largest_leg"] = worst[PORTFOLIO_NAMES].abs().idxmax(axis=1)
    worst.index = worst.index.date
    worst.round(2).to_csv(OUTPUT / "rmw_daily_largest_gaps.csv", index_label="date")
    print("\nThe 10 days with the largest RMW gap (ours - French, %), and each portfolio's gap:")
    print(worst.round(2).to_string())

    # --- 3. Daily compounded to months vs monthly ----------------------------------------------
    monthly = pd.read_parquet(FACTORS / "RMW.parquet")[["rmw"]]
    old = pd.read_parquet(MONTHLY_PIPELINE_REF / "RMW.parquet")[["rmw"]]
    french_monthly = pd.read_parquet(INTERMEDIATE / "french_ff5.parquet").set_index("date")[["rmw"]]
    for df in (monthly, old, french_monthly):
        df.index = df.index.to_period("M")
    months = monthly.index.intersection(old.index).intersection(french_monthly.index)
    pairs = {
        "from daily vs old monthly pipeline": (monthly, old),
        "from daily vs French monthly": (monthly, french_monthly),
        "old monthly pipeline vs French monthly": (old, french_monthly),
    }
    table = pd.DataFrame({name: {
        "correlation": a.loc[months, "rmw"].corr(b.loc[months, "rmw"]),
        "tracking_error_%": (a.loc[months, "rmw"] - b.loc[months, "rmw"]).std() * 12 ** 0.5 * 100,
        "mean_first_%": a.loc[months, "rmw"].mean() * 12 * 100,
        "mean_second_%": b.loc[months, "rmw"].mean() * 12 * 100,
    } for name, (a, b) in pairs.items()}).T
    table.round(3).to_csv(OUTPUT / "rmw_daily_vs_monthly.csv")
    print(f"\nMonthly RMW ({len(months)} months): daily compounded vs old monthly vs French:")
    print(table.round(3).to_string())

    # --- Figures -----------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(both.index, (1 + both["rmw_ff"]).cumprod(), color=COLOR_FRENCH, linewidth=2,
            label="Ken French RMW (daily)")
    ax.plot(both.index, (1 + both["rmw"]).cumprod(), color=COLOR_MINE, linewidth=2,
            label=SERIES["rmw"] + " (daily)")
    ax.plot(both.index, (1 + both["rmw_gp"]).cumprod(), color=COLOR_EXTRA[0], linewidth=1.5,
            label=SERIES["rmw_gp"] + " (daily)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}"))
    ax.yaxis.set_minor_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:g}" if str(y)[0] in "25" else ""))
    ax.set_ylabel("Value of $1 invested (log scale)")
    ax.set_title(f"Profitability factor, daily: growth of $1, {start:%d %b %Y} - {end:%d %b %Y}",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "rmw_daily_growth_of_1.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for column, color in [("rmw", COLOR_MINE), ("rmw_gp", COLOR_EXTRA[0])]:
        rolling = both[column].rolling(DAYS).corr(both["rmw_ff"])
        ax.plot(rolling.index, rolling, color=color, linewidth=1.5, label=SERIES[column])
    ax.axhline(0.95, color=COLOR_TEXT, linewidth=1, linestyle="--", label="0.95 target")
    ax.set_ylabel("Correlation with French's RMW")
    ax.set_title("Rolling one-year (252-day) correlation with Ken French's daily RMW",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "rmw_daily_rolling_correlation.png", dpi=200)
    plt.close(fig)

    beyond = ff5[ff5.index > end]
    if len(beyond) > 0:
        print(f"\nFrench's daily file already has {len(beyond)} later trading days "
              f"({beyond.index.min():%d %b %Y} - {beyond.index.max():%d %b %Y}).")


if __name__ == "__main__":
    main()
