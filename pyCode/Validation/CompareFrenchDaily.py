# ABOUTME: Compares our daily factors and daily 2x3 portfolios with Ken French's daily files (tables and figures).
"""
Run:     python Validation/CompareFrenchDaily.py   (from pyCode/)
Inputs:  pyData/Factors/FF3Daily.parquet
         pyData/Portfolios/FF2x3Daily_BMdec.parquet, FF2x3Daily_BMdec_holdings.parquet
         pyData/Intermediate/french_ff3_daily.parquet
         pyData/Intermediate/french_6_portfolios_daily.parquet
Outputs: output/daily_summary.csv                correlation, mean, std, Sharpe, regression, tracking error
         output/daily_correlation_by_period.csv  correlation in three sub-periods
         output/daily_portfolio_match.csv        the six portfolios, ours vs French
         output/daily_largest_hml_gaps.csv       the 10 days where our HML misses French's most
         output/daily_recent_days.csv            our last month of data, ours vs French
         output/daily_growth_of_1.png            cumulative returns, one panel per factor
         output/daily_rolling_correlation.png    one-year (252-day) rolling correlation

The daily counterpart of CompareFrench.py. Everything is compared day against day:
annualized with 252 trading days, never mixed with monthly numbers (daily x 252 is not
the same as monthly x 12, see CompareDailyMonthly.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, PORTFOLIO_NAMES, PORTFOLIOS, make_folders  # noqa: E402
from utils.plot_style import COLOR_FRENCH, COLOR_MINE, COLOR_TEXT, apply_style, plt  # noqa: E402

FACTOR_NAMES = ["mkt_rf", "smb", "hml"]
LABELS = {"mkt_rf": "MKT-RF", "smb": "SMB", "hml": "HML"}
DAYS = 252  # trading days per year, to annualize


def summary_table(both):
    """One row per factor: how close is ours to French's, day by day?"""
    # Same measures as the monthly CompareFrench.py, annualized with 252 days.
    # The regression's standard errors are Newey-West (HAC, 5 lags = one trading week):
    # daily differences between two series can be slightly autocorrelated (a stock with a
    # stale price catches up the next day), which makes plain OLS t-stats too large.
    rows = []
    for f in FACTOR_NAMES:
        mine, ff = both[f], both[f + "_ff"]
        regression = sm.OLS(mine, sm.add_constant(ff)).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({
            "factor": LABELS[f],
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
            "r_squared": regression.rsquared,
            "tracking_error_%": (mine - ff).std() * DAYS ** 0.5 * 100,
        })
    return pd.DataFrame(rows).set_index("factor")


def portfolio_table(ours, french, our_counts, french_counts):
    """The six portfolios, ours vs French: which one does a factor gap come from?"""
    rows = []
    for name in PORTFOLIO_NAMES:
        gap = ours[name] - french[name]
        # HML = 1/2 (SH + BH) - 1/2 (SL + BL), so a portfolio's gap moves HML's gap by
        # +1/2 (SH, BH), -1/2 (SL, BL) or 0 (SN, BN). Same logic for SMB with 1/3.
        hml_weight = {"SH": 0.5, "BH": 0.5, "SL": -0.5, "BL": -0.5}.get(name, 0.0)
        smb_weight = 1 / 3 if name.startswith("S") else -1 / 3
        rows.append({
            "portfolio": name,
            "correlation": ours[name].corr(french[name]),
            "mean_gap_%": gap.mean() * DAYS * 100,
            "tracking_error_%": gap.std() * DAYS ** 0.5 * 100,
            "contribution_to_smb_gap_%": smb_weight * gap.mean() * DAYS * 100,
            "contribution_to_hml_gap_%": hml_weight * gap.mean() * DAYS * 100,
            "avg_stocks_mine": our_counts[name].mean(),
            "avg_stocks_french": french_counts[name].mean(),
        })
    return pd.DataFrame(rows).set_index("portfolio")


def plot_growth_of_1(both, start, end):
    # Compounding shows whether small daily differences add up over 62 years.
    # Log scale: equal vertical distances are equal percentage gains.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, f in zip(axes, FACTOR_NAMES):
        growth_mine = (1 + both[f]).cumprod()
        growth_ff = (1 + both[f + "_ff"]).cumprod()
        ax.plot(growth_ff.index, growth_ff, color=COLOR_FRENCH, linewidth=2, label="Ken French (daily)")
        ax.plot(growth_mine.index, growth_mine, color=COLOR_MINE, linewidth=2,
                label="Our replication (daily)")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}"))
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(
            lambda y, _: f"${y:g}" if str(y)[0] in "25" else ""))
        ax.set_title(f"{LABELS[f]}: growth of $1", loc="left", fontweight="bold")
        ax.set_ylabel("Value of $1 invested (log scale)")
    # Same two lines in every panel; one legend where it covers no data (MKT-RF's top left).
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Daily factors, replicated vs official, {start:%d %b %Y} - {end:%d %b %Y}",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "daily_growth_of_1.png", dpi=200)
    plt.close(fig)


def plot_rolling_correlation(both):
    # One trading year (252 days) per window: spots periods where the replication drifts.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, f in zip(axes, FACTOR_NAMES):
        rolling = both[f].rolling(DAYS).corr(both[f + "_ff"])
        ax.plot(rolling.index, rolling, color=COLOR_MINE, linewidth=1.5,
                label="1-year (252-day) rolling correlation")
        ax.axhline(0.95, color=COLOR_TEXT, linewidth=1, linestyle="--", label="0.95 target")
        ax.set_title(LABELS[f], loc="left", fontweight="bold")
    axes[0].legend(loc="lower left")
    axes[0].set_ylabel("Correlation with Ken French")
    fig.suptitle("Rolling one-year correlation of daily factors, our replication vs Ken French",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "daily_rolling_correlation.png", dpi=200)
    plt.close(fig)


def main():
    make_folders()
    apply_style()
    print("Comparing our daily factors with Ken French's daily factors...", flush=True)
    factors = pd.read_parquet(FACTORS / "FF3Daily.parquet")
    official = pd.read_parquet(INTERMEDIATE / "french_ff3_daily.parquet").set_index("date")[FACTOR_NAMES]

    both = factors[FACTOR_NAMES].join(official, rsuffix="_ff", how="inner")
    start, end = both.index.min(), both.index.max()
    print(f"overlap: {start.date()} to {end.date()} ({len(both):,} trading days; "
          f"ours has {len(factors):,})")

    # --- Summary table --------------------------------------------------------------------
    summary = summary_table(both)
    summary.round(3).to_csv(OUTPUT / "daily_summary.csv")
    print("\nSummary (means, std and tracking error in % per year, 252 days):")
    print(summary.round(3).T.to_string())

    # --- Correlation by sub-period -------------------------------------------------------------
    periods = {"1963-1979": ("1963", "1979"), "1980-1999": ("1980", "1999"),
               "2000-2025": ("2000", "2025")}
    by_period = pd.DataFrame({
        name: {LABELS[f]: both.loc[a:b, f].corr(both.loc[a:b, f + "_ff"]) for f in FACTOR_NAMES}
        for name, (a, b) in periods.items()
    })
    by_period.round(3).to_csv(OUTPUT / "daily_correlation_by_period.csv")
    print("\nCorrelation by sub-period:")
    print(by_period.round(3).to_string())

    # --- The six portfolios --------------------------------------------------------------------
    # As in the monthly HMLGapDiagnostics.py: if a factor gap comes from one portfolio,
    # it shows up here.
    ours = pd.read_parquet(PORTFOLIOS / "FF2x3Daily_BMdec.parquet")[PORTFOLIO_NAMES]
    french_6 = pd.read_parquet(INTERMEDIATE / "french_6_portfolios_daily.parquet")
    french_returns = french_6[french_6["table"] == "vw_return"].set_index("date")[PORTFOLIO_NAMES]
    french_counts = french_6[french_6["table"] == "n_firms"].set_index("date")[PORTFOLIO_NAMES]
    holdings = pd.read_parquet(PORTFOLIOS / "FF2x3Daily_BMdec_holdings.parquet")
    our_counts = holdings.pivot(index="date", columns="portfolio", values="n_stocks")[PORTFOLIO_NAMES]

    days = ours.index.intersection(french_returns.index)
    portfolios = portfolio_table(ours.loc[days], french_returns.loc[days],
                                 our_counts.loc[days], french_counts.loc[days])
    portfolios.round(3).to_csv(OUTPUT / "daily_portfolio_match.csv")
    print(f"\nThe six portfolios, ours vs French ({len(days):,} days; gaps in % per year):")
    print(portfolios.round(3).to_string())
    print(f"sum of contributions: SMB gap {portfolios['contribution_to_smb_gap_%'].sum():.3f}%, "
          f"HML gap {portfolios['contribution_to_hml_gap_%'].sum():.3f}% "
          f"(= mean_mine - mean_french in the summary)")

    # --- The worst days for HML -----------------------------------------------------------------
    # Which portfolio drives the biggest daily misses? (Monthly: it was BH, see TEAM_NOTES.)
    gaps = (ours.loc[days] - french_returns.loc[days]) * 100
    gaps["hml_gap"] = (both.loc[days, "hml"] - both.loc[days, "hml_ff"]) * 100
    worst = gaps.loc[gaps["hml_gap"].abs().nlargest(10).index].sort_index()
    worst["largest_leg"] = worst[PORTFOLIO_NAMES].abs().idxmax(axis=1)
    worst.index = worst.index.date
    worst.round(2).to_csv(OUTPUT / "daily_largest_hml_gaps.csv", index_label="date")
    print("\nThe 10 days with the largest HML gap (ours - French, in %), and each portfolio's gap:")
    print(worst.round(2).to_string())

    # --- Figures ---------------------------------------------------------------------------------
    plot_growth_of_1(both, start, end)
    plot_rolling_correlation(both)

    # --- The most recent days ----------------------------------------------------------------------
    # Part 2: connect to live results. Our data ends with the latest CRSP annual release.
    last_month = both.loc[both.index >= end.replace(day=1)]
    recent = last_month[[c for f in FACTOR_NAMES for c in (f, f + "_ff")]] * 100  # in %
    recent.index = recent.index.date
    recent.round(2).to_csv(OUTPUT / "daily_recent_days.csv", index_label="date")
    print(f"\nOur last month of data, {end:%B %Y} (daily returns in %, '_ff' = French):")
    print(recent.round(2).to_string())

    beyond = official[official.index > end]
    if len(beyond) > 0:
        print(f"\nKen French's daily file already has {len(beyond)} later trading days "
              f"({beyond.index.min():%d %b %Y} - {beyond.index.max():%d %b %Y}) that our CRSP "
              f"data does not cover yet. They appear after the next CRSP update "
              f"(raise CRSP_END_YEAR and rerun the daily download).")


if __name__ == "__main__":
    main()
