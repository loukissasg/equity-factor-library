# ABOUTME: Compares our replicated factors with Ken French's official ones (tables and figures for the presentation).
"""
Run:     python Validation/CompareFrench.py   (from pyCode/)
Inputs:  pyData/Factors/FF3.parquet   (monthly FF3 compounded from the daily portfolios,
                                       written by Factors/FF3Monthly.py)
         pyData/Intermediate/french_ff3.parquet
Outputs: output/summary.csv                correlation, mean, std, Sharpe, regression, tracking error
         output/correlation_by_period.csv  correlation in three sub-periods
         output/recent_months.csv          last 12 months, ours vs French
         output/growth_of_1.png            cumulative returns, one panel per factor
         output/rolling_correlation.png    12-month rolling correlation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, make_folders  # noqa: E402
from utils.plot_style import COLOR_FRENCH, COLOR_MINE, COLOR_TEXT, apply_style, plt  # noqa: E402

FACTOR_NAMES = ["mkt_rf", "smb", "hml"]
LABELS = {"mkt_rf": "MKT-RF", "smb": "SMB", "hml": "HML"}


def summary_table(both):
    """One row per factor: how close is ours to French's?"""
    # correlation        do the two series move together month by month?
    # mean, std, Sharpe  same average premium and risk?
    # regression mine = a + b * official: b near 1 = same exposure; a near 0 = no
    #                    systematic gap in returns
    # tracking error     volatility of the monthly difference, in % per year
    rows = []
    for f in FACTOR_NAMES:
        mine, ff = both[f], both[f + "_ff"]
        regression = sm.OLS(mine, sm.add_constant(ff)).fit()
        rows.append({
            "factor": LABELS[f],
            "correlation": mine.corr(ff),
            "mean_mine_%": mine.mean() * 12 * 100,
            "mean_french_%": ff.mean() * 12 * 100,
            "std_mine_%": mine.std() * 12 ** 0.5 * 100,
            "std_french_%": ff.std() * 12 ** 0.5 * 100,
            "sharpe_mine": mine.mean() / mine.std() * 12 ** 0.5,
            "sharpe_french": ff.mean() / ff.std() * 12 ** 0.5,
            "slope": regression.params.iloc[1],
            "intercept_annual_%": regression.params.iloc[0] * 12 * 100,
            "intercept_tstat": regression.tvalues.iloc[0],
            "r_squared": regression.rsquared,
            "tracking_error_%": (mine - ff).std() * 12 ** 0.5 * 100,
        })
    return pd.DataFrame(rows).set_index("factor")


def plot_growth_of_1(both, start, end):
    # Compounding shows whether small monthly differences add up to a different long-run
    # picture. Log scale: equal vertical distances are equal percentage gains.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, f in zip(axes, FACTOR_NAMES):
        growth_mine = (1 + both[f]).cumprod()
        growth_ff = (1 + both[f + "_ff"]).cumprod()
        ax.plot(growth_ff.index, growth_ff, color=COLOR_FRENCH, linewidth=2, label="Ken French")
        ax.plot(growth_mine.index, growth_mine, color=COLOR_MINE, linewidth=2,
                label="Our replication")
        ax.set_yscale("log")
        # Plain dollars ($1, $2, $5, ...) instead of powers of ten.
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}"))
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(
            lambda y, _: f"${y:g}" if str(y)[0] in "25" else ""))
        ax.set_title(f"{LABELS[f]}: growth of $1", loc="left", fontweight="bold")
        ax.set_ylabel("Value of $1 invested (log scale)")
        ax.legend(loc="upper left")
    fig.suptitle(f"Replicated vs official Fama-French factors, {start:%b %Y} - {end:%b %Y}",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "growth_of_1.png", dpi=200)
    plt.close(fig)


def plot_rolling_correlation(both):
    # Spots specific periods where the replication drifts away from the official series.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, f in zip(axes, FACTOR_NAMES):
        rolling = both[f].rolling(12).corr(both[f + "_ff"])
        ax.plot(rolling.index, rolling, color=COLOR_MINE, linewidth=2,
                label="12-month rolling correlation")
        ax.axhline(0.95, color=COLOR_TEXT, linewidth=1, linestyle="--", label="0.95 target")
        ax.set_title(LABELS[f], loc="left", fontweight="bold")
    axes[0].legend(loc="lower left")  # same two lines in every panel; MKT-RF has free space
    axes[0].set_ylabel("Correlation with Ken French")
    fig.suptitle("Rolling 12-month correlation, our replication vs Ken French",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "rolling_correlation.png", dpi=200)
    plt.close(fig)


def main():
    make_folders()
    apply_style()
    print("Comparing with Ken French's factors...", flush=True)
    factors = pd.read_parquet(FACTORS / "FF3.parquet")
    official = pd.read_parquet(INTERMEDIATE / "french_ff3.parquet").set_index("date")[FACTOR_NAMES]

    both = factors[FACTOR_NAMES].join(official, rsuffix="_ff", how="inner")
    start, end = both.index.min(), both.index.max()
    print(f"overlap: {start.date()} to {end.date()} ({len(both)} months)")

    # --- Summary table --------------------------------------------------------------------
    summary = summary_table(both)
    summary.round(3).to_csv(OUTPUT / "summary.csv")
    print("\nSummary (means, std and tracking error in % per year):")
    print(summary.round(3).T.to_string())

    # --- Correlation by sub-period -------------------------------------------------------------
    # Shows *when* the replication is weaker (we expect the 1960s, where Fama-French use
    # hand-collected book equity that is not in Compustat).
    periods = {"1963-1979": ("1963", "1979"), "1980-1999": ("1980", "1999"),
               "2000-2025": ("2000", "2025")}
    by_period = pd.DataFrame({
        name: {LABELS[f]: both.loc[a:b, f].corr(both.loc[a:b, f + "_ff"]) for f in FACTOR_NAMES}
        for name, (a, b) in periods.items()
    })
    by_period.round(3).to_csv(OUTPUT / "correlation_by_period.csv")
    print("\nCorrelation by sub-period:")
    print(by_period.round(3).to_string())

    # --- Figures ---------------------------------------------------------------------------------
    plot_growth_of_1(both, start, end)
    plot_rolling_correlation(both)

    # --- The most recent months ------------------------------------------------------------------
    # Part 2 of the project: connect the factors to live results. Our sample ends with the
    # latest CRSP release; Ken French's file may already include later months.
    recent = both.tail(12).copy()
    recent = recent[[c for f in FACTOR_NAMES for c in (f, f + "_ff")]] * 100  # in %
    recent.index = recent.index.strftime("%Y-%m")
    recent.round(2).to_csv(OUTPUT / "recent_months.csv")
    print("\nLast 12 months of overlap (monthly returns in %, '_ff' = French):")
    print(recent.round(2).to_string())

    beyond = official[official.index > end]
    if len(beyond) > 0:
        print(f"\nKen French's file already has {len(beyond)} later months "
              f"({beyond.index.min():%b %Y} - {beyond.index.max():%b %Y}) that our CRSP data "
              f"does not cover yet. They appear after the next CRSP update "
              f"(REFRESH_DOWNLOADS = True in config.py).")


if __name__ == "__main__":
    main()
