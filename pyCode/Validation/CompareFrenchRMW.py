# ABOUTME: Validates our profitability factors (RMW from OperProf, RMW_GP from GP) against Ken French's RMW.
"""
Run:     python Validation/CompareFrenchRMW.py   (from pyCode/)
Inputs:  pyData/Factors/RMW.parquet, RMW_GP.parquet
         pyData/Portfolios/FF2x3_OperProf.parquet, FF2x3_OperProf_holdings.parquet
         pyData/Intermediate/french_ff5.parquet, french_6_portfolios_op.parquet
Outputs: output/rmw_summary.csv                correlation, mean, std, Sharpe, regression
         output/rmw_correlation_by_period.csv  correlation in three sub-periods
         output/rmw_portfolio_match.csv        our six size x OP portfolios vs French's
         output/rmw_portfolio_contents.csv     number of firms per portfolio, every 5th July
         output/rmw_growth_of_1.png            cumulative returns: ours, GP version, French
         output/rmw_rolling_correlation.png    36-month rolling correlation with French

Two questions:
  1. Does our RMW (size x OperProf) replicate French's RMW? Same targets as FF3:
     correlation above 0.95, similar mean and Sharpe, slope near 1, intercept near 0.
  2. Does the GP version tell the same story? It is a *different* signal, so we expect a
     positive but clearly lower correlation; the point is whether the premium survives.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, PORTFOLIO_NAMES, PORTFOLIOS, make_folders  # noqa: E402
from utils.plot_style import (COLOR_EXTRA, COLOR_FRENCH, COLOR_MINE, COLOR_TEXT,  # noqa: E402
                              apply_style, plt)

SERIES = {"rmw": "RMW (OperProf)", "rmw_gp": "RMW_GP (gross profitability)"}
PERIODS = {"1963-1979": ("1963", "1979"), "1980-1999": ("1980", "1999"),
           "2000-2025": ("2000", "2025")}


def compare(mine, ff, label):
    """One row: how close is `mine` to French's RMW?"""
    regression = sm.OLS(mine, sm.add_constant(ff)).fit()  # mine = a + b * French
    return {
        "factor": label,
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
    }


def plot_growth_of_1(both, start, end):
    # Log scale: equal vertical distances are equal percentage gains.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(both.index, (1 + both["rmw_ff"]).cumprod(), color=COLOR_FRENCH, linewidth=2,
            label="Ken French RMW")
    ax.plot(both.index, (1 + both["rmw"]).cumprod(), color=COLOR_MINE, linewidth=2,
            label=SERIES["rmw"])
    ax.plot(both.index, (1 + both["rmw_gp"]).cumprod(), color=COLOR_EXTRA[0], linewidth=1.5,
            label=SERIES["rmw_gp"])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}"))
    ax.yaxis.set_minor_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:g}" if str(y)[0] in "25" else ""))
    ax.set_ylabel("Value of $1 invested (log scale)")
    ax.set_title(f"Profitability factor: growth of $1, {start:%b %Y} - {end:%b %Y}",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "rmw_growth_of_1.png", dpi=200)
    plt.close(fig)


def plot_rolling_correlation(both):
    # 36 months rather than 12: RMW is a low-volatility factor, so 12-month correlations
    # are very noisy. Spots periods where the replication drifts.
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for column, color in [("rmw", COLOR_MINE), ("rmw_gp", COLOR_EXTRA[0])]:
        rolling = both[column].rolling(36).corr(both["rmw_ff"])
        ax.plot(rolling.index, rolling, color=color, linewidth=2, label=SERIES[column])
    ax.axhline(0.95, color=COLOR_TEXT, linewidth=1, linestyle="--", label="0.95 target")
    ax.set_ylabel("Correlation with French's RMW")
    ax.set_title("Rolling 36-month correlation with Ken French's RMW", loc="left",
                 fontweight="bold")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "rmw_rolling_correlation.png", dpi=200)
    plt.close(fig)


def main():
    make_folders()
    apply_style()
    print("Comparing our profitability factors with Ken French's RMW...", flush=True)
    rmw = pd.read_parquet(FACTORS / "RMW.parquet")
    rmw_gp = pd.read_parquet(FACTORS / "RMW_GP.parquet")
    ff5 = pd.read_parquet(INTERMEDIATE / "french_ff5.parquet").set_index("date")

    both = (rmw[["rmw"]].join(rmw_gp[["rmw_gp"]], how="inner")
            .join(ff5[["rmw"]].rename(columns={"rmw": "rmw_ff"}), how="inner"))
    start, end = both.index.min(), both.index.max()
    print(f"overlap: {start.date()} to {end.date()} ({len(both)} months)")

    # --- Summary table --------------------------------------------------------------------
    summary = pd.DataFrame([compare(both[c], both["rmw_ff"], SERIES[c]) for c in SERIES])
    summary = summary.set_index("factor")
    summary.round(3).to_csv(OUTPUT / "rmw_summary.csv")
    print("\nSummary vs French's RMW (means, std and tracking error in % per year):")
    print(summary.round(3).T.to_string())
    print(f"\ncorrelation between our two versions, RMW vs RMW_GP: "
          f"{both['rmw'].corr(both['rmw_gp']):.3f}")

    # --- Correlation by sub-period -------------------------------------------------------------
    by_period = pd.DataFrame({
        name: {SERIES[c]: both.loc[a:b, c].corr(both.loc[a:b, "rmw_ff"]) for c in SERIES}
        for name, (a, b) in PERIODS.items()
    })
    by_period.round(3).to_csv(OUTPUT / "rmw_correlation_by_period.csv")
    print("\nCorrelation with French's RMW by sub-period:")
    print(by_period.round(3).to_string())

    # --- The six size x OP portfolios, one by one -------------------------------------------------
    # As for HML: RMW = 1/2 (SH + BH) - 1/2 (SL + BL), so a gap in RMW must come from one
    # of four portfolios. This shows which.
    mine_6 = pd.read_parquet(PORTFOLIOS / "FF2x3_OperProf.parquet")[PORTFOLIO_NAMES]
    french_6 = pd.read_parquet(INTERMEDIATE / "french_6_portfolios_op.parquet")
    french_returns = french_6[french_6["table"] == "vw_return"].set_index("date")[PORTFOLIO_NAMES]
    common = mine_6.index.intersection(french_returns.index)
    gap = mine_6.loc[common] - french_returns.loc[common]
    match = pd.DataFrame({
        "correlation": {p: mine_6.loc[common, p].corr(french_returns.loc[common, p])
                        for p in PORTFOLIO_NAMES},
        "tracking_error_%": gap.std() * 12 ** 0.5 * 100,
        "mean_gap_%_pa": gap.mean() * 12 * 100,
    })
    match.round(3).to_csv(OUTPUT / "rmw_portfolio_match.csv")
    print("\nOur size x OP portfolios vs French's (L = weak, H = robust profitability):")
    print(match.round(3).T.to_string())

    # Number of firms: tells us whether our universe is as large as French's.
    holdings = pd.read_parquet(PORTFOLIOS / "FF2x3_OperProf_holdings.parquet")
    mine_n = holdings.pivot(index="date", columns="portfolio", values="n_stocks")[PORTFOLIO_NAMES]
    french_n = french_6[french_6["table"] == "n_firms"].set_index("date")[PORTFOLIO_NAMES]
    julys = [d for d in mine_n.index if d.month == 7 and d.year % 5 == 0 and d in french_n.index]
    contents = pd.concat({"n_mine": mine_n.loc[julys], "n_french": french_n.loc[julys]}, axis=1)
    contents.index = contents.index.year
    contents.to_csv(OUTPUT / "rmw_portfolio_contents.csv")
    print("\nNumber of firms in July, ours vs French:")
    print(contents.astype("Int64").to_string())

    # --- Figures -----------------------------------------------------------------------------------
    plot_growth_of_1(both, start, end)
    plot_rolling_correlation(both)

    beyond = ff5[ff5.index > end]
    if len(beyond) > 0:
        print(f"\nFrench's file already has {len(beyond)} later months "
              f"({beyond.index.min():%b %Y} - {beyond.index.max():%b %Y}); our CRSP data ends "
              f"{end:%b %Y}.")


if __name__ == "__main__":
    main()
