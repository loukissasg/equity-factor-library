# ABOUTME: Explains the gap between our HML and Ken French's by comparing the six 2x3 portfolios one by one.
"""
Run:     python Validation/HMLGapDiagnostics.py   (from pyCode/)
Inputs:  pyData/Portfolios/FF2x3_BMdec.parquet, FF2x3_BMdec_holdings.parquet,
         FF2x3_BMdec_assignments.parquet   (the size x B/M sort behind HML)
         pyData/Intermediate/french_6_portfolios.parquet, crsp_names.parquet
Outputs: output/portfolio_match.csv          correlation, tracking error, return gap per portfolio
         output/hml_gap_decomposition.csv    HML gap split into its four legs
         output/hml_gap_by_portfolio.png     cumulative contribution of each leg
         output/portfolio_contents.csv       number of firms and average size, ours vs French
         output/bh_concentration.csv         how concentrated big value (BH) is
         output/bh_borderline_giants.csv     largest big firms near the BH/BN cutoff

HML = 1/2 (SH + BH) - 1/2 (SL + BL), so the HML gap (ours - French) splits exactly into
the return gaps of four portfolios:
    HML gap = 1/2 gap(SH) + 1/2 gap(BH) - 1/2 gap(SL) - 1/2 gap(BL)
Comparing each portfolio with French's tells us which one drives the difference; the
portfolio contents tell us why.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, OUTPUT, PORTFOLIO_NAMES, PORTFOLIOS, make_folders  # noqa: E402
from utils.plot_style import (COLOR_EXTRA, COLOR_INK, COLOR_MINE, COLOR_TEXT,  # noqa: E402
                              apply_style, plt)

# 2007 splits the sample where the HML gap becomes visible on the growth-of-$1 chart.
GAP_PERIODS = {"1963-2025": ("1963", "2025"), "1963-2006": ("1963", "2006"),
               "2007-2025": ("2007", "2025")}
NEAR_CUTOFF_PCT = 10  # "near the cutoff" = B/M within +/-10% of the 70th percentile


def portfolio_match(mine_6, french_6_returns, gap_6):
    rows = []
    for period, (a, b) in GAP_PERIODS.items():
        for port in PORTFOLIO_NAMES:
            rows.append({
                "period": period,
                "portfolio": port,
                "correlation": mine_6.loc[a:b, port].corr(french_6_returns.loc[a:b, port]),
                "tracking_error_%": gap_6.loc[a:b, port].std() * 12 ** 0.5 * 100,
                "mean_gap_%_pa": gap_6.loc[a:b, port].mean() * 12 * 100,
            })
    return pd.DataFrame(rows)


def plot_leg_contributions(contributions):
    # How each leg's contribution adds up over time (sum of monthly gaps, in percentage
    # points). The leg that drifts furthest from zero is the culprit.
    leg_colors = {"SH (+1/2)": COLOR_EXTRA[0], "BH (+1/2)": COLOR_MINE,
                  "SL (-1/2)": COLOR_EXTRA[1], "BL (-1/2)": COLOR_EXTRA[2]}
    cumulative = contributions.cumsum() * 100
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0, color=COLOR_TEXT, linewidth=1)
    for leg, color in leg_colors.items():
        ax.plot(cumulative.index, cumulative[leg], color=color, linewidth=2, label=leg)
        ax.annotate(leg, xy=(cumulative.index[-1], cumulative[leg].iloc[-1]),
                    xytext=(6, 0), textcoords="offset points", va="center",
                    color=COLOR_TEXT, fontsize=9)
    ax.plot(cumulative.index, cumulative.sum(axis=1), color=COLOR_INK, linewidth=2,
            linestyle="--", label="Total HML gap")
    ax.set_ylabel("Cumulative gap, ours - French (percentage points)")
    ax.set_title("Where the HML gap comes from: cumulative contribution of each portfolio",
                 loc="left", fontweight="bold")
    ax.legend(loc="lower left")
    ax.set_xlim(right=cumulative.index[-1] + pd.DateOffset(years=5))  # room for end labels
    fig.tight_layout()
    fig.savefig(OUTPUT / "hml_gap_by_portfolio.png", dpi=200)
    plt.close(fig)


def main():
    make_folders()
    apply_style()
    print("Diagnosing the HML gap, portfolio by portfolio...", flush=True)
    portfolio_returns = pd.read_parquet(PORTFOLIOS / "FF2x3_BMdec.parquet")[PORTFOLIO_NAMES]
    holdings = pd.read_parquet(PORTFOLIOS / "FF2x3_BMdec_holdings.parquet")
    # The generic sort stores the signal as "signal" and size as "Size"; rename to B/M terms.
    assignments = pd.read_parquet(PORTFOLIOS / "FF2x3_BMdec_assignments.parquet").rename(
        columns={"signal": "bm", "signal_70": "bm_70", "Size": "me_june"})
    french_6 = pd.read_parquet(INTERMEDIATE / "french_6_portfolios.parquet")
    names = pd.read_parquet(INTERMEDIATE / "crsp_names.parquet")

    french_6_returns = french_6[french_6["table"] == "vw_return"].set_index("date")[PORTFOLIO_NAMES]
    common_dates = portfolio_returns.index.intersection(french_6_returns.index)
    mine_6 = portfolio_returns.loc[common_dates]
    gap_6 = mine_6 - french_6_returns.loc[common_dates]

    # --- (1) How well does each portfolio match? ---------------------------------------------
    match = portfolio_match(mine_6, french_6_returns, gap_6)
    match.round(3).to_csv(OUTPUT / "portfolio_match.csv", index=False)
    print("\nPortfolio match (correlation / tracking error % p.a. / mean return gap % p.a.):")
    print(match.pivot(index="portfolio", columns="period",
                      values=["correlation", "tracking_error_%", "mean_gap_%_pa"])
          .loc[PORTFOLIO_NAMES].round(3).to_string())

    # --- (2) Split the HML gap into the four legs ------------------------------------------------
    contributions = pd.DataFrame({
        "SH (+1/2)": 0.5 * gap_6["SH"],
        "BH (+1/2)": 0.5 * gap_6["BH"],
        "SL (-1/2)": -0.5 * gap_6["SL"],
        "BL (-1/2)": -0.5 * gap_6["BL"],
    })
    decomposition = pd.DataFrame({
        period: contributions.loc[a:b].mean() * 12 * 100 for period, (a, b) in GAP_PERIODS.items()
    }).T
    decomposition["total"] = decomposition.sum(axis=1)
    decomposition.round(3).to_csv(OUTPUT / "hml_gap_decomposition.csv")
    print("\nContribution of each leg to the HML gap, ours - French (% p.a.):")
    print(decomposition.round(2).to_string())
    plot_leg_contributions(contributions)

    # --- (3) Portfolio contents: number of firms and average size, in July --------------------------
    french_n = french_6[french_6["table"] == "n_firms"].set_index("date")[PORTFOLIO_NAMES]
    french_me = french_6[french_6["table"] == "avg_me"].set_index("date")[PORTFOLIO_NAMES]
    mine_n = holdings.pivot(index="date", columns="portfolio", values="n_stocks")[PORTFOLIO_NAMES]
    mine_me = holdings.pivot(index="date", columns="portfolio",
                             values="avg_me_lag")[PORTFOLIO_NAMES] / 1000  # $ millions

    julys = [d for d in mine_n.index if d.month == 7 and d.year % 5 == 0 and d in french_n.index]
    contents = pd.concat({
        "n_mine": mine_n.loc[julys], "n_french": french_n.loc[julys],
        "avg_me_mine_$m": mine_me.loc[julys], "avg_me_french_$m": french_me.loc[julys],
    }, axis=1)
    contents.index = contents.index.year
    contents.round(0).to_csv(OUTPUT / "portfolio_contents.csv")
    print("\nBig-firm portfolios in July: number of firms and average ME ($m), ours vs French:")
    print(contents.loc[:, (slice(None), ["BN", "BH"])].round(0).astype(int).to_string())

    # --- (4) Why BH is fragile: concentration, and value sitting near the B/M cutoff ------------------
    big_firms = assignments[assignments["size_group"] == "B"].copy()
    big_firms["distance_to_cutoff_%"] = (big_firms["bm"] / big_firms["bm_70"] - 1) * 100
    bh_firms = big_firms[big_firms["portfolio"] == "BH"]

    bh_value = bh_firms.groupby("sort_year")["me_june"].sum()
    top5_value = (bh_firms.sort_values("me_june", ascending=False)
                  .groupby("sort_year")["me_june"].apply(lambda s: s.head(5).sum()))
    near_cutoff = big_firms[big_firms["distance_to_cutoff_%"].abs() <= NEAR_CUTOFF_PCT]
    near_value = near_cutoff.groupby("sort_year")["me_june"].sum()

    concentration = pd.DataFrame({
        "bh_firms": bh_firms.groupby("sort_year").size(),
        "top5_share_of_bh_value": top5_value / bh_value,
        # Value of big firms near the cutoff, relative to BH's total value. Above ~0.5, a
        # small change in book equity can reshuffle much of the portfolio.
        "value_near_cutoff_vs_bh": near_value / bh_value,
    })
    concentration.round(3).to_csv(OUTPUT / "bh_concentration.csv")
    print("\nBH concentration, every 5th June:")
    print(concentration[concentration.index % 5 == 0].round(2).to_string())

    # --- (5) The borderline giants: largest big firms near the cutoff each June ----------------------
    borderline = near_cutoff.merge(names, on="permno", how="left")
    borderline["me_june_$bn"] = borderline["me_june"] / 1e6  # $ thousands -> $ billions
    borderline = (borderline.sort_values(["sort_year", "me_june"], ascending=[True, False])
                  .groupby("sort_year").head(5))
    borderline = borderline[["sort_year", "ticker", "issuernm", "portfolio", "me_june_$bn",
                             "bm", "bm_70", "distance_to_cutoff_%"]]
    borderline.round(2).to_csv(OUTPUT / "bh_borderline_giants.csv", index=False)
    for year in [2008, 2020, 2023, borderline["sort_year"].max()]:
        print(f"\nJune {year}: largest big firms within {NEAR_CUTOFF_PCT}% of the BH/BN cutoff")
        print(borderline[borderline["sort_year"] == year].drop(columns="sort_year")
              .round(2).to_string(index=False))


if __name__ == "__main__":
    main()
