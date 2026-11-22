# ABOUTME: Daily vs monthly FF3: compounds the daily factors to months and compares with our monthly FF3 and French's monthly file.
"""
Run:     python Validation/CompareDailyMonthly.py   (from pyCode/)
Inputs:  pyData/Portfolios/FF2x3Daily_BMdec.parquet   our daily six portfolios
         pyData/Factors/FF3Daily.parquet               our daily market return
         pyData/reference/monthly_pipeline/FF3.parquet our FF3 from the old monthly pipeline
                                                       (crsp.msf_v2, frozen 2026-11-21)
         pyData/Intermediate/french_ff3.parquet, french_ff3_daily.parquet
         pyData/Intermediate/french_6_portfolios_daily.parquet
Outputs: output/daily_vs_monthly_agreement.csv   correlation and tracking error, 4 pairs
         output/daily_vs_monthly_means.csv       annualized means: daily x 252 vs monthly x 12
         output/daily_vs_monthly_growth_of_1.png growth of $1: daily compounded vs monthly

Two questions:
  1. Do the daily factors, compounded to months, give the same factors as the monthly
     pipeline? (They should: same sort, same stocks; only the return frequency differs.)
  2. Why does SMB earn less per year when measured with daily data (daily mean x 252)
     than with monthly data (monthly mean x 12)? French's own files show the same gap.

How to compound a factor: SMB and HML are *differences* of portfolios, and the compounded
difference is not the difference of the compounded legs. So we compound each of the six
portfolios (and the market) to months first, then build the factors, exactly as the
monthly pipeline does. MKT-RF uses French's monthly RF.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402

from config import (FACTORS, INTERMEDIATE, MONTHLY_PIPELINE_REF, OUTPUT, PORTFOLIO_NAMES,  # noqa: E402
                    PORTFOLIOS, make_folders)
from Portfolios.FF2x3Daily import compound_to_months  # noqa: E402
from utils.plot_style import COLOR_EXTRA, COLOR_FRENCH, COLOR_MINE, apply_style, plt  # noqa: E402

FACTOR_NAMES = ["mkt_rf", "smb", "hml"]
LABELS = {"mkt_rf": "MKT-RF", "smb": "SMB", "hml": "HML"}


def monthly_factors_from_daily(portfolios, market, rf_monthly):
    """Compound six daily portfolios and the daily market to months, then build the factors."""
    legs = compound_to_months(portfolios[PORTFOLIO_NAMES])
    factors = pd.DataFrame(index=legs.index)
    factors["smb"] = legs[["SL", "SN", "SH"]].mean(axis=1) - legs[["BL", "BN", "BH"]].mean(axis=1)
    factors["hml"] = legs[["SH", "BH"]].mean(axis=1) - legs[["SL", "BL"]].mean(axis=1)
    factors["mkt_rf"] = compound_to_months(market) - rf_monthly
    return factors[FACTOR_NAMES]


def agreement(a, b):
    """Correlation and tracking error (% per year, monthly data) of two monthly factor tables."""
    months = a.index.intersection(b.index)
    out = {}
    for f in FACTOR_NAMES:
        gap = a.loc[months, f] - b.loc[months, f]
        out[f"{LABELS[f]} corr"] = a.loc[months, f].corr(b.loc[months, f])
        out[f"{LABELS[f]} TE %"] = gap.std() * 12 ** 0.5 * 100
    return out


def main():
    make_folders()
    apply_style()
    print("Comparing daily and monthly FF3...", flush=True)

    # --- Load everything as monthly periods (Jul 1963 - Dec 2025) ----------------------------------
    rf = pd.read_parquet(INTERMEDIATE / "french_ff3.parquet").set_index("date")
    rf.index = rf.index.to_period("M")
    french_monthly = rf[FACTOR_NAMES]
    rf = rf["rf"]

    ours_daily_ports = pd.read_parquet(PORTFOLIOS / "FF2x3Daily_BMdec.parquet")
    ours_daily = pd.read_parquet(FACTORS / "FF3Daily.parquet")
    ours_from_daily = monthly_factors_from_daily(ours_daily_ports, ours_daily["mkt"], rf)

    ours_monthly = pd.read_parquet(MONTHLY_PIPELINE_REF / "FF3.parquet")[FACTOR_NAMES]
    ours_monthly.index = ours_monthly.index.to_period("M")

    # French's daily portfolios compounded the same way: how far apart are *his* daily and
    # monthly files? That is the benchmark for how far apart ours may be.
    start, end = ours_daily.index.min(), ours_daily.index.max()
    french_daily = pd.read_parquet(INTERMEDIATE / "french_ff3_daily.parquet").set_index("date")
    french_daily = french_daily.loc[start:end]
    french_6 = pd.read_parquet(INTERMEDIATE / "french_6_portfolios_daily.parquet")
    french_6 = french_6[french_6["table"] == "vw_return"].set_index("date").loc[start:end]
    french_from_daily = monthly_factors_from_daily(
        french_6, french_daily["mkt_rf"] + french_daily["rf"], rf)

    months = ours_from_daily.index
    french_monthly = french_monthly.loc[months.min():months.max()]
    print(f"months: {len(months)}   {months.min()} to {months.max()}")

    # --- 1. Agreement: correlation and tracking error ------------------------------------------------
    pairs = {
        "ours: daily->monthly vs old monthly pipeline": (ours_from_daily, ours_monthly),
        "French: daily->monthly vs monthly": (french_from_daily, french_monthly),
        "ours daily->monthly vs French monthly": (ours_from_daily, french_monthly),
        "ours old monthly pipeline vs French monthly": (ours_monthly, french_monthly),
    }
    table = pd.DataFrame({name: agreement(a, b) for name, (a, b) in pairs.items()}).T
    table.round(4).to_csv(OUTPUT / "daily_vs_monthly_agreement.csv")
    print("\nAgreement of monthly factors (tracking error in % per year):")
    print(table.round(4).to_string())

    # --- 2. Average premium: daily x 252 vs monthly x 12 ----------------------------------------------
    # Annualizing a daily mean with 252 ignores compounding *within* the month. For stocks
    # whose daily returns are positively autocorrelated (small, less-traded stocks react
    # to news with a lag), a month compounds to more than 21 x the daily mean. Small stocks
    # are on SMB's long side, so SMB looks weaker in daily data.
    means = pd.DataFrame({
        "ours daily x 252": ours_daily[FACTOR_NAMES].mean() * 252,
        "ours daily->monthly x 12": ours_from_daily.mean() * 12,
        "ours old monthly pipeline x 12": ours_monthly.loc[months.min():months.max()].mean() * 12,
        "French daily x 252": french_daily[FACTOR_NAMES].mean() * 252,
        "French daily->monthly x 12": french_from_daily.mean() * 12,
        "French monthly x 12": french_monthly.mean() * 12,
    }).T.astype(float) * 100
    means.columns = [LABELS[f] for f in FACTOR_NAMES]
    means.round(2).to_csv(OUTPUT / "daily_vs_monthly_means.csv")
    print("\nAverage premium, % per year, Jul 1963 - Dec 2025:")
    print(means.round(2).to_string())

    # Evidence for the explanation: day-to-day (lag-1) autocorrelation of each portfolio.
    autocorr = pd.DataFrame({
        "ours": {p: ours_daily_ports[p].autocorr() for p in PORTFOLIO_NAMES},
        "French": {p: french_6[p].autocorr() for p in PORTFOLIO_NAMES},
    }).T
    print("\nLag-1 autocorrelation of daily portfolio returns (small > big):")
    print(autocorr.round(3).to_string())

    # --- 3. Growth of $1: the three monthly versions ---------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    series = [
        (french_monthly, "Ken French (monthly)", COLOR_FRENCH),
        (ours_monthly.loc[months.min():months.max()], "Ours, old monthly pipeline", COLOR_EXTRA[0]),
        (ours_from_daily, "Ours, daily compounded to months", COLOR_MINE),
    ]
    for ax, f in zip(axes, FACTOR_NAMES):
        for data, label, color in series:
            growth = (1 + data[f]).cumprod()
            ax.plot(growth.index.to_timestamp(), growth, color=color, linewidth=2, label=label)
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}"))
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(
            lambda y, _: f"${y:g}" if str(y)[0] in "25" else ""))
        ax.set_title(f"{LABELS[f]}: growth of $1", loc="left", fontweight="bold")
        ax.set_ylabel("Value of $1 invested (log scale)")
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Daily vs monthly construction, {months.min().strftime('%b %Y')} - "
                 f"{months.max().strftime('%b %Y')}",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT / "daily_vs_monthly_growth_of_1.png", dpi=200)
    plt.close(fig)
    print("\nSaved output/daily_vs_monthly_growth_of_1.png")


if __name__ == "__main__":
    main()
