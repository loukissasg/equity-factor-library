# ABOUTME: Compares SMB (FF3) with the quality-controlled size factors SMBQ and SMBQ_GP: statistics, correlations, regressions on QMJ.
"""
Run:     python Validation/SMBQAnalysis.py   (from pyCode/)
Inputs:  pyData/Factors/FF3.parquet, SMBQ.parquet, SMBQ_GP.parquet
         pyData/Intermediate/french_umd.parquet, aqr_qmj.parquet
Outputs: output/smbq_stats.csv          mean, volatility, Sharpe, per factor and period
         output/smbq_correlations.csv   correlation matrix, per period
         output/smbq_regressions.csv    alphas, t-stats, loadings, R^2, per factor/model/period
         output/smbq_growth_of_1.png    cumulative returns of SMB, SMBQ, SMBQ_GP

The question (Asness, Frazzini, Israel, Moskowitz and Pedersen 2018): is the weak size
premium after the 1980s really gone, or is it hidden because small stocks are junky?
If SMB is short quality, then
  - regressing SMB on QMJ gives a negative QMJ loading, and adding QMJ raises SMB's alpha;
  - a size factor built *within* quality quintiles (SMBQ) should have a less negative
    QMJ loading and a cleaner, larger premium.

Regressions (as in the paper): factor = a + b1 MKT-RF(t) + b2 MKT-RF(t-1) + h HML + u UMD
[+ q QMJ] + e. Lagged MKT-RF picks up small stocks reacting to market news a month late
(they trade less often), which would otherwise show up in the alpha.
Periods: 1963-latest, and 1963-2012 to compare with the paper (their sample is 1957-2012).
t-statistics are plain OLS. b_qmj_vol_matched = QMJ loading after scaling the factor to
SMB's volatility (raw loadings are not comparable across factors of different volatility).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import matplotlib.ticker as mticker  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, make_folders  # noqa: E402
from utils.plot_style import COLOR_EXTRA, COLOR_MINE, COLOR_TEXT, apply_style, plt  # noqa: E402

SIZE_FACTORS = {"smb": "SMB (FF3)", "smbq": "SMBQ (OperProf)", "smbq_gp": "SMBQ_GP (GP)"}
OTHER_FACTORS = {"mkt_rf": "MKT-RF", "hml": "HML", "umd": "UMD", "qmj": "QMJ"}
PERIODS = {"1963-2025": ("1963-07", "2025-12"), "1963-2012": ("1963-07", "2012-12")}
BASE_REGRESSORS = ["mkt_rf", "mkt_rf_lag", "hml", "umd"]


def load_factors():
    """One monthly table with every factor used in this script (decimals)."""
    ff3 = pd.read_parquet(FACTORS / "FF3.parquet")[["mkt_rf", "smb", "hml"]]
    smbq = pd.read_parquet(FACTORS / "SMBQ.parquet")[["smbq"]]
    smbq_gp = pd.read_parquet(FACTORS / "SMBQ_GP.parquet")[["smbq_gp"]]
    umd = pd.read_parquet(INTERMEDIATE / "french_umd.parquet").set_index("date")[["umd"]]
    qmj = pd.read_parquet(INTERMEDIATE / "aqr_qmj.parquet").set_index("date")[["qmj"]]

    data = ff3.join([smbq, smbq_gp], how="inner").join([umd, qmj], how="left")
    # Lag within our own MKT-RF series; the first month (Jul 1963) has no lag and drops
    # out of the regressions (not out of the statistics).
    data["mkt_rf_lag"] = data["mkt_rf"].shift(1)
    missing = data[["umd", "qmj"]].isna().sum()
    if missing.any():
        raise ValueError(f"UMD/QMJ missing for some months of our sample: {missing.to_dict()}")
    return data


def summary_stats(data, period, start, end):
    rows = []
    for column, label in {**SIZE_FACTORS, **OTHER_FACTORS}.items():
        x = data.loc[start:end, column]
        rows.append({"period": period, "factor": label,
                     "mean_annual_%": x.mean() * 12 * 100,
                     "std_annual_%": x.std() * 12 ** 0.5 * 100,
                     "sharpe": x.mean() / x.std() * 12 ** 0.5,
                     "months": len(x)})
    return rows


def regressions(data, period, start, end):
    """Each size factor on MKT-RF, lagged MKT-RF, HML, UMD; then the same plus QMJ."""
    rows = []
    sample = data.loc[start:end].dropna(subset=["mkt_rf_lag"])
    for y, label in SIZE_FACTORS.items():
        for model, regressors in [("without QMJ", BASE_REGRESSORS),
                                  ("with QMJ", BASE_REGRESSORS + ["qmj"])]:
            fit = sm.OLS(sample[y], sm.add_constant(sample[regressors])).fit()
            row = {"period": period, "factor": label, "model": model,
                   "alpha_annual_%": fit.params["const"] * 12 * 100,
                   "alpha_tstat": fit.tvalues["const"]}
            for r in regressors:
                row[f"b_{r}"] = fit.params[r]
                row[f"t_{r}"] = fit.tvalues[r]
            # A loading scales with the volatility of the left-hand factor: SMBQ compares the
            # extreme size quintiles and is ~35% more volatile than SMB, so its raw QMJ beta is
            # inflated. Rescaling the factor to SMB's volatility makes the betas comparable.
            if "qmj" in regressors:
                row["b_qmj_vol_matched"] = fit.params["qmj"] * sample["smb"].std() / sample[y].std()
            row["r_squared"] = fit.rsquared
            row["months"] = int(fit.nobs)
            rows.append(row)
    return rows


def plot_growth_of_1(data):
    # Log scale: equal vertical distances are equal percentage gains. SMB in grey (the
    # benchmark), SMBQ in our blue, the GP version in green.
    fig, ax = plt.subplots(figsize=(10, 5))
    styles = {"smb": (COLOR_TEXT, 2.0), "smbq": (COLOR_MINE, 2.4), "smbq_gp": (COLOR_EXTRA[0], 1.6)}
    for column, (color, width) in styles.items():
        growth = (1 + data[column]).cumprod()
        ax.plot(growth.index, growth, color=color, linewidth=width, label=SIZE_FACTORS[column])
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}"))
    ax.yaxis.set_minor_formatter(mticker.FuncFormatter(
        lambda y, _: f"${y:g}" if str(y)[0] in "25" else ""))
    ax.axvline(pd.Timestamp("2012-12-31"), color=COLOR_TEXT, linewidth=1, linestyle=":")
    ax.annotate("end of the paper's sample (2012)", xy=(pd.Timestamp("2012-12-31"), 1.02),
                xycoords=("data", "axes fraction"), ha="right", va="top", fontsize=8,
                color=COLOR_TEXT, xytext=(-4, 0), textcoords="offset points")
    ax.set_ylabel("Value of $1 invested (log scale)")
    ax.set_title(f"Size factors: growth of $1, {data.index.min():%b %Y} - {data.index.max():%b %Y}",
                 loc="left", fontweight="bold")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUTPUT / "smbq_growth_of_1.png", dpi=200)
    plt.close(fig)


def main():
    make_folders()
    apply_style()
    print("Comparing SMB, SMBQ and SMBQ_GP...", flush=True)
    data = load_factors()
    print(f"sample: {data.index.min().date()} to {data.index.max().date()} ({len(data)} months)")

    stats_rows, corr_tables, reg_rows = [], [], []
    for period, (start, end) in PERIODS.items():
        stats_rows += summary_stats(data, period, start, end)
        reg_rows += regressions(data, period, start, end)
        labels = {**SIZE_FACTORS, **OTHER_FACTORS}
        corr = data.loc[start:end, list(labels)].corr().rename(index=labels, columns=labels)
        corr.insert(0, "period", period)
        corr_tables.append(corr)

    stats = pd.DataFrame(stats_rows)
    correlations = pd.concat(corr_tables)
    regs = pd.DataFrame(reg_rows)
    stats.round(3).to_csv(OUTPUT / "smbq_stats.csv", index=False)
    correlations.round(3).to_csv(OUTPUT / "smbq_correlations.csv")
    regs.round(3).to_csv(OUTPUT / "smbq_regressions.csv", index=False)

    for period in PERIODS:
        print(f"\n===== {period} =====")
        print("Annualized statistics:")
        print(stats[stats["period"] == period].drop(columns="period").set_index("factor")
              .round(2).to_string())
        print("\nCorrelations:")
        print(correlations[correlations["period"] == period].drop(columns="period")
              .round(2).to_string())
        print("\nRegressions on MKT-RF, MKT-RF(t-1), HML, UMD [+ QMJ] (alpha in % p.a.):")
        show = regs[regs["period"] == period].set_index(["factor", "model"])
        show = show[["alpha_annual_%", "alpha_tstat", "b_mkt_rf", "b_mkt_rf_lag", "b_hml",
                     "b_umd", "b_qmj", "t_qmj", "b_qmj_vol_matched", "r_squared"]]
        print(show.round(2).to_string())

    plot_growth_of_1(data)
    print("\nsaved output/smbq_stats.csv, smbq_correlations.csv, smbq_regressions.csv, "
          "smbq_growth_of_1.png")


if __name__ == "__main__":
    main()
