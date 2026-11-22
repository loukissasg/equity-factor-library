# ABOUTME: Tests whether more junk-neutral versions of SMBQ beat SMB: premium, QMJ exposure, alpha, pricing of the 25 portfolios.
"""
Run:     python Validation/SMBQJunkNeutral.py   (from pyCode/)
Inputs:  pyData/Factors/FF3.parquet, SMBQ.parquet, SMBQ_JunkNeutral.parquet
         pyData/Intermediate/french_umd.parquet, aqr_qmj.parquet,
         french_25_portfolios.parquet, french_ff3.parquet (RF)
Outputs: output/smbq_junk_neutral.csv        one row per size factor and period
         output/smbq_junk_neutral_growth.png growth of $1: each version vs SMB

The question: is SMBQ only weak because it controls for junk badly? A size factor that
is really junk-neutral should have (a) a QMJ loading near zero, (b) a positive alpha on
MKT-RF, MKT-RF(t-1), HML and UMD *without* needing QMJ, and (c) price the 25 size x B/M
portfolios better than SMB in the model MKT-RF + size factor + HML.

Tests for each size factor (same definitions as SMBQAnalysis.py and ModelComparison25.py):
  mean, volatility, Sharpe; correlation with SMB and with QMJ
  regression without / with QMJ: alpha (t), QMJ loading, QMJ loading at SMB's volatility
  model MKT-RF + factor + HML on French's 25 portfolios: average |alpha|, GRS
  max-Sharpe mix of MKT-RF + factor + HML (in-sample)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import matplotlib.ticker as mticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, make_folders  # noqa: E402
from utils.plot_style import COLOR_GRID, COLOR_MINE, COLOR_TEXT, apply_style, plt  # noqa: E402
from Validation.ModelComparison25 import evaluate  # noqa: E402  (same GRS code as the main test)

SIZE_FACTORS = {
    "smb": "SMB (FF3)",
    "smbq": "SMBQ (5x5 OperProf)",
    "smbq_d10": "SMBQ_D10 (OperProf deciles)",
    "smbq_exq1": "SMBQ_exQ1 (drop junk quintile)",
    "smbq_comp": "SMBQ_Comp (profit + safety)",
}
PERIODS = {"1963-2025": ("1963-07", "2025-12"), "1963-2012": ("1963-07", "2012-12")}
BASE = ["mkt_rf", "mkt_rf_lag", "hml", "umd"]


def load():
    ff3 = pd.read_parquet(FACTORS / "FF3.parquet")[["mkt_rf", "smb", "hml"]]
    smbq = pd.read_parquet(FACTORS / "SMBQ.parquet")[["smbq"]]
    variants = pd.read_parquet(FACTORS / "SMBQ_JunkNeutral.parquet")
    umd = pd.read_parquet(INTERMEDIATE / "french_umd.parquet").set_index("date")[["umd"]]
    qmj = pd.read_parquet(INTERMEDIATE / "aqr_qmj.parquet").set_index("date")[["qmj"]]
    data = ff3.join([smbq, variants], how="inner").join([umd, qmj], how="left").astype("float64")
    data["mkt_rf_lag"] = data["mkt_rf"].shift(1)  # first month has no lag (dropped in regressions)

    rf = pd.read_parquet(INTERMEDIATE / "french_ff3.parquet").set_index("date")["rf"]
    p25 = pd.read_parquet(INTERMEDIATE / "french_25_portfolios.parquet").set_index("date")
    excess = p25.sub(rf, axis=0).astype("float64")
    return data, excess


def max_sharpe(f):
    """Sharpe ratio of the maximum-Sharpe (tangency) mix of the columns of f, annualized."""
    weights = np.linalg.solve(f.cov().values, f.mean().values)
    mix = f.values @ weights
    return mix.mean() / mix.std(ddof=1) * 12 ** 0.5


def assess(data, excess, column, period, start, end):
    sample = data.loc[start:end]
    regs = sample.dropna(subset=["mkt_rf_lag"])
    y = sample[column]
    fit0 = sm.OLS(regs[column], sm.add_constant(regs[BASE])).fit()
    fit1 = sm.OLS(regs[column], sm.add_constant(regs[BASE + ["qmj"]])).fit()
    model_factors = sample[["mkt_rf", column, "hml"]]
    pricing, _ = evaluate(excess.loc[sample.index], model_factors, list(model_factors.columns))
    return {
        "period": period,
        "factor": SIZE_FACTORS[column],
        "mean_%_pa": y.mean() * 1200,
        "vol_%_pa": y.std() * 12 ** 0.5 * 100,
        "sharpe": y.mean() / y.std() * 12 ** 0.5,
        "corr_SMB": y.corr(sample["smb"]),
        "corr_QMJ": y.corr(sample["qmj"]),
        "alpha_no_QMJ_%_pa": fit0.params["const"] * 1200,
        "t_alpha_no_QMJ": fit0.tvalues["const"],
        "alpha_with_QMJ_%_pa": fit1.params["const"] * 1200,
        "t_alpha_with_QMJ": fit1.tvalues["const"],
        "b_QMJ": fit1.params["qmj"],
        "t_b_QMJ": fit1.tvalues["qmj"],
        # QMJ loading after scaling the factor to SMB's volatility (raw betas grow with vol).
        "b_QMJ_vol_matched": fit1.params["qmj"] * regs["smb"].std() / regs[column].std(),
        "GRS_MKT_factor_HML": pricing["GRS"],
        "avg_abs_alpha_25_%_month": pricing["avg_abs_alpha_%_month"],
        "max_sharpe_MKT_factor_HML": max_sharpe(model_factors),
    }


def plot_growth(data):
    # Small multiples: one panel per SMBQ version, each against SMB (grey, the benchmark),
    # so every panel has just two lines and the same scale.
    versions = [c for c in SIZE_FACTORS if c != "smb"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True, sharey=True,
                             layout="constrained")
    smb_growth = (1 + data["smb"]).cumprod()
    for ax, column in zip(axes.flat, versions):
        growth = (1 + data[column]).cumprod()
        ax.plot(smb_growth.index, smb_growth, color=COLOR_TEXT, linewidth=1.4, label="SMB (FF3)")
        ax.plot(growth.index, growth, color=COLOR_MINE, linewidth=2.2, label=SIZE_FACTORS[column])
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"${y:g}"))
        ax.yaxis.set_minor_formatter(mticker.FuncFormatter(
            lambda y, _: f"${y:g}" if str(y)[0] in "25" else ""))
        # "\$": two plain $ signs in one title would switch matplotlib into math mode.
        ax.set_title(f"{SIZE_FACTORS[column]}: \\${growth.iloc[-1]:.1f} vs SMB "
                     f"\\${smb_growth.iloc[-1]:.1f}", loc="left", fontweight="bold", fontsize=10)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(color=COLOR_GRID)
    fig.suptitle(f"Growth of $1: junk-neutral size factors vs SMB, "
                 f"{data.index.min():%b %Y} - {data.index.max():%b %Y}", x=0.01, ha="left",
                 fontsize=12)
    fig.savefig(OUTPUT / "smbq_junk_neutral_growth.png", dpi=200)
    plt.close(fig)


def main():
    make_folders()
    apply_style()
    print("Testing the junk-neutral SMBQ versions...", flush=True)
    data, excess = load()
    rows = [assess(data, excess, column, period, start, end)
            for period, (start, end) in PERIODS.items() for column in SIZE_FACTORS]
    table = pd.DataFrame(rows)
    table.round(3).to_csv(OUTPUT / "smbq_junk_neutral.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        for period in PERIODS:
            print(f"\n===== {period} =====")
            print(table[table["period"] == period].drop(columns="period").set_index("factor")
                  .round(2).T.to_string())

    plot_growth(data)
    print("\nsaved output/smbq_junk_neutral.csv, smbq_junk_neutral_growth.png")


if __name__ == "__main__":
    main()
