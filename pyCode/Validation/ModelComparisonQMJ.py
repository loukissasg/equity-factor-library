# ABOUTME: Compares FF3 + QMJ with the same model using SMBQ_Comp instead of SMB: growth of $1, max Sharpe, GRS, spanning tests.
"""
Run:     python Validation/ModelComparisonQMJ.py   (from pyCode/)
Inputs:  pyData/Factors/FF3.parquet, SMBQ_JunkNeutral.parquet (smbq_comp)
         pyData/Intermediate/aqr_qmj.parquet, french_25_portfolios.parquet, french_ff3.parquet (RF)
Outputs: output/model_qmj_comparison.csv   per model and period: max Sharpe, weights, GRS
         output/model_qmj_spanning.csv     does each size factor add to the other model?
         output/model_qmj_growth_of_1.png  growth of $1 of each model's max-Sharpe mix

Models:
  D  FF3 + QMJ:            MKT-RF, SMB,       HML, QMJ
  E  SMBQ_Comp + QMJ:      MKT-RF, SMBQ_Comp, HML, QMJ   (SMB replaced, all else equal)

Three ways to compare them:
  1. Max-Sharpe ("tangency") mix of each model's factors, scaled to 10% volatility, as a
     growth-of-$1 line (same method as model_growth_of_1.png). In-sample weights.
  2. GRS test on Ken French's 25 size x B/M portfolios (same code as ModelComparison25.py).
  3. Spanning tests, the direct test of "should we swap?": regress SMBQ_Comp on model D's
     other factors (MKT-RF, SMB, HML, QMJ). A significant alpha means SMBQ_Comp earns
     something model D can't reproduce, so adding it raises the model's max Sharpe. Then
     the reverse: SMB on MKT-RF, SMBQ_Comp, HML, QMJ. If SMB has no alpha there, nothing
     is lost by dropping it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, make_folders  # noqa: E402
from utils.plot_style import COLOR_INK, apply_style, plt  # noqa: E402
from Validation.ModelComparison25 import MODEL_COLORS, TARGET_VOL, evaluate  # noqa: E402

MODELS = {
    "D: FF3 + QMJ (MKT-RF, SMB, HML, QMJ)": ["mkt_rf", "smb", "hml", "qmj"],
    "E: MKT-RF, SMBQ_Comp, HML, QMJ": ["mkt_rf", "smbq_comp", "hml", "qmj"],
}
SHORT = {"D: FF3 + QMJ (MKT-RF, SMB, HML, QMJ)": "D: FF3 + QMJ",
         "E: MKT-RF, SMBQ_Comp, HML, QMJ": "E: SMBQ_Comp + QMJ"}
PERIODS = {"1963-2025": ("1963-07", "2025-12"), "1963-2012": ("1963-07", "2012-12")}


def load():
    ff3 = pd.read_parquet(FACTORS / "FF3.parquet")[["mkt_rf", "smb", "hml"]]
    comp = pd.read_parquet(FACTORS / "SMBQ_JunkNeutral.parquet")[["smbq_comp"]]
    qmj = pd.read_parquet(INTERMEDIATE / "aqr_qmj.parquet").set_index("date")[["qmj"]]
    factors = ff3.join(comp, how="inner").join(qmj, how="inner").astype("float64")
    rf = pd.read_parquet(INTERMEDIATE / "french_ff3.parquet").set_index("date")["rf"]
    p25 = pd.read_parquet(INTERMEDIATE / "french_25_portfolios.parquet").set_index("date")
    excess = p25.sub(rf, axis=0).astype("float64")
    return factors, excess


def tangency_mix(f):
    """Max-Sharpe weights (proportional to inverse covariance x means) and the scaled mix."""
    weights = np.linalg.solve(f.cov().values, f.mean().values)
    mix = f.values @ weights
    mix = pd.Series(mix * TARGET_VOL / (mix.std(ddof=1) * 12 ** 0.5), index=f.index)
    return weights / np.abs(weights).sum(), mix


def spanning(sample, y, others):
    """Alpha (% p.a.) and t-stat of factor y on the other factors."""
    fit = sm.OLS(sample[y], sm.add_constant(sample[others])).fit()
    return {"factor": y, "regressed_on": ", ".join(others),
            "alpha_%_pa": fit.params["const"] * 1200, "t_alpha": fit.tvalues["const"],
            **{f"b_{c}": fit.params[c] for c in others}, "r_squared": fit.rsquared}


def plot_growth(factors, period):
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    growths = {}
    for model, columns in MODELS.items():
        _, mix = tangency_mix(factors[columns])
        growths[model] = (mix.mean() / mix.std() * 12 ** 0.5, (1 + mix).cumprod())
    # The two lines end close together: push the higher line's label up and the lower
    # one's down, decided from the data (a fixed order would mislabel them).
    ending = {m: g.iloc[-1] for m, (_, g) in growths.items()}
    top = max(ending, key=ending.get)
    for color, (model, (sharpe, growth)) in zip(MODEL_COLORS, growths.items()):
        nudge = 7 if model == top else -7
        ax.plot(growth.index, growth, color=color, linewidth=2,
                label=f"{SHORT[model]} (Sharpe {sharpe:.2f})")
        ax.annotate(f"{SHORT[model]}  ${growth.iloc[-1]:,.0f}",
                    xy=(growth.index[-1], growth.iloc[-1]), xytext=(6, nudge),
                    textcoords="offset points", va="center", fontsize=9, color=COLOR_INK)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}" if y >= 1 else f"${y:g}"))
    ax.set_xlim(right=factors.index[-1] + pd.DateOffset(years=11))  # room for the end labels
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(1970, 2030, 10)],
                  [str(y) for y in range(1970, 2030, 10)])
    ax.set_ylabel("Value of $1 (log scale), excess return")
    ax.set_title(f"FF3 + QMJ vs SMBQ_Comp instead of SMB: growth of $1 in each model's best "
                 f"factor mix, {period}\nMaximum-Sharpe weights estimated on the full sample "
                 f"(in-sample); every mix scaled to 10% annual volatility",
                 loc="left", fontweight="bold", fontsize=11)
    ax.legend(loc="upper left")
    fig.savefig(OUTPUT / "model_qmj_growth_of_1.png", dpi=200)
    plt.close(fig)


def main():
    make_folders()
    apply_style()
    print("Comparing FF3 + QMJ with SMBQ_Comp + QMJ...", flush=True)
    factors, excess = load()

    model_rows, span_rows = [], []
    for period, (start, end) in PERIODS.items():
        sample = factors.loc[start:end]
        for model, columns in MODELS.items():
            weights, mix = tangency_mix(sample[columns])
            pricing, _ = evaluate(excess.loc[sample.index], sample[columns], columns)
            model_rows.append({
                "period": period, "model": model,
                "max_sharpe": mix.mean() / mix.std() * 12 ** 0.5,
                "growth_of_1": float((1 + mix).prod()),
                **{f"w_{c}": w for c, w in zip(columns, weights)},
                "GRS": pricing["GRS"], "GRS_p_value": pricing["GRS_p_value"],
                "avg_abs_alpha_25_%_month": pricing["avg_abs_alpha_%_month"],
                "avg_R2": pricing["avg_R2"],
            })
        for y, others in [("smbq_comp", ["mkt_rf", "smb", "hml", "qmj"]),
                          ("smb", ["mkt_rf", "smbq_comp", "hml", "qmj"])]:
            span_rows.append({"period": period, **spanning(sample, y, others)})

    models = pd.DataFrame(model_rows)
    spans = pd.DataFrame(span_rows)
    models_out = models.round(4)
    models_out["GRS_p_value"] = models["GRS_p_value"].map("{:.2e}".format)
    models_out.to_csv(OUTPUT / "model_qmj_comparison.csv", index=False)
    spans.round(3).to_csv(OUTPUT / "model_qmj_spanning.csv", index=False)

    with pd.option_context("display.width", 250, "display.max_columns", 30):
        show = models.set_index(["period", "model"]).round(3)
        show["GRS_p_value"] = models["GRS_p_value"].map("{:.1e}".format).values
        print("\nModels (weights = shares of the total position):")
        print(show.to_string())
        print("\nSpanning tests (alpha in % p.a.):")
        print(spans.set_index(["period", "factor"]).round(2).to_string())

    start, end = PERIODS["1963-2025"]
    plot_growth(factors.loc[start:end], "1963-2025")
    print("\nsaved output/model_qmj_comparison.csv, model_qmj_spanning.csv, model_qmj_growth_of_1.png")


if __name__ == "__main__":
    main()
