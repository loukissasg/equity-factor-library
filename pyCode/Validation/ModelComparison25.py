# ABOUTME: Compares FF3 with SMBQ-based models on Ken French's 25 size x B/M portfolios: average |alpha|, GRS test, average R^2.
"""
Run:     python Validation/ModelComparison25.py   (from pyCode/)
Inputs:  pyData/Factors/FF3.parquet, SMBQ.parquet
         pyData/Intermediate/aqr_qmj.parquet, french_25_portfolios.parquet, french_ff3.parquet (RF)
Outputs: output/model_comparison_25.csv     one row per model and period
         output/model_alphas_25.csv         the 25 alphas (% per month) per model and period
         output/model_alpha_tstats_25.csv   their t-statistics
         output/model_comparison_25.png     fit statistics of the three models, side by side
         output/model_alphas_25.png         5x5 alpha heatmaps: where each model fails
         output/model_growth_of_1.png       growth of $1 of each model's max-Sharpe factor mix
         output/model_tangency_weights.csv  the weights and Sharpe ratio behind that chart

Models (all factors are ours except QMJ, which is AQR's):
  A  FF3:  MKT-RF, SMB,  HML
  B        MKT-RF, SMBQ, HML          (HML unchanged)
  C        MKT-RF, SMBQ, HML, QMJ

Test: for each of the 25 portfolios, regress its excess return on the model's factors.
If the model prices the portfolios, every intercept (alpha) should be zero.
  average |alpha|  how far the model is off, on average (% per month)
  GRS              Gibbons, Ross and Shanken (1989) test that all 25 alphas are jointly
                   zero. It weights the alphas by how precisely they are estimated and
                   accounts for correlated errors. Large GRS / small p-value = rejected.
  average R^2      how much of the portfolios' variation the factors explain

GRS = (T/N) * ((T - N - L) / (T - L - 1)) * (a' S^-1 a) / (1 + m' W^-1 m)  ~  F(N, T - N - L)
  T months, N = 25 portfolios, L factors, a = vector of alphas,
  S = residual covariance (divided by T - L - 1), m = factor means,
  W = factor covariance (divided by T - 1).

Caveat: French's 25 portfolios are sorted on size and B/M, so they reward a model that
prices size and value; they say little about quality. A better size factor can still
leave the famous "small growth" alpha (ME1_BM1) unexplained.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from scipy import stats  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, make_folders  # noqa: E402
from utils.plot_style import COLOR_INK, COLOR_MINE, COLOR_TEXT, apply_style, plt  # noqa: E402

MODELS = {
    "A: FF3 (MKT-RF, SMB, HML)": ["mkt_rf", "smb", "hml"],
    "B: MKT-RF, SMBQ, HML": ["mkt_rf", "smbq", "hml"],
    "C: MKT-RF, SMBQ, HML, QMJ": ["mkt_rf", "smbq", "hml", "qmj"],
}
PERIODS = {"1963-2025": ("1963-07", "2025-12"), "1963-2012": ("1963-07", "2012-12")}
SHORT_NAMES = {"A: FF3 (MKT-RF, SMB, HML)": "A: FF3",
               "B: MKT-RF, SMBQ, HML": "B: SMBQ",
               "C: MKT-RF, SMBQ, HML, QMJ": "C: SMBQ + QMJ"}
# Diverging scale for alphas: red = negative, blue = positive, neutral gray at zero, so a
# cell's colour says "how far from zero" and its hue says the sign.
DIVERGING = ["#e34948", "#f0efec", "#2a78d6"]
# One colour per model for the growth chart, in a fixed order (validated as a set for
# colour-blind readers). Orange here is model B, not "Ken French" as in the other charts.
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
TARGET_VOL = 0.10  # every model's mix is scaled to 10% annual volatility


def grs_test(alphas, residuals, factors):
    """GRS statistic and p-value (formula in the docstring above)."""
    T, N = residuals.shape
    L = factors.shape[1]
    sigma = residuals.T @ residuals / (T - L - 1)           # residual covariance, N x N
    mu = factors.mean().values                                # factor means, L
    omega = np.atleast_2d(np.cov(factors.values, rowvar=False))  # factor covariance, L x L
    a = alphas.values
    grs = ((T / N) * ((T - N - L) / (T - L - 1))
           * (a @ np.linalg.solve(sigma, a)) / (1 + mu @ np.linalg.solve(omega, mu)))
    p_value = stats.f.sf(grs, N, T - N - L)
    return grs, p_value


def evaluate(excess, factor_data, columns):
    """Time-series regressions of the 25 portfolios on one model's factors."""
    X = sm.add_constant(factor_data[columns])
    alphas, r2, residuals = {}, {}, {}
    for portfolio in excess.columns:
        fit = sm.OLS(excess[portfolio], X).fit()
        alphas[portfolio] = fit.params["const"]
        r2[portfolio] = fit.rsquared
        residuals[portfolio] = fit.resid
    alphas = pd.Series(alphas)
    residuals = pd.DataFrame(residuals)
    grs, p_value = grs_test(alphas, residuals.values, factor_data[columns])
    return {
        "avg_abs_alpha_%_month": alphas.abs().mean() * 100,
        "GRS": grs,
        "GRS_p_value": p_value,
        "avg_R2": pd.Series(r2).mean(),
        "n_alpha_|t|>2": None,  # filled below
    }, alphas


def plot_fit_statistics(summary, period):
    """Four small panels (one per statistic, each on its own scale), one bar per model."""
    # Small multiples instead of one chart: the four statistics have different units, and
    # mixing them on one axis would be misleading. The model is on the x-axis and every
    # bar is labelled, so one colour is enough.
    table = summary[summary["period"] == period].set_index("model")
    panels = [
        ("avg_abs_alpha_%_month", "Average |alpha|, % per month", "{:.3f}", "lower is better"),
        ("GRS", "GRS statistic", "{:.2f}", "lower is better"),
        ("avg_R2", "Average R²", "{:.3f}", "higher is better"),
        ("n_alpha_|t|>2", "Alphas with |t| > 2 (of 25)", "{:.0f}", "lower is better"),
    ]
    labels = [SHORT_NAMES[m] for m in table.index]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), layout="constrained")
    for ax, (column, title, fmt, direction) in zip(axes, panels):
        values = table[column].astype(float).values
        ax.set_axisbelow(True)  # gridlines behind the bars, not across them
        bars = ax.bar(labels, values, color=COLOR_MINE, width=0.6)
        for bar, value in zip(bars, values):
            ax.annotate(fmt.format(value), xy=(bar.get_x() + bar.get_width() / 2, value),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9, color=COLOR_INK)
        ax.set_title(f"{title}\n({direction})", loc="left", fontweight="bold", fontsize=10)
        if column == "avg_R2":
            ax.set_ylim(0.85, 0.93)  # all near 0.9: zoom in, and say so in the axis label
            ax.set_ylabel("axis starts at 0.85", fontsize=8, color=COLOR_TEXT)
        else:
            ax.set_ylim(0, values.max() * 1.18)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", labelsize=9)
    p_values = ", ".join(f"{SHORT_NAMES[m]} p = {float(p):.0e}"
                         for m, p in table["GRS_p_value"].items())
    fig.suptitle(f"Pricing Ken French's 25 size x B/M portfolios, {period} "
                 f"(GRS rejects all three: {p_values})", x=0.01, ha="left", fontsize=11)
    fig.savefig(OUTPUT / "model_comparison_25.png", dpi=200)
    plt.close(fig)


def plot_alpha_heatmaps(alphas, tstats, period):
    """One 5x5 heatmap per model, same colour scale; outlined cells have |t| > 2."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("alpha", DIVERGING)
    limit = float(alphas.loc[period].abs().max().max())  # symmetric scale around zero

    fig, axes = plt.subplots(1, 3, figsize=(15, 6.0), layout="constrained")
    fig.get_layout_engine().set(h_pad=0.12)  # room between the main title and panel titles
    for k, (ax, model) in enumerate(zip(axes, MODELS)):
        grid = alphas.loc[(period, model)].values.reshape(5, 5)
        tgrid = tstats.loc[(period, model)].values.reshape(5, 5)
        image = ax.imshow(grid, cmap=cmap, vmin=-limit, vmax=limit)
        for i in range(5):
            for j in range(5):
                value = grid[i, j]
                # Dark text on light cells, white text on the darkest ones.
                text_color = "white" if abs(value) > 0.6 * limit else COLOR_INK
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                        color=text_color,
                        fontweight="bold" if abs(tgrid[i, j]) > 2 else "normal")
                if abs(tgrid[i, j]) > 2:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                               edgecolor=COLOR_INK, linewidth=1.6))
        ax.set_xticks(range(5), ["Growth\nBM1", "BM2", "BM3", "BM4", "Value\nBM5"], fontsize=8)
        # Row labels only on the first panel: the rows are the same in all three.
        ax.set_yticks(range(5), ["Small ME1", "ME2", "ME3", "ME4", "Big ME5"] if k == 0 else [""] * 5,
                      fontsize=8)
        ax.grid(False)
        n_sig = int((abs(tgrid) > 2).sum())
        ax.set_title(f"{SHORT_NAMES[model]}\navg |alpha| {abs(grid).mean():.3f}, "
                     f"{n_sig} of 25 with |t| > 2", loc="left", fontweight="bold", fontsize=10)
    colorbar = fig.colorbar(image, ax=axes, shrink=0.85)
    colorbar.set_label("Alpha, % per month")
    fig.suptitle(f"Alphas of the 25 size x B/M portfolios, {period}. Outlined and bold: |t| > 2.\n"
                 f"Red = the model over-predicts the return, blue = under-predicts.",
                 x=0.01, ha="left", fontsize=11)
    # bbox_inches="tight": square heatmap cells leave the layout engine unsure of the height,
    # so let matplotlib crop to the content (keeps the two-line x labels from being cut).
    fig.savefig(OUTPUT / "model_alphas_25.png", dpi=200, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def plot_model_growth(factors, period):
    """Growth of $1 in each model's maximum-Sharpe ("tangency") mix of its factors."""
    # Why this turns a model into one line: if a model's factors explain expected returns,
    # the best portfolio an investor can build is a mix of those factors. The highest
    # achievable Sharpe ratio is therefore a natural score for a model (Barillas and
    # Shanken 2017): a model whose factors combine into a better portfolio prices more.
    # Tangency weights: w proportional to inverse(covariance) x mean excess returns.
    # Each mix is then scaled to TARGET_VOL so the lines differ in return, not in risk.
    # In-sample: the weights use the whole period, which an investor could not have known.
    weight_rows = []
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    # Vertical nudge (points) for each end label: A and B end almost at the same value,
    # so A's label goes slightly below its line and B's slightly above.
    label_nudge = [-7, 7, 0]
    for color, nudge, (model, columns) in zip(MODEL_COLORS, label_nudge, MODELS.items()):
        f = factors[columns]
        weights = np.linalg.solve(f.cov().values, f.mean().values)
        mix = f.values @ weights
        scale = TARGET_VOL / (mix.std(ddof=1) * 12 ** 0.5)
        mix = pd.Series(mix * scale, index=f.index)
        sharpe = mix.mean() / mix.std() * 12 ** 0.5
        normalized = weights / np.abs(weights).sum()  # shares of the total position
        weight_rows.append({"period": period, "model": model, "sharpe": sharpe,
                            "mean_%_pa": mix.mean() * 1200,
                            **{f"w_{c}": w for c, w in zip(columns, normalized)}})

        growth = (1 + mix).cumprod()
        label = f"{SHORT_NAMES[model]} (Sharpe {sharpe:.2f})"
        ax.plot(growth.index, growth, color=color, linewidth=2, label=label)
        # Direct label at the line's end, in ink (not the line colour) for contrast.
        ax.annotate(f"{SHORT_NAMES[model]}  ${growth.iloc[-1]:,.0f}",
                    xy=(growth.index[-1], growth.iloc[-1]), xytext=(6, nudge),
                    textcoords="offset points", va="center", fontsize=9, color=COLOR_INK)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"${y:,.0f}" if y >= 1 else f"${y:g}"))
    ax.set_xlim(right=factors.index[-1] + pd.DateOffset(years=9))  # room for the end labels
    # Ticks only where there is data (the extra room on the right is for labels only).
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(1970, 2030, 10)],
                  [str(y) for y in range(1970, 2030, 10)])
    ax.set_ylabel("Value of $1 (log scale), excess return")
    ax.set_title(f"Growth of $1 in each model's best factor mix, {period}\n"
                 "Maximum-Sharpe weights estimated on the full sample (in-sample); "
                 "every mix scaled to 10% annual volatility",
                 loc="left", fontweight="bold", fontsize=11)
    ax.legend(loc="upper left")
    fig.savefig(OUTPUT / "model_growth_of_1.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(weight_rows)


def main():
    make_folders()
    apply_style()
    print("Model comparison on Ken French's 25 size x B/M portfolios...", flush=True)
    ff3 = pd.read_parquet(FACTORS / "FF3.parquet")[["mkt_rf", "smb", "hml"]]
    smbq = pd.read_parquet(FACTORS / "SMBQ.parquet")[["smbq"]]
    qmj = pd.read_parquet(INTERMEDIATE / "aqr_qmj.parquet").set_index("date")[["qmj"]]
    rf = pd.read_parquet(INTERMEDIATE / "french_ff3.parquet").set_index("date")["rf"]
    p25 = pd.read_parquet(INTERMEDIATE / "french_25_portfolios.parquet").set_index("date")

    # Test assets in excess of the T-bill rate (French's RF, the same RF as in our MKT-RF).
    excess_all = p25.sub(rf, axis=0)
    # astype(float64): FF3 is stored with pandas' nullable Float64 type, which NumPy's
    # matrix functions (used in the GRS test) cannot handle.
    factors_all = ff3.join(smbq, how="inner").join(qmj, how="inner").astype("float64")
    excess_all = excess_all.astype("float64")

    summary_rows, alpha_rows, tstat_rows = [], [], []
    for period, (start, end) in PERIODS.items():
        factors = factors_all.loc[start:end]
        excess = excess_all.loc[factors.index]
        if excess.isna().any().any():
            raise ValueError("missing portfolio returns in the test period")
        print(f"\n===== {period}: {len(factors)} months, 25 portfolios =====")
        for model, columns in MODELS.items():
            result, alphas = evaluate(excess, factors, columns)
            # Count individually significant alphas, a more intuitive companion to GRS.
            X = sm.add_constant(factors[columns])
            tstats = pd.Series({p: sm.OLS(excess[p], X).fit().tvalues["const"] for p in excess})
            result["n_alpha_|t|>2"] = int((tstats.abs() > 2).sum())
            summary_rows.append({"period": period, "model": model, **result})
            alpha_rows.append(pd.Series(alphas * 100, name=(period, model)))
            tstat_rows.append(pd.Series(tstats, name=(period, model)))

        table = pd.DataFrame([r for r in summary_rows if r["period"] == period]).set_index("model")
        # p-values are tiny; scientific notation instead of rounding them to 0.000.
        show = table.drop(columns="period").round(3)
        show["GRS_p_value"] = table["GRS_p_value"].map("{:.1e}".format)
        print(show.to_string())

    summary = pd.DataFrame(summary_rows)
    tstats_all = pd.DataFrame(tstat_rows)
    tstats_all.index = pd.MultiIndex.from_tuples(tstats_all.index, names=["period", "model"])
    tstats_all.round(3).to_csv(OUTPUT / "model_alpha_tstats_25.csv")
    plot_fit_statistics(summary, "1963-2025")
    p_values = summary["GRS_p_value"]
    summary = summary.round(4)
    summary["GRS_p_value"] = p_values.map("{:.2e}".format)
    summary.to_csv(OUTPUT / "model_comparison_25.csv", index=False)
    alphas = pd.DataFrame(alpha_rows)
    alphas.index = pd.MultiIndex.from_tuples(alphas.index, names=["period", "model"])
    alphas.round(4).to_csv(OUTPUT / "model_alphas_25.csv")
    plot_alpha_heatmaps(alphas, tstats_all, "1963-2025")

    start, end = PERIODS["1963-2025"]
    weights = plot_model_growth(factors_all.loc[start:end], "1963-2025")
    weights.round(3).to_csv(OUTPUT / "model_tangency_weights.csv", index=False)
    print("\nMaximum-Sharpe factor mix per model, 1963-2025 (weights as shares of |total|):")
    print(weights.drop(columns="period").set_index("model").round(2).to_string())

    # The 5x5 alpha grids for the full sample: where does each model fail?
    for model in MODELS:
        grid = alphas.loc[("1963-2025", model)].values.reshape(5, 5)
        print(f"\nAlphas, % per month, 1963-2025, {model} (rows ME1 small..ME5 big, cols BM1 growth..BM5 value):")
        print(pd.DataFrame(grid, index=[f"ME{s}" for s in range(1, 6)],
                           columns=[f"BM{b}" for b in range(1, 6)]).round(2).to_string())
    print("\nsaved output/model_comparison_25.csv, model_alphas_25.csv, model_alpha_tstats_25.csv, "
          "model_comparison_25.png, model_alphas_25.png, model_growth_of_1.png, "
          "model_tangency_weights.csv")


if __name__ == "__main__":
    main()
