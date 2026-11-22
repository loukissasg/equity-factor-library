# ABOUTME: Collects the Stage 3 (SMBQ) results into two short tables for a presentation slide.
"""
Run:     python Validation/Stage3SlideSummary.py   (from pyCode/, after SMBQAnalysis.py and
         ModelComparison25.py)
Inputs:  output/smbq_stats.csv, smbq_regressions.csv, model_comparison_25.csv
Outputs: output/stage3_slide_factors.csv   SMB vs SMBQ vs SMBQ_GP, 1963-2025 and 1963-2012
         output/stage3_slide_models.csv    models A/B/C on the 25 size x B/M portfolios
         output/stage3_slide_summary.md    both tables, formatted for a slide

Reads only the tables written by the other two Stage 3 scripts, so the numbers on the
slide are exactly the ones in the detailed outputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import OUTPUT, make_folders  # noqa: E402

SIZE_FACTORS = ["SMB (FF3)", "SMBQ (OperProf)", "SMBQ_GP (GP)"]


def factor_table():
    stats = pd.read_csv(OUTPUT / "smbq_stats.csv")
    regs = pd.read_csv(OUTPUT / "smbq_regressions.csv")
    with_qmj = regs[regs["model"] == "with QMJ"].set_index(["period", "factor"])
    without_qmj = regs[regs["model"] == "without QMJ"].set_index(["period", "factor"])

    rows = []
    for period in stats["period"].unique():
        for factor in SIZE_FACTORS:
            s = stats[(stats["period"] == period) & (stats["factor"] == factor)].iloc[0]
            a = without_qmj.loc[(period, factor)]
            b = with_qmj.loc[(period, factor)]
            rows.append({
                "period": period,
                "factor": factor,
                "mean_%_pa": round(s["mean_annual_%"], 2),
                "vol_%_pa": round(s["std_annual_%"], 1),
                "sharpe": round(s["sharpe"], 2),
                "alpha_no_QMJ_%_pa (t)": f"{a['alpha_annual_%']:.2f} ({a['alpha_tstat']:.1f})",
                "alpha_with_QMJ_%_pa (t)": f"{b['alpha_annual_%']:.2f} ({b['alpha_tstat']:.1f})",
                "QMJ_loading (t)": f"{b['b_qmj']:.2f} ({b['t_qmj']:.1f})",
                "QMJ_loading_vol_matched": round(b["b_qmj_vol_matched"], 2),
                "R2_with_QMJ": round(b["r_squared"], 2),
            })
    return pd.DataFrame(rows)


def model_table():
    models = pd.read_csv(OUTPUT / "model_comparison_25.csv")
    return pd.DataFrame({
        "period": models["period"],
        "model": models["model"],
        "avg_|alpha|_%_month": models["avg_abs_alpha_%_month"].round(3),
        "GRS": models["GRS"].round(2),
        "GRS_p": models["GRS_p_value"].astype(float).map("{:.1e}".format),
        "avg_R2": models["avg_R2"].round(3),
        "alphas_|t|>2 (of 25)": models["n_alpha_|t|>2"],
    })


def to_markdown(df):
    """A plain markdown table (no extra dependency needed)."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main():
    make_folders()
    factors = factor_table()
    models = model_table()
    factors.to_csv(OUTPUT / "stage3_slide_factors.csv", index=False)
    models.to_csv(OUTPUT / "stage3_slide_models.csv", index=False)

    text = ["# Stage 3: SMBQ, size controlling for junk (Asness et al. 2018)", "",
            "## Size factors (regressions on MKT-RF, MKT-RF(t-1), HML, UMD [+ QMJ])", "",
            to_markdown(factors), "",
            "## Pricing Ken French's 25 size x B/M portfolios", "",
            to_markdown(models), ""]
    (OUTPUT / "stage3_slide_summary.md").write_text("\n".join(text), encoding="utf-8")

    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(factors.to_string(index=False))
        print()
        print(models.to_string(index=False))
    print("\nsaved output/stage3_slide_factors.csv, stage3_slide_models.csv, stage3_slide_summary.md")


if __name__ == "__main__":
    main()
