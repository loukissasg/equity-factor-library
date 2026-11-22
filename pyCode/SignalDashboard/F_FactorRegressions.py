# ABOUTME: Regresses each long-short signal on CAPM, our FF3, FF5 + UMD and FF5 + UMD + QMJ: alphas, t-stats, loadings.
"""
Run:     python SignalDashboard/F_FactorRegressions.py   (from pyCode/)
Inputs:  pyData/OpenAP/ls_returns.parquet, factors_monthly.parquet (all in % per month)
Outputs: output/stats_regressions.csv  one row per signal x model: alpha (% per month and
                                       p.a.), t-stat, every loading and its t-stat, R2, N
         output/stats_alpha_summary.csv  signal x model alpha t-stats, plus "survives"

The question: is the signal's return just exposure to factors we already know? The
regression is LS_t = alpha + sum(beta_k * factor_k,t) + e_t. Alpha is the average return
left over after paying for those exposures. If alpha's t-stat stays above ~2, the signal
"survives" that model.

Choices (the "why"):
- CAPM and FF3 use OUR replicated factors (MKT-RF, SMB, HML), as asked in the brief. FF5,
  UMD and QMJ come from Ken French and AQR (we did not replicate them).
- Same months for every model: the window where all factors exist (1963-07 onwards, or later
  if the signal starts later). Otherwise alphas would differ partly because the samples
  differ, not because of the models.
- No risk-free rate is subtracted from LS: it is already a zero-cost (excess) return.
- Plain OLS t-stats, consistent with the t-stats of the means in E_Performance.py.
- Thresholds: |t| > 1.96 is the classic 5% level. Harvey, Liu and Zhu (2016, Review of
  Financial Studies 29(1), 5-68) argue that after decades of data mining a new factor
  should clear t > 3, so we report both.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402

from config import DASHBOARD_SIGNALS, OUTPUT, make_folders  # noqa: E402
from utils.dashboard_data import load_factors, load_ls  # noqa: E402

MODELS = {
    "CAPM": ["mkt_rf_mine"],
    "FF3": ["mkt_rf_mine", "smb_mine", "hml_mine"],
    "FF5+UMD": ["mkt_rf", "smb", "hml", "rmw", "cma", "umd"],
    "FF5+UMD+QMJ": ["mkt_rf", "smb", "hml", "rmw", "cma", "umd", "qmj"],
}
ALL_FACTORS = sorted({f for cols in MODELS.values() for f in cols})


def run(y, X):
    fit = sm.OLS(y, sm.add_constant(X)).fit()
    row = {
        "alpha_monthly": fit.params["const"], "alpha_annual": 12 * fit.params["const"],
        "t_alpha": fit.tvalues["const"], "r2": fit.rsquared, "months": int(fit.nobs),
    }
    for factor in X.columns:
        row[f"beta_{factor}"] = fit.params[factor]
        row[f"t_{factor}"] = fit.tvalues[factor]
    return row


def main():
    make_folders()
    ls, factors = load_ls(), load_factors()
    rows = []
    for signal in DASHBOARD_SIGNALS:
        # One common window per signal for all four models (see docstring).
        data = pd.concat([ls[signal], factors[ALL_FACTORS]], axis=1).dropna()
        start, end = data.index.min().strftime("%Y-%m"), data.index.max().strftime("%Y-%m")
        raw_mean = data[signal].mean()
        for model, cols in MODELS.items():
            rows.append({"signal": signal, "model": model, "start": start, "end": end,
                         "mean_monthly_same_window": raw_mean,
                         **run(data[signal], data[cols])})
    table = pd.DataFrame(rows)
    table.to_csv(OUTPUT / "stats_regressions.csv", index=False)

    summary = table.pivot(index="signal", columns="model", values="t_alpha")[list(MODELS)]
    summary = summary.loc[DASHBOARD_SIGNALS]
    summary.columns = [f"t_alpha_{m}" for m in summary.columns]
    summary["survives_FF5+UMD_t>1.96"] = summary["t_alpha_FF5+UMD"].abs() > 1.96
    summary["survives_FF5+UMD+QMJ_t>1.96"] = summary["t_alpha_FF5+UMD+QMJ"].abs() > 1.96
    summary["survives_FF5+UMD+QMJ_t>3"] = summary["t_alpha_FF5+UMD+QMJ"].abs() > 3
    summary.to_csv(OUTPUT / "stats_alpha_summary.csv")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("ALPHA (% per month) and t-stat, same window for all models")
    alpha = table.pivot(index="signal", columns="model", values="alpha_monthly")[list(MODELS)]
    print(pd.concat({"alpha": alpha, "t": summary[[f"t_alpha_{m}" for m in MODELS]]
                     .set_axis(list(MODELS), axis=1)}, axis=1).loc[DASHBOARD_SIGNALS].round(2).to_string())
    print("\nWindows:", ", ".join(f"{s} {g['start'].iat[0]}-{g['end'].iat[0]}"
                                  for s, g in table.groupby("signal", sort=False)))
    print("\nLOADINGS in FF5+UMD+QMJ (t-stats in brackets)")
    full = table[table["model"] == "FF5+UMD+QMJ"].set_index("signal").loc[DASHBOARD_SIGNALS]
    for signal, r in full.iterrows():
        parts = [f"{f}={r[f'beta_{f}']:+.2f} [{r[f't_{f}']:+.1f}]" for f in MODELS["FF5+UMD+QMJ"]]
        print(f"  {signal:<19} R2={r['r2']:.2f}  " + "  ".join(parts))
    print("\nSURVIVES?")
    print(summary.iloc[:, 4:].to_string())


if __name__ == "__main__":
    main()
