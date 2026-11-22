# ABOUTME: Performance statistics of the 10 long-short signals: full sample, publication periods, recent, crises, correlations.
"""
Run:     python SignalDashboard/E_Performance.py   (from pyCode/)
Inputs:  pyData/OpenAP/ls_returns.parquet, factors_monthly.parquet, output/signal_info.csv
Outputs (output/):
    stats_full_sample.csv     mean, t-stat, vol, Sharpe, max drawdown (+ dates), worst month, % positive
    stats_periods.csv         in-sample / pre-publication / post-publication (+ pre-sample) and declines
    stats_recent.csv          last 5 years and last 12 months
    stats_market_phases.csv   cumulative return in 5 market phases (signals + the market)
    stats_best_worst_12m.csv  best and worst rolling 12-month window per signal
    stats_correlations.csv    correlation matrix of the 10 LS returns (common sample)
  pyData/OpenAP/timeseries.parquet  date, signal, ret, wealth, drawdown, rolling 36m Sharpe
                                    (the dashboard plots these; it computes nothing itself)

Conventions (the "why"):
- Returns are long-short, zero-cost, % per month. Zero-cost means LS is already an excess
  return, so the Sharpe ratio is mean / std with no risk-free rate subtracted.
- Annualizing: mean x 12, volatility x sqrt(12), Sharpe x sqrt(12). Simple, standard, and
  how OpenAP and most papers report it.
- t-stat of the mean = mean / (std / sqrt(N)), the classic iid t-stat that the original
  papers and OpenAP report, so ours are comparable with SignalDoc's.
- Cumulative returns compound the monthly LS return: $1 that is put into the long-short
  portfolio every month and reinvested. That is how "growth of $1" and drawdowns are
  computed.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import DASHBOARD_SIGNALS, OPENAP, OUTPUT, make_folders  # noqa: E402
from utils.dashboard_data import load_factors, load_info, load_ls  # noqa: E402

# Market phases from the brief: (name, first month, last month).
MARKET_PHASES = [
    ("1973-74 oil crisis bear market", "1973-01", "1974-12"),
    ("2000-02 dot-com bust", "2000-01", "2002-12"),
    ("2007-09 financial crisis", "2007-10", "2009-02"),
    ("2020 COVID crash", "2020-02", "2020-03"),
    ("2022 inflation / rate hikes", "2022-01", "2022-12"),
]
ROLLING_SHARPE_MONTHS = 36


def month_end(yyyy_mm):
    return pd.Timestamp(yyyy_mm) + pd.offsets.MonthEnd(0)


def basic_stats(r):
    """Mean, t-stat, volatility and Sharpe of a monthly % series (NaNs already removed)."""
    n = len(r)
    if n < 2:
        return {"months": n, "mean_monthly": np.nan, "mean_annual": np.nan, "tstat": np.nan,
                "vol_annual": np.nan, "sharpe_annual": np.nan}
    mean, std = r.mean(), r.std()
    return {
        "months": n,
        "mean_monthly": mean,
        "mean_annual": 12 * mean,
        "tstat": mean / (std / np.sqrt(n)),
        "vol_annual": std * np.sqrt(12),
        "sharpe_annual": mean / std * np.sqrt(12),
    }


def wealth_index(r):
    """Growth of $1, starting at 1.0 the month before the first return (so a first-month
    loss also counts as a drawdown)."""
    if (r <= -100).any():
        raise ValueError(f"{r.name}: a monthly LS return of -100% or worse; compounding breaks")
    start = pd.Series([1.0], index=[r.index[0] - pd.offsets.MonthEnd(1)])
    return pd.concat([start, (1 + r / 100).cumprod()])


def max_drawdown(r):
    """Largest peak-to-trough fall of the wealth index, with its dates."""
    wealth = wealth_index(r)
    drawdown = wealth / wealth.cummax() - 1
    trough = drawdown.idxmin()
    peak = wealth.loc[:trough].idxmax()
    after = wealth.loc[trough:]
    recovered = after[after >= wealth.loc[peak]]
    return {
        "max_drawdown_pct": 100 * drawdown.min(),
        "dd_peak": peak.strftime("%Y-%m"),
        "dd_trough": trough.strftime("%Y-%m"),
        "dd_recovered": recovered.index[0].strftime("%Y-%m") if len(recovered) else "not recovered",
        "dd_months_peak_to_trough": (trough.to_period("M") - peak.to_period("M")).n,
    }


def compounded_pct(r):
    return 100 * ((1 + r / 100).prod() - 1)


def full_sample_table(ls):
    rows = []
    for signal in DASHBOARD_SIGNALS:
        r = ls[signal].dropna()
        rows.append({
            "signal": signal,
            "first": r.index.min().strftime("%Y-%m"), "last": r.index.max().strftime("%Y-%m"),
            **basic_stats(r),
            **max_drawdown(r),
            "worst_month_pct": r.min(), "worst_month": r.idxmin().strftime("%Y-%m"),
            "best_month_pct": r.max(), "best_month": r.idxmax().strftime("%Y-%m"),
            "share_positive_pct": 100 * (r > 0).mean(),
        })
    return pd.DataFrame(rows)


def period_bounds(info_row, first, last):
    """The publication-period split agreed after Stage 1 (see config.py)."""
    start_yr, end_yr, pub_yr = int(info_row["sample_start"]), int(info_row["sample_end"]), int(info_row["year"])
    return [
        ("pre-sample", first, month_end(f"{start_yr - 1}-12")),
        ("in-sample", month_end(f"{start_yr}-01"), month_end(f"{end_yr}-12")),
        ("pre-publication", month_end(f"{end_yr + 1}-01"), month_end(f"{pub_yr}-12")),
        ("post-publication", month_end(f"{pub_yr + 1}-01"), last),
    ]


def periods_table(ls, info):
    rows = []
    for signal in DASHBOARD_SIGNALS:
        r = ls[signal].dropna()
        for name, start, end in period_bounds(info.loc[signal], r.index.min(), r.index.max()):
            piece = r.loc[max(start, r.index.min()):min(end, r.index.max())]
            stats = basic_stats(piece)
            rows.append({
                "signal": signal, "period": name,
                "start": piece.index.min().strftime("%Y-%m") if len(piece) else "",
                "end": piece.index.max().strftime("%Y-%m") if len(piece) else "",
                "months": stats["months"], "mean_monthly": stats["mean_monthly"],
                "mean_annual": stats["mean_annual"], "tstat": stats["tstat"],
                "sharpe_annual": stats["sharpe_annual"],
            })
    table = pd.DataFrame(rows)

    # McLean-Pontiff style decline: how much smaller the mean is than in-sample.
    # Only meaningful if the in-sample mean is positive (it is the benchmark).
    wide = table.pivot(index="signal", columns="period", values="mean_monthly")
    declines = pd.DataFrame(index=wide.index)
    for later in ["pre-publication", "post-publication"]:
        ok = wide["in-sample"] > 0
        declines[f"decline_{later}_pct"] = np.where(
            ok, 100 * (1 - wide[later] / wide["in-sample"]), np.nan)
    table = table.merge(declines, left_on="signal", right_index=True)
    # SignalDoc's 'Return' is what the original paper reported. Its definition varies
    # (e.g. size-adjusted for Accruals), so it is a rough sanity check, not a target.
    table["paper_return_signaldoc"] = table["signal"].map(
        pd.read_parquet(OPENAP / "signal_doc_full.parquet").set_index("Acronym")["Return"])
    table.loc[table["period"] != "in-sample", "paper_return_signaldoc"] = np.nan
    return table


def recent_table(ls):
    rows = []
    last = ls.index.max()
    for label, months in [("last 5 years", 60), ("last 12 months", 12)]:
        start = last - pd.offsets.MonthEnd(months - 1)
        for signal in DASHBOARD_SIGNALS:
            r = ls.loc[start:last, signal].dropna()
            rows.append({"signal": signal, "window": label,
                         "start": start.strftime("%Y-%m"), "end": last.strftime("%Y-%m"),
                         **basic_stats(r), "cumulative_pct": compounded_pct(r)})
    return pd.DataFrame(rows)


def market_phase_table(ls, factors):
    # The market's own total return (MKT-RF + RF, Ken French) as context for each phase.
    series = ls.copy()
    series["Market (total return)"] = factors["mkt_rf"] + factors["rf"]
    rows = []
    for name, first, last in MARKET_PHASES:
        start, end = month_end(first), month_end(last)
        expected = len(pd.date_range(start, end, freq="ME"))
        for col in series.columns:
            r = series.loc[start:end, col].dropna()
            complete = len(r) == expected
            rows.append({
                "phase": name, "start": first, "end": last, "series": col,
                "months": len(r),
                # A phase only counts if every month is available; otherwise show missing.
                "cumulative_pct": compounded_pct(r) if complete else np.nan,
                "note": "" if complete else f"only {len(r)} of {expected} months available",
            })
    return pd.DataFrame(rows)


def best_worst_12m(ls):
    rows = []
    for signal in DASHBOARD_SIGNALS:
        r = ls[signal].dropna()
        # Sum of log returns over 12 months = log of the compounded 12-month return.
        rolling = np.expm1(np.log1p(r / 100).rolling(12, min_periods=12).sum()) * 100
        for label, end in [("best", rolling.idxmax()), ("worst", rolling.idxmin())]:
            rows.append({"signal": signal, "window": label,
                         "start": (end - pd.offsets.MonthEnd(11)).strftime("%Y-%m"),
                         "end": end.strftime("%Y-%m"), "return_12m_pct": rolling.loc[end]})
    return pd.DataFrame(rows)


def correlations(ls):
    # Common sample (all 10 available), so every pair is measured over the same months.
    common = ls.dropna()
    return common.corr(), common.index.min(), common.index.max(), len(common)


def time_series(ls):
    frames = []
    for signal in DASHBOARD_SIGNALS:
        r = ls[signal].dropna()
        wealth = wealth_index(r)
        roll = r.rolling(ROLLING_SHARPE_MONTHS, min_periods=ROLLING_SHARPE_MONTHS)
        frames.append(pd.DataFrame({
            "date": wealth.index, "signal": signal,
            "ret": r.reindex(wealth.index),               # NaN on the $1 starting row
            "wealth": wealth.values,
            "drawdown_pct": 100 * (wealth / wealth.cummax() - 1).values,
            "rolling_sharpe_36m": (roll.mean() / roll.std() * np.sqrt(12)).reindex(wealth.index).values,
        }))
    return pd.concat(frames, ignore_index=True)


def main():
    make_folders()
    ls, factors, info = load_ls(), load_factors(), load_info()

    full = full_sample_table(ls)
    periods = periods_table(ls, info)
    recent = recent_table(ls)
    phases = market_phase_table(ls, factors)
    extremes = best_worst_12m(ls)
    corr, c_start, c_end, c_n = correlations(ls)
    ts = time_series(ls)

    full.to_csv(OUTPUT / "stats_full_sample.csv", index=False)
    periods.to_csv(OUTPUT / "stats_periods.csv", index=False)
    recent.to_csv(OUTPUT / "stats_recent.csv", index=False)
    phases.to_csv(OUTPUT / "stats_market_phases.csv", index=False)
    extremes.to_csv(OUTPUT / "stats_best_worst_12m.csv", index=False)
    corr.to_csv(OUTPUT / "stats_correlations.csv")
    ts.to_parquet(OPENAP / "timeseries.parquet", index=False)
    # Metadata the dashboard displays (so it never has to recompute anything itself).
    (OPENAP / "stats_meta.json").write_text(json.dumps({
        "correlation_start": c_start.strftime("%Y-%m"), "correlation_end": c_end.strftime("%Y-%m"),
        "correlation_months": int(c_n),
        "market_phases": [{"name": n, "start": s, "end": e} for n, s, e in MARKET_PHASES],
        "rolling_sharpe_months": ROLLING_SHARPE_MONTHS,
    }, indent=2))

    # ---- Printed summary (sanity checks) ----
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("FULL SAMPLE (mean/vol in % p.a.)")
    print(full[["signal", "first", "months", "mean_annual", "tstat", "vol_annual", "sharpe_annual",
                "max_drawdown_pct", "dd_peak", "dd_trough", "dd_recovered", "worst_month_pct",
                "worst_month", "share_positive_pct"]].round(2).to_string(index=False))

    print("\nPERIODS (mean % per month, t-stat)")
    wide = periods.pivot(index="signal", columns="period", values=["mean_monthly", "tstat", "months"])
    order = ["pre-sample", "in-sample", "pre-publication", "post-publication"]
    show = pd.concat({k: wide[k][order] for k in ["mean_monthly", "tstat", "months"]}, axis=1)
    print(show.loc[DASHBOARD_SIGNALS].round(2).to_string())
    dec = periods[periods["period"] == "in-sample"].set_index("signal")
    print("\nDecline vs in-sample (%; negative = higher than in-sample). Paper return = SignalDoc:")
    print(dec[["mean_monthly", "paper_return_signaldoc", "decline_pre-publication_pct",
               "decline_post-publication_pct"]].loc[DASHBOARD_SIGNALS].round(2).to_string())
    # Median as well as mean: one or two signals that improved after publication (large
    # negative declines) pull the mean around a lot with only 10 signals.
    post = dec["decline_post-publication_pct"]
    print(f"Post-publication decline: mean {post.mean():.0f}%, median {post.median():.0f}% "
          f"(McLean-Pontiff 2016: about 58% across 97 predictors)")

    print("\nRECENT")
    print(recent.pivot(index="signal", columns="window", values=["mean_annual", "tstat", "cumulative_pct"])
          .loc[DASHBOARD_SIGNALS].round(2).to_string())

    print("\nMARKET PHASES (cumulative %)")
    print(phases.pivot(index="series", columns="phase", values="cumulative_pct")
          [[p[0] for p in MARKET_PHASES]].round(1).to_string())

    print("\nBEST / WORST 12 MONTHS")
    print(extremes.to_string(index=False, float_format="%.1f"))

    print(f"\nCORRELATIONS (common sample {c_start:%Y-%m} to {c_end:%Y-%m}, {c_n} months)")
    print(corr.round(2).to_string())
    print(f"\nSaved 6 tables to output/ and timeseries.parquet ({len(ts):,} rows)")


if __name__ == "__main__":
    main()
