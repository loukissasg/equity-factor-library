# ABOUTME: Stage 1 report: signal metadata from SignalDoc and data coverage of every signal and factor.
"""
Run:     python SignalDashboard/C_DataReport.py   (from pyCode/)
Inputs:  pyData/OpenAP/op_portfolios.parquet, ls_returns.parquet, signal_doc_full.parquet,
         factors_monthly.parquet (from A_OpenAPDownload.py and B_Benchmarks.py)
Outputs: output/signal_info.csv         one row per signal: the SignalDoc fields we use
         output/stage1_data_coverage.csv first/last month, months, gaps, thin months

This script only describes the data. It computes no performance statistics (Stage 3).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import DASHBOARD_SIGNALS, OPENAP, OUTPUT, make_folders  # noqa: E402

# SignalDoc column -> our name. Only the fields we need for the cards and the period split.
DOC_FIELDS = {
    "Acronym": "signal",
    "LongDescription": "full_name",
    "Authors": "authors",
    "Year": "year",                       # publication year of the original paper
    "Journal": "journal",
    "SampleStartYear": "sample_start",    # original paper's sample (years only)
    "SampleEndYear": "sample_end",
    "Cat.Economic": "economic_category",
    "Cat.Data": "data_category",
    "T-Stat": "tstat_original",           # t-stat reported in the original paper
    "Portfolio Period": "portfolio_period_months",   # months between rebalancings
    "Stock Weight": "stock_weight",       # EW = equal-weighted, VW = value-weighted
    # Extras that the Stage 2 cards need to explain the construction:
    "Sign": "sign",                       # +1: long high signal; -1: long low signal
    "LS Quantile": "ls_quantile",         # 0.1 = deciles, 0.2 = quintiles, blank = not stated
    "Filter": "filter",                   # extra stock filter in the original paper
    "Detailed Definition": "definition",
}

# A long-short leg with fewer stocks than this is a thin, noisy portfolio. Flag it; do not drop.
THIN_LEG = 20


def signal_info(doc, ports):
    info = doc[list(DOC_FIELDS)].rename(columns=DOC_FIELDS)
    # How many quantile portfolios the OP file actually has (fills the blanks in ls_quantile).
    n_ports = (ports[ports["port"] != "LS"].groupby("signalname")["port"].nunique()
               .rename("n_portfolios"))
    info = info.merge(n_ports, left_on="signal", right_index=True)
    for col in ["year", "sample_start", "sample_end", "portfolio_period_months", "sign"]:
        info[col] = info[col].astype("Int64")
    # Keep the order of config.DASHBOARD_SIGNALS so every table lists signals the same way.
    return info.set_index("signal").loc[DASHBOARD_SIGNALS].reset_index()


def coverage(ls, ports, factors):
    ls_rows = ports[ports["port"] == "LS"].set_index(["signalname", "date"])
    rows = []
    for signal in DASHBOARD_SIGNALS:
        s = ls[signal].dropna()
        expected_months = len(pd.date_range(s.index.min(), s.index.max(), freq="ME"))
        legs = ls_rows.loc[signal]
        thin = legs[(legs["Nlong"] < THIN_LEG) | (legs["Nshort"] < THIN_LEG)]
        rows.append({
            "series": signal, "type": "signal (OpenAP LS)",
            "first": s.index.min().strftime("%Y-%m"), "last": s.index.max().strftime("%Y-%m"),
            "months": len(s), "missing_months_inside_range": expected_months - len(s),
            "min_stocks_long": int(legs["Nlong"].min()),
            "min_stocks_short": int(legs["Nshort"].min()),
            "thin_months": len(thin),
            "thin_month_list": ", ".join(thin.index.strftime("%Y-%m")),
        })
    for col in factors.columns:
        s = factors[col].dropna()
        expected_months = len(pd.date_range(s.index.min(), s.index.max(), freq="ME"))
        rows.append({
            "series": col, "type": "factor",
            "first": s.index.min().strftime("%Y-%m"), "last": s.index.max().strftime("%Y-%m"),
            "months": len(s), "missing_months_inside_range": expected_months - len(s),
        })
    table = pd.DataFrame(rows)
    # Int64 (nullable integers): factors have no stock counts, so these cells stay blank.
    for col in ["min_stocks_long", "min_stocks_short", "thin_months"]:
        table[col] = table[col].astype("Int64")
    return table


def main():
    make_folders()
    ports = pd.read_parquet(OPENAP / "op_portfolios.parquet")
    ls = pd.read_parquet(OPENAP / "ls_returns.parquet")
    doc = pd.read_parquet(OPENAP / "signal_doc_full.parquet")
    factors = pd.read_parquet(OPENAP / "factors_monthly.parquet")

    info = signal_info(doc, ports)
    info.to_csv(OUTPUT / "signal_info.csv", index=False)
    show = ["signal", "full_name", "authors", "year", "journal", "sample_start", "sample_end",
            "economic_category", "data_category", "tstat_original", "portfolio_period_months",
            "stock_weight", "sign", "n_portfolios"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\nSignalDoc fields (output/signal_info.csv)\n")
        print(info[show].to_string(index=False))

    cov = coverage(ls, ports, factors)
    cov.to_csv(OUTPUT / "stage1_data_coverage.csv", index=False)
    print("\nCoverage (output/stage1_data_coverage.csv)\n")
    cols = ["series", "first", "last", "months", "missing_months_inside_range",
            "min_stocks_long", "min_stocks_short", "thin_months"]
    print(cov[cols].to_string(index=False, na_rep=""))
    for _, row in cov[cov["thin_months"] > 0].iterrows():
        print(f"  {row['series']}: months with a leg under {THIN_LEG} stocks: "
              f"{row['thin_month_list']}")

    # The regression window is limited by the shortest series it uses.
    needed = ["mkt_rf_mine", "smb_mine", "hml_mine", "mkt_rf", "smb", "hml", "rmw", "cma",
              "umd", "qmj"]
    common = factors[needed].dropna()
    print(f"\nAll factors available together: {common.index.min():%Y-%m} to "
          f"{common.index.max():%Y-%m} ({len(common):,} months). OpenAP LS ends "
          f"{ls.index.max():%Y-%m}, so regressions can run to {min(common.index.max(), ls.index.max()):%Y-%m}.")


if __name__ == "__main__":
    main()
