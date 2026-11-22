# ABOUTME: Checks that every number embedded in signal_dashboard.html equals the Stage 3 tables exactly.
"""
Run:     python SignalDashboard/H_CheckDashboard.py   (from pyCode/)
Inputs:  output/signal_dashboard.html, output/stats_*.csv, pyData/OpenAP/timeseries.parquet
Outputs: a PASS/FAIL report; exits with an error on any mismatch (stops master_dashboard.py)

Why: the brief requires that the dashboard matches the Stage 3 tables exactly. We read the
JSON block back OUT of the finished HTML (not from the builder's memory) and compare it,
value by value, with the CSV files. "Exactly" means bit-for-bit equal floats. Displayed
numbers are these values rounded in the browser (usually to 1-2 decimals).
"""

import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import DASHBOARD_SIGNALS, OPENAP, OUTPUT  # noqa: E402

HTML_FILE = OUTPUT / "signal_dashboard.html"
TABLES = {"full": "stats_full_sample.csv", "periods": "stats_periods.csv",
          "recent": "stats_recent.csv", "regressions": "stats_regressions.csv",
          "phases": "stats_market_phases.csv", "extremes": "stats_best_worst_12m.csv",
          "alpha_summary": "stats_alpha_summary.csv"}


def same(a, b):
    """Exact equality, treating missing (None / NaN / empty) as equal to missing."""
    missing_a = a is None or (isinstance(a, float) and math.isnan(a))
    missing_b = b is None or (isinstance(b, float) and math.isnan(b)) or b == ""
    if missing_a or missing_b:
        return missing_a and missing_b
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return str(a) == str(b)


def extract_data(html):
    match = re.search(r'<script id="dashboard-data" type="application/json">(.*?)</script>', html, re.S)
    if not match:
        raise SystemExit("No dashboard-data block found in the HTML")
    return json.loads(match.group(1))


def main():
    data = extract_data(HTML_FILE.read_text(encoding="utf-8"))
    checked, failures = 0, []

    for name, filename in TABLES.items():
        csv = pd.read_csv(OUTPUT / filename, keep_default_na=False, na_values=[""])
        page = data[name]
        if len(page) != len(csv):
            failures.append(f"{name}: {len(page)} rows in page vs {len(csv)} in {filename}")
            continue
        for i, row in enumerate(csv.to_dict("records")):
            for col, value in row.items():
                checked += 1
                if not same(page[i].get(col), value.item() if hasattr(value, "item") else value):
                    failures.append(f"{name} row {i} {col}: page {page[i].get(col)!r} vs csv {value!r}")

    corr = pd.read_csv(OUTPUT / "stats_correlations.csv", index_col=0).loc[DASHBOARD_SIGNALS, DASHBOARD_SIGNALS]
    for i, row in enumerate(corr.to_numpy()):
        for j, value in enumerate(row):
            checked += 1
            if not same(data["corr"]["values"][i][j], float(value)):
                failures.append(f"corr [{i}][{j}] differs")

    ts = pd.read_parquet(OPENAP / "timeseries.parquet")
    for signal in DASHBOARD_SIGNALS:
        g = ts[ts["signal"] == signal].sort_values("date")
        page = data["timeseries"][signal]
        for col in ["wealth", "drawdown_pct", "rolling_sharpe_36m"]:
            for a, b in zip(page[col], g[col].tolist()):
                checked += 1
                if not same(a, b):
                    failures.append(f"timeseries {signal} {col} differs")
                    break
        if page["date"] != g["date"].dt.strftime("%Y-%m-%d").tolist():
            failures.append(f"timeseries {signal} dates differ")

    # Every card in the page must be the Stage 2 card, unchanged.
    cards = json.loads((OPENAP / "signal_cards.json").read_text(encoding="utf-8"))
    if data["cards"] != cards["cards"]:
        failures.append("signal cards differ from signal_cards.json")

    if failures:
        print(f"FAIL: {len(failures)} mismatches out of {checked:,} values")
        for f in failures[:20]:
            print("  ", f)
        sys.exit(1)
    print(f"PASS: all {checked:,} embedded values equal the Stage 3 tables exactly; cards identical.")


if __name__ == "__main__":
    main()
