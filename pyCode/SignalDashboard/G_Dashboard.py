# ABOUTME: Builds output/signal_dashboard.html: packs the Stage 2 cards and Stage 3 tables into one HTML file.
"""
Run:     python SignalDashboard/G_Dashboard.py                 (from pyCode/; Plotly from the CDN)
         python SignalDashboard/G_Dashboard.py --embed-plotly  (Plotly inlined: works fully offline)
Inputs:  output/stats_*.csv, output/signal_info.csv           (Stage 1 and Stage 3)
         pyData/OpenAP/signal_cards.json, timeseries.parquet, stats_meta.json, release_info.json
         SignalDashboard/dashboard_template.html               (layout, styles, chart code)
Outputs: output/signal_dashboard.html

This script computes NO statistics. It only reads the Stage 3 outputs and writes them into
the page as one JSON block, so the dashboard cannot disagree with the tables.
H_CheckDashboard.py re-reads the HTML and proves it.

About "offline": the data is always inside the file. Plotly (about 3.5 MB) loads from the
jsDelivr CDN at a pinned version by default, as the brief asks, so the charts need an
internet connection when the file is opened (the tables do not). With --embed-plotly the
library is downloaded once, cached in pyData/OpenAP/, and inlined, so the email attachment
works on a plane too.
"""

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from config import DASHBOARD_SIGNALS, OPENAP, OUTPUT, make_folders  # noqa: E402

TEMPLATE = Path(__file__).resolve().parent / "dashboard_template.html"
OUTPUT_FILE = OUTPUT / "signal_dashboard.html"

# Pinned Plotly version: the page always gets exactly this library, whatever the date.
PLOTLY_VERSION = "2.35.2"
# jsDelivr serves the official npm build. (cdnjs lists 2.35.2 but returned 404 for the file
# on 2026-11-05, so we use jsDelivr.)
PLOTLY_URL = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{PLOTLY_VERSION}/plotly.min.js"
PLOTLY_CACHE = OPENAP / f"plotly-{PLOTLY_VERSION}.min.js"

# Tables embedded in the page: name in the page -> CSV in output/.
TABLES = {
    "full": "stats_full_sample.csv",
    "periods": "stats_periods.csv",
    "recent": "stats_recent.csv",
    "regressions": "stats_regressions.csv",
    "phases": "stats_market_phases.csv",
    "extremes": "stats_best_worst_12m.csv",
}


def clean(value):
    """Make one value JSON-safe. NaN -> null (JSON has no NaN). Python floats are written
    with repr, which round-trips exactly, so the check script can demand exact equality."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or (isinstance(value, str)):
        return value
    if pd.isna(value):
        return None
    return value


def records(df):
    return [{k: clean(v) for k, v in row.items()} for row in df.to_dict("records")]


def load_tables():
    data = {}
    for name, filename in TABLES.items():
        # keep_default_na=False + na_values=[""]: text such as "not recovered" stays text.
        df = pd.read_csv(OUTPUT / filename, keep_default_na=False, na_values=[""])
        data[name] = records(df)
    alpha = pd.read_csv(OUTPUT / "stats_alpha_summary.csv")
    data["alpha_summary"] = records(alpha)
    corr = pd.read_csv(OUTPUT / "stats_correlations.csv", index_col=0).loc[DASHBOARD_SIGNALS, DASHBOARD_SIGNALS]
    data["corr"] = {"labels": DASHBOARD_SIGNALS,
                    "values": [[clean(v) for v in row] for row in corr.to_numpy()]}
    return data


def load_timeseries():
    ts = pd.read_parquet(OPENAP / "timeseries.parquet")
    out = {}
    for signal in DASHBOARD_SIGNALS:
        g = ts[ts["signal"] == signal].sort_values("date")
        out[signal] = {
            "date": g["date"].dt.strftime("%Y-%m-%d").tolist(),
            "wealth": [clean(v) for v in g["wealth"]],
            "drawdown_pct": [clean(v) for v in g["drawdown_pct"]],
            "rolling_sharpe_36m": [clean(v) for v in g["rolling_sharpe_36m"]],
        }
    return out


def plotly_tag(embed):
    if not embed:
        return f'<script src="{PLOTLY_URL}"></script>'
    if not PLOTLY_CACHE.exists():
        print(f"Downloading Plotly {PLOTLY_VERSION} from jsDelivr (once)...", flush=True)
        response = requests.get(PLOTLY_URL, timeout=120)
        response.raise_for_status()
        PLOTLY_CACHE.write_bytes(response.content)
    # "</script" inside the library would end our tag early; Plotly's minified file has
    # none, but guard anyway.
    library = PLOTLY_CACHE.read_text(encoding="utf-8").replace("</script", "<\\/script")
    return f"<script>{library}</script>"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--embed-plotly", action="store_true", help="inline Plotly for full offline use")
    args = parser.parse_args()
    make_folders()

    info = pd.read_csv(OUTPUT / "signal_info.csv").set_index("signal").loc[DASHBOARD_SIGNALS]
    cards = json.loads((OPENAP / "signal_cards.json").read_text(encoding="utf-8"))
    release = json.loads((OPENAP / "release_info.json").read_text())
    stats_meta = json.loads((OPENAP / "stats_meta.json").read_text())
    tables = load_tables()

    reg = pd.read_csv(OUTPUT / "stats_regressions.csv")
    reg_window = (f"{reg['start'].min()} to {reg['end'].max()} "
                  f"(signals starting later use their own first month)")
    full = pd.read_csv(OUTPUT / "stats_full_sample.csv")

    data = {
        "meta": {
            "signals": DASHBOARD_SIGNALS,
            "release": release["release"], "hf_branch": release["hugging_face_branch"],
            "package_version": release["openassetpricing_package_version"],
            "downloaded_on": release["downloaded_on"],
            "data_end": full["last"].max(),
            "reg_window": reg_window,
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "plotly_version": PLOTLY_VERSION, "plotly_embedded": args.embed_plotly,
            **stats_meta,
        },
        "info": {s: {k: clean(v) for k, v in row.items()} for s, row in info.iterrows()},
        "cards": cards["cards"], "references": cards["references"],
        "timeseries": load_timeseries(),
        **tables,
    }

    # "</" must not appear inside a <script> block; "<\/" is the same string to JSON.parse.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    for placeholder in ["__PLOTLY_TAG__", "__DATA__"]:
        if html.count(placeholder) != 1:
            raise ValueError(f"Template must contain {placeholder} exactly once")
    html = html.replace("__PLOTLY_TAG__", plotly_tag(args.embed_plotly)).replace("__DATA__", payload)
    OUTPUT_FILE.write_text(html, encoding="utf-8")

    size_mb = OUTPUT_FILE.stat().st_size / 1e6
    print(f"Wrote {OUTPUT_FILE.relative_to(OUTPUT.parent)} ({size_mb:.1f} MB, Plotly "
          f"{'embedded' if args.embed_plotly else 'from CDN ' + PLOTLY_URL})")


if __name__ == "__main__":
    main()
