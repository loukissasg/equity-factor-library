# ABOUTME: Downloads OpenAP's original-paper (OP) portfolio returns and SignalDoc for the 10 dashboard signals.
"""
Run:     python SignalDashboard/A_OpenAPDownload.py   (from pyCode/)
Inputs:  Open Source Asset Pricing (Chen and Zimmermann), release OPENAP_RELEASE in config.py,
         read with the official `openassetpricing` package (pip install openassetpricing).
         The package reads public parquet/csv files from the Hugging Face dataset
         OpenSourceAP/CrossSection (no account needed).
Outputs: pyData/OpenAP/op_portfolios.parquet   long: signalname, port, date, ret, signallag, Nlong, Nshort
         pyData/OpenAP/ls_returns.parquet      wide: date x signal, long-short return (% per month)
         pyData/OpenAP/signal_doc_full.parquet all SignalDoc columns, our 10 signals only
         pyData/OpenAP/release_info.json       which release, files and package version we used

Why OpenAP's "OP" portfolios: they follow each original paper's construction (its
quantiles, its equal/value weighting, its holding period), so the returns are the
closest public replication of what the authors reported. The alternative sets (deciles,
NYSE breakpoints, ...) change the recipe, which we may use later as robustness checks.

What the file contains (checked on 2026-11-05 against the data, not just the docs):
- port "01" ... "05" or "10": quantile portfolios, sorted on the SIGNED signal
  (signal x SignalDoc 'Sign'). E.g. IdioVol3F has Sign = -1, so its signallag is negative
  in every portfolio and port 05 holds the LOWEST-volatility stocks.
- port "LS": highest portfolio minus lowest. Because the sort uses the signed signal, LS is
  always "long the side the original paper says earns more, short the other side".
- ret: monthly return in PERCENT (1.0 = 1%). LS is a zero-cost (self-financing) return, so
  no risk-free rate should be subtracted from it.
- signallag: average signal of the stocks in the portfolio (signed, lagged).
- Nlong / Nshort: number of stocks in the long / short leg (Nshort = 0 for ports 01..10).
- date: the last trading day of the month; we convert it to the calendar month-end so it
  merges with our other monthly files.
"""

import io
import json
import sys
from datetime import date
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import openassetpricing as oap  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

from config import (DASHBOARD_SIGNALS, OPENAP, OPENAP_RELEASE, REFRESH_DOWNLOADS,  # noqa: E402
                    make_folders)
from utils.checks import require_columns, require_unique  # noqa: E402

PORTS_FILE = OPENAP / "op_portfolios.parquet"
LS_FILE = OPENAP / "ls_returns.parquet"
DOC_FILE = OPENAP / "signal_doc_full.parquet"
INFO_FILE = OPENAP / "release_info.json"

# Hugging Face branch of the pinned release (as printed by oap.list_release()). Only used
# for the cross-check with the wide long-short file, which the package does not download.
HF_BRANCH = "Data-Release-2026.10"
HF_BASE = f"https://huggingface.co/datasets/OpenSourceAP/CrossSection/resolve/{HF_BRANCH}"
LS_WIDE_PATH = "Portfolios/Full Sets OP/PredictorLSretWide.parquet"


def download():
    openap = oap.OpenAP(OPENAP_RELEASE)

    # 'op' = Portfolios/Full Sets OP/PredictorPortsFull.parquet (see the package's hf_parse.py).
    ports = openap.dl_port("op", "pandas", DASHBOARD_SIGNALS)
    doc = openap.dl_signal_doc("pandas")

    require_columns(ports, ["signalname", "port", "date", "ret", "signallag", "Nlong", "Nshort"],
                    "PredictorPortsFull")
    missing = sorted(set(DASHBOARD_SIGNALS) - set(ports["signalname"]))
    if missing:
        raise SystemExit(f"Signals not found in the OP portfolio file: {missing}")

    # Calendar month-end dates, so OpenAP merges with French/AQR/our FF3 files.
    ports["date"] = pd.to_datetime(ports["date"]) + pd.offsets.MonthEnd(0)
    ports = ports.sort_values(["signalname", "port", "date"]).reset_index(drop=True)

    doc = doc[doc["Acronym"].isin(DASHBOARD_SIGNALS)].reset_index(drop=True)
    if len(doc) != len(DASHBOARD_SIGNALS):
        raise SystemExit(f"SignalDoc has {len(doc)} rows for our {len(DASHBOARD_SIGNALS)} signals")

    info = {
        "source": "Open Source Asset Pricing (Chen and Zimmermann), openassetpricing.com",
        "release": OPENAP_RELEASE,
        "hugging_face_branch": HF_BRANCH,
        "files": ["Portfolios/Full Sets OP/PredictorPortsFull.parquet", "SignalDoc.csv",
                  f"{LS_WIDE_PATH} (cross-check only)"],
        "openassetpricing_package_version": version("openassetpricing"),
        "downloaded_on": date.today().isoformat(),
    }
    return ports, doc, info


def check_ls_definition(ports):
    """Stop if LS is not 'top portfolio minus bottom portfolio' (what every later stage assumes)."""
    wide = ports.pivot_table(index=["signalname", "date"], columns="port", values="ret")
    for signal, block in wide.groupby(level="signalname"):
        block = block.dropna(axis=1, how="all")
        quantile_ports = sorted(c for c in block.columns if c != "LS")
        top_minus_bottom = block[quantile_ports[-1]] - block[quantile_ports[0]]
        gap = (block["LS"] - top_minus_bottom).abs().max()
        if gap > 1e-8:
            raise ValueError(f"{signal}: LS differs from port {quantile_ports[-1]} - port "
                             f"{quantile_ports[0]} by up to {gap}")


def cross_check_wide_file(ls):
    """Compare our LS series with OpenAP's own wide long-short file (two files, one answer)."""
    url = f"{HF_BASE}/{LS_WIDE_PATH.replace(' ', '%20')}"
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    wide = pd.read_parquet(io.BytesIO(response.content))
    wide["date"] = pd.to_datetime(wide["date"]) + pd.offsets.MonthEnd(0)
    wide = wide.set_index("date")[DASHBOARD_SIGNALS]
    aligned_ours, aligned_wide = ls.align(wide, join="outer")
    max_gap = (aligned_ours - aligned_wide).abs().max().max()
    same_missing = (aligned_ours.isna() == aligned_wide.isna()).all().all()
    print(f"Cross-check vs PredictorLSretWide.parquet: max abs difference {max_gap:.2e} "
          f"(% per month), same missing months: {same_missing}")
    if max_gap > 1e-8 or not same_missing:
        raise ValueError("LS rows of PredictorPortsFull do not match PredictorLSretWide")


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not PORTS_FILE.exists():
        print(f"Downloading OpenAP release {OPENAP_RELEASE} (OP portfolios + SignalDoc)...",
              flush=True)
        ports, doc, info = download()
        require_unique(ports, ["signalname", "port", "date"], "OP portfolios")
        check_ls_definition(ports)

        # Wide LS table: the main series for every later stage.
        ls = (ports[ports["port"] == "LS"]
              .pivot(index="date", columns="signalname", values="ret")[DASHBOARD_SIGNALS])
        ls.columns.name = None
        cross_check_wide_file(ls)

        ports.to_parquet(PORTS_FILE, index=False)
        ls.to_parquet(LS_FILE)
        doc.to_parquet(DOC_FILE, index=False)
        INFO_FILE.write_text(json.dumps(info, indent=2))
    else:
        print(f"Using cached {PORTS_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        ports = pd.read_parquet(PORTS_FILE)
        ls = pd.read_parquet(LS_FILE)
        info = json.loads(INFO_FILE.read_text())

    print(f"Release {info['release']} ({info['hugging_face_branch']}), package "
          f"openassetpricing {info['openassetpricing_package_version']}, downloaded "
          f"{info['downloaded_on']}")
    print(f"OP portfolio rows: {len(ports):,}   LS table: {ls.shape[0]:,} months x "
          f"{ls.shape[1]} signals")
    print("LS = highest minus lowest portfolio of the SIGNED signal, checked for all signals.")


if __name__ == "__main__":
    main()
