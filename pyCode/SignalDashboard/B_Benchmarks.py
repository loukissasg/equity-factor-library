# ABOUTME: Combines our FF3, Ken French's FF5 and momentum, and AQR's QMJ into one monthly factor panel in percent.
"""
Run:     python SignalDashboard/B_Benchmarks.py   (from pyCode/)
Inputs:  pyData/Factors/FF3.parquet                 our replicated MKT-RF, SMB, HML (+ French RF)
         pyData/Intermediate/french_ff5.parquet     DataDownloads/G_French5Factors.py
         pyData/Intermediate/french_umd.parquet     DataDownloads/K_FrenchMomentum.py
         pyData/Intermediate/aqr_qmj.parquet        DataDownloads/J_AQRQMJ.py
Outputs: pyData/OpenAP/factors_monthly.parquet      date + one column per factor, % per month

We reuse the files the FF3 project already downloaded instead of downloading again, so
the dashboard regressions use exactly the factors we validated there. To refresh them,
run those download scripts with REFRESH_DOWNLOADS = True.

Why percent: OpenAP reports returns in percent (1.0 = 1%), our other files use decimals
(0.01 = 1%). Mixing them would make every alpha and loading 100x off, so we convert the
factors to percent here, once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, INTERMEDIATE, OPENAP, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402

OUTPUT_FILE = OPENAP / "factors_monthly.parquet"

# file -> (script that creates it, {column in file: column in our panel})
SOURCES = {
    FACTORS / "FF3.parquet": ("Factors/FF3Monthly.py",
                              {"mkt_rf": "mkt_rf_mine", "smb": "smb_mine", "hml": "hml_mine"}),
    INTERMEDIATE / "french_ff5.parquet": ("DataDownloads/G_French5Factors.py",
                                          {"mkt_rf": "mkt_rf", "smb": "smb", "hml": "hml",
                                           "rmw": "rmw", "cma": "cma", "rf": "rf"}),
    INTERMEDIATE / "french_umd.parquet": ("DataDownloads/K_FrenchMomentum.py", {"umd": "umd"}),
    INTERMEDIATE / "aqr_qmj.parquet": ("DataDownloads/J_AQRQMJ.py", {"qmj": "qmj"}),
}


def load(path, script, columns):
    if not path.exists():
        raise SystemExit(f"{path} not found. Run {script} first (from pyCode/).")
    df = pd.read_parquet(path)
    if "date" not in df.columns:   # our FF3 file stores the date as the index
        df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]) + pd.offsets.MonthEnd(0)
    require_unique(df, ["date"], path.name)
    # astype(float): our FF3 file uses pandas' nullable Float64; plain floats are simpler.
    return df.set_index("date")[list(columns)].rename(columns=columns).astype(float)


def main():
    make_folders()
    pieces = [load(path, script, cols) for path, (script, cols) in SOURCES.items()]
    # Outer join: keep every month any source has; each factor's own range is reported below.
    panel = pd.concat(pieces, axis=1, join="outer").sort_index() * 100   # decimals -> percent

    # Sanity check on units: monthly MKT-RF should average roughly 0.5% (i.e. now in percent).
    print(f"Mean MKT-RF (French): {panel['mkt_rf'].mean():.2f}% per month (expect ~0.5)")
    print(f"{'factor':<12}{'first':>12}{'last':>12}{'months':>8}{'mean %':>8}{'std %':>8}")
    for col in panel.columns:
        s = panel[col].dropna()
        print(f"{col:<12}{s.index.min():%Y-%m}{'':>5}{s.index.max():%Y-%m}{'':>5}"
              f"{len(s):>8,}{s.mean():>8.2f}{s.std():>8.2f}")

    panel.to_parquet(OUTPUT_FILE)
    print(f"Saved {OUTPUT_FILE.relative_to(OPENAP.parents[1])}")


if __name__ == "__main__":
    main()
