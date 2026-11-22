# ABOUTME: Downloads Ken French's monthly momentum factor (UMD, "Mom" in his file).
"""
Run:     python DataDownloads/K_FrenchMomentum.py   (from pyCode/)
Inputs:  Ken French data library, F-F_Momentum_Factor_CSV.zip
Outputs: pyData/Intermediate/french_umd.parquet   date (month-end), umd (decimal)

UMD (up minus down) = 1/2 (small winners + big winners) - 1/2 (small losers + big
losers), prior return from month -12 to -2, rebuilt monthly. Asness et al. (2018)
include it in their SMB regressions; we download it rather than build it (momentum is
not part of this project).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_umd.parquet"


def download():
    lines = download_french_csv("F-F_Momentum_Factor_CSV.zip")
    # Check the column, then keep the monthly block (6-digit date; the annual block has 4).
    if not any(line.strip().replace(" ", "") == ",Mom" for line in lines):
        raise SystemExit("momentum file: header ',Mom' not found")
    rows = []
    for line in lines:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) == 2 and len(fields[0]) == 6 and fields[0].isdigit():
            rows.append(fields)
    umd = pd.DataFrame(rows, columns=["yyyymm", "umd"])
    umd["umd"] = umd["umd"].astype(float)
    umd = umd[~umd["umd"].isin([-99.99, -999])]  # the file's missing-value codes
    umd["umd"] = umd["umd"] / 100  # percent -> decimal
    umd["date"] = pd.to_datetime(umd["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return umd[["date", "umd"]].reset_index(drop=True)


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Ken French's momentum factor...", flush=True)
        umd = download()
        require_unique(umd, ["date"], "French UMD")
        umd.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        umd = pd.read_parquet(OUTPUT_FILE)

    print(f"months: {len(umd):,}   dates: {umd['date'].min().date()} to {umd['date'].max().date()}   "
          f"mean {umd['umd'].mean() * 1200:.2f}% p.a.")


if __name__ == "__main__":
    main()
