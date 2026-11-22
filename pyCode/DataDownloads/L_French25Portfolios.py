# ABOUTME: Downloads Ken French's 25 value-weighted size x B/M portfolios, the test assets for the model comparison.
"""
Run:     python DataDownloads/L_French25Portfolios.py   (from pyCode/)
Inputs:  Ken French data library, 25_Portfolios_5x5_CSV.zip
Outputs: pyData/Intermediate/french_25_portfolios.parquet
         date (month-end) and 25 columns ME1_BM1 ... ME5_BM5 (raw returns, decimals)
         ME1 = smallest size quintile, BM1 = lowest B/M (growth), BM5 = highest (value)

Why these: they are the classic test assets for asset-pricing models (Fama-French
1993). A good model should explain their average returns with alphas close to zero;
the GRS test asks whether all 25 alphas are jointly zero.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.french_library import download_french_csv  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "french_25_portfolios.parquet"
TITLE = "Average Value Weighted Returns -- Monthly"
COLUMNS = [f"ME{s}_BM{b}" for s in range(1, 6) for b in range(1, 6)]
# French's names, in his order: size quintile outer loop, B/M inner loop.
FRENCH_COLUMNS = (["SMALL LoBM", "ME1 BM2", "ME1 BM3", "ME1 BM4", "SMALL HiBM"]
                  + [f"ME{s} BM{b}" for s in range(2, 5) for b in range(1, 6)]
                  + ["BIG LoBM", "ME5 BM2", "ME5 BM3", "ME5 BM4", "BIG HiBM"])


def download():
    lines = download_french_csv("25_Portfolios_5x5_CSV.zip")

    # Read only the first table (monthly value-weighted returns): from its title to the
    # next title. Check the column order against French's names.
    start = next(i for i, line in enumerate(lines) if line.strip() == TITLE)
    header = [f.strip() for f in lines[start + 1].split(",")][1:]
    if header != FRENCH_COLUMNS:
        raise SystemExit(f"unexpected columns in the 25-portfolio file: {header}")
    rows = []
    for line in lines[start + 2:]:
        fields = [f.strip() for f in line.split(",")]
        if not (len(fields) == 26 and len(fields[0]) == 6 and fields[0].isdigit()):
            break  # end of the monthly value-weighted table
        rows.append(fields)

    p25 = pd.DataFrame(rows, columns=["yyyymm"] + COLUMNS)
    p25[COLUMNS] = p25[COLUMNS].astype(float)
    p25[COLUMNS] = p25[COLUMNS].mask(p25[COLUMNS].isin([-99.99, -999])) / 100  # missing; % -> decimal
    p25["date"] = pd.to_datetime(p25["yyyymm"], format="%Y%m") + pd.offsets.MonthEnd(0)
    return p25[["date"] + COLUMNS]


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading Ken French's 25 size x B/M portfolios...", flush=True)
        p25 = download()
        require_unique(p25, ["date"], "French 25 portfolios")
        p25.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        p25 = pd.read_parquet(OUTPUT_FILE)

    print(f"months: {len(p25):,}   dates: {p25['date'].min().date()} to {p25['date'].max().date()}   "
          f"missing values: {int(p25[COLUMNS].isna().sum().sum())}")
    print("Mean monthly return (%), 1963-07 on (rows = size quintile, columns = B/M quintile):")
    recent = p25[p25["date"] >= "1963-07-01"][COLUMNS].mean() * 100
    print(recent.values.reshape(5, 5).round(2))


if __name__ == "__main__":
    main()
