# ABOUTME: Downloads AQR's monthly US Quality-Minus-Junk (QMJ) factor from the AQR data library.
"""
Run:     python DataDownloads/J_AQRQMJ.py   (from pyCode/)
Inputs:  AQR data library, Quality-Minus-Junk-Factors-Monthly.xlsx
Outputs: pyData/Intermediate/aqr_qmj.parquet   date (month-end), qmj (US, decimal)

QMJ (Asness, Frazzini and Pedersen 2019) is long high-quality stocks (profitable,
growing, safe, high payout) and short "junk". Asness et al. (2018) "Size matters, if
you control your junk" show that small stocks are on average junky, so SMB is short
quality; once you control for that, the size premium reappears. We use AQR's QMJ as
the quality benchmark in the SMBQ regressions.

File layout (checked on 2026-11-05): sheet "QMJ Factors", a text header, then a header
row "DATE, AUS, AUT, ..., USA, Global, ..." (row 19 in Excel), then monthly rows with
dates as text "MM/DD/YYYY" and returns as decimals (US mean ~0.3% per month).
AQR reconstructs the full history at every update, so past values can change slightly
between downloads.
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402
import requests  # noqa: E402

from config import INTERMEDIATE, REFRESH_DOWNLOADS, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402

OUTPUT_FILE = INTERMEDIATE / "aqr_qmj.parquet"
URL = ("https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
       "Quality-Minus-Junk-Factors-Monthly.xlsx")
SHEET = "QMJ Factors"


def download():
    response = requests.get(URL, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    raw = pd.read_excel(io.BytesIO(response.content), sheet_name=SHEET, header=None,
                        engine="openpyxl")

    # Find the header row instead of hard-coding its position (it is the row starting
    # with "DATE"), and check that the US column is there.
    header_row = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "DATE"][0]
    header = raw.iloc[header_row].astype(str).str.strip().tolist()
    if "USA" not in header:
        raise SystemExit(f"AQR file: no USA column in the header {header}")

    data = raw.iloc[header_row + 1:].copy()
    data.columns = header
    qmj = pd.DataFrame({
        "date": pd.to_datetime(data["DATE"], format="%m/%d/%Y", errors="coerce"),
        "qmj": pd.to_numeric(data["USA"], errors="coerce"),
    }).dropna()
    # Month-end dates, to match our CRSP dates (AQR already uses month-ends).
    qmj["date"] = qmj["date"] + pd.offsets.MonthEnd(0)
    return qmj.reset_index(drop=True)


def main():
    make_folders()
    if REFRESH_DOWNLOADS or not OUTPUT_FILE.exists():
        print("Downloading AQR's QMJ factor...", flush=True)
        qmj = download()
        require_unique(qmj, ["date"], "AQR QMJ")
        qmj.to_parquet(OUTPUT_FILE, index=False)
    else:
        print(f"Using cached {OUTPUT_FILE.name} (REFRESH_DOWNLOADS = False)", flush=True)
        qmj = pd.read_parquet(OUTPUT_FILE)

    print(f"months: {len(qmj):,}   dates: {qmj['date'].min().date()} to {qmj['date'].max().date()}")
    print(f"US QMJ mean {qmj['qmj'].mean() * 1200:.2f}% p.a., std "
          f"{qmj['qmj'].std() * 12 ** 0.5 * 100:.2f}% p.a. (sanity check: decimals, not percent)")


if __name__ == "__main__":
    main()
