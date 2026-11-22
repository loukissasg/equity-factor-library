# ABOUTME: QualityComp signal: a simple QMJ-style composite of profitability and safety (OperProf, low leverage, low volatility).
"""
Run:     python Predictors/QualityComp.py   (from pyCode/, after OperProf.py)
Inputs:  pyData/Predictors/OperProf.parquet
         pyData/Intermediate/compustat_annual.parquet, ccm_links.parquet, crsp_monthly_from_daily.parquet
         pyData/Intermediate/SignalMasterTable.parquet   (returns, June PERMNO)
Outputs: pyData/Predictors/QualityComp.parquet   [permno, yyyymm, QualityComp]
         yyyymm = YYYY06 (the June t sort that uses the signal)

Why: AQR's QMJ defines quality as profitability + growth + safety + payout. Sorting on
OperProf alone controls only the first, so SMBQ stayed short "safety" (small stocks are
also riskier and more indebted). This composite adds two safety measures we can build
from our own data. It is a simplified version of QMJ, not a replica.

Components, each measured with data known at the end of June t:
  1. OperProf             fiscal year t-1 (Predictors/OperProf.py)        high = quality
  2. Book leverage lt/at  fiscal year t-1, same CCM link as BMdec        low  = quality
  3. Return volatility    std of monthly returns, July t-3 to June t     low  = quality
                          (36 months, at least 24 required)

Combining (as AQR does): each June, rank every stock on each component, turn the ranks
into z-scores (mean 0, std 1), flip leverage and volatility so that high = safe, and
average the three. Ranks rather than raw values, so one extreme ratio cannot dominate,
and each component has the same weight. A stock needs all three components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIRST_SORT_YEAR, INTERMEDIATE, PREDICTORS, make_folders  # noqa: E402
from Predictors.BMdec import book_equity, link_to_crsp  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.save_predictor import save_predictor  # noqa: E402

VOL_WINDOW = 36   # months of returns for the volatility component
VOL_MIN_OBS = 24  # minimum returns in the window


def june_permno(panel):
    """PERMNO the master table carries for each firm in June (same as Size and BMdec)."""
    june = panel[panel["date"].dt.month == 6][["permco", "permno", "date"]].copy()
    june["sort_year"] = june["date"].dt.year
    return june[["permco", "sort_year", "permno"]]


def leverage(comp, ccm, crsp, panel):
    """Book leverage lt/at of fiscal year t-1, keyed by June t PERMNO."""
    firm_years = book_equity(comp)  # one record per firm and calendar year, as for BMdec
    firm_years["sort_year"] = firm_years["fy_calendar_year"] + 1
    valid = firm_years["lt"].notna() & (firm_years["at"] > 0)
    firm_years.loc[valid, "leverage"] = firm_years.loc[valid, "lt"] / firm_years.loc[valid, "at"]
    with_lev = firm_years[firm_years["leverage"].notna()].copy()
    linked = link_to_crsp(with_lev[["gvkey", "sort_year", "datadate", "be",
                                    "years_in_compustat"]].copy(), ccm, crsp)
    linked = linked.merge(with_lev[["gvkey", "sort_year", "leverage"]],
                          on=["gvkey", "sort_year"], how="left")
    linked = linked.merge(june_permno(panel), on=["permco", "sort_year"], how="inner")
    linked["yyyymm"] = linked["sort_year"] * 100 + 6
    return linked[["permno", "yyyymm", "leverage"]]


def volatility(panel):
    """Std of monthly returns over the 36 months up to June t, keyed by June t PERMNO."""
    # Per firm (PERMCO): the master table has one row per firm-month. The rolling window
    # counts rows, so it equals 36 calendar months only without gaps; months_spanned
    # checks that the window really covers at most 36 months.
    panel = panel.sort_values(["permco", "date"])
    grouped = panel.groupby("permco")["ret"]
    panel["vol"] = grouped.transform(lambda r: r.rolling(VOL_WINDOW, min_periods=VOL_MIN_OBS).std())
    first_date = panel.groupby("permco")["date"].shift(VOL_WINDOW - 1)
    months_spanned = ((panel["date"].dt.year - first_date.dt.year) * 12
                      + panel["date"].dt.month - first_date.dt.month + 1)
    panel.loc[months_spanned > VOL_WINDOW, "vol"] = np.nan

    june = panel[panel["date"].dt.month == 6].copy()
    june["yyyymm"] = june["date"].dt.year * 100 + 6
    return june[["permno", "yyyymm", "vol"]]


def rank_z(series):
    """Cross-sectional rank turned into a z-score (mean 0, std 1)."""
    ranks = series.rank()
    return (ranks - ranks.mean()) / ranks.std()


def main():
    make_folders()
    print("Building the QualityComp signal...", flush=True)
    comp = pd.read_parquet(INTERMEDIATE / "compustat_annual.parquet")
    ccm = pd.read_parquet(INTERMEDIATE / "ccm_links.parquet")
    crsp = pd.read_parquet(INTERMEDIATE / "crsp_monthly_from_daily.parquet")
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet")
    operprof = pd.read_parquet(PREDICTORS / "OperProf.parquet")

    lev = leverage(comp, ccm, crsp, panel)
    vol = volatility(panel)

    q = operprof.merge(lev, on=["permno", "yyyymm"], how="inner")
    q = q.merge(vol, on=["permno", "yyyymm"], how="inner")
    q = q[q["vol"].notna() & (q["yyyymm"] // 100 >= FIRST_SORT_YEAR)].copy()
    print(f"stock-Junes with OperProf: {len(operprof):,}   with all three components: {len(q):,}")

    # Each June: rank-z of each component; leverage and volatility enter with a minus sign
    # (low leverage, low volatility = safe = quality). Equal-weighted average.
    by_june = q.groupby("yyyymm")
    q["z_profit"] = by_june["OperProf"].transform(rank_z)
    q["z_low_leverage"] = -by_june["leverage"].transform(rank_z)
    q["z_low_vol"] = -by_june["vol"].transform(rank_z)
    q["QualityComp"] = q[["z_profit", "z_low_leverage", "z_low_vol"]].mean(axis=1)

    require_unique(q, ["permno", "yyyymm"], "QualityComp")
    save_predictor(q, "QualityComp")

    # --- Sanity checks ----------------------------------------------------------------------------
    print("Correlation between the components (pooled over all Junes):")
    print(q[["z_profit", "z_low_leverage", "z_low_vol", "QualityComp"]].corr().round(2).to_string())
    print("Median components by June, every 10th year:")
    med = q.groupby(q["yyyymm"] // 100)[["OperProf", "leverage", "vol"]].median()
    print(med[med.index % 10 == 5].round(3).to_string())


if __name__ == "__main__":
    main()
