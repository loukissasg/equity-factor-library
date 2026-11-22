# ABOUTME: GP signal (Novy-Marx 2013): gross profits of fiscal year t-1 over total assets. Robustness check for OperProf.
"""
Run:     python Predictors/GP.py   (from pyCode/)
Inputs:  pyData/Intermediate/compustat_annual.parquet
         pyData/Intermediate/compustat_names.parquet    (SIC, to drop financial firms)
         pyData/Intermediate/ccm_links.parquet
         pyData/Intermediate/crsp_monthly_from_daily.parquet       (PERMNO -> PERMCO map)
         pyData/Intermediate/SignalMasterTable.parquet  (December ME, June PERMNO)
Outputs: pyData/Predictors/GP.parquet   [permno, yyyymm, GP]
         yyyymm = YYYY06 (the June t sort that uses the signal)

Definition: GP = (revt - cogs) / at, fiscal year ending in calendar year t-1.

Why gross profits, scaled by assets: Novy-Marx argues that gross profit is the
"cleanest" accounting measure of economic profitability. Items further down the income
statement (SG&A, which includes R&D and advertising; interest; taxes) mix in
investment and financing choices. Assets, not book equity, are the denominator so the
ratio does not depend on how the firm is financed (debt vs equity).

Why a robustness check: OperProf (French's RMW) and GP measure the same idea in two
different ways. If a size x GP sort gives a similar factor, the profitability premium
is not an artefact of one accounting definition.

Timing and sample: the same rules as BMdec and OperProf: fiscal year t-1 used in June t,
December t-1 ME required, two years in Compustat. Unlike OperProf, no positive-BE
requirement: GP does not use book equity.

Financial firms are dropped, exactly as in OpenAP's GP.py (and Novy-Marx 2013): SIC from
comp.names, converted to a number, keep SIC < 6000 or SIC >= 7000. For a bank, "COGS" is
mostly interest expense and assets are huge, so GP is mechanically tiny and says nothing
about profitability. Kept in, banks made up 50-57% of the big low-GP portfolio's value
after 1995, so RMW_GP was partly a short position in big banks.

Remaining differences from OpenAP's Predictors/GP.py:
  1. They update monthly: each annual record is used from 6 months after the fiscal
     year-end, for 12 months. We use the June-t value for a July-June holding period.
  2. They require revt, cogs and at all present but do not drop at <= 0; we require
     at > 0 (a ratio over zero or negative assets is meaningless).
  3. Their sample is all CCM-linked firms in Compustat; we also require December t-1 ME
     and two years in Compustat, as for BMdec.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIRST_SORT_YEAR, INTERMEDIATE, REQUIRE_2Y_COMPUSTAT, make_folders  # noqa: E402
# book_equity() also keeps one record per firm and calendar year and counts years in
# Compustat; link_to_crsp() is the CCM link. Reusing both gives GP the same firm-years and
# the same links as BMdec and OperProf.
from Predictors.BMdec import book_equity, link_to_crsp  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.save_predictor import save_predictor  # noqa: E402


def main():
    make_folders()
    print("Building the GP signal...", flush=True)
    comp = pd.read_parquet(INTERMEDIATE / "compustat_annual.parquet")
    names = pd.read_parquet(INTERMEDIATE / "compustat_names.parquet")
    ccm = pd.read_parquet(INTERMEDIATE / "ccm_links.parquet")
    crsp = pd.read_parquet(INTERMEDIATE / "crsp_monthly_from_daily.parquet")
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet")

    firm_years = book_equity(comp)

    # --- Drop financial firms (exactly as OpenAP's GP.py) -------------------------------------
    # SIC is text in comp.names; to_numeric(errors="coerce") turns anything non-numeric into
    # missing. Keeping SIC < 6000 or >= 7000 drops finance, insurance and real estate
    # (6000-6999). As in OpenAP, a missing SIC fails both comparisons, so it is dropped too.
    # The years_in_compustat count is taken before this filter, so it still counts all of
    # the firm's Compustat years.
    firm_years = firm_years.merge(names[["gvkey", "sic"]], on="gvkey", how="left")
    firm_years["sic"] = pd.to_numeric(firm_years["sic"], errors="coerce")
    n_before = len(firm_years)
    is_financial = firm_years["sic"].between(6000, 6999)
    firm_years = firm_years[(firm_years["sic"] < 6000) | (firm_years["sic"] >= 7000)].copy()
    print(f"firm-years: {n_before:,}   financials dropped: {is_financial.sum():,}   "
          f"missing SIC dropped: {n_before - len(firm_years) - is_financial.sum():,}")

    # --- Gross profitability per firm-year -------------------------------------------------------
    # Both revenue and COGS must be reported: a missing COGS set to 0 would turn revenue
    # into gross profit and make the firm look very profitable.
    valid = firm_years["revt"].notna() & firm_years["cogs"].notna() & (firm_years["at"] > 0)
    firm_years["GP"] = np.nan
    firm_years.loc[valid, "GP"] = ((firm_years.loc[valid, "revt"] - firm_years.loc[valid, "cogs"])
                                   / firm_years.loc[valid, "at"])
    print(f"firm-years: {len(firm_years):,}   with revt, cogs and at > 0: {valid.mean():.1%}")

    firm_years["sort_year"] = firm_years["fy_calendar_year"] + 1  # fiscal year t-1 -> June t
    with_gp = firm_years[firm_years["GP"].notna()].copy()

    # --- Link to CRSP --------------------------------------------------------------------------------
    linked = link_to_crsp(with_gp[["gvkey", "sort_year", "datadate", "be",
                                   "years_in_compustat"]].copy(), ccm, crsp)
    linked = linked.merge(with_gp[["gvkey", "sort_year", "GP"]], on=["gvkey", "sort_year"],
                          how="left")

    # --- December t-1 ME: same universe rule as BMdec and OperProf ---------------------------------
    december = panel[panel["date"].dt.month == 12].copy()
    december["sort_year"] = december["date"].dt.year + 1
    december = december[december["sort_year"] >= FIRST_SORT_YEAR]
    december = december.rename(columns={"me": "me_dec"})[["permco", "sort_year", "me_dec"]]
    gp = december.merge(linked, on=["permco", "sort_year"], how="left")
    gp.loc[~(gp["me_dec"] > 0), "GP"] = np.nan

    # --- At least two years in Compustat (survivorship bias, see BMdec.py) -----------------------
    n_young = (gp["GP"].notna() & (gp["years_in_compustat"] < 2)).sum()
    if REQUIRE_2Y_COMPUSTAT:
        gp.loc[gp["years_in_compustat"] < 2, "GP"] = np.nan
    print(f"firm-years with GP but fewer than 2 years in Compustat: {n_young:,} "
          f"({'removed' if REQUIRE_2Y_COMPUSTAT else 'kept, REQUIRE_2Y_COMPUSTAT = False'})")

    # --- Standard format: [permno, yyyymm, GP] ------------------------------------------------------
    june = panel[panel["date"].dt.month == 6][["permco", "permno", "date"]].copy()
    june["sort_year"] = june["date"].dt.year
    gp = gp.merge(june[["permco", "sort_year", "permno"]], on=["permco", "sort_year"], how="inner")
    gp["yyyymm"] = gp["sort_year"] * 100 + 6
    require_unique(gp, ["permco", "sort_year"], "GP")

    saved = save_predictor(gp, "GP")

    # --- Sanity checks ----------------------------------------------------------------------------
    per_year = gp.groupby("sort_year").agg(
        firms_with_dec_me=("me_dec", lambda s: s.notna().sum()),
        with_gp=("GP", lambda s: s.notna().sum()),
        median_gp=("GP", "median"),
    )
    per_year["median_gp"] = per_year["median_gp"].round(3)
    print("GP per sort year, every 5th year (firms with June and December ME):")
    print(per_year[(per_year.index % 5 == 0) | (per_year.index == per_year.index.max())]
          .to_string())
    print(f"GP quantiles (1/5/50/95/99%): "
          f"{saved['GP'].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(3).tolist()}")


if __name__ == "__main__":
    main()
