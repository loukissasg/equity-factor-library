# ABOUTME: BMdec signal (Fama-French 1993): book equity of fiscal year t-1 over market equity of December t-1.
"""
Run:     python Predictors/BMdec.py   (from pyCode/)
Inputs:  pyData/Intermediate/compustat_annual.parquet
         pyData/Intermediate/ccm_links.parquet
         pyData/Intermediate/crsp_monthly_from_daily.parquet       (PERMNO -> PERMCO map)
         pyData/Intermediate/SignalMasterTable.parquet  (December ME, June PERMNO)
Outputs: pyData/Predictors/BMdec.parquet   [permno, yyyymm, BMdec]
         yyyymm = YYYY06 (the June t sort that uses the signal)

Why "BMdec" and not "BM": in OpenAP, "BM" is Stattman (1980) book-to-market (common
equity over *current* ME, updated monthly). "BMdec" is their name for the Fama-French
version: book equity over *December* ME. That is the signal behind HML.

Timing: firms publish annual accounts with a delay of up to several months. Using the
fiscal year ending in calendar year t-1 in June t guarantees (at least 6 months later)
that book equity was public when portfolios are formed: no look-ahead bias. Book and
market equity are both measured at the end of year t-1, so B/M is not distorted by
price moves between December and June.

Missing BMdec (not saved) when: BE <= 0, no CCM link, no December ME, or fewer than two
years in Compustat (see REQUIRE_2Y_COMPUSTAT in config.py).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIRST_SORT_YEAR, INTERMEDIATE, REQUIRE_2Y_COMPUSTAT, make_folders  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.save_predictor import save_predictor  # noqa: E402


def book_equity(comp):
    """Ken French's definition: BE = shareholders' equity + deferred taxes - preferred stock."""
    be = comp.copy()

    # Shareholders' equity: seq; else common equity + preferred stock; else assets - liabilities.
    be["se"] = be["seq"]
    be["se"] = be["se"].fillna(be["ceq"] + be["pstk"])
    be["se"] = be["se"].fillna(be["at"] - be["lt"])

    # Preferred stock: redemption, else liquidating, else par value, else 0. Preferred
    # shareholders are paid before common shareholders, so their claim is removed: BE
    # should measure what belongs to common shareholders only.
    be["ps"] = be["pstkrv"].fillna(be["pstkl"]).fillna(be["pstk"]).fillna(0)

    # Deferred taxes are added back (an accounting liability unlikely to be paid in full,
    # so economically equity). Missing counts as 0.
    be["be"] = be["se"] + be["txditc"].fillna(0) - be["ps"]

    # The year that matters is the calendar year in which the fiscal year ended (June and
    # December 2010 year-ends both feed the June 2011 sort). If a firm changed its fiscal
    # year-end and has two records ending in the same calendar year, keep the later one.
    be["fy_calendar_year"] = be["datadate"].dt.year
    be = be.sort_values(["gvkey", "datadate"])
    be = be.drop_duplicates(["gvkey", "fy_calendar_year"], keep="last")

    # Years in Compustat: 1 for a firm's first record, 2 for its second, and so on.
    be["years_in_compustat"] = be.groupby("gvkey").cumcount() + 1
    return be


def link_to_crsp(be, ccm, crsp):
    """Attach a PERMCO to each Compustat firm-year via the CCM link table."""
    # A link is valid only within its linkdt-linkenddt window. We check it at the
    # portfolio formation date (end of June of the sort year), when the data is used.
    be["june_date"] = pd.to_datetime(be["sort_year"].astype(str) + "-06-30")
    linked = be.merge(ccm, on="gvkey", how="inner")
    linked = linked[(linked["june_date"] >= linked["linkdt"]) &
                    (linked["june_date"] <= linked["linkenddt"])]

    # CCM links to a PERMNO, but the master table has one row per firm (PERMCO), kept
    # under the share class with the largest ME, which is not always the linked class.
    # Linking at the PERMCO level avoids losing those firms.
    permno_to_permco = crsp[["permno", "permco"]].drop_duplicates("permno", keep="last")
    linked = linked.merge(permno_to_permco, on="permno", how="inner")

    # Rarely, two gvkeys map to the same firm in the same year. Prefer the primary link
    # (linkprim 'P'), then the larger book equity.
    n_linked = len(linked)
    linked = linked.sort_values(["permco", "sort_year", "linkprim", "be"],
                                ascending=[True, True, False, False])
    linked = linked.drop_duplicates(["permco", "sort_year"], keep="first")
    print(f"Compustat firm-years with BE: {len(be):,}   linked to CRSP: {n_linked:,}   "
          f"after removing duplicate firm-years: {len(linked):,}")
    return linked[["permco", "sort_year", "gvkey", "be", "years_in_compustat"]]


def main():
    make_folders()
    print("Building the BMdec signal...", flush=True)
    comp = pd.read_parquet(INTERMEDIATE / "compustat_annual.parquet")
    ccm = pd.read_parquet(INTERMEDIATE / "ccm_links.parquet")
    crsp = pd.read_parquet(INTERMEDIATE / "crsp_monthly_from_daily.parquet")
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet")

    # --- Book equity ----------------------------------------------------------------------
    be = book_equity(comp)
    print(f"firm-years: {len(be):,}   with BE: {be['be'].notna().mean():.1%}   "
          f"BE > 0 among those: {(be['be'] > 0).sum() / be['be'].notna().sum():.1%}")

    be["sort_year"] = be["fy_calendar_year"] + 1  # fiscal year t-1 feeds the June t sort
    be = be[be["be"].notna()][["gvkey", "sort_year", "datadate", "be", "years_in_compustat"]]

    # --- Link to CRSP -------------------------------------------------------------------------
    linked = link_to_crsp(be, ccm, crsp)

    # --- December ME and B/M ---------------------------------------------------------------------
    december = panel[panel["date"].dt.month == 12].copy()
    december["sort_year"] = december["date"].dt.year + 1  # December t-1 feeds the June t sort
    december = december[december["sort_year"] >= FIRST_SORT_YEAR]
    december = december.rename(columns={"me": "me_dec"})[["permco", "sort_year", "me_dec"]]

    bm = december.merge(linked, on=["permco", "sort_year"], how="left")

    # B/M only for positive book equity. A firm with negative BE has accumulated losses
    # larger than its paid-in capital; its B/M would be negative and meaningless as a
    # value signal, so Fama-French leave these firms out of the B/M sort.
    # Units: Compustat BE is in $ millions, our ME in $ thousands, hence ME / 1,000.
    positive = (bm["be"] > 0) & (bm["me_dec"] > 0)
    bm["BMdec"] = np.nan
    bm.loc[positive, "BMdec"] = bm.loc[positive, "be"] / (bm.loc[positive, "me_dec"] / 1000)

    # --- At least two years in Compustat ------------------------------------------------------------
    # Fama-French (1993) use a firm only once it has appeared in Compustat for two years.
    # When Compustat added a firm it often back-filled its earlier history, and the firms it
    # chose to add were mostly ones that had survived and done well. A firm's first
    # Compustat year is therefore biased towards winners (survivorship bias). This is a
    # property of the *accounting data*, so it belongs here, in the signal, and not in the
    # portfolio code (which must work for any signal).
    n_young = (bm["BMdec"].notna() & (bm["years_in_compustat"] < 2)).sum()
    if REQUIRE_2Y_COMPUSTAT:
        bm.loc[bm["years_in_compustat"] < 2, "BMdec"] = np.nan
    print(f"firm-years with a B/M but fewer than 2 years in Compustat: {n_young:,} "
          f"({'removed' if REQUIRE_2Y_COMPUSTAT else 'kept, REQUIRE_2Y_COMPUSTAT = False'})")

    # --- Standard format: [permno, yyyymm, BMdec] ---------------------------------------------------
    # The signal is a firm (PERMCO) property, but the standard format is keyed by PERMNO.
    # We use the PERMNO the master table carries for that firm in June t (its largest share
    # class), which is the same PERMNO the size signal and the portfolio code use.
    june = panel[panel["date"].dt.month == 6][["permco", "permno", "date"]].copy()
    june["sort_year"] = june["date"].dt.year
    bm = bm.merge(june[["permco", "sort_year", "permno"]], on=["permco", "sort_year"], how="inner")
    bm["yyyymm"] = bm["sort_year"] * 100 + 6
    require_unique(bm, ["permco", "sort_year"], "BMdec")

    saved = save_predictor(bm, "BMdec")

    # --- Sanity checks ----------------------------------------------------------------------------
    per_year = bm.groupby("sort_year").agg(
        firms_with_dec_me=("me_dec", lambda s: s.notna().sum()),
        with_be=("be", lambda s: s.notna().sum()),
        with_bmdec=("BMdec", lambda s: s.notna().sum()),
        median_bmdec=("BMdec", "median"),
    )
    per_year["median_bmdec"] = per_year["median_bmdec"].round(2)
    print("BMdec per sort year, every 5th year (firms with June and December ME):")
    print(per_year[(per_year.index % 5 == 0) | (per_year.index == per_year.index.max())]
          .to_string())
    no_be = bm["me_dec"].notna() & bm["be"].isna()
    print(f"CRSP firms with December ME but no linked Compustat BE: {no_be.mean():.1%}")
    print(f"firm-years with BE <= 0 (no B/M): {(bm['be'] <= 0).sum():,}")
    print(f"BMdec quantiles (5/50/95%): "
          f"{saved['BMdec'].quantile([0.05, 0.5, 0.95]).round(2).tolist()}")


if __name__ == "__main__":
    main()
