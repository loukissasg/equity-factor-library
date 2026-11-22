# ABOUTME: OperProf signal (Fama-French 2015, Ken French's RMW definition): operating profits of fiscal year t-1 over book equity plus minority interest.
"""
Run:     python Predictors/OperProf.py   (from pyCode/)
Inputs:  pyData/Intermediate/compustat_annual.parquet
         pyData/Intermediate/ccm_links.parquet
         pyData/Intermediate/crsp_monthly_from_daily.parquet       (PERMNO -> PERMCO map)
         pyData/Intermediate/SignalMasterTable.parquet  (December ME, June PERMNO)
Outputs: pyData/Predictors/OperProf.parquet   [permno, yyyymm, OperProf]
         yyyymm = YYYY06 (the June t sort that uses the signal)

Definition (Ken French's variable definitions page, as used for RMW):
    OP = (revenues - cost of goods sold - SG&A - interest expense)
         / (book equity + minority interest),   fiscal year ending in calendar year t-1
    Compustat: (revt - cogs - xsga - xint) / (BE + mib)

Sample rules (French's 5-factor page): the firm needs December t-1 and June t ME,
positive BE for t-1, and "non-missing revenues and at least one of the following: cost
of goods sold, selling, general and administrative expenses, or interest expense".

Why profitability: a firm that turns its equity into more operating profit is, all
else equal, worth more per unit of book. Fama-French (2015) find that robust-profit
firms earn higher returns than weak-profit firms of the same size and B/M.

Timing: same rule as BMdec. The fiscal year ending in t-1 is at least six months old in
June t, so the accounts were public when the portfolios are formed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import FIRST_SORT_YEAR, INTERMEDIATE, REQUIRE_2Y_COMPUSTAT, make_folders  # noqa: E402
# Book equity and the CCM link come from BMdec.py, so OperProf uses *exactly* the same BE
# and the same gvkey -> firm link as HML (one definition, not two copies to keep in sync).
from Predictors.BMdec import book_equity, link_to_crsp  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.save_predictor import save_predictor  # noqa: E402

# French's own documentation is not consistent about the denominator. His variable
# definitions page says "book equity and minority interest"; the header of his
# 6_Portfolios_ME_OP_2x3 file says OP is "divided by book equity". True follows the
# definitions page. (See Stage 2 notes: we checked which version tracks his RMW better.)
ADD_MINORITY_INTEREST = True


def operating_profitability(be):
    """Add the OP numerator and denominator to the firm-year table from book_equity()."""
    op = be.copy()

    # Expenses: a missing item counts as 0, as long as at least one of the three is
    # reported. Many firms report SG&A but no separate COGS (or the reverse), and treating
    # one missing line as "no profit at all" would throw away much of the sample.
    has_revenue = op["revt"].notna()
    has_an_expense = op[["cogs", "xsga", "xint"]].notna().any(axis=1)
    op["op_profit"] = (op["revt"] - op["cogs"].fillna(0) - op["xsga"].fillna(0)
                       - op["xint"].fillna(0))
    op.loc[~(has_revenue & has_an_expense), "op_profit"] = np.nan

    # Denominator: book equity plus minority interest (French's current definition; the
    # 2015 paper used BE alone). Consolidated revenues and expenses include the
    # subsidiaries' minority shareholders' share, so the equity base should include it too.
    # Missing mib counts as 0 (most firms have no minority interest).
    if ADD_MINORITY_INTEREST:
        op["op_equity"] = op["be"] + op["mib"].fillna(0)
    else:
        op["op_equity"] = op["be"]
    return op, has_revenue, has_an_expense


def main():
    make_folders()
    print("Building the OperProf signal...", flush=True)
    comp = pd.read_parquet(INTERMEDIATE / "compustat_annual.parquet")
    ccm = pd.read_parquet(INTERMEDIATE / "ccm_links.parquet")
    crsp = pd.read_parquet(INTERMEDIATE / "crsp_monthly_from_daily.parquet")
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet")

    # --- Operating profitability per firm-year -----------------------------------------------
    be = book_equity(comp)
    op, has_revenue, has_an_expense = operating_profitability(be)
    print(f"firm-years: {len(op):,}   with revenue: {has_revenue.mean():.1%}   "
          f"with revenue and >= 1 expense item: {(has_revenue & has_an_expense).mean():.1%}")
    print("share missing among firm-years with revenue (filled with 0 if another is present):")
    print(op.loc[has_revenue, ["cogs", "xsga", "xint", "mib"]].isna().mean()
          .map("{:.1%}".format).to_string())

    op["sort_year"] = op["fy_calendar_year"] + 1  # fiscal year t-1 feeds the June t sort

    # --- Link to CRSP (same rows and same link choice as BMdec) ---------------------------------
    with_be = op[op["be"].notna()][["gvkey", "sort_year", "datadate", "be", "years_in_compustat"]]
    linked = link_to_crsp(with_be, ccm, crsp)
    linked = linked.merge(op[["gvkey", "sort_year", "op_profit", "op_equity"]],
                          on=["gvkey", "sort_year"], how="left")

    # --- December t-1 ME (required by French's sample rule) ---------------------------------------
    # OP itself does not use ME, but French's RMW universe is the same as HML's: firms with
    # December t-1 and June t ME. Requiring it keeps our RMW universe comparable.
    december = panel[panel["date"].dt.month == 12].copy()
    december["sort_year"] = december["date"].dt.year + 1
    december = december[december["sort_year"] >= FIRST_SORT_YEAR]
    december = december.rename(columns={"me": "me_dec"})[["permco", "sort_year", "me_dec"]]
    opf = december.merge(linked, on=["permco", "sort_year"], how="left")

    # --- OperProf, only for positive book equity --------------------------------------------------
    # French require positive BE for RMW: with BE <= 0 the ratio is negative or explodes and
    # says nothing about profitability (a loss-making firm with negative BE would get a
    # *positive* OP). The denominator must also be positive after adding minority interest.
    valid = (opf["be"] > 0) & (opf["op_equity"] > 0) & opf["op_profit"].notna() & (opf["me_dec"] > 0)
    opf["OperProf"] = np.nan
    opf.loc[valid, "OperProf"] = opf.loc[valid, "op_profit"] / opf.loc[valid, "op_equity"]

    # --- At least two years in Compustat ------------------------------------------------------------
    # Same survivorship-bias rule as BMdec (see the comment there): back-filled first
    # Compustat years are biased towards firms that did well. Applied for the same reason,
    # and so RMW and HML are built on the same firms.
    n_young = (opf["OperProf"].notna() & (opf["years_in_compustat"] < 2)).sum()
    if REQUIRE_2Y_COMPUSTAT:
        opf.loc[opf["years_in_compustat"] < 2, "OperProf"] = np.nan
    print(f"firm-years with OperProf but fewer than 2 years in Compustat: {n_young:,} "
          f"({'removed' if REQUIRE_2Y_COMPUSTAT else 'kept, REQUIRE_2Y_COMPUSTAT = False'})")

    # --- Standard format: [permno, yyyymm, OperProf] -----------------------------------------------
    # Same PERMNO as the size signal: the firm's largest share class in June t.
    june = panel[panel["date"].dt.month == 6][["permco", "permno", "date"]].copy()
    june["sort_year"] = june["date"].dt.year
    opf = opf.merge(june[["permco", "sort_year", "permno"]], on=["permco", "sort_year"], how="inner")
    opf["yyyymm"] = opf["sort_year"] * 100 + 6
    require_unique(opf, ["permco", "sort_year"], "OperProf")

    saved = save_predictor(opf, "OperProf")

    # --- Sanity checks ----------------------------------------------------------------------------
    per_year = opf.groupby("sort_year").agg(
        firms_with_dec_me=("me_dec", lambda s: s.notna().sum()),
        with_positive_be=("be", lambda s: (s > 0).sum()),
        with_operprof=("OperProf", lambda s: s.notna().sum()),
        median_operprof=("OperProf", "median"),
    )
    per_year["median_operprof"] = per_year["median_operprof"].round(3)
    print("OperProf per sort year, every 5th year (firms with June and December ME):")
    print(per_year[(per_year.index % 5 == 0) | (per_year.index == per_year.index.max())]
          .to_string())
    lost = opf["be"].gt(0) & opf["OperProf"].isna() & opf["years_in_compustat"].ge(2)
    print(f"firm-years with positive BE (2+ years) but no OperProf (missing revenue/expenses): "
          f"{lost.sum():,}")
    print(f"OperProf quantiles (1/5/50/95/99%): "
          f"{saved['OperProf'].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(3).tolist()}")


if __name__ == "__main__":
    main()
