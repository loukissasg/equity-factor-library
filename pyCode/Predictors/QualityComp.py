# ABOUTME: QualityComp signal: a simple QMJ-style composite of profitability and safety (OperProf, low leverage, low volatility).
"""
Run:     python Predictors/QualityComp.py   (from pyCode/, after OperProf.py)
Inputs:  pyData/Predictors/OperProf.parquet
         pyData/Intermediate/compustat_annual.parquet, ccm_links.parquet, crsp_monthly_from_daily.parquet
         pyData/Intermediate/SignalMasterTable.parquet   (June PERMNO of each firm)
         pyData/Intermediate/DailyMasterTable/DailyMasterTable_YYYY.parquet   (daily returns)
         pyData/Intermediate/french_ff3_daily.parquet    (daily RF)
Outputs: pyData/Predictors/QualityComp.parquet   [permno, yyyymm, QualityComp]
         yyyymm = YYYY06 (the June t sort that uses the signal)

Why: AQR's QMJ defines quality as profitability + growth + safety + payout. Sorting on
OperProf alone controls only the first, so SMBQ stayed short "safety" (small stocks are
also riskier and more indebted). This composite adds two safety measures we can build
from our own data. It is a simplified version of QMJ, not a replica.

Components, each measured with data known at the end of June t:
  1. OperProf             fiscal year t-1 (Predictors/OperProf.py)        high = quality
  2. Book leverage lt/at  fiscal year t-1, same CCM link as BMdec        low  = quality
  3. Return volatility    std of daily excess returns over the last 252  low  = quality
                          trading days up to the end of June t (at least 100 required)

Volatility from daily data follows Open Source Asset Pricing, which computes every
volatility signal from daily returns: RealizedVol (Ang et al. 2006) is the std of daily
excess returns (ret - rf) over a month; IdioVolAHT uses a 252-trading-day window with at
least 100 observations. We take RealizedVol's definition (total, not residual, volatility)
over IdioVolAHT's one-year window: a score held for a whole year should not rest on a
single month of data. (Until 2026-11-22: std of 36 monthly returns; that version is in
archive/Predictors/QualityComp.py.)

Combining (as AQR does): each June, rank every stock on each component, turn the ranks
into z-scores (mean 0, std 1), flip leverage and volatility so that high = safe, and
average the three. Ranks rather than raw values, so one extreme ratio cannot dominate,
and each component has the same weight. A stock needs all three components.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import (CRSP_END_YEAR, DAILY_MASTER_DIR, FIRST_SORT_YEAR, INTERMEDIATE,  # noqa: E402
                    MONTHLY_METHOD_REF, PREDICTORS, make_folders)
from Predictors.BMdec import book_equity, link_to_crsp  # noqa: E402
from utils.checks import require_unique  # noqa: E402
from utils.save_predictor import save_predictor  # noqa: E402

VOL_WINDOW = 252   # trading days of returns for the volatility component (about one year)
VOL_MIN_OBS = 100  # minimum daily returns in the window (as OSAP's IdioVolAHT)


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
    """
    Std of daily excess returns over the last 252 trading days up to the end of June t,
    for every firm and June t, keyed by June t PERMNO.
    """
    rf = pd.read_parquet(INTERMEDIATE / "french_ff3_daily.parquet").set_index("date")["rf"]
    rows = []
    for year in range(FIRST_SORT_YEAR, CRSP_END_YEAR + 1):
        # The window reaches back into the previous calendar year, so read both years.
        days = pd.concat([
            pd.read_parquet(DAILY_MASTER_DIR / f"DailyMasterTable_{y}.parquet",
                            columns=["permco", "date", "ret"])
            for y in (year - 1, year)])
        days = days[days["date"] <= pd.Timestamp(f"{year}-06-30")]

        # The window is the last 252 *market* trading days, the same calendar dates for
        # every firm. A firm with gaps simply has fewer returns in it (at least 100 needed).
        trading_days = days["date"].drop_duplicates().sort_values()
        window = days[days["date"] >= trading_days.iloc[-VOL_WINDOW]].copy()

        # Excess returns, as OSAP's RealizedVol. Subtracting a (nearly constant) daily RF
        # hardly changes a standard deviation, but it keeps the definition the same.
        window["rf"] = window["date"].map(rf)
        if window["rf"].isna().any():
            raise ValueError(f"QualityComp: no French RF for some days of the {year} window")
        window["excess"] = window["ret"] - window["rf"]

        stats = window.groupby("permco")["excess"].agg(vol="std", n_returns="count").reset_index()
        stats = stats[stats["n_returns"] >= VOL_MIN_OBS]
        stats["sort_year"] = year
        rows.append(stats)

    vol = pd.concat(rows, ignore_index=True)
    # Daily returns are per firm (PERMCO, the largest share class); the signal is stored
    # under the PERMNO the master table carries for the firm in June, as Size and BMdec.
    vol = vol.merge(june_permno(panel), on=["permco", "sort_year"], how="inner")
    vol["yyyymm"] = vol["sort_year"] * 100 + 6
    print(f"firm-Junes with daily volatility: {len(vol):,}   median daily returns in the "
          f"window: {vol['n_returns'].median():.0f}")
    return vol[["permno", "yyyymm", "vol"]]


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

    # How much did the switch to daily volatility change the score? (Frozen copy of the
    # version with 36 monthly returns, if present.)
    old_file = MONTHLY_METHOD_REF / "QualityComp.parquet"
    if old_file.exists():
        old = pd.read_parquet(old_file)
        both = q[["permno", "yyyymm", "QualityComp"]].merge(
            old, on=["permno", "yyyymm"], how="outer", suffixes=("", "_old"), indicator=True)
        common = both[both["_merge"] == "both"].copy()
        # Quintile within each June: the 5x5 sort uses NYSE breakpoints, but plain
        # quintiles are enough to see how many stocks would change group.
        for column in ["QualityComp", "QualityComp_old"]:
            common[column + "_q"] = common.groupby("yyyymm")[column].transform(
                lambda s: pd.qcut(s, 5, labels=False))
        moved = (common["QualityComp_q"] != common["QualityComp_old_q"]).mean()
        print(f"\nvs the old score (monthly volatility): firm-Junes in both {len(common):,}, "
              f"only new {(both['_merge'] == 'left_only').sum():,}, "
              f"only old {(both['_merge'] == 'right_only').sum():,}")
        print(f"correlation of scores {common['QualityComp'].corr(common['QualityComp_old']):.3f}, "
              f"share changing quintile {moved:.1%}")


if __name__ == "__main__":
    main()
