# ABOUTME: Checks that the signals built from daily CRSP equal the ones built from monthly CRSP.
"""
Run:     python Validation/CheckSignalsDailyVsMonthly.py   (from pyCode/)
Inputs:  pyData/Predictors/<Name>.parquet                       (now built from daily data)
         pyData/reference/monthly_pipeline/<Name>.parquet       (saved before the switch)
         pyData/Intermediate/SignalMasterTable.parquet and its saved copy
Outputs: printed report only

The switch to daily data should not change the annual signals: they only use month-end
ME (June, December) and Compustat, and month-end ME is identical in both CRSP sources.
QualityComp is the exception we expect: its volatility component uses 36 monthly
returns, and a few monthly returns differ (mostly delisting returns that msf_v2 lacks).

Expected extra rows: BMdec/OperProf/GP get a few firm-years for firms that left our
universe during June (they have a June row in the master table but no Size, so the 2x3
sort drops them). GP also *loses* 525 firm-years (1966-84) with no December ME: the old
monthly data used pandas' nullable Float64, where `~(me_dec > 0)` is NA for a missing
value and .loc treats NA as False, so GP.py's "December ME required" rule silently
didn't apply. With plain float64 it does.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import DATA_ROOT, INTERMEDIATE, PREDICTORS  # noqa: E402

OLD = DATA_ROOT / "reference" / "monthly_pipeline"
SIGNALS = ["Size", "BMdec", "OperProf", "GP", "QualityComp"]


def compare(name, new, old, keys, value, abs_tol=None):
    """Rows on each side and share of equal values (relative 1e-9, or abs_tol if given)."""
    both = new.merge(old, on=keys, how="outer", suffixes=("_new", "_old"), indicator=True)
    matched = both[both["_merge"] == "both"]
    diff = (matched[f"{value}_new"] - matched[f"{value}_old"]).abs()
    if abs_tol is None:
        tol = 1e-9 * matched[f"{value}_old"].abs().clip(lower=1e-12)
    else:
        tol = abs_tol
    equal = (diff <= tol) | (matched[f"{value}_new"].isna() & matched[f"{value}_old"].isna())
    print(f"{name:<18} rows new {len(new):>9,}  old {len(old):>9,}  "
          f"only new {(both['_merge'] == 'left_only').sum():>5,}  "
          f"only old {(both['_merge'] == 'right_only').sum():>5,}  "
          f"equal values {equal.mean():.4%}  max |diff| {diff.max():.3g}")
    return both, matched[~equal]


def main():
    print("Signals from daily CRSP vs from monthly CRSP (saved copies)\n", flush=True)
    for name in SIGNALS:
        new = pd.read_parquet(PREDICTORS / f"{name}.parquet")
        old = pd.read_parquet(OLD / f"{name}.parquet")
        both, different = compare(name, new, old, ["permno", "yyyymm"], name)
        if len(different) or (both["_merge"] != "both").any():
            years = sorted(set(both.loc[both["_merge"] != "both", "yyyymm"] // 100)
                           | set(different["yyyymm"] // 100))
            print(f"{'':<18} years with differences: {years[:15]}{' ...' if len(years) > 15 else ''}")
        if name == "QualityComp":
            # A z-scored rank average: a tiny change in one stock's volatility can move its
            # rank and nudge others. So measure how much the score moves, not just whether.
            matched = both[both["_merge"] == "both"]
            change = (matched["QualityComp_new"] - matched["QualityComp_old"]).abs()
            print(f"{'':<18} correlation new vs old: "
                  f"{matched['QualityComp_new'].corr(matched['QualityComp_old']):.6f}   "
                  f"|change| < 0.001: {(change < 0.001).mean():.2%}   "
                  f"< 0.01: {(change < 0.01).mean():.2%}   (score std "
                  f"{matched['QualityComp_old'].std():.2f})")

    print("\nMaster table (firm-months):")
    new = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet")
    old = pd.read_parquet(OLD / "SignalMasterTable.parquet")
    for value in ["me", "me_lag"]:
        compare(f"  {value}", new[["permco", "date", value]], old[["permco", "date", value]],
                ["permco", "date"], value)
    # CRSP stores mthret with 6 decimals, so compounded daily returns match to ~1e-6, not
    # to machine precision. 1 basis point, as in CheckDailyVsMonthlyCRSP.py.
    compare("  ret (1bp)", new[["permco", "date", "ret"]], old[["permco", "date", "ret"]],
            ["permco", "date"], "ret", abs_tol=1e-4)


if __name__ == "__main__":
    main()
