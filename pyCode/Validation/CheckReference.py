# ABOUTME: Regression test: checks that existing factors (FF3 daily and monthly, RMW, SMBQ_Comp) are unchanged versus their saved reference copies.
"""
Run:     python Validation/CheckReference.py          (from pyCode/; run by master.py)
         python Validation/CheckReference.py --save   (accept the current factors as the
                                                       new references, after reviewing them)
Inputs:  pyData/Factors/<Factor>.parquet         (current pipeline)
         pyData/reference/<Factor>.parquet       (saved copy of an accepted version)
Outputs: none; prints the max absolute difference per factor and stops with an error if
         any is too big. With --save: overwrites pyData/reference/<Factor>.parquet.

Why: the rule of the factor library is that adding a new factor never changes an
existing one. After any code change, every factor with a reference copy must be identical
up to floating-point noise (1e-10). References:
    FF3Daily.parquet          daily FF3, accepted after Stage 6 of the move to daily data
    FF3.parquet               monthly FF3 compounded from daily, accepted at the same time
    RMW.parquet               RMW (still monthly), re-accepted at the same time
    SMBQ_JunkNeutral.parquet  only the smbq_comp column is checked (SMBQ_Comp was chosen as
                              the best version; the other two columns are exploratory)
The references from before the move to daily data are frozen in
pyData/reference/monthly_pipeline/.

After a CRSP/Compustat data refresh the numbers change legitimately: review the
differences, then accept them with --save. Never use --save to silence a failure you
cannot explain.
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import DATA_ROOT, FACTORS  # noqa: E402

TOLERANCE = 1e-10
REFERENCE = DATA_ROOT / "reference"
CHECKS = {
    "FF3Daily": ["mkt_rf", "smb", "hml", "mkt", "rf"],
    "FF3": ["mkt_rf", "smb", "hml", "mkt", "rf"],
    "RMW": ["rmw", "smb_op"],
    "SMBQ_JunkNeutral": ["smbq_comp"],
}


def check(name, columns):
    """True if pyData/Factors/<name>.parquet equals its reference copy."""
    reference_file = REFERENCE / f"{name}.parquet"
    if not reference_file.exists():
        print(f"\nFAIL {name}: no reference copy yet. Review the factor, then accept it with "
              f"'python Validation/CheckReference.py --save'.")
        return False
    reference = pd.read_parquet(reference_file)
    current = pd.read_parquet(FACTORS / f"{name}.parquet")

    same_dates = reference.index.equals(current.index)
    print(f"\n{name}: dates reference {len(reference):,}, current {len(current):,}, "
          f"identical dates: {same_dates}")
    if not same_dates:
        print(f"FAIL {name}: the factor dates differ from the reference")
        return False

    max_diff = (current[columns] - reference[columns]).abs().max()
    print("max absolute difference, current - reference:")
    print(max_diff.to_string())
    if (max_diff > TOLERANCE).any():
        print(f"FAIL {name}: difference above {TOLERANCE:g}")
        return False
    print(f"PASS {name}: identical to the reference (all differences <= {TOLERANCE:g})")
    return True


def save_references():
    """Copy the current factor files over the reference copies (an explicit, reviewed step)."""
    for name in CHECKS:
        shutil.copyfile(FACTORS / f"{name}.parquet", REFERENCE / f"{name}.parquet")
        print(f"saved pyData/reference/{name}.parquet")


def main():
    if "--save" in sys.argv:
        save_references()
        return
    results = [check(name, columns) for name, columns in CHECKS.items()]
    if not all(results):
        raise SystemExit("Regression test failed: an existing factor has changed.")


if __name__ == "__main__":
    main()
