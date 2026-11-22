# ABOUTME: Saves a signal in the standard predictor format [permno, yyyymm, <Name>], like OpenAP's save_predictor.
"""
Every script in Predictors/ ends by calling save_predictor(). Because all signals share
one format, the portfolio code can sort on any of them without knowing how they were
built: adding a new factor means adding one Predictors/ script, nothing else.

Mirrors OpenAP's utils/save_standardized.py: keep only permno, yyyymm and the signal,
drop rows where the signal is missing. One difference: we save parquet (smaller and
faster, keeps dtypes) instead of CSV.

yyyymm is the month in which the signal is used. Our signals are annual and used for
the June sort, so yyyymm = YYYY06.
"""

from config import PREDICTORS
from utils.checks import require_columns, require_unique


def save_predictor(df, name):
    """Save df[[permno, yyyymm, name]] to pyData/Predictors/<name>.parquet."""
    require_columns(df, ["permno", "yyyymm", name], f"predictor {name}")
    out = df[["permno", "yyyymm", name]].copy()

    # A missing signal carries no information for the sort, so it is not stored (as in
    # OpenAP). The portfolio code then only sees stocks that have the signal.
    out = out[out[name].notna()]
    out["permno"] = out["permno"].astype("int64")
    out["yyyymm"] = out["yyyymm"].astype("int64")
    out[name] = out[name].astype("float64")
    out = out.sort_values(["permno", "yyyymm"]).reset_index(drop=True)

    require_unique(out, ["permno", "yyyymm"], f"predictor {name}")
    PREDICTORS.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTORS / f"{name}.parquet", index=False)
    print(f"saved {name}: {len(out):,} rows, {out['permno'].nunique():,} permnos, "
          f"yyyymm {out['yyyymm'].min()} to {out['yyyymm'].max()}")
    return out
