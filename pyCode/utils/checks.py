# ABOUTME: Integrity checks that stop the pipeline when the data is not as expected.
"""
Printing a warning is easy to miss; raising an error is not. These checks guard the
assumptions later scripts rely on (e.g. one row per firm and month).
"""


def require_unique(df, keys, name):
    """Raise ValueError if `df` has more than one row for any combination of `keys`."""
    n_duplicates = df.duplicated(keys).sum()
    if n_duplicates > 0:
        raise ValueError(f"{name}: {n_duplicates:,} duplicate rows on {keys}")


def require_columns(df, columns, name):
    """Raise ValueError if any of `columns` is missing from `df`."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
