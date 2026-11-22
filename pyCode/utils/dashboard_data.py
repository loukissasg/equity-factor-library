# ABOUTME: Loaders shared by the signal-dashboard scripts, so every stage uses the same sample.
"""
load_ls() applies config.SIGNAL_START_OVERRIDE (e.g. AnnouncementReturn starts 1971-11).
Doing it in one place guarantees that the statistics, regressions and the dashboard all
use exactly the same months.
"""

import pandas as pd

from config import DASHBOARD_SIGNALS, OPENAP, OUTPUT, SIGNAL_START_OVERRIDE


def load_ls():
    """Monthly long-short returns, % per month, date index (month-end), one column per signal."""
    ls = pd.read_parquet(OPENAP / "ls_returns.parquet")[DASHBOARD_SIGNALS]
    for signal, first_month in SIGNAL_START_OVERRIDE.items():
        first_date = pd.Timestamp(first_month) + pd.offsets.MonthEnd(0)
        ls.loc[ls.index < first_date, signal] = float("nan")
    return ls


def load_factors():
    """Benchmark factors from B_Benchmarks.py, % per month."""
    return pd.read_parquet(OPENAP / "factors_monthly.parquet")


def load_info():
    """SignalDoc fields from C_DataReport.py, indexed by signal."""
    return pd.read_csv(OUTPUT / "signal_info.csv").set_index("signal")
