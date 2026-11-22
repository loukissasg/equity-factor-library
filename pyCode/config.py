# ABOUTME: Central settings for the FF3 replication: paths, sample dates and run flags.
"""
Imported by every script in pyCode/. Change settings here, not in the scripts.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "pyData"
INTERMEDIATE = DATA_ROOT / "Intermediate"  # raw downloads and the master table
PREDICTORS = DATA_ROOT / "Predictors"      # signals, one file each: [permno, yyyymm, <Name>]
PORTFOLIOS = DATA_ROOT / "Portfolios"      # FF2x3_<signal>*.parquet: returns, assignments
FACTORS = DATA_ROOT / "Factors"            # SMB, HML, MKT-RF
OUTPUT = PROJECT_ROOT / "output"           # tables and figures for the presentation
# Daily data is ~75M stock-days, too big for one file in memory, so it is stored one
# parquet file per year and processed year by year.
CRSP_DAILY_DIR = INTERMEDIATE / "crsp_daily"            # raw download: crsp_daily_YYYY.parquet
DAILY_MASTER_DIR = INTERMEDIATE / "DailyMasterTable"    # firm-days: DailyMasterTable_YYYY.parquet
ENV_FILE = PROJECT_ROOT / ".env"           # WRDS_USERNAME (never committed)
# Outputs of the old monthly pipeline (crsp.msf_v2, before the move to daily data),
# frozen on 2026-11-21. Used only to compare daily vs monthly; the scripts that made
# them are in pyCode/archive/.
MONTHLY_PIPELINE_REF = DATA_ROOT / "reference" / "monthly_pipeline"

# --- Sample ----------------------------------------------------------------------------
# The first portfolios are formed in June 1963 (Compustat coverage becomes reliable).
# That sort needs December 1962 ME and fiscal-year 1962 book equity, so CRSP starts
# earlier. Compustat starts at its beginning (1950) so we can count each firm's years
# in Compustat (Fama-French require two, to limit survivorship bias in early data).
CRSP_START = "1960-01-01"
COMPUSTAT_START = "1950-01-01"
FIRST_SORT_YEAR = 1963
# Last year of the current CRSP annual release (dsf_v2 ends 2025-12-31). Raise it after
# a new release; the daily download then fetches only the new year(s).
CRSP_END_YEAR = 2025

# --- Run flags ---------------------------------------------------------------------------
# False: download scripts reuse the parquet files in pyData/Intermediate.
# True: download everything again (e.g. after a new CRSP release).
REFRESH_DOWNLOADS = False

# Print the CIZ schema report (DataDownloads/CheckCIZSchema.py). Needed once, or after a
# CRSP update.
RUN_SCHEMA_CHECK = False

# Fama-French (1993) use a firm only once it has appeared in Compustat for two years
# (applied in Predictors/BMdec.py, since it is a property of the accounting data).
# Compustat back-filled histories of firms it added (mostly successful ones), so a
# firm's first record is biased towards survivors. Set to False to see the effect.
REQUIRE_2Y_COMPUSTAT = True

# --- Portfolio names, in Ken French's column order -----------------------------------------
PORTFOLIO_NAMES = ["SL", "SN", "SH", "BL", "BN", "BH"]


# --- Signal dashboard (separate project: OpenAP's published portfolios, no WRDS) ------------
# Scripts in pyCode/SignalDashboard/. We do NOT rebuild these signals; we read Chen and
# Zimmermann's published original-paper ("OP") long-short portfolios.
OPENAP = DATA_ROOT / "OpenAP"              # OpenAP downloads and the benchmark factor panel
# Pin the release so results do not change silently when OpenAP publishes a new vintage.
# 202610 = Hugging Face branch "Data-Release-2026.10" (code v2.1.0, data to Dec 2025).
OPENAP_RELEASE = 202610
DASHBOARD_SIGNALS = ["Mom12m", "STreversal", "LRreversal", "GP", "AssetGrowth",
                     "ShareIss1Y", "Accruals", "IdioVol3F", "Illiquidity",
                     "AnnouncementReturn"]
# Decided 2026-11-05 after Stage 1: AnnouncementReturn's first two months have 1 and 13
# stocks per leg (too thin to be a portfolio), so its statistics start in 1971-11.
# Illiquidity's one thin month (1932-06, 14 stocks long) is kept.
SIGNAL_START_OVERRIDE = {"AnnouncementReturn": "1971-11"}
# Period split (McLean and Pontiff 2016 style), SignalDoc gives years only:
#   in-sample        = Jan(SampleStartYear) .. Dec(SampleEndYear)
#   pre-publication  = Jan(SampleEndYear + 1) .. Dec(publication Year)
#   post-publication = Jan(Year + 1) onwards
# "Publication" = the journal year in SignalDoc; working papers may have circulated earlier.


def make_folders():
    """Create the data and output folders if they do not exist yet."""
    for folder in [INTERMEDIATE, PREDICTORS, PORTFOLIOS, FACTORS, OUTPUT, OPENAP,
                   CRSP_DAILY_DIR, DAILY_MASTER_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
