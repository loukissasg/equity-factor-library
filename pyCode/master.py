# ABOUTME: Runs the whole FF3 replication end to end by calling each stage script in order.
"""
Run:     cd pyCode
         python master.py
Inputs:  WRDS (CRSP, Compustat, CCM) and Ken French's data library
Outputs: pyData/ (intermediate data, parquet) and output/ (tables and figures)

Each stage is a separate script that reads its inputs from disk, does one job and
writes its outputs to disk, so any stage can also be rerun on its own, e.g.
    python Factors/FF3Daily.py
"""

import os
import subprocess
import sys
from pathlib import Path

from config import ENV_FILE, RUN_SCHEMA_CHECK, make_folders

# Stages in order. Each list entry is one script, run from the pyCode/ folder.
# Since 2026-11-21 the pipeline is built from daily CRSP data only. The monthly-only
# scripts are kept in archive/ (see archive/README.md).
STAGES = [
    ("1. Data downloads", [
        "DataDownloads/A_CRSPDaily.py",         # crsp.dsf_v2, one file per year
        "DataDownloads/B_CompustatAnnual.py",
        "DataDownloads/C_CCMLinks.py",
        "DataDownloads/D_CRSPNames.py",
        "DataDownloads/E_FrenchFactors.py",
        "DataDownloads/F_French6Portfolios.py",
        "DataDownloads/G_French5Factors.py",
        "DataDownloads/H_French6PortfoliosOP.py",
        "DataDownloads/I_CompustatNames.py",
        "DataDownloads/J_AQRQMJ.py",
        "DataDownloads/K_FrenchMomentum.py",
        "DataDownloads/L_French25Portfolios.py",
        "DataDownloads/M_FrenchFactorsDaily.py",       # daily RF and daily benchmark
        "DataDownloads/N_French6PortfoliosDaily.py",
        "DataDownloads/O_French5FactorsDaily.py",      # daily RMW benchmark
        "DataDownloads/P_French6PortfoliosOPDaily.py",
    ]),
    # DailyMasterTable: firm-days for the daily portfolios, plus a month-end file built
    # from daily data, which SignalMasterTable turns into the firm-month panel the
    # (annual) signals need.
    ("2. Master tables", ["DailyMasterTable.py", "SignalMasterTable.py"]),
    # Every Predictors/ script writes pyData/Predictors/<Name>.parquet [permno, yyyymm, <Name>].
    ("3. Signals", ["Predictors/Size.py", "Predictors/BMdec.py",
                    "Predictors/OperProf.py", "Predictors/GP.py",
                    "Predictors/QualityComp.py"]),   # QualityComp needs OperProf first
    # FF3: FF3Daily.py calls ff2x3_daily() from Portfolios/FF2x3Daily.py (same June sort,
    # daily returns); FF3Monthly.py compounds those daily portfolios to monthly FF3.
    # RMW and RMW_GP also use ff2x3_daily() and write daily and monthly files themselves;
    # the SMBQ scripts do the same with size_quality_5x5_daily() (Portfolios/SizeQuality5x5Daily.py).
    ("4. Portfolios and factors", ["Factors/FF3Daily.py",
                                   "Factors/FF3Monthly.py",   # needs FF3Daily.py first
                                   "Factors/RMW.py", "Factors/RMW_GP.py",
                                   "Factors/SMBQ.py", "Factors/SMBQ_GP.py",
                                   "Factors/SMBQ_JunkNeutral.py"]),   # needs SMBQ.py first
    ("5. Validation", ["Validation/CompareFrenchDaily.py",
                       "Validation/CompareDailyMonthly.py",
                       "Validation/CompareFrench.py",   # monthly FF3 (from daily) vs French
                       "Validation/CompareFrenchRMW.py",
                       "Validation/CompareFrenchRMWDaily.py",
                       "Validation/SMBQAnalysis.py", "Validation/ModelComparison25.py",
                       "Validation/Stage3SlideSummary.py", "Validation/SMBQJunkNeutral.py",
                       "Validation/ModelComparisonQMJ.py",
                       "Validation/CheckReference.py"]),
]
if RUN_SCHEMA_CHECK:
    STAGES.insert(0, ("0. CIZ schema check", ["DataDownloads/CheckCIZSchemaDaily.py"]))


def check_environment():
    """Stop early with a clear message if the WRDS username is not set up."""
    if not ENV_FILE.exists():
        print(f"ERROR: {ENV_FILE} not found. Copy dotenv.template to .env and add WRDS_USERNAME.")
        sys.exit(1)


def main():
    pycode_dir = Path(__file__).resolve().parent
    os.chdir(pycode_dir)  # scripts are referenced relative to pyCode/
    check_environment()
    make_folders()

    for stage_name, scripts in STAGES:
        print("\n" + "=" * 80)
        print(stage_name)
        print("=" * 80, flush=True)
        for script in scripts:
            print(f"\n--- {script} ---", flush=True)
            try:
                # -u: unbuffered, so progress appears immediately.
                # check=True: stop the pipeline if a script fails.
                subprocess.run([sys.executable, "-u", script], check=True)
            except subprocess.CalledProcessError as error:
                print(f"\nERROR: {script} failed (exit code {error.returncode}). Pipeline stopped.")
                sys.exit(1)
            print(f"OK  {script}", flush=True)

    print("\nPipeline complete. Tables and figures are in output/.")


if __name__ == "__main__":
    main()
