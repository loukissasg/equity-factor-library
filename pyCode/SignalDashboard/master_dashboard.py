# ABOUTME: Runs the signal-dashboard pipeline (OpenAP data -> cards -> statistics -> HTML) in order.
"""
Run:     cd pyCode
         python SignalDashboard/master_dashboard.py
Inputs:  OpenAP's published data (no WRDS needed) and the factor files of the FF3 project
Outputs: pyData/OpenAP/ and output/

Separate from pyCode/master.py on purpose: the dashboard does not need to rerun the
whole FF3 replication. Stages are added here as they are built.
"""

import os
import subprocess
import sys
from pathlib import Path

STAGES = [
    ("1. Data", ["SignalDashboard/A_OpenAPDownload.py",
                 "SignalDashboard/B_Benchmarks.py",
                 "SignalDashboard/C_DataReport.py"]),
    ("2. Signal cards", ["SignalDashboard/D_SignalCards.py"]),
    ("3. Statistics", ["SignalDashboard/E_Performance.py",
                       "SignalDashboard/F_FactorRegressions.py"]),
    # G builds the HTML (Plotly from the CDN; run G with --embed-plotly by hand for a fully
    # offline file). H re-reads the HTML and stops the pipeline if any number differs.
    ("4. Dashboard", ["SignalDashboard/G_Dashboard.py",
                      "SignalDashboard/H_CheckDashboard.py"]),
]


def main():
    os.chdir(Path(__file__).resolve().parents[1])   # scripts are referenced from pyCode/
    for stage_name, scripts in STAGES:
        print("\n" + "=" * 80 + f"\n{stage_name}\n" + "=" * 80, flush=True)
        for script in scripts:
            print(f"\n--- {script} ---", flush=True)
            try:
                subprocess.run([sys.executable, "-u", script], check=True)
            except subprocess.CalledProcessError as error:
                print(f"\nERROR: {script} failed (exit code {error.returncode}). Pipeline stopped.")
                sys.exit(1)
            print(f"OK  {script}", flush=True)


if __name__ == "__main__":
    main()
