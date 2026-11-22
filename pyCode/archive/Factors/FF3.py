# ABOUTME: Builds the three Fama-French factors (MKT-RF, SMB, HML) from the size x BMdec 2x3 sort and the market.
"""
Run:     python Factors/FF3.py   (from pyCode/)
Inputs:  pyData/Predictors/Size.parquet, BMdec.parquet   (via ff2x3)
         pyData/Intermediate/SignalMasterTable.parquet
         pyData/Intermediate/french_ff3.parquet   (RF: one-month T-bill rate)
Outputs: pyData/Portfolios/FF2x3_BMdec*.parquet  (written by ff2x3)
         pyData/Factors/FF3.parquet   monthly mkt_rf, smb, hml, mkt, rf (decimals)
         output/ff3_replicated.csv    the same, for Excel

Each factor is a zero-investment long-short portfolio: buy one group of stocks, short
another, so the return measures the *premium* of one characteristic over the other.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, make_folders  # noqa: E402
from Portfolios.FF2x3 import ff2x3  # noqa: E402


def summary_stats(df):
    """Mean and volatility (in % per year) and annualized Sharpe ratio for each column."""
    return pd.DataFrame({
        "mean_annual_%": df.mean() * 12 * 100,
        "std_annual_%": df.std() * (12 ** 0.5) * 100,
        "sharpe": df.mean() / df.std() * (12 ** 0.5),
    }).round(2)


def main():
    make_folders()

    # --- SMB and HML from the size x B/M sort ---------------------------------------------------
    # sign +1: high B/M ("value") is the long side, as in Fama-French.
    p = ff2x3("BMdec", +1)

    print("\nBuilding SMB, HML and MKT-RF...", flush=True)
    panel = pd.read_parquet(INTERMEDIATE / "SignalMasterTable.parquet")
    french = pd.read_parquet(INTERMEDIATE / "french_ff3.parquet").set_index("date")

    factors = pd.DataFrame(index=p.index)

    # SMB = 1/3 (SL + SN + SH) - 1/3 (BL + BN + BH): small minus big, holding B/M constant.
    factors["smb"] = p["smb"]

    # HML = 1/2 (SH + BH) - 1/2 (SL + BL): value minus growth, holding size constant.
    factors["hml"] = p["spread"]

    # Market: value-weighted return of *all* stocks in our CRSP universe, not only those in
    # the 2x3 sort. A stock without Compustat data or with negative book equity is still
    # part of the market investors hold.
    market = panel[panel["ret"].notna() & panel["me_lag"].notna() & (panel["me_lag"] > 0)].copy()
    market["ret_x_weight"] = market["ret"] * market["me_lag"]
    grouped = market.groupby("date")
    factors["mkt"] = grouped["ret_x_weight"].sum() / grouped["me_lag"].sum()

    # MKT-RF: subtract the one-month T-bill rate (from Ken French's file) to get the
    # premium for holding stocks instead of a risk-free asset.
    factors = factors.join(french[["rf"]], how="left")
    factors["mkt_rf"] = factors["mkt"] - factors["rf"]
    if factors["rf"].isna().any():
        raise ValueError("FF3: risk-free rate missing for some months")

    factors = factors[["mkt_rf", "smb", "hml", "mkt", "rf"]]
    factors.index.name = "date"
    factors.to_parquet(FACTORS / "FF3.parquet")
    factors.to_csv(OUTPUT / "ff3_replicated.csv", float_format="%.6f")

    # --- Sanity checks -----------------------------------------------------------------------
    print(f"months: {len(factors)}   dates: {factors.index.min().date()} to "
          f"{factors.index.max().date()}")
    print("Full sample:")
    print(summary_stats(factors[["mkt_rf", "smb", "hml"]]).to_string())

    # Average annual premium by decade: a first look at how the factors behaved over time
    # (part 3 of the project: the fading size premium, value's drawdown after 2007).
    by_decade = (factors[["mkt_rf", "smb", "hml"]]
                 .groupby((factors.index.year // 10) * 10).mean() * 12 * 100)
    by_decade.index = [f"{d}s" for d in by_decade.index]
    print("Average annual premium by decade (%):")
    print(by_decade.round(1).to_string())


if __name__ == "__main__":
    main()
