# ABOUTME: Monthly FF3 built from the daily portfolios: compounds the six daily portfolios and the daily market to months.
"""
Run:     python Factors/FF3Monthly.py   (from pyCode/, after Factors/FF3Daily.py)
Inputs:  pyData/Portfolios/FF2x3Daily_BMdec.parquet   daily SL..BH
         pyData/Factors/FF3Daily.parquet               daily market return
         pyData/Intermediate/french_ff3.parquet        monthly RF
Outputs: pyData/Factors/FF3.parquet   monthly mkt_rf, smb, hml, mkt, rf (decimals), month-end dates
         output/ff3_replicated.csv    the same, for Excel

The pipeline is daily only: there is one source of truth, the daily factors. Many
analyses still need a monthly FF3 (French's 25 portfolios, AQR's QMJ and the other
factors are monthly), so this script turns the daily data into monthly factors. It does
not re-sort or re-weight anything.

Why compound the portfolios and not the factors: SMB and HML are differences of
portfolios, and compounding a difference is not the same as the difference of the
compounded portfolios. So each portfolio (and the market) is compounded to a monthly
return first, then the factors are formed from those, exactly as Fama-French do monthly.
Checked in Stage 5 (Validation/CompareDailyMonthly.py): this reproduces the old monthly
pipeline at correlation >= 0.9996 and matches French's monthly factors equally well.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import FACTORS, INTERMEDIATE, OUTPUT, PORTFOLIO_NAMES, PORTFOLIOS, make_folders  # noqa: E402
from Portfolios.FF2x3Daily import compound_to_months  # noqa: E402
from utils.checks import require_unique  # noqa: E402


def main():
    make_folders()
    print("Compounding the daily portfolios and market to monthly FF3...", flush=True)
    portfolios = pd.read_parquet(PORTFOLIOS / "FF2x3Daily_BMdec.parquet")
    daily = pd.read_parquet(FACTORS / "FF3Daily.parquet")

    # (1 + r1)(1 + r2)... - 1 over the trading days of each month.
    legs = compound_to_months(portfolios[PORTFOLIO_NAMES])
    factors = pd.DataFrame(index=legs.index)

    # Same formulas as the daily and the old monthly version, applied to monthly legs.
    factors["smb"] = legs[["SL", "SN", "SH"]].mean(axis=1) - legs[["BL", "BN", "BH"]].mean(axis=1)
    factors["hml"] = legs[["SH", "BH"]].mean(axis=1) - legs[["SL", "BL"]].mean(axis=1)
    factors["mkt"] = compound_to_months(daily["mkt"])

    # MKT-RF: French's *monthly* RF (the one-month T-bill return for that month).
    rf = pd.read_parquet(INTERMEDIATE / "french_ff3.parquet").set_index("date")["rf"]
    rf.index = rf.index.to_period("M")
    factors = factors.join(rf, how="left")
    if factors["rf"].isna().any():
        raise ValueError("FF3Monthly: risk-free rate missing for some months")
    factors["mkt_rf"] = factors["mkt"] - factors["rf"]

    # Month-end dates, the format every monthly script expects (as the old FF3.py wrote).
    factors.index = factors.index.to_timestamp(how="end").normalize()
    factors.index.name = "date"
    factors = factors[["mkt_rf", "smb", "hml", "mkt", "rf"]]
    require_unique(factors.reset_index(), ["date"], "FF3 monthly")
    factors.to_parquet(FACTORS / "FF3.parquet")
    factors.to_csv(OUTPUT / "ff3_replicated.csv", float_format="%.6f")

    # --- Sanity checks -----------------------------------------------------------------------
    # Every month must be complete: a month whose first or last trading days were missing
    # would compound to a wrong return without any error.
    days_per_month = portfolios.groupby(portfolios.index.to_period("M")).size()
    print(f"months: {len(factors)}   dates: {factors.index.min().date()} to "
          f"{factors.index.max().date()}   trading days per month: "
          f"{days_per_month.min()} to {days_per_month.max()}")
    print("Full sample (% per year from monthly means x 12; Sharpe annualized):")
    stats = pd.DataFrame({
        "mean_annual_%": factors[["mkt_rf", "smb", "hml"]].mean() * 12 * 100,
        "std_annual_%": factors[["mkt_rf", "smb", "hml"]].std() * 12 ** 0.5 * 100,
        "sharpe": factors[["mkt_rf", "smb", "hml"]].mean()
                  / factors[["mkt_rf", "smb", "hml"]].std() * 12 ** 0.5,
    }).round(2)
    print(stats.to_string())

    by_decade = (factors[["mkt_rf", "smb", "hml"]]
                 .groupby((factors.index.year // 10) * 10).mean() * 12 * 100)
    by_decade.index = [f"{d}s" for d in by_decade.index]
    print("Average annual premium by decade (%):")
    print(by_decade.round(1).to_string())


if __name__ == "__main__":
    main()
