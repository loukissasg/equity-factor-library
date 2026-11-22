"""
Fama-French 3-factor replication (MKT-RF, SMB, HML) from CRSP v2 (CIZ) and Compustat.
Requires a WRDS username in .env (WRDS_USERNAME=...) and a pgpass file for the password.
Output: ff3_factors.csv (monthly, decimals) and correlations with Ken French's factors.
"""

import io
import os
import zipfile

import pandas as pd
import requests
import wrds
from dotenv import load_dotenv

FIRST_SORT_YEAR = 1963

# ------------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------------
load_dotenv()
db = wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"))

# Common shares of US firms on NYSE/AMEX/NASDAQ (CIZ equivalent of shrcd 10/11, exchcd 1-3).
# mthret already includes delisting returns.
crsp = db.raw_sql("""
    SELECT permno, permco, mthcaldt, mthret, mthprc, shrout, primaryexch
    FROM crsp.msf_v2
    WHERE mthcaldt >= '1960-01-01'
      AND sharetype = 'NS' AND securitytype = 'EQTY' AND securitysubtype = 'COM'
      AND usincflg = 'Y' AND issuertype IN ('ACOR', 'CORP')
      AND primaryexch IN ('N', 'A', 'Q')
      AND conditionaltype = 'RW' AND tradingstatusflg = 'A'
""", date_cols=["mthcaldt"])
crsp[["permno", "permco"]] = crsp[["permno", "permco"]].astype(int)

comp = db.raw_sql("""
    SELECT gvkey, datadate, seq, ceq, pstk, at, lt, pstkrv, pstkl, txditc
    FROM comp.funda
    WHERE indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
""", date_cols=["datadate"])

ccm = db.raw_sql("""
    SELECT gvkey, lpermno AS permno, linkprim, linkdt, linkenddt
    FROM crsp.ccmxpf_lnkhist
    WHERE linktype IN ('LU', 'LC') AND linkprim IN ('P', 'C')
""", date_cols=["linkdt", "linkenddt"])
ccm["permno"] = ccm["permno"].astype(int)
ccm["linkenddt"] = ccm["linkenddt"].fillna(pd.Timestamp.today())
db.close()

# Ken French's factors: RF for MKT-RF, and the benchmark for the comparison.
url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
       "F-F_Research_Data_Factors_CSV.zip")
with zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=60).content)) as z:
    text = z.read(z.namelist()[0]).decode("utf-8", errors="ignore")
rows = [[f.strip() for f in line.split(",")] for line in text.splitlines()]
rows = [r for r in rows if len(r) == 5 and len(r[0]) == 6 and r[0].isdigit()]  # monthly rows
french = pd.DataFrame(rows, columns=["yyyymm", "mkt_rf", "smb", "hml", "rf"])
french.index = pd.to_datetime(french.pop("yyyymm"), format="%Y%m") + pd.offsets.MonthEnd(0)
french = french.astype(float) / 100

# ------------------------------------------------------------------------------------
# Market equity: one row per firm (PERMCO) and month
# ------------------------------------------------------------------------------------
crsp["date"] = crsp["mthcaldt"] + pd.offsets.MonthEnd(0)
crsp["me_sec"] = crsp["mthprc"].abs() * crsp["shrout"]  # $ thousands
crsp.loc[crsp["me_sec"] <= 0, "me_sec"] = None

# Sum ME across share classes of the same firm; keep the largest class's row and return.
crsp["me"] = crsp.groupby(["permco", "date"])["me_sec"].transform("sum", min_count=1)
panel = crsp.sort_values(["permco", "date", "me_sec"], ascending=[True, True, False],
                         na_position="last")
panel = panel.drop_duplicates(["permco", "date"]).sort_values(["permco", "date"])

# Portfolio weight = ME at the end of the previous month (only if exactly one month earlier).
prev = panel.groupby("permco")[["date", "me"]].shift(1)
months_apart = ((panel["date"].dt.year - prev["date"].dt.year) * 12
                + panel["date"].dt.month - prev["date"].dt.month)
panel["me_lag"] = prev["me"].where(months_apart == 1)

# ------------------------------------------------------------------------------------
# Book equity (fiscal year ending in calendar year t-1, used in June t)
# ------------------------------------------------------------------------------------
comp["se"] = comp["seq"].fillna(comp["ceq"] + comp["pstk"]).fillna(comp["at"] - comp["lt"])
comp["ps"] = comp["pstkrv"].fillna(comp["pstkl"]).fillna(comp["pstk"]).fillna(0)
comp["be"] = comp["se"] + comp["txditc"].fillna(0) - comp["ps"]  # $ millions

comp["year"] = comp["datadate"].dt.year
comp = comp.sort_values(["gvkey", "datadate"]).drop_duplicates(["gvkey", "year"], keep="last")
comp["years_in_comp"] = comp.groupby("gvkey").cumcount() + 1
comp = comp[comp["be"].notna()]
comp["sort_year"] = comp["year"] + 1

# Link to CRSP: link valid at the June sort date, mapped to the firm (PERMCO).
comp["june_date"] = pd.to_datetime(comp["sort_year"].astype(str) + "-06-30")
link = comp.merge(ccm, on="gvkey")
link = link[(link["june_date"] >= link["linkdt"]) & (link["june_date"] <= link["linkenddt"])]
link = link.merge(crsp[["permno", "permco"]].drop_duplicates("permno", keep="last"), on="permno")
link = link.sort_values(["permco", "sort_year", "linkprim", "be"],
                        ascending=[True, True, False, False])
link = link.drop_duplicates(["permco", "sort_year"])

# ------------------------------------------------------------------------------------
# June sort: size = ME June t, B/M = BE t-1 / ME December t-1
# ------------------------------------------------------------------------------------
june = panel[panel["date"].dt.month == 6].copy()
june["sort_year"] = june["date"].dt.year
june = june.rename(columns={"me": "me_june"})[["permco", "sort_year", "me_june", "primaryexch"]]

dec = panel[panel["date"].dt.month == 12].copy()
dec["sort_year"] = dec["date"].dt.year + 1
dec = dec.rename(columns={"me": "me_dec"})[["permco", "sort_year", "me_dec"]]

sort = june.merge(dec, on=["permco", "sort_year"])
sort = sort.merge(link[["permco", "sort_year", "be", "years_in_comp"]], on=["permco", "sort_year"])
sort = sort[(sort["sort_year"] >= FIRST_SORT_YEAR) & sort["me_june"].notna()
            & (sort["be"] > 0) & (sort["me_dec"] > 0)
            & (sort["years_in_comp"] >= 2)]  # FF: at least two years in Compustat
sort["bm"] = sort["be"] / (sort["me_dec"] / 1000)  # BE in $m, ME in $k

# NYSE breakpoints: size median, B/M 30th and 70th percentiles.
nyse = sort[sort["primaryexch"] == "N"]
bp = nyse.groupby("sort_year").agg(size_med=("me_june", "median"),
                                   bm30=("bm", lambda s: s.quantile(0.3)),
                                   bm70=("bm", lambda s: s.quantile(0.7))).reset_index()
sort = sort.merge(bp, on="sort_year")

sort["size"] = "B"
sort.loc[sort["me_june"] <= sort["size_med"], "size"] = "S"
sort["value"] = "N"
sort.loc[sort["bm"] <= sort["bm30"], "value"] = "L"
sort.loc[sort["bm"] > sort["bm70"], "value"] = "H"
sort["portfolio"] = sort["size"] + sort["value"]

# ------------------------------------------------------------------------------------
# Value-weighted returns, held July t to June t+1
# ------------------------------------------------------------------------------------
rets = panel[panel["mthret"].notna() & (panel["me_lag"] > 0)].copy()
rets["sort_year"] = rets["date"].dt.year - (rets["date"].dt.month <= 6).astype(int)
rets["ret_w"] = rets["mthret"] * rets["me_lag"]

port = rets.merge(sort[["permco", "sort_year", "portfolio"]], on=["permco", "sort_year"])
g = port.groupby(["date", "portfolio"])
p = (g["ret_w"].sum() / g["me_lag"].sum()).unstack()

# ------------------------------------------------------------------------------------
# Factors
# ------------------------------------------------------------------------------------
ff3 = pd.DataFrame(index=p.index)
ff3["smb"] = (p["SL"] + p["SN"] + p["SH"]) / 3 - (p["BL"] + p["BN"] + p["BH"]) / 3
ff3["hml"] = (p["SH"] + p["BH"]) / 2 - (p["SL"] + p["BL"]) / 2

mkt = rets.groupby("date")["ret_w"].sum() / rets.groupby("date")["me_lag"].sum()
ff3["mkt_rf"] = mkt - french["rf"]
ff3 = ff3[["mkt_rf", "smb", "hml"]]
ff3.index.name = "date"
ff3.to_csv("ff3_factors.csv", float_format="%.6f")

# ------------------------------------------------------------------------------------
# Comparison with Ken French
# ------------------------------------------------------------------------------------
both = ff3.join(french, rsuffix="_ff", how="inner")
print(f"{both.index.min():%Y-%m} to {both.index.max():%Y-%m}, {len(both)} months")
for f in ["mkt_rf", "smb", "hml"]:
    print(f"{f:<7} corr {both[f].corr(both[f + '_ff']):.3f}   "
          f"mean p.a. {both[f].mean() * 1200:5.2f}% (French {both[f + '_ff'].mean() * 1200:5.2f}%)")
