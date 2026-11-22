# ABOUTME: Writes the plain-English signal cards (definition, long/short side, economic story, practice, caveats).
"""
Run:     python SignalDashboard/D_SignalCards.py   (from pyCode/)
Inputs:  output/signal_info.csv (C_DataReport.py): paper, sample, t-stat, weighting, rebalancing
Outputs: output/signal_cards.md          the cards, for reading and presenting
         pyData/OpenAP/signal_cards.json the same content, read by the dashboard (Stage 4)

Where the content comes from (all checked on 2026-11-05):
- Definitions and timing: OpenAP code at tag v2.1.0 (the code behind release 202610):
  Signals/pyCode/Predictors/<Signal>.py (IdioVol3F is built in
  ZZ0_RealizedVol_IdioVol3F_ReturnSkew3F.py, AnnouncementReturn in ZZ2_AnnouncementReturn.py),
  Signals/pyCode/DataDownloads/CompustatAnnual.py (6-month reporting lag) and
  Portfolios/Code/01_PortfolioFunction.R (sorting, 1-month lag, rebalancing, long-short).
- Numbers about each paper (sample, t-stat, journal, weighting): SignalDoc.csv, read from
  signal_info.csv, NOT typed in here, so they cannot drift from the data.
- Economic stories and caveats: the papers in REFERENCES. Each claim was checked against
  the paper's published abstract (EconPapers / IDEAS listings). We did not read every full
  text, so anything we could not confirm is listed under "Not verified" on the card.

Timing convention used on every card: OpenAP lags every signal by one month before
sorting (01_PortfolioFunction.R: yyyymm + 1). We call the month whose return the portfolio
earns the "holding month" h. A signal "dated month t" is used for returns in h = t + 1.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make config/utils importable

import pandas as pd  # noqa: E402

from config import DASHBOARD_SIGNALS, OPENAP, OUTPUT, SIGNAL_START_OVERRIDE, make_folders  # noqa: E402

MD_FILE = OUTPUT / "signal_cards.md"
JSON_FILE = OPENAP / "signal_cards.json"

# Turnover rating from the rebalancing frequency (SignalDoc 'Portfolio Period'), as asked
# for in the brief. This is a first guess: real turnover also depends on how quickly the
# signal itself changes, which each card discusses where the code makes it clear.
TURNOVER_BY_PERIOD = {1: "High", 3: "Medium", 12: "Low"}
REBALANCING_TEXT = {1: "monthly", 3: "every 3 months", 12: "once a year (in June)"}
JOURNALS = {"JF": "Journal of Finance", "JFE": "Journal of Financial Economics",
            "AR": "The Accounting Review", "JFM": "Journal of Financial Markets"}

REFERENCES = {
    "JT1993": "Jegadeesh, N., and S. Titman (1993). Returns to Buying Winners and Selling "
              "Losers: Implications for Stock Market Efficiency. Journal of Finance 48(1), 65-91.",
    "DM2016": "Daniel, K., and T. J. Moskowitz (2016). Momentum Crashes. Journal of "
              "Financial Economics 122(2), 221-247.",
    "J1990": "Jegadeesh, N. (1990). Evidence of Predictable Behavior of Security Returns. "
             "Journal of Finance 45(3), 881-898.",
    "N2012": "Nagel, S. (2012). Evaporating Liquidity. Review of Financial Studies 25(7), "
             "2005-2039.",
    "DBT1985": "De Bondt, W. F. M., and R. Thaler (1985). Does the Stock Market Overreact? "
               "Journal of Finance 40(3), 793-805.",
    "FF1996": "Fama, E. F., and K. R. French (1996). Multifactor Explanations of Asset "
              "Pricing Anomalies. Journal of Finance 51(1), 55-84.",
    "NM2013": "Novy-Marx, R. (2013). The Other Side of Value: The Gross Profitability "
              "Premium. Journal of Financial Economics 108(1), 1-28.",
    "FF2015": "Fama, E. F., and K. R. French (2015). A Five-Factor Asset Pricing Model. "
              "Journal of Financial Economics 116(1), 1-22.",
    "CGS2008": "Cooper, M. J., H. Gulen, and M. J. Schill (2008). Asset Growth and the "
               "Cross-Section of Stock Returns. Journal of Finance 63(4), 1609-1651.",
    "HXZ2015": "Hou, K., C. Xue, and L. Zhang (2015). Digesting Anomalies: An Investment "
               "Approach. Review of Financial Studies 28(3), 650-705.",
    "PW2008": "Pontiff, J., and A. Woodgate (2008). Share Issuance and Cross-sectional "
              "Returns. Journal of Finance 63(2), 921-945.",
    "S1996": "Sloan, R. G. (1996). Do Stock Prices Fully Reflect Information in Accruals and "
             "Cash Flows about Future Earnings? The Accounting Review 71(3), 289-315.",
    "GHS2011": "Green, J., J. R. M. Hand, and M. T. Soliman (2011). Going, Going, Gone? The "
               "Apparent Demise of the Accruals Anomaly. Management Science 57(5), 797-816.",
    "AHXZ2006": "Ang, A., R. J. Hodrick, Y. Xing, and X. Zhang (2006). The Cross-Section of "
                "Volatility and Expected Returns. Journal of Finance 61(1), 259-299.",
    "SYY2015": "Stambaugh, R. F., J. Yu, and Y. Yuan (2015). Arbitrage Asymmetry and the "
               "Idiosyncratic Volatility Puzzle. Journal of Finance 70(5), 1903-1948.",
    "BC2008": "Bali, T. G., and N. Cakici (2008). Idiosyncratic Volatility and the Cross "
              "Section of Expected Returns. Journal of Financial and Quantitative Analysis "
              "43(1), 29-58.",
    "A2002": "Amihud, Y. (2002). Illiquidity and Stock Returns: Cross-Section and "
             "Time-Series Effects. Journal of Financial Markets 5(1), 31-56.",
    "BKW2015": "Ben-Rephael, A., O. Kadan, and A. Wohl (2015). The Diminishing Liquidity "
               "Premium. Journal of Financial and Quantitative Analysis 50(1-2), 197-229.",
    "CJL1996": "Chan, L. K. C., N. Jegadeesh, and J. Lakonishok (1996). Momentum "
               "Strategies. Journal of Finance 51(5), 1681-1713.",
    "MP2016": "McLean, R. D., and J. Pontiff (2016). Does Academic Research Destroy Stock "
              "Return Predictability? Journal of Finance 71(1), 5-32.",
    "CZ2022": "Chen, A. Y., and T. Zimmermann (2022). Open Source Cross-Sectional Asset "
              "Pricing. Critical Finance Review 11(2), 207-264.",
    "OAP": "Open Source Asset Pricing, code v2.1.0 (github.com/OpenSourceAP/CrossSection) "
           "and SignalDoc.csv, data release 202610.",
}

# ---------------------------------------------------------------------------------------
# The cards. Each list item is one bullet. [KEY] at the end of a sentence points to
# REFERENCES. "practice" items are our suggestions for how the signal could be used, not
# empirical claims, unless they carry a reference.
# ---------------------------------------------------------------------------------------
CARDS = {
    "Mom12m": {
        "in_one_line": "Stocks that went up most over the past year (skipping last month) "
                       "keep outperforming those that went down most.",
        "measures": [
            "Cumulative return over 11 months: compounded monthly returns from 12 months "
            "to 2 months before the holding month. The most recent month is skipped [OAP].",
            "In OpenAP's code: product of (1 + ret) over lags 1 to 11 of the signal month, "
            "minus 1; missing monthly returns are treated as 0 (Mom12m.py) [OAP].",
            "Data: CRSP monthly returns only.",
        ],
        "long_short": "Long the top decile (past winners), short the bottom decile (past "
                      "losers). The paper finds winners keep outperforming for 3 to 12 "
                      "months [JT1993].",
        "story_type": "Behavioral (underreaction)",
        "story": [
            "The original paper finds the profits are not due to systematic risk or to "
            "delayed reactions to common factors, and says the evidence is consistent with "
            "delayed price reactions to firm-specific information [JT1993].",
        ],
        "follow_up": [
            "Momentum has rare but severe crashes. They happen in 'panic' states, after "
            "market declines and when volatility is high, as the market rebounds. Scaling "
            "momentum by forecasts of its mean and variance roughly doubles its Sharpe "
            "ratio [DM2016].",
        ],
        "practice": [
            "Long-short factor (the UMD factor in Ken French's library is the standard "
            "version), or a tilt towards winners in a long-only portfolio.",
            "Combination: momentum is commonly combined with value in practice. Whether the "
            "two diversify each other here is something the Stage 3 correlation table will "
            "show; we do not assume it.",
            "Risk management matters more than for most signals: see the crash evidence "
            "[DM2016].",
        ],
        "turnover_note": "OpenAP rebalances every 3 months. The 11-month window shifts "
                         "every month, so many winners stay winners, but the extreme "
                         "deciles still change a lot.",
        "caveats": [
            "Crash risk when the market rebounds after declines and volatility is high "
            "[DM2016]. Stage 3's worst months will show where this hits in our data.",
            "OpenAP's version rebalances every 3 months (Portfolio Period = 3, "
            "01_PortfolioFunction.R), to match the paper's 3-month holding period "
            "(Table 1A, K = 3) [OAP].",
        ],
        "not_verified": [
            "Whether the paper's Table 1A also skips a month between ranking and holding. "
            "OpenAP's version does.",
        ],
    },
    "STreversal": {
        "in_one_line": "Last month's biggest losers tend to bounce back next month, and "
                       "last month's biggest winners tend to fall back.",
        "measures": [
            "The stock's return in the month just before the holding month (STreversal.py) "
            "[OAP]. Missing returns are set to 0.",
            "Universe: CRSP common stocks (share codes 10, 11, 12) on NYSE, AMEX and "
            "NASDAQ [OAP].",
            "Data: CRSP monthly returns only.",
        ],
        "long_short": "Long the bottom decile of last month's return (losers), short the top "
                      "decile (winners). SignalDoc Sign = -1 [OAP].",
        "story_type": "Not settled in the original paper; liquidity provision in follow-up "
                      "work",
        "story": [
            "The original paper documents negative first-order serial correlation in monthly "
            "returns: a high return this month predicts a lower return next month [J1990].",
        ],
        "follow_up": [
            "Reversal returns can be read as the return to liquidity provision: you buy "
            "what others are selling. These returns are highly predictable with the VIX and "
            "jump during financial turmoil, which the paper interprets as intermediaries "
            "withdrawing liquidity in crises [N2012].",
        ],
        "practice": [
            "As a stand-alone long-short factor it needs a full turnover of the extreme "
            "deciles every month, so it is mainly a trading-desk signal.",
            "More practical as a timing filter: avoid buying a stock right after an extreme "
            "one-month run-up, or delay selling one right after a crash, as part of the "
            "execution of other signals.",
            "The liquidity-provision view suggests it pays most when markets are stressed "
            "(high VIX) [N2012].",
        ],
        "turnover_note": "Monthly rebalancing on a signal that is completely new each month: "
                         "the highest turnover of all 10 signals.",
        "caveats": [
            "Paper portfolio, equal-weighted (SignalDoc), so small stocks get as much weight "
            "as large ones [OAP].",
        ],
        "not_verified": [
            "Which explanation (overreaction, price pressure or microstructure effects) the "
            "1990 paper itself favours. Its abstract only documents the predictability.",
        ],
    },
    "LRreversal": {
        "in_one_line": "Stocks that did worst over the past few years tend to beat those "
                       "that did best (a contrarian signal).",
        "measures": [
            "Compounded return over 24 months, from 36 months to 13 months before the "
            "signal month: lags 13 to 36 in LRreversal.py [OAP]. With the 1-month portfolio "
            "lag, the window ends about 14 months before the holding month, so the "
            "momentum year is left out.",
            "Data: CRSP monthly returns only.",
        ],
        "long_short": "Long the bottom quintile (long-run losers), short the top quintile "
                      "(long-run winners). SignalDoc Sign = -1 [OAP].",
        "story_type": "Behavioral (overreaction)",
        "story": [
            "The paper tests whether investors overreact to unexpected news, so that "
            "extreme past losers become too cheap and extreme past winners too expensive. "
            "Loser portfolios earn exceptionally large January returns, even five years "
            "after the portfolios are formed [DBT1985].",
        ],
        "follow_up": [
            "Fama and French report that most of the anomalies they test, including "
            "De Bondt and Thaler's long-term reversal, largely disappear in the three-factor "
            "model, i.e. they are explained by exposure to the market, SMB and HML "
            "[FF1996].",
        ],
        "practice": [
            "Little to gain as a separate factor if you already hold SMB and HML, because "
            "the three-factor model largely captures it [FF1996]. We will test this with our "
            "own FF3 factors in Stage 3.",
            "Could serve as a contrarian screen in a value strategy.",
        ],
        "turnover_note": "OpenAP rebalances monthly, but the 24-month window moves by only "
                         "one month at a time, so the signal changes slowly. True turnover "
                         "should be well below what 'monthly' suggests.",
        "caveats": [
            "OpenAP: insignificant at the 12-month horizon. The original evidence is at "
            "3-year horizons (SignalDoc Notes) [OAP].",
            "Strong January seasonality on the long (loser) side [DBT1985].",
        ],
        "not_verified": [],
    },
    "GP": {
        "in_one_line": "Firms with high gross profits relative to their assets earn higher "
                       "returns, even though they look expensive.",
        "measures": [
            "(Revenue - cost of goods sold) / total assets: (revt - cogs) / at in GP.py "
            "[OAP]. SignalDoc's text says 'sale', but the code uses revt.",
            "Financial firms (SIC 6000-6999) are excluded [OAP].",
            "Timing: annual Compustat data counts as available 6 months after the fiscal "
            "year-end (CompustatAnnual.py). Portfolios are formed in June and held July to "
            "June, so each June uses fiscal years that ended by the previous December "
            "[OAP].",
            "Data: Compustat annual (CRSP for returns and market cap).",
        ],
        "long_short": "Long the top quintile (most profitable), short the bottom quintile. "
                      "Value-weighted, as in the paper [OAP].",
        "story_type": "Not settled; the paper argues against the usual risk stories",
        "story": [
            "Gross profits-to-assets predicts returns about as well as book-to-market. "
            "Profitable firms earn higher returns despite higher valuation ratios [NM2013].",
            "The paper argues this is hard to square with popular explanations of the value "
            "premium: profitable firms are less prone to distress, have longer cash-flow "
            "durations and lower operating leverage [NM2013].",
            "Why gross profits: the paper argues it is a cleaner measure of economic "
            "profitability than measures further down the income statement [NM2013]. "
            "(See 'Not verified'.)",
        ],
        "follow_up": [
            "Profitability became a standard factor: RMW in the Fama-French five-factor "
            "model [FF2015]. Our own RMW and RMW_GP replications are in the FF3 project.",
        ],
        "practice": [
            "Best used together with value. Controlling for profitability dramatically "
            "improves value strategies, especially among the largest, most liquid stocks "
            "[NM2013].",
            "Works as a quality tilt or screen in a long-only portfolio (e.g. avoid the "
            "least profitable quintile).",
        ],
        "turnover_note": "Rebalanced once a year on a slow-moving accounting ratio: low "
                         "turnover.",
        "caveats": [
            "OpenAP uses all-stock breakpoints. The paper's Table 2a mentions NYSE "
            "breakpoints, but OpenAP matches the paper's result more closely without them "
            "(SignalDoc Notes) [OAP].",
            "Excludes financial firms, so it says nothing about banks and insurers [OAP].",
            "Its original t-stat is the lowest of the 10 signals (see the facts table, from "
            "SignalDoc) [OAP].",
        ],
        "not_verified": [
            "The 'cleaner measure' argument is from our reading of the paper, not from its "
            "abstract.",
        ],
    },
    "AssetGrowth": {
        "in_one_line": "Firms that grew their total assets the least over the past year "
                       "outperform the fastest growers.",
        "measures": [
            "Annual growth in total assets: (at - at one year earlier) / at one year earlier "
            "(AssetGrowth.py) [OAP].",
            "Timing: annual Compustat with a 6-month reporting lag; formed each June, held "
            "July to June [OAP].",
            "Data: Compustat annual.",
        ],
        "long_short": "Long the bottom decile (lowest asset growth), short the top decile "
                      "(highest growth). SignalDoc Sign = -1 [OAP].",
        "story_type": "Debated: mispricing vs investment-based (risk) explanations",
        "story": [
            "The paper documents that asset growth strongly predicts future abnormal "
            "returns, and that it keeps its forecasting power even among large stocks, "
            "where many other predictors lose theirs [CGS2008].",
        ],
        "follow_up": [
            "The q-factor model gives an investment-based (rational) explanation: firms "
            "invest more when their cost of capital is low, so high investment goes with low "
            "expected returns. Its investment factor is built on investment-to-assets "
            "(asset growth) [HXZ2015].",
        ],
        "practice": [
            "Long-short factor: close cousin of CMA (Fama-French) and the q-factor "
            "investment factor [HXZ2015]. Expect high correlation with them.",
            "Screen in a long-only portfolio: avoid firms with extreme asset growth (e.g. "
            "big acquirers).",
        ],
        "turnover_note": "Rebalanced once a year: low turnover.",
        "caveats": [
            "Equal-weighted in OpenAP's main version. OpenAP notes it also works "
            "value-weighted (t = 5 in the paper's Table 2.B.2, SignalDoc Notes) [OAP].",
        ],
        "not_verified": [
            "Which explanation the original authors favour (we believe they lean towards "
            "investors over-extrapolating past growth). The abstract does not say, so this "
            "is left open.",
        ],
    },
    "ShareIss1Y": {
        "in_one_line": "Firms that issued many new shares over the past year underperform; "
                       "firms that bought shares back outperform.",
        "measures": [
            "Growth in split-adjusted shares outstanding (shrout x cfacshr) from 18 to 6 "
            "months before the signal month (ShareIss1Y.py) [OAP].",
            "Because it is rebalanced each June, the June signal measures share growth "
            "from December two years earlier to December of the previous year.",
            "Data: CRSP monthly shares and adjustment factors only. SignalDoc labels the "
            "data category 'Accounting', but the code uses no Compustat data [OAP].",
        ],
        "long_short": "Long the bottom quintile (lowest issuance, including buybacks), short "
                      "the top quintile (heavy issuers). SignalDoc Sign = -1 [OAP].",
        "story_type": "Not verified (see below)",
        "story": [
            "After 1970, share issuance predicts returns (negatively) more significantly "
            "than size, book-to-market or momentum do individually. Before 1970 the paper "
            "finds no significant predictive ability for most holding periods [PW2008].",
            "The result survives excluding the data used in earlier studies of long-run "
            "returns after buybacks, seasoned offerings and mergers [PW2008].",
        ],
        "follow_up": [],
        "practice": [
            "Screen or tilt: avoid heavy issuers. This is easy to implement because it only "
            "needs share counts.",
            "Natural partner of the investment signals (AssetGrowth): firms that issue "
            "shares often grow their assets with the money.",
        ],
        "turnover_note": "Rebalanced once a year: low turnover.",
        "caveats": [
            "OpenAP's series starts in 1927, but the paper finds no significant effect "
            "before 1970 [PW2008]. Full-sample statistics mix the two periods, so the "
            "period split in Stage 3 matters here.",
            "The original evidence is a univariate regression, not a portfolio sort, so the quintile long-short is OpenAP's own construction (SignalDoc "
            "'Test in OP') [OAP].",
        ],
        "not_verified": [
            "The paper's own explanation (e.g. managers issuing when shares are overvalued). "
            "The abstract does not state one, so we do not attribute one.",
        ],
    },
    "Accruals": {
        "in_one_line": "Firms whose earnings come mostly from accruals rather than cash "
                       "underperform firms with low accruals.",
        "measures": [
            "Working-capital accruals scaled by average total assets: (change in current "
            "assets - change in cash) - (change in current liabilities - change in "
            "short-term debt - change in taxes payable) - depreciation, all divided by the "
            "average of this year's and last year's total assets (Accruals.py) [OAP].",
            "SignalDoc's text leaves out depreciation; the code subtracts it, as in Sloan's "
            "definition [OAP].",
            "Filter: price above $5 at formation (SignalDoc Filter) [OAP].",
            "Timing: annual Compustat with a 6-month reporting lag; formed each June, held "
            "July to June [OAP].",
            "Data: Compustat annual.",
        ],
        "long_short": "Long the bottom decile (lowest accruals), short the top decile "
                      "(highest accruals). SignalDoc Sign = -1 [OAP].",
        "story_type": "Behavioral (earnings fixation)",
        "story": [
            "Future earnings depend on how much of current earnings comes from cash flows "
            "versus accruals, but stock prices behave as if investors fixate on total "
            "earnings. The information in the two components only reaches prices once it "
            "shows up in future earnings [S1996].",
        ],
        "follow_up": [
            "The hedge returns to the accruals anomaly decayed in the US until they were, on "
            "average, no longer reliably positive. The paper links this partly to more "
            "hedge-fund capital trading on it [GHS2011].",
        ],
        "practice": [
            "Earnings-quality screen: flag firms whose earnings are mostly accruals.",
            "As a long-short factor, check the post-publication returns in Stage 3 before "
            "relying on it [GHS2011].",
        ],
        "turnover_note": "Rebalanced once a year: low turnover.",
        "caveats": [
            "Reported to have largely disappeared in the 2000s [GHS2011]. Stage 3 will show "
            "whether our post-publication numbers agree.",
            "The original test is a size-adjusted hedge return (Table 6), not a raw "
            "long-short return (SignalDoc) [OAP].",
        ],
        "not_verified": [],
    },
    "IdioVol3F": {
        "in_one_line": "Stocks whose returns jumped around a lot for firm-specific reasons "
                       "last month earn low returns afterwards.",
        "measures": [
            "Standard deviation of the residuals from a Fama-French three-factor regression "
            "of the stock's daily excess returns, estimated within one calendar month; at "
            "least 15 daily returns are required "
            "(ZZ0_RealizedVol_IdioVol3F_ReturnSkew3F.py) [OAP].",
            "Used for the next month's return (1-month portfolio lag) [OAP].",
            "Data: CRSP daily returns and Ken French's daily factors.",
        ],
        "long_short": "Long the bottom quintile (lowest idiosyncratic volatility), short the "
                      "top quintile. Value-weighted, as in the paper. SignalDoc Sign = -1 "
                      "[OAP].",
        "story_type": "A puzzle in the original paper; mispricing in follow-up work",
        "story": [
            "Size, book-to-market, momentum and liquidity effects cannot account for the "
            "low average returns of stocks with high idiosyncratic volatility [AHXZ2006].",
            "Why it is called a puzzle (textbook CAPM reasoning, our addition): idiosyncratic "
            "risk can be diversified away, so it should not be priced at all, let alone "
            "negatively.",
        ],
        "follow_up": [
            "Arbitrage asymmetry: buying is easier than shorting, so overpricing persists "
            "more than underpricing. The volatility-return relation is negative among "
            "overpriced stocks and positive among underpriced ones; the negative side is "
            "stronger, especially in stocks that are hard to short [SYY2015].",
        ],
        "practice": [
            "Low-volatility tilt or screen in long-only portfolios: avoiding the highest-IVOL "
            "stocks does not require shorting them.",
            "As a long-short factor it needs monthly rebalancing.",
        ],
        "turnover_note": "Monthly rebalancing, and the signal is re-estimated from one month "
                         "of daily data each month, so it is noisy: high turnover.",
        "caveats": [
            "Fragile to construction choices: one replication finds no robust relation once "
            "the data frequency, weighting scheme, breakpoints and screens for size, price "
            "and liquidity are varied [BC2008].",
            "OpenAP follows the original (value-weighted quintiles, one month of daily "
            "data) [OAP].",
        ],
        "not_verified": [],
    },
    "Illiquidity": {
        "in_one_line": "Hard-to-trade stocks (whose price moves a lot per dollar traded) earn "
                       "higher returns than liquid ones.",
        "measures": [
            "Amihud's measure: for each day, |return| / dollar volume (|price| x volume). "
            "These are averaged within each month, then over the last 12 months (the signal "
            "month and the 11 months before), and all 12 months must be available "
            "(Illiquidity.py) [OAP].",
            "Filter: NYSE stocks only, price above $5 (SignalDoc Filter: abs(prc)>5, "
            "exchcd==1) [OAP].",
            "Formed each June, held July to June [OAP].",
            "Data: CRSP daily returns, prices and volume.",
        ],
        "long_short": "Long the top quintile (most illiquid), short the bottom quintile "
                      "(most liquid). SignalDoc Sign = +1 [OAP].",
        "story_type": "Compensation for illiquidity (risk/cost based)",
        "story": [
            "Across stocks, illiquidity is positively related to expected returns. Over "
            "time, expected market illiquidity raises ex-ante excess returns, while "
            "unexpected illiquidity lowers contemporaneous returns: the return is partly an "
            "illiquidity premium [A2002].",
            "The effects are stronger for small stocks [A2002].",
        ],
        "follow_up": [
            "Stock liquidity has improved over four decades and the premium for stock-level "
            "illiquidity has diminished [BKW2015].",
        ],
        "practice": [
            "Hard to harvest directly: the long side is, by construction, the stocks that "
            "are most expensive to trade. In practice it is more a cost to manage than a "
            "factor to buy.",
            "Useful as a control: check whether another signal's returns come mostly from "
            "illiquid stocks.",
        ],
        "turnover_note": "Rebalanced once a year on a 12-month average: low turnover, but "
                         "each trade in the long leg is expensive.",
        "caveats": [
            "The premium has declined over time [BKW2015].",
            "The original evidence is a cross-sectional regression, so the quintile long-short is OpenAP's own construction (SignalDoc) [OAP].",
            "1932-06 has only 14 stocks in the long leg (Stage 1). Kept, as agreed.",
        ],
        "not_verified": [],
    },
    "AnnouncementReturn": {
        "in_one_line": "Stocks whose price jumped around their last earnings announcement "
                       "keep drifting in the same direction.",
        "measures": [
            "Sum of daily market-adjusted returns (stock return - market return) over the "
            "4 trading days from 2 days before to 1 day after the quarterly earnings "
            "announcement date (Compustat 'rdq') (ZZ2_AnnouncementReturn.py) [OAP].",
            "The latest announcement return is carried forward for up to 6 months, so each "
            "stock's signal changes roughly once a quarter [OAP].",
            "SignalDoc's text says 'from one day before to 2 days after' and 'IBES dates', "
            "but the v2.1.0 code uses days -2 to +1 and Compustat dates. The card follows "
            "the code.",
            "Data: CRSP daily returns, Compustat quarterly announcement dates, Ken French's "
            "daily market factor.",
        ],
        "long_short": "Long the top quintile (best announcement reaction), short the bottom "
                      "quintile. SignalDoc Sign = +1 [OAP].",
        "story_type": "Behavioral (underreaction to earnings news)",
        "story": [
            "Past returns and past earnings surprises each predict large drifts in future "
            "returns, after controlling for each other. Market risk, size and book-to-market "
            "do not explain the drifts, and there is little evidence of later reversals "
            "[CJL1996].",
            "The authors relate this to the market underreacting to information, especially "
            "earnings news [CJL1996].",
        ],
        "follow_up": [],
        "practice": [
            "An event-driven tilt: update positions after each earnings season.",
            "Combination: pairs naturally with price momentum (Mom12m). The paper treats "
            "the two as separate effects [CJL1996].",
        ],
        "turnover_note": "Rebalanced monthly. Each stock's signal changes only when it "
                         "announces (about quarterly), but in every earnings season many "
                         "stocks enter and leave the extreme quintiles: high turnover.",
        "caveats": [
            "The first two months (1971-09/10) have 1 and 13 stocks per leg. Our statistics "
            "start in 1971-11, as agreed after Stage 1.",
            "The original evidence is a multivariate regression (Table 7). OpenAP's quintile long-short is its own construction (SignalDoc) [OAP].",
        ],
        "not_verified": [],
    },
}


def build_cards(info):
    """Combine the hand-written text with the SignalDoc fields (numbers come from data)."""
    cards = []
    for signal in DASHBOARD_SIGNALS:
        row = info.loc[signal]
        text = CARDS[signal]
        period = int(row["portfolio_period_months"])
        quantile = "deciles" if row["n_portfolios"] == 10 else "quintiles"
        card = {
            "signal": signal,
            "full_name": row["full_name"],
            "paper": f"{row['authors']} ({row['year']}), {JOURNALS.get(row['journal'], row['journal'])}",
            "facts": {
                "Original sample": f"{row['sample_start']}-{row['sample_end']}",
                "Published": str(row["year"]),
                "Original t-stat": f"{row['tstat_original']:.2f}",
                "Economic category": row["economic_category"],
                "Data category (SignalDoc)": row["data_category"],
                "Portfolios": f"{quantile}, {'value' if row['stock_weight'] == 'VW' else 'equal'}-weighted",
                "Rebalancing": REBALANCING_TEXT[period],
                "Turnover / trading costs": TURNOVER_BY_PERIOD[period],
                "Stats start": SIGNAL_START_OVERRIDE.get(signal, "first month available"),
            },
            **text,
        }
        # Collect every reference key used on this card, in REFERENCES order.
        all_text = json.dumps(text)
        card["references"] = [k for k in REFERENCES if f"[{k}]" in all_text]
        cards.append(card)
    return cards


def card_to_markdown(card):
    lines = [f"## {card['signal']}: {card['full_name']}", "",
             f"*{card['in_one_line']}*", "",
             f"**Paper:** {card['paper']}", "",
             "| | |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in card["facts"].items()]
    sections = [("What it measures", card["measures"]),
                ("Long and short side", [card["long_short"]]),
                (f"Economic story: {card['story_type']}", card["story"]),
                ("Follow-up evidence", card["follow_up"]),
                ("How it could be used in practice", card["practice"] + [
                    f"Turnover: **{card['facts']['Turnover / trading costs']}**. "
                    f"{card['turnover_note']}"]),
                ("Caveats", card["caveats"]),
                ("Not verified", card["not_verified"])]
    for title, items in sections:
        if items:
            lines += ["", f"**{title}**", ""] + [f"- {item}" for item in items]
    lines += ["", "**References**", ""] + [f"- [{k}] {REFERENCES[k]}" for k in card["references"]]
    return "\n".join(lines)


INTRO = """# Signal cards: 10 OpenAP anomalies

How to read a card:

- **Long-short (LS)** = return of the top portfolio minus the bottom portfolio of the signal,
  after multiplying the signal by the sign from the original paper. So LS is always long
  the side the paper says earns more. It is a paper portfolio: zero cost, before trading
  costs.
- **Timing:** OpenAP lags every signal by one month. The "holding month" is the month
  whose return the portfolio earns.
- **Turnover rating** comes from the rebalancing frequency (monthly = High, every 3 months
  = Medium, yearly = Low). Each card adds how fast the signal itself changes. For measured
  trading costs, see Novy-Marx, R., and M. Velikov (2016), A Taxonomy of Anomalies and
  Their Trading Costs, Review of Financial Studies 29(1), 104-147. We did not check that
  paper per signal, so no card quotes it.
- **Sources:** definitions come from OpenAP's code (v2.1.0) and SignalDoc.csv (release
  202610) [OAP], [CZ2022]. Economic claims come from the cited papers and were checked
  against their published abstracts. Anything we could not confirm is under
  "Not verified".
- **General caveat:** across 97 predictors, returns were about 26% lower out of sample and
  about 58% lower after publication [MP2016]. Stage 3 measures this for each signal.

References used above:
- [OAP] {OAP}
- [CZ2022] {CZ2022}
- [MP2016] {MP2016}
""".format(**REFERENCES)


def main():
    make_folders()
    info = pd.read_csv(OUTPUT / "signal_info.csv").set_index("signal")
    missing = sorted(set(DASHBOARD_SIGNALS) - set(CARDS))
    if missing:
        raise SystemExit(f"No card text for: {missing}")
    cards = build_cards(info)

    # Every [KEY] in the text must exist in REFERENCES (catches typos in citations).
    all_text = json.dumps(CARDS)
    for key in set(pd.Series(all_text).str.findall(r"\[([A-Za-z]+\d{4}|OAP)\]")[0]):
        if key not in REFERENCES:
            raise ValueError(f"Unknown reference key [{key}]")

    markdown = INTRO + "\n---\n\n" + "\n\n---\n\n".join(card_to_markdown(c) for c in cards) + "\n"
    MD_FILE.write_text(markdown, encoding="utf-8")
    JSON_FILE.write_text(json.dumps({"cards": cards, "references": REFERENCES}, indent=2),
                         encoding="utf-8")

    print(f"Wrote {len(cards)} cards to {MD_FILE.name} and {JSON_FILE.name}")
    for c in cards:
        print(f"  {c['signal']:<19} {c['facts']['Rebalancing']:<22} turnover "
              f"{c['facts']['Turnover / trading costs']:<7} refs: {', '.join(c['references'])}")


if __name__ == "__main__":
    main()
