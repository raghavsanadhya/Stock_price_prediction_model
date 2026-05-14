"""Central configuration for tickers, paths, and model defaults."""

from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent

# Cached OHLCV under project root
DATA_DIR = ROOT / "data" / "cache"

# Yahoo Finance symbols for front-month futures (liquid, widely used proxies)
COMMODITIES = {
    "WTI Crude Oil (1,000 barrels)": "CL=F",
    "Brent Crude Oil (1,000 barrels)": "BZ=F",
    "Natural Gas (10,000 MMBtu)": "NG=F",
    "Copper (25,000 pounds)": "HG=F",
    "Gold (100 troy ounces)": "GC=F",
    "Silver (5,000 troy ounces)": "SI=F",
    "Platinum (50 troy ounces)": "PL=F",
    "Palladium (100 troy ounces)": "PA=F",
    "Coffee (37,500 pounds)": "KC=F",
    "Sugar (112,000 pounds)": "SB=F",
    "Wheat (5,000 bushels)": "ZW=F",
    "Corn (5,000 bushels)": "ZC=F",
    "Soybeans (5,000 bushels)": "ZS=F",
}

# Top NASDAQ stocks by market cap and liquidity
NASDAQ_STOCKS = {
    "Apple (AAPL)": "AAPL",
    "Microsoft (MSFT)": "MSFT",
    "Amazon (AMZN)": "AMZN",
    "Tesla (TSLA)": "TSLA",
    "Meta (META)": "META",
    "Nvidia (NVDA)": "NVDA",
    "Alphabet/Google (GOOGL)": "GOOGL",
    "Broadcom (AVGO)": "AVGO",
    "Costco (COST)": "COST",
    "Invesco QQQ (QQQ)": "QQQ",
    "Netflix (NFLX)": "NFLX",
    "AMD (AMD)": "AMD",
    "Qualcomm (QCOM)": "QCOM",
    "PepsiCo (PEP)": "PEP",
    "Walmart (WMT)": "WMT",
    "Airbnb (ABNB)": "ABNB",
    "ASML (ASML)": "ASML",
    "Adobe (ADBE)": "ADBE",
    "Salesforce (CRM)": "CRM",
    "ServiceNow (NOW)": "NOW",
    "Accenture (ACN)": "ACN",
    "Cisco (CSCO)": "CSCO",
    "Intel (INTC)": "INTC",
}

# BSE Sensex and other major Indian stocks
INDIAN_STOCKS = {
    "Tata Consultancy Services (TCS)": "TCS.NS",
    "Reliance Industries": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "Infosys": "INFY.NS",
    "Wipro": "WIPRO.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "Axis Bank": "AXISBANK.NS",
    "HCL Technologies": "HCLTECH.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "State Bank of India": "SBIN.NS",
    "NTPC Limited": "NTPC.NS",
    "Coal India": "COALINDIA.NS",
    "Power Grid": "POWERGRID.NS",
    "Larsen & Toubro": "LT.NS",
    "Titan Company": "TITAN.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Nestlé India": "NESTLEIND.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Sun Pharmaceutical": "SUNPHARMA.NS",
    "Bajaj Finance": "BAJAJFINSV.NS",
    "Tata Steel": "TATASTEEL.NS",
    "Divi's Laboratories": "DIVISLAB.NS",
    "Adani Green Energy": "ADANIGREEN.NS",
    "Adani Ports": "ADANIPORTS.NS",
    "GAIL (India)": "GAIL.NS",
    "Indian Oil": "IOC.NS",
    "BPCL": "BPCL.NS",
    "Muthoot Finance Ltd": "MUTHOOTFIN.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "Hindustan Unilever": "HINDUNILVR.NS",
    "Eicher Motors Ltd": " EICHERMOT.NS",
    "Mahindra & Mahindra": "M&M.NS",
    "TVS Motor": "TVSMOTOR.NS",
    "Mazagon Dock Shipbuilders Ltd": "MAZDOCK.NS",
}

# Combined assets dictionary
ASSETS = {
    **{f"Commodity: {k}": v for k, v in COMMODITIES.items()},
    **{f"American: {k}": v for k, v in NASDAQ_STOCKS.items()},
    **{f"Indian: {k}": v for k, v in INDIAN_STOCKS.items()},
}

DEFAULT_PERIOD = "2y"
DEFAULT_INTERVAL = "1d"
DEFAULT_FORECAST_DAYS = 30

# Feature / model defaults
LAG_DAYS = list(range(1, 8))  # 1..7 day lags of close
ROLL_WINDOWS = (5, 10)
TEST_SIZE_FRAC = 0.2  # hold out last 20% for backtest metrics
RANDOM_STATE = 42
