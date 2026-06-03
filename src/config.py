from datetime import date, timedelta

START_DATE = "2012-01-01"
END_DATE = (date.today() - timedelta(days=1)).isoformat()

sector_dict = {
    "Banking_Financials": [
        "HDFCBANK.NS",
        "ICICIBANK.NS",
        "SBIN.NS",
        "KOTAKBANK.NS",
        "AXISBANK.NS",
        "BAJFINANCE.NS",
        "^NSEI"
    ],
    "IT": [
        "TCS.NS",
        "INFY.NS",
        "HCLTECH.NS",
        "WIPRO.NS",
        "TECHM.NS",
        "^NSEI"
    ],
    "Energy_Oil_Gas": [
        "RELIANCE.NS",
        "ONGC.NS",
        "NTPC.NS",
        "POWERGRID.NS",
        "BPCL.NS",
        "^NSEI"
    ],
    "FMCG": [
        "ITC.NS",
        "HINDUNILVR.NS",
        "NESTLEIND.NS",
        "BRITANNIA.NS",
        "TATACONSUM.NS",
        "^NSEI"
    ],
    "Auto": [
        "MARUTI.NS",
        "BAJAJ-AUTO.NS",
        "EICHERMOT.NS",
        "^NSEI"
    ],
    "Metals_Mining": [
        "TATASTEEL.NS",
        "JSWSTEEL.NS",
        "HINDALCO.NS",
        "COALINDIA.NS",
        "^NSEI"
    ],
    "Pharma_Healthcare": [
        "SUNPHARMA.NS",
        "CIPLA.NS",
        "DRREDDY.NS",
        "APOLLOHOSP.NS",
        "^NSEI"
    ]
}

horizon = 5