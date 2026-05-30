import pandas as pd
import yfinance as yf
from config import START_DATE,END_DATE

def download_data(tickers):
    
    data = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False
    )
    
    data.columns = [
        f"{ticker}_{feature}"
        for feature, ticker in data.columns
    ]
    
    data = data.drop(
        columns=[
            col for col in data.columns
            if "Adj Close" in col
        ],
        errors="ignore"
    )
    
    data = data.dropna(how="all")
    
    return data


def select_core_columns(data):
    
    selected_cols = [
        col for col in data.columns
        if (
            "Close" in col
            or "Volume" in col
        )
    ]
    
    filtered_data = data[selected_cols]
    
    return filtered_data


    