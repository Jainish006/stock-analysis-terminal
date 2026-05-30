from data_collection import *
from feature_engineering import *

def create_future_returns(data,target_ticker,horizon=5):
    future_returns = (
        data[f"{target_ticker}_Close"].pct_change(horizon).shift(-horizon)
    )

    return future_returns



def create_labels(future_returns, train_size, quantile=0.70):
    train_returns = future_returns.iloc[:train_size]   # training rows only
    threshold     = train_returns.quantile(quantile)   # threshold from train only
    labels        = (future_returns > threshold).astype(int)
    return labels