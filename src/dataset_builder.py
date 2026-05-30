from labels import *
from data_collection import *
from feature_engineering import *
from config import horizon
def get_sector_tickers(sector_name,sector_dict):
    return sector_dict[sector_name]



def prepare_ticker_universe(target_ticker,sector_name,sector_dict):
    
    tickers = sector_dict[sector_name].copy()
    
    if target_ticker not in tickers:
        tickers.insert(0, target_ticker)
    
    tickers = list(dict.fromkeys(tickers))
    
    return tickers


def build_feature_matrix(data, tickers):

    returns = create_returns(data)
    momentum = create_momentum_features(data)
    volatility = create_volatility_features(returns)
    ma_distance = create_ma_distance_features(data, tickers)
    rolling_mean = create_rolling_mean_return_features(returns)
    rel_volume = create_relative_volume_features(data, tickers)
    all_features = pd.concat(
        [returns, momentum, volatility, ma_distance, rolling_mean, rel_volume],
        axis=1
    )
 

    raw_cols = [
        col for col in all_features.columns
        if col.endswith("_Close") or col.endswith("_Volume")
    ]
    bad_cols = [
        col for col in all_features.columns
        if "^NSEI_Volume" in col or "^NSEI_rel_volume" in col
    ]
 
    all_features = all_features.drop(columns=raw_cols + bad_cols, errors="ignore")
    all_features = all_features.dropna()
 
    return all_features

def prepare_dataset(target_ticker, sector_name, sector_dict,horizons=horizon, quantile=0.7):

    tickers        = prepare_ticker_universe(target_ticker, sector_name, sector_dict)
    data           = download_data(tickers)
    features       = build_feature_matrix(data, tickers)
    future_returns = create_future_returns(data, target_ticker, horizon=horizons)

    aligned_returns = future_returns.reindex(features.index)

    # compute train_size here and pass it in
    train_size      = int(len(aligned_returns) * 0.8)

    label = create_labels(aligned_returns, train_size=train_size, quantile=quantile)

    dataset           = features.copy()
    dataset['Target'] = label
    dataset           = dataset.dropna()

    return dataset