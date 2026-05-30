from data_collection import *



def create_returns(data):
    
    core_data = select_core_columns(data)
    core_data = core_data[[col for col in core_data.columns if 'Close' in col]]
    returns = core_data.pct_change()
    returns.columns = [col.replace('_Close', '') for col in returns.columns]
    
    return returns


def create_momentum_features(data,windows=[3, 5, 10, 20]):
    
    core_data = select_core_columns(data)

    core_data = core_data[[col for col in core_data.columns if 'Close' in col]]
    
    momentum_features = []
    
    for window in windows:
        
        momentum = (core_data.pct_change(window))
        
        momentum.columns = [
            f"{col.replace('_Close', '')}_mom_{window}"
            for col in momentum.columns
        ]
        
        momentum_features.append(momentum)


    momentum_df = pd.concat(momentum_features,axis=1)
    
    return momentum_df



def create_volatility_features(returns,windows=[5, 20]):
    
    volatility_features = []

    for window in windows:
        
        volatility = returns.rolling(window).std()
        
        volatility.columns = [
            f"{col.replace('_Close', '')}_vol_{window}"
            for col in volatility.columns
        ]
        
        volatility_features.append(volatility)
    
    volatility_df = pd.concat(
        volatility_features,
        axis=1
    )
    
    return volatility_df


def create_intraday_returns(data,tickers):
    
    intraday_returns = pd.DataFrame()
    
    for ticker in tickers:
        
        intraday_returns[
            f"{ticker}_intraday_return"
        ] = (
            (
                data[f"{ticker}_Close"]
                - data[f"{ticker}_Open"]
            )
            / data[f"{ticker}_Open"]
        )
    
    return intraday_returns


def create_hl_range(data,tickers):
    
    hl_range = pd.DataFrame()
    
    for ticker in tickers:
        
        hl_range[
            f"{ticker}_hl_range"
        ] = (
            (
                data[f"{ticker}_High"]
                - data[f"{ticker}_Low"]
            )
            / data[f"{ticker}_Close"]
        )
    
    return hl_range



def create_ma_distance_features(data,tickers,windows=[10, 20, 50]):
    
    ma_features = []
    
    for window in windows:
        
        ma_distance = pd.DataFrame()
        
        for ticker in tickers:
            
            ma = (
                data[f"{ticker}_Close"]
                .rolling(window)
                .mean()
            )
            
            ma_distance[
                f"{ticker}_ma_dist_{window}"
            ] = (
                (
                    data[f"{ticker}_Close"]
                    - ma
                )
                / ma
            )
        
        ma_features.append(ma_distance)
    
    ma_df = pd.concat(
        ma_features,
        axis=1
    )
    
    return ma_df


def create_relative_volume_features(data,tickers,window=20):
    
    relative_volume = pd.DataFrame()
    
    for ticker in tickers:
        
        if ticker == "^NSEI":
            continue
        
        volume_ma = (
            data[f"{ticker}_Volume"]
            .rolling(window)
            .mean()
        )
        
        relative_volume[
            f"{ticker}_rel_volume_{window}"
        ] = (
            data[f"{ticker}_Volume"]
            / volume_ma
        )
    
    return relative_volume


def create_rolling_mean_return_features(returns,windows=[5, 20]):
    
    rolling_mean_features = []
    
    for window in windows:
        
        rolling_mean = (
            returns
            .rolling(window)
            .mean()
        )
        
        rolling_mean.columns = [
        f"{col.replace('_Close', '')}_mean_ret_{window}"
        for col in rolling_mean.columns
        ]
        
        rolling_mean_features.append(
            rolling_mean
        )
    
    rolling_mean_df = pd.concat(
        rolling_mean_features,
        axis=1
    )
    
    return rolling_mean_df