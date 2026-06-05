from workflow import run_training_pipeline
from config import sector_dict
import pandas as pd

def initialize_model(target_ticker,sector_name,horizon_val=5):
    return run_training_pipeline(target_ticker,sector_name,sector_dict,horizon_val,save_models=False)


def get_latest_features(feature_matrix):
    return feature_matrix.iloc[[-1]]

def get_predictions(lr_model,xgb_model,lr_weight,xgb_weight,latest_row):
    lr_prob = lr_model.predict_proba(latest_row)[0][1]
    xgb_prob = xgb_model.predict_proba(latest_row)[0][1]

    consensus = (lr_weight*lr_prob + xgb_weight*xgb_prob) 

    return {
        "lr_prob":lr_prob,
        "xgb_prob":xgb_prob,
        "consensus":consensus
    }


def get_recommendation(consensus_prob):
    if consensus_prob > 0.65:
        return "Strong Bullish"
    elif consensus_prob > 0.55 and consensus_prob <= 0.65:
        return "Moderately Bullish"
    elif consensus_prob > 0.45 and consensus_prob <=0.55:
        return "Neutral"
    else:
        return "Bearish"


def get_agreement(lr_prob,xgb_prob):
    agreement = abs(lr_prob - xgb_prob)
    
    if agreement <= 0.10:
        return "High Agreement"
    else:
        return "Low Agreement"


def get_signal(lr_prob,xgb_prob,consensus_prob):
    agreement = get_agreement(lr_prob,xgb_prob)
    recommendation = get_recommendation(consensus_prob)

    if recommendation == "Strong Bullish" and agreement == "High Agreement":
        return "Strong Buy"
    elif recommendation == "Strong Bullish" and agreement == "Low Agreement":
        return "Cautious Buy"
    elif recommendation == "Moderately Bullish" and agreement == "High Agreement":
        return "Moderate Buy"
    elif recommendation == "Moderately Bullish" and agreement == "Low Agreement":
        return "Cautious Buy"
    elif recommendation == "Neutral" and agreement == "High Agreement":
        return "Hold"
    elif recommendation == "Neutral" and agreement == "Low Agreement":
        return "Hold"
    elif recommendation == "Bearish" and agreement == "High Agreement":
        return "Strong Sell"
    elif recommendation == "Bearish" and agreement == "Low Agreement":
        return "Cautious Sell"
        

def get_feature_summary(latest_row,target_ticker):
    features = {
        'momentum': [x for x in latest_row.columns if f'{target_ticker}_mom_' in x],
        'volatility': [x for x in latest_row.columns if f'{target_ticker}_vol_' in x],
        'ma_distance': [x for x in latest_row.columns if f'{target_ticker}_ma_dist_' in x],
        'rolling_mean': [x for x in latest_row.columns if f'{target_ticker}_mean_ret_' in x]
    }

    result = {}
    for category,val in features.items():
        result[category] = {col : latest_row[col].iloc[0] for col in val}

    return result



def get_sector_strength(latest_row,sector_tickers):
    mom_5_sum = 0
    mom_20_sum = 0
    count = 0 
    weighted_sum = 0
    for i in sector_tickers:
        if '^NSEI' in i:
            continue
        mom_5_sum += latest_row[f'{i}_mom_5'].iloc[0] if f'{i}_mom_5' in latest_row.columns else 0
        mom_20_sum += latest_row[f'{i}_mom_20'].iloc[0] if f'{i}_mom_20' in latest_row.columns else 0
        weighted_sum = (0.4*mom_5_sum) + (0.6*mom_20_sum)
        count+=1
    
    weighted = weighted_sum/count

    if weighted > 0.05:
        strength = "Strong Sector Strength"
    elif weighted > 0.02 and weighted <=0.05:
        strength = "Moderate Sector Strength"
    elif weighted > -0.02 and weighted <= 0.02:
        strength = "Neutral Sector Strength"
    else:
        strength = "Weak Sector Strength"


    return {
        'strength':strength,
        'value': weighted
    }


def get_nifty_context(latest_row):
    features = {}
    for i in latest_row.columns:
        if '^NSEI' in  i:
            features[i] = latest_row[i].iloc[0]
        else:
            continue

    return features


def get_relative_performance(latest_row,sector_tickers):
    data = []

    for ticker in sector_tickers:
        if '^NSEI' in ticker:
            continue
        
        try:
            column_name = f'{ticker}_mom_20'
            value = latest_row[column_name].iloc[0]
            
            data.append((ticker,value))
        except(KeyError, IndexError, TypeError):
            continue

    df = pd.DataFrame(data,columns=['Stocks','20D Momentum'])

    df['20D Momentum'] = df['20D Momentum'] * 100

    df = df.sort_values(by="20D Momentum", ascending=False).reset_index(drop=True)

    return df


def get_health_score(latest_row,target_ticker):
    score = 0
    if latest_row[f'{target_ticker}_mom_20'].values[0] > 0:
        score+=20
    if latest_row[f'{target_ticker}_mom_5'].values[0] > 0:
        score+=15
    if latest_row[f'{target_ticker}_ma_dist_20'].values[0] > 0:
        score+=20
    if latest_row[f'{target_ticker}_ma_dist_50'].values[0] > 0:
        score+=20
    if latest_row[f'{target_ticker}_mean_ret_20'].values[0] > 0:
        score+=15
    if latest_row[f'{target_ticker}_vol_20'].values[0] < 20:
        score+=10



    if latest_row[f'{target_ticker}_mom_20'].values[0] < -0.10:   
        score -= 20
    if latest_row[f'{target_ticker}_mom_5'].values[0] < -0.05:    
        score -= 15
    if latest_row[f'{target_ticker}_ma_dist_20'].values[0] < -0.07:  
        score -= 20
    if latest_row[f'{target_ticker}_ma_dist_50'].values[0] < -0.15:  
        score -= 20
    if latest_row[f'{target_ticker}_mean_ret_20'].values[0] < 0:
        score -= 15
    if latest_row[f'{target_ticker}_vol_20'].values[0] > 0.40:    
        score -= 10

    score = max(0, min(100, score))
    
    if score >= 70:
        label = "Strong"
    elif score >= 40 and score < 70:
        label = "Moderate"
    else:
        label = "Weak"

    
    return {
        "score" : score,
        "label" : label
    }


def get_drawdown_metrics(data, target_ticker):
    Close_prices = data[f'{target_ticker}_Close']
    rolling_max = Close_prices.cummax()
    drawdown_series = (Close_prices - rolling_max) / rolling_max

    current_dd = drawdown_series.iloc[-1]
    max_dd = drawdown_series.min()

    return {
        'current_drawdown': current_dd,
        'max_drawdown': max_dd,
    }

def get_buy_risk_score(latest_row, target_ticker, drawdown_metrics):
    score = 0

    # Volatility component
    if latest_row[f'{target_ticker}_vol_5'].values[0] > 0.015:
        score += 25
    if latest_row[f'{target_ticker}_vol_20'].values[0] > 0.012:
        score += 25
    if drawdown_metrics['current_drawdown'] < -0.05:
        score += 25
    if drawdown_metrics['current_drawdown'] < -0.10:
        score += 25

    if latest_row[f'{target_ticker}_vol_5'].values[0] < 0.008:
        score -= 15
    if latest_row[f'{target_ticker}_vol_20'].values[0] < 0.007:
        score -= 15
    if drawdown_metrics['current_drawdown'] > -0.01:
        score -= 20
    if drawdown_metrics['max_drawdown'] > -0.08:
        score -= 10

    # Volume Shock component
    vol_shock_col = f'{target_ticker}_rel_volume_20'
    if vol_shock_col in latest_row.columns:
        vol_shock = latest_row[vol_shock_col].values[0]
        if vol_shock >= 3.0:
            score += 30   # heavy institutional selling — very dangerous
        elif vol_shock >= 2.0:
            score += 20
        elif vol_shock >= 1.5:
            score += 10
        else:
            score += 0
    else:
        vol_shock = None

    score = max(0, min(100, score))

    if score >= 70:
        label = "High Risk"
    elif score >= 40 and score < 70:
        label = "Moderate Risk"
    else:
        label = "Low Risk"

    return {
        "score": score,
        "label": label,
        "vol_shock": vol_shock       
    }


def get_sector_ranking(sector_tickers,sector_name,sector_dict):
    data = []
    consensus = []
    for ticker in sector_tickers:
        if '^NSEI' in ticker:
            continue
        
        result = initialize_model(ticker,sector_name)

        latest_row = get_latest_features(result['main_feature_matrix'])

        pred = get_predictions(result['lr_model'],result['xgb_model'],result['lr_weight'],result['xgb_weight'],latest_row)

        data.append(ticker)
        consensus.append(pred['consensus'])

    df = pd.DataFrame({'Stocks': data, 'Bullish Probability': consensus})

    df['Bullish Probability'] = df['Bullish Probability'] * 100

    df = df.sort_values(by="Bullish Probability", ascending=False).reset_index(drop=True)

    return df


def get_risk_reward(risk_score, consensus_prob):
    estimated_reward = consensus_prob * 100        
    estimated_risk = risk_score                     

    if estimated_risk == 0:
        ratio = float('inf')
    else:
        ratio = round(estimated_reward / estimated_risk, 2)

    if ratio >= 1.5:
        label = "Favorable"
    elif ratio >= 0.8 and ratio < 1.5:
        label = "Neutral"
    else:
        label = "Unfavorable"

    return {
        "estimated_reward": round(estimated_reward, 2),
        "estimated_risk": round(estimated_risk, 2),
        "ratio": ratio,
        "label": label
    }



def get_sector_price_comparison(data, sector_tickers):
    rebased = {}

    for ticker in sector_tickers:
        col = f'{ticker}_Close'
        if col not in data.columns:
            continue

        prices = data[col].dropna()
        if prices.empty:
            continue

        rebased[ticker] = (prices / prices.iloc[0]) * 100

    df = pd.DataFrame(rebased)
    
    return df