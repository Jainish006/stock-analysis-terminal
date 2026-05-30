from main import get_risk_reward
from main import get_predictions
from main import get_buy_risk_score
from main import get_drawdown_metrics
from main import get_latest_features
from main import *
import streamlit as st
import pandas as pd
from config import sector_dict,horizon

st.set_page_config(page_title="Stock Analysis Terminal", layout='wide', page_icon="📈")

# Session state initialization
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = None
if 'sector' not in st.session_state:
    st.session_state['sector'] = None

# Sidebar navigation
pages = [
    "🏠 Home",
    "📈 Prediction",
    "🔍 Feature Analysis",
    "🌐 Market Context",
    "⚠️ Entry Risk (with Risk/Reward prediction)",
    "📉 Charts"
]

if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

st.sidebar.title("📊 Analysis Menu")
st.sidebar.markdown("---")

for p in pages:
    if st.sidebar.button(p, use_container_width=True):
        st.session_state.page = p

st.sidebar.markdown("---")

current_page = st.session_state.page


# ── HOME ──────────────────────────────────────────────────────────────────────
if current_page == "🏠 Home":
    st.title('Stock Analysis Terminal')
    st.markdown('An ML-powered stock analysis terminal for NSE equities. Enter a stock ticker and sector to get model predictions, technical health, risk analysis, and sector context.')

    col1, col2 = st.columns(2)
    with col1:
        target_ticker = st.text_input('Enter NSE stock ticker', placeholder='e.g. HDFCBANK.NS')
    with col2:
        sector_name = st.selectbox('Select the sector of the stock', list(sector_dict.keys()))

    button_1 = st.button('Run Analysis')
    if button_1:
        if not target_ticker or not sector_name:
            st.error("Please enter both a ticker and select a sector")
        else:
            with st.spinner('Training models... This may take a minute'):
                result = initialize_model(target_ticker, sector_name)
                st.session_state['result'] = result
                st.session_state['ticker'] = target_ticker
                st.session_state['sector'] = sector_name

    if st.session_state['result'] is not None:
        raw_data = st.session_state['result']['raw_data']
        ticker = st.session_state['ticker']
        x = raw_data[f'{ticker}_Close'].copy()
        x.index = pd.to_datetime(x.index).tz_localize(None)
        df_plot = pd.DataFrame({'Price': x})
        st.subheader(f'{ticker} - Price History')
        st.line_chart(df_plot)


# ── PREDICTION ────────────────────────────────────────────────────────────────
elif current_page == "📈 Prediction":
    if st.session_state['result'] is None:
        st.warning("Please go to Home and run the analysis first")
        st.stop()

    result = st.session_state['result']
    ticker = st.session_state['ticker']
    sector = st.session_state['sector']
    
    st.title("📈 Prediction Dashboard")
    st.markdown(f"**{ticker}** | {sector} Sector")
    st.info(f"ℹ️ All probabilities and signals on this page reflect the likelihood of the stock being Bullish over the next {horizon} days. A higher probability means the model sees stronger buying conditions ahead.")
    st.markdown("---")
    latest_row = get_latest_features(result['main_feature_matrix'])
    preds = get_predictions(result['lr_model'], result['xgb_model'], result['lr_weight'], result['xgb_weight'], latest_row)
    lr_prob = preds['lr_prob']
    xgb_prob = preds['xgb_prob']
    consensus = preds['consensus']
    recommendation = get_recommendation(consensus)
    agreement = get_agreement(lr_prob, xgb_prob)
    signal = get_signal(lr_prob, xgb_prob, consensus)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
    <div style="background-color:#1e1e2e; padding:20px; border-radius:10px; border-left:4px solid #4CAF50;">
        <p style="color:gray; margin:0; font-size:13px;">LR Probability</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{lr_prob:.2%}</p>
    </div>
""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
    <div style="background-color:#1e1e2e; padding:20px; border-radius:10px; border-left:4px solid #4CAF50;">
        <p style="color:gray; margin:0; font-size:13px;">XGB Probability</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{xgb_prob:.2%}</p>
    </div>
""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
    <div style="background-color:#1e1e2e; padding:20px; border-radius:10px; border-left:4px solid #4CAF50;">
        <p style="color:gray; margin:0; font-size:13px;">Consensus Probability</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{consensus:.2%}</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(f"Model weights — LR: {result['lr_weight']:.2f} | XGB: {result['xgb_weight']:.2f}")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
    <div style="background-color:#1e1e2e; padding:20px; border-radius:10px; border-left:4px solid #4CAF50;">
        <p style="color:gray; margin:0; font-size:13px;">Recommendation</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
    <div style="background-color:#1e1e2e; padding:20px; border-radius:10px; border-left:4px solid #4CAF50;">
        <p style="color:gray; margin:0; font-size:13px;">Prediction Horizon</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{horizon} Days</p>
    </div>
    """, unsafe_allow_html=True)
        st.caption("the number of days ahead model is predicting for")
    with col3:
        st.markdown(f"""
    <div style="background-color:#1e1e2e; padding:20px; border-radius:10px; border-left:4px solid #4CAF50;">
        <p style="color:gray; margin:0; font-size:13px;">Model Agreement</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{agreement}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Final Signal")

    signal_descriptions = {
        "Strong Buy": "Both models show strong bullish conviction with high agreement.",
        "Cautious Buy": "Bullish signal but models disagree — proceed with caution.",
        "Moderate Buy": "Moderate bullish conviction with good model agreement.",
        "Hold": "No strong directional conviction. Wait for a clearer signal.",
        "Cautious Sell": "Bearish signal but models disagree — monitor closely.",
        "Strong Sell": "Both models show strong bearish conviction with high agreement."
    }

    if "Buy" in signal:
        st.success(f"🟢 {signal}")
    elif signal == "Hold":
        st.warning(f"🟡 {signal}")
    else:
        st.error(f"🔴 {signal}")

    st.info(signal_descriptions.get(signal, ""))

# ── FEATURE ANALYSIS ──────────────────────────────────────────────────────────
elif current_page == "🔍 Feature Analysis":
    if st.session_state['result'] is None:
        st.warning("Please go to Home and run the analysis first")
        st.stop()

    result = st.session_state['result']
    ticker = st.session_state['ticker']
    sector = st.session_state['sector']
    
    st.title("🔍 Feature Analysis")
    st.markdown(f"**{ticker}** | {sector} Sector")
    st.info("Latest engineered feature values for the stock. Green indicates a positive signal, red indicates a negative signal.")
    st.markdown("---")
    
    latest_row = get_latest_features(result['main_feature_matrix'])
    feature_summary = get_feature_summary(latest_row, ticker)

    def kpi_box(label, value):
        if value >= 0:
            color = "#4CAF50"
        else:
            color = "#F44336"
        return f"""
        <div style="background-color:#1e1e2e; padding:20px; border-radius:10px; border-left:4px solid {color}; margin-bottom:15px;">
        <p style="color:gray; margin:0; font-size:13px;">{label}</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value:.2%}</p>
        </div>
        """

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Momentum")
        st.caption("The tendency of assets that have performed well recently to continue performing well.")
        st.markdown(kpi_box("3 Day Momentum", feature_summary['momentum'][f'{ticker}_mom_3']), unsafe_allow_html=True)
        st.markdown(kpi_box("5 Day Momentum", feature_summary['momentum'][f'{ticker}_mom_5']), unsafe_allow_html=True)
        st.markdown(kpi_box("10 Day Momentum", feature_summary['momentum'][f'{ticker}_mom_10']), unsafe_allow_html=True)
        st.markdown(kpi_box("20 Day Momentum", feature_summary['momentum'][f'{ticker}_mom_20']), unsafe_allow_html=True)

    with col2:
        st.subheader("⚡ Volatility")
        st.caption("The degree of variation of a trading price series over time.")
        st.markdown(kpi_box("5 Day Volatility", feature_summary['volatility'][f'{ticker}_vol_5']), unsafe_allow_html=True)
        st.markdown(kpi_box("20 Day Volatility", feature_summary['volatility'][f'{ticker}_vol_20']), unsafe_allow_html=True)

    st.markdown("---")

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("📊 Moving Average Distance")
        st.caption("The difference between the current price and the moving average of the price.")
        st.markdown(kpi_box("10 Day MA Distance", feature_summary['ma_distance'][f'{ticker}_ma_dist_10']), unsafe_allow_html=True)
        st.markdown(kpi_box("20 Day MA Distance", feature_summary['ma_distance'][f'{ticker}_ma_dist_20']), unsafe_allow_html=True)
        st.markdown(kpi_box("50 Day MA Distance", feature_summary['ma_distance'][f'{ticker}_ma_dist_50']), unsafe_allow_html=True)

    with col4:
        st.subheader("📉 Rolling Mean Return")
        st.caption("The average return of the price over a specific period.")
        st.markdown(kpi_box("5 Day Mean Return", feature_summary['rolling_mean'][f'{ticker}_mean_ret_5']), unsafe_allow_html=True)
        st.markdown(kpi_box("20 Day Mean Return", feature_summary['rolling_mean'][f'{ticker}_mean_ret_20']), unsafe_allow_html=True)



# ── MARKET CONTEXT ────────────────────────────────────────────────────────────
elif current_page == "🌐 Market Context":
    if st.session_state['result'] is None:
        st.warning("Please go to Home and run the analysis first")
        st.stop()

    result = st.session_state['result']
    ticker = st.session_state['ticker']
    sector = st.session_state['sector']

    st.title("🌐 Market Context")
    st.markdown(f"**{ticker}** | {sector} Sector")
    st.info("Sector strength, NIFTY market context, and relative performance of stocks in the same sector.")
    st.markdown("---")

    latest_row = get_latest_features(result['main_feature_matrix'])
    sector_tickers = sector_dict[sector]
    sector_strength = get_sector_strength(latest_row, sector_tickers)
    nifty_context = get_nifty_context(latest_row)
    relative_performance = get_relative_performance(latest_row, sector_tickers)

    label_map = {
        "mom_3": "3 Day Momentum",
        "mom_5": "5 Day Momentum",
        "mom_10": "10 Day Momentum",
        "mom_20": "20 Day Momentum",
        "vol_5": "5 Day Volatility",
        "vol_20": "20 Day Volatility",
        "ma_dist_10": "MA10 Distance",
        "ma_dist_20": "MA20 Distance",
        "ma_dist_50": "MA50 Distance",
        "mean_ret_5": "5 Day Mean Return",
        "mean_ret_20": "20 Day Mean Return"
    }

    def kpi_box_str(label, value, color):
        return f"""
        <div style="background-color:#1e1e2e; padding:20px; border-radius:10px; border-left:4px solid {color}; margin-bottom:15px;">
            <p style="color:gray; margin:0; font-size:13px;">{label}</p>
            <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value}</p>
        </div>
        """

    def kpi_box_float(label, value, is_volatility=False):
        if is_volatility:
            color = "#2196F3"
        elif value >= 0:
            color = "#4CAF50"
        else:
            color = "#F44336"
        return f"""
        <div style="background-color:#1e1e2e; padding:12px 16px; border-radius:10px; border-left:4px solid {color}; margin-bottom:15px;">
            <p style="color:gray; margin:0; font-size:13px;">{label}</p>
            <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value:.2%}</p>
        </div>
        """

    # Sector Strength
    st.subheader("🏭 Sector Strength")
    col1, col2 = st.columns(2)

    strength_color_map = {
        "Strong Sector Strength": "#4CAF50",
        "Moderate Sector Strength": "#8BC34A",
        "Neutral Sector Strength": "#FF9800",
        "Weak Sector Strength": "#F44336"
    }
    strength_color = strength_color_map.get(sector_strength['strength'], "gray")

    with col1:
        st.markdown(kpi_box_str("Sector Strength", sector_strength['strength'], strength_color), unsafe_allow_html=True)

    with col2:
        st.markdown(kpi_box_float("Sector Momentum Value", sector_strength['value']), unsafe_allow_html=True)

    st.markdown("---")


    st.subheader("📊 NIFTY Context")

    momentum_keys = {k: v for k, v in nifty_context.items() if '_mom_' in k}
    volatility_keys = {k: v for k, v in nifty_context.items() if '_vol_' in k}
    ma_keys = {k: v for k, v in nifty_context.items() if '_ma_dist_' in k}
    mean_ret_keys = {k: v for k, v in nifty_context.items() if '_mean_ret_' in k}

    st.caption("Momentum")
    cols = st.columns(len(momentum_keys))
    for i, (key, value) in enumerate(momentum_keys.items()):
        label = label_map.get(key.replace("^NSEI_", ""), key)
        with cols[i]:
            st.markdown(kpi_box_float(label, value), unsafe_allow_html=True)

    st.caption("Volatility")
    cols = st.columns(4)
    for i, (key, value) in enumerate(volatility_keys.items()):
        label = label_map.get(key.replace("^NSEI_", ""), key)
        with cols[i]:
            st.markdown(kpi_box_float(label, value, is_volatility=True), unsafe_allow_html=True) 

    st.caption("MA Distance")
    col_ma = st.columns(len(ma_keys))
    for i, (key, value) in enumerate(ma_keys.items()):
        label = label_map.get(key.replace("^NSEI_", ""), key)
        with col_ma[i]:
            st.markdown(kpi_box_float(label, value), unsafe_allow_html=True)    

    st.caption("Mean Return")
    cols = st.columns(4)
    for i, (key, value) in enumerate(mean_ret_keys.items()):
        label = label_map.get(key.replace("^NSEI_", ""), key)
        with cols[i]:
            st.markdown(kpi_box_float(label, value), unsafe_allow_html=True)



    st.markdown("---")

    
    st.subheader("📈 Relative Performance")
    st.caption("20 Day Momentum comparison across all stocks in the sector.")
    st.dataframe(
        relative_performance.style.format({"20D Momentum": "{:.2f}%"}),
        use_container_width=True
    )



# ── ENTRY RISK ────────────────────────────────────────────────────────────────
elif current_page == "⚠️ Entry Risk (with Risk/Reward prediction)":
    if st.session_state['result'] is None:
        st.warning("Please go to Home and run the analysis first")
        st.stop()

    result = st.session_state['result']
    ticker = st.session_state['ticker']
    sector = st.session_state['sector']
    st.title("⚠️ Entry Risk")
    st.markdown(f"**{ticker}** | {sector} Sector")
    # st.info("Entry Risk measures how risky it is to buy this stock right now. It is based on two factors — volatility (how much the price is fluctuating) and drawdown (how much the stock has already fallen from its peak). A high score means the stock is volatile and/or in a significant drawdown, making it a riskier time to enter. A low score means the stock is relatively stable and near its highs, making it a safer entry point. Score ranges from 0 to 100.")
    st.markdown("---")

    latest_row = get_latest_features(result['main_feature_matrix'])
    raw_data = result['raw_data']
    drawdown = get_drawdown_metrics(raw_data,ticker)
    buy_risk = get_buy_risk_score(latest_row,ticker,drawdown)
    preds = get_predictions(result['lr_model'],result['xgb_model'],result['lr_weight'],result['xgb_weight'],latest_row)
    consensus = preds['consensus']
    risk_reward = get_risk_reward(buy_risk['score'],consensus)

    if buy_risk['label'] == "Strong":
        color = "#F44336"
    elif buy_risk['label'] == "Moderate":
        color = "#FF9800"
    else:
        color = "#4CAF50"

    st.markdown(f"""
    <div style="background-color:#1e1e2e; padding:40px; border-radius:15px; 
    border: 3px solid {color}; text-align:center; margin-bottom:20px;">
    <p style="color:gray; margin:0; font-size:16px;">Technical Health Score</p>
    <p style="color:{color}; margin:0; font-size:72px; font-weight:bold;">{buy_risk['score']}/100</p>
    <p style="color:{color}; margin:0; font-size:24px;">{buy_risk['label']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("📊 Volatility Metrics")
    def kpi_box_float(label, value, is_volatility=False):
        if is_volatility:
            color = "#2196F3"
        elif value >= 0:
            color = "#4CAF50"
        else:
            color = "#F44336"
        return f"""
        <div style="background-color:#1e1e2e; padding:12px 16px; border-radius:10px; border-left:4px solid {color}; margin-bottom:15px;">
            <p style="color:gray; margin:0; font-size:13px;">{label}</p>
            <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value:.2%}</p>
        </div>
        """

    def kpi_box_drawdown(label, value):
       
        if value >= 0.05:
            color = "#4CAF50"
        else:
            color = "#F44336"
        return f"""
            <div style="background-color:#1e1e2e; padding:12px 16px; border-radius:10px; border-left:4px solid {color}; margin-bottom:15px;">
            <p style="color:gray; margin:0; font-size:13px;">{label}</p>
            <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value:.2%}</p>
            </div>
    
        """
    
    col1,col2 = st.columns(2)

    with col1:
        
        st.markdown(kpi_box_float("5D Volatility",buy_risk[f'{ticker}_vol_5'],is_volatility=True),unsafe_allow_html=True)

    with col2:
        st.markdown(kpi_box_float("20D Volatility",buy_risk[f'{ticker}_vol_20'],is_volatility=True),unsafe_allow_html=True)
        # st.markdown(kpi_box_float("Current Drawdown",buy_risk['current_dd'],is_volatility=True),unsafe_allow_html=True)
        # st.markdown(kpi_box_float("Max Drawdown",buy_risk['max_dd'],is_volatility=True),unsafe_allow_html=True)    

    col1,col2 = st.columns(2)

    with col1:
        
        st.markdown(kpi_box_drawdown("Current Drawdown",buy_risk['current_dd']),unsafe_allow_html=True)

    with col2:
        st.markdown(kpi_box_drawdown("Max Drawdown",buy_risk['max_dd']),unsafe_allow_html=True)

    




# ── CHARTS ────────────────────────────────────────────────────────────────────
elif current_page == "📉 Charts":
    if st.session_state['result'] is None:
        st.warning("Please go to Home and run the analysis first")
        st.stop()

    result = st.session_state['result']
    ticker = st.session_state['ticker']
    sector = st.session_state['sector']
    st.title("📉 Charts")
    st.write("Coming soon")


