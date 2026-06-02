from main import get_risk_reward
from main import get_predictions
from main import get_buy_risk_score
from main import get_drawdown_metrics
from main import get_latest_features
from main import *
import streamlit as st
import pandas as pd
from config import sector_dict, horizon
import plotly.express as px
import plotly.graph_objects as go
st.set_page_config(page_title="Stock Analysis Terminal", layout='wide', page_icon="📈")

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if 'result' not in st.session_state:
    st.session_state['result'] = None
if 'ticker' not in st.session_state:
    st.session_state['ticker'] = None
if 'sector' not in st.session_state:
    st.session_state['sector'] = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
pages = [
    "🏠 Home",
    "📖 Page Guide",
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

# ── SHARED HELPERS ────────────────────────────────────────────────────────────
def kpi_box_float(label, value, is_volatility=False):
    if is_volatility:
        color = "#2196F3"
    elif value >= 0:
        color = "#4CAF50"
    else:
        color = "#F44336"
    return f"""
    <div style="background-color:#1e1e2e; padding:12px 16px; border-radius:10px;
    border-left:4px solid {color}; margin-bottom:15px;">
        <p style="color:gray; margin:0; font-size:13px;">{label}</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value:.2%}</p>
    </div>
    """

def kpi_box_str(label, value, color):
    return f"""
    <div style="background-color:#1e1e2e; padding:20px; border-radius:10px;
    border-left:4px solid {color}; margin-bottom:15px;">
        <p style="color:gray; margin:0; font-size:13px;">{label}</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value}</p>
    </div>
    """

def kpi_box_drawdown(label, value):
    # Drawdown values are negative; closer to 0 is better
    color = "#4CAF50" if value >= -0.05 else "#F44336"
    return f"""
    <div style="background-color:#1e1e2e; padding:12px 16px; border-radius:10px;
    border-left:4px solid {color}; margin-bottom:15px;">
        <p style="color:gray; margin:0; font-size:13px;">{label}</p>
        <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value:.2%}</p>
    </div>
    """


# ── PAGE GUIDE ────────────────────────────────────────────────────────────────
if current_page == "📖 Page Guide":
    st.title("📖 Page Guide")
    st.markdown("A quick reference for what each page shows and when to use it.")
    st.markdown("---")
 
    pages_info = [
        {
            "icon": "🏠",
            "title": "Home",
            "nav": "🏠 Home",
            "what": "The starting point. Enter the NSE ticker and select the sector, then click Run Analysis to train the models.",
            "when": "Always start here. No other page works until you run the analysis.",
            "shows": ["Stock ticker input", "Sector selector", "Price history chart after analysis is run"],
        },
        {
            "icon": "📈",
            "title": "Prediction",
            "nav": "📈 Prediction",
            "what": f"Shows the ML model output — Logistic Regression and XGBoost probabilities of the stock being bullish over the next {horizon} days, combined into a consensus score.",
            "when": "Use this when you want a direct buy/sell/hold signal backed by the model.",
            "shows": ["LR Probability", "XGB Probability", "Consensus Probability", "Model Agreement", "Final Signal (Strong Buy → Strong Sell)"],
        },
        {
            "icon": "🔍",
            "title": "Feature Analysis",
            "nav": "🔍 Feature Analysis",
            "what": "Displays the latest values of every engineered feature the model uses — momentum, volatility, MA distance, and rolling mean returns.",
            "when": "Use this to understand why the model gave the signal it did. Green = positive, Red = negative.",
            "shows": ["3/5/10/20 Day Momentum", "5/20 Day Volatility", "MA Distance (10/20/50)", "Rolling Mean Return (5/20 Day)"],
        },
        {
            "icon": "🌐",
            "title": "Market Context",
            "nav": "🌐 Market Context",
            "what": "Zooms out from the individual stock to show how the overall sector and NIFTY are behaving.",
            "when": "Use this to check whether the stock's signal is supported by broader market tailwinds or is swimming against the tide.",
            "shows": ["Sector Strength label + momentum value", "NIFTY momentum / volatility / MA distance / mean return", "Relative 20D momentum of all sector peers"],
        },
        {
            "icon": "⚠️",
            "title": "Entry Risk & Risk/Reward",
            "nav": "⚠️ Entry Risk (with Risk/Reward prediction)",
            "what": "Measures how risky it is to enter the stock right now, based on volatility, drawdown from peak, and volume shock. Then combines that with the consensus probability to give a Risk/Reward rating.",
            "when": "Use this after you see a Buy signal — it tells you whether the timing is safe or whether you should wait for a better entry.",
            "shows": ["Entry Risk Score (0–100)", "Current & Max Drawdown", "Volume Shock", "Risk/Reward Rating + Score"],
        },
        {
            "icon": "📉",
            "title": "Charts",
            "nav": "📉 Charts",
            "what": "Visual charts for the stock and its sector over the last 2 years.",
            "when": "Use this to visually validate the signals — see price trends, when momentum turned, how deep the drawdowns were.",
            "shows": ["Normalised sector price comparison", "Momentum trend over time", "Volatility trend", "Drawdown from peak", "Sector 20D momentum bar chart"],
        },
    ]
 
    for info in pages_info:
        with st.container():
            col_icon, col_body = st.columns([0.04, 0.96])
            with col_icon:
                st.markdown(f"<p style='font-size:36px; margin:0;'>{info['icon']}</p>", unsafe_allow_html=True)
            with col_body:
                st.markdown(f"### {info['title']}")
                st.markdown(f"**What it shows:** {info['what']}")
                st.markdown(f"**When to use it:** {info['when']}")
                st.markdown("**Includes:** " + " &nbsp;·&nbsp; ".join([f"`{s}`" for s in info['shows']]))
                if st.button(f"Go to {info['title']} →", key=f"guide_{info['nav']}"):
                    st.session_state.page = info['nav']
                    st.rerun()
        st.markdown("---")
 
 

# ── HOME ──────────────────────────────────────────────────────────────────────
elif current_page == "🏠 Home":
    st.title('Stock Analysis Terminal')
    st.markdown(
        'An ML-powered stock analysis terminal for NSE equities. '
        'Enter a stock ticker and sector to get model predictions, '
        'technical health, risk analysis, and sector context.'
    )

    col1, col2 = st.columns(2)
    with col1:
        target_ticker = st.text_input('Enter NSE stock ticker', placeholder='e.g. HDFCBANK.NS')
    with col2:
        sector_name = st.selectbox('Select the sector of the stock', list(sector_dict.keys()))

    if st.button('Run Analysis'):
        if not target_ticker or not sector_name:
            st.error("Please enter both a ticker and select a sector")
        else:
            with st.spinner('Training models… This may take a minute'):
                result = initialize_model(target_ticker, sector_name)
                st.session_state['result'] = result
                st.session_state['ticker'] = target_ticker
                st.session_state['sector'] = sector_name

    if st.session_state['result'] is not None:
        raw_data = st.session_state['result']['raw_data']
        ticker = st.session_state['ticker']
        x = raw_data[f'{ticker}_Close'].copy()
        x.index = pd.to_datetime(x.index).tz_localize(None)

        price_df = pd.DataFrame({
            'Date': x.index,
            'Price': x.values
        })
        fig = px.line(price_df, x='Date', y='Price', title=f'{ticker} — Price History')
        st.plotly_chart(fig,use_container_width=True)
        


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
    st.info(
        f"ℹ️ All probabilities and signals on this page reflect the likelihood of the stock being "
        f"Bullish over the next **{horizon} days**. A higher probability means the model sees "
        f"stronger buying conditions ahead."
    )
    st.markdown("---")

    latest_row = get_latest_features(result['main_feature_matrix'])
    preds = get_predictions(result['lr_model'], result['xgb_model'],result['lr_weight'], result['xgb_weight'], latest_row)
    lr_prob    = preds['lr_prob']
    xgb_prob   = preds['xgb_prob']
    consensus  = preds['consensus']
    recommendation = get_recommendation(consensus)
    agreement  = get_agreement(lr_prob, xgb_prob)
    signal     = get_signal(lr_prob, xgb_prob, consensus)

    col1, col2, col3 = st.columns(3)
    for col, label, val in zip(
        [col1, col2, col3],
        ["LR Probability", "XGB Probability", "Consensus Probability"],
        [lr_prob, xgb_prob, consensus]
    ):
        with col:
            st.markdown(f"""
            <div style="background-color:#1e1e2e; padding:20px; border-radius:10px;
            border-left:4px solid #4CAF50;">
                <p style="color:gray; margin:0; font-size:13px;">{label}</p>
                <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{val:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

    st.caption(f"Model weights — LR: {result['lr_weight']:.2f} | XGB: {result['xgb_weight']:.2f}")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(kpi_box_str("Recommendation", recommendation, "#4CAF50"), unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style="background-color:#1e1e2e; padding:20px; border-radius:10px;
        border-left:4px solid #4CAF50;">
            <p style="color:gray; margin:0; font-size:13px;">Prediction Horizon</p>
            <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{horizon} Days</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("The number of days ahead the model is predicting for")
    with col3:
        st.markdown(kpi_box_str("Model Agreement", agreement, "#4CAF50"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Final Signal")

    signal_descriptions = {
        "Strong Buy":    "Both models show strong bullish conviction with high agreement.",
        "Cautious Buy":  "Bullish signal but models disagree — proceed with caution.",
        "Moderate Buy":  "Moderate bullish conviction with good model agreement.",
        "Hold":          "No strong directional conviction. Wait for a clearer signal.",
        "Cautious Sell": "Bearish signal but models disagree — monitor closely.",
        "Strong Sell":   "Both models show strong bearish conviction with high agreement."
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

    latest_row    = get_latest_features(result['main_feature_matrix'])
    feature_summary = get_feature_summary(latest_row, ticker)

    def kpi_box(label, value):
        color = "#4CAF50" if value >= 0 else "#F44336"
        return f"""
        <div style="background-color:#1e1e2e; padding:12px 16px; border-radius:10px;
        border-left:4px solid {color}; margin-bottom:15px;">
            <p style="color:gray; margin:0; font-size:13px;">{label}</p>
            <p style="color:white; margin:0; font-size:28px; font-weight:bold;">{value:.2%}</p>
        </div>
        """

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Momentum")
        st.caption("The tendency of assets that have performed well recently to continue performing well.")
        st.markdown(kpi_box("3 Day Momentum",  feature_summary['momentum'][f'{ticker}_mom_3']),  unsafe_allow_html=True)
        st.markdown(kpi_box("5 Day Momentum",  feature_summary['momentum'][f'{ticker}_mom_5']),  unsafe_allow_html=True)
        st.markdown(kpi_box("10 Day Momentum", feature_summary['momentum'][f'{ticker}_mom_10']), unsafe_allow_html=True)
        st.markdown(kpi_box("20 Day Momentum", feature_summary['momentum'][f'{ticker}_mom_20']), unsafe_allow_html=True)

    with col2:
        st.subheader("⚡ Volatility")
        st.caption("The degree of variation of a trading price series over time.")
        st.markdown(kpi_box("5 Day Volatility",  feature_summary['volatility'][f'{ticker}_vol_5']),  unsafe_allow_html=True)
        st.markdown(kpi_box("20 Day Volatility", feature_summary['volatility'][f'{ticker}_vol_20']), unsafe_allow_html=True)

    st.markdown("---")
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
        st.markdown(kpi_box("5 Day Mean Return",  feature_summary['rolling_mean'][f'{ticker}_mean_ret_5']),  unsafe_allow_html=True)
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

    latest_row          = get_latest_features(result['main_feature_matrix'])
    sector_tickers      = sector_dict[sector]
    sector_strength     = get_sector_strength(latest_row, sector_tickers)
    nifty_context       = get_nifty_context(latest_row)
    relative_performance = get_relative_performance(latest_row, sector_tickers)

    label_map = {
        "mom_3":       "3 Day Momentum",
        "mom_5":       "5 Day Momentum",
        "mom_10":      "10 Day Momentum",
        "mom_20":      "20 Day Momentum",
        "vol_5":       "5 Day Volatility",
        "vol_20":      "20 Day Volatility",
        "ma_dist_10":  "MA10 Distance",
        "ma_dist_20":  "MA20 Distance",
        "ma_dist_50":  "MA50 Distance",
        "mean_ret_5":  "5 Day Mean Return",
        "mean_ret_20": "20 Day Mean Return"
    }

    # Sector Strength
    st.subheader("🏭 Sector Strength")
    strength_color_map = {
        "Strong Sector Strength":   "#4CAF50",
        "Moderate Sector Strength": "#8BC34A",
        "Neutral Sector Strength":  "#FF9800",
        "Weak Sector Strength":     "#F44336"
    }
    strength_color = strength_color_map.get(sector_strength['strength'], "gray")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(kpi_box_str("Sector Strength", sector_strength['strength'], strength_color), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_box_float("Sector Momentum Value", sector_strength['value']), unsafe_allow_html=True)

    st.markdown("---")


    # ── NIFTY CONTEXT ─────────────────────────────────────────────

    st.subheader("📊 NIFTY Market Context")

    # -----------------------------
    # Market Regime Calculation
    # -----------------------------

    mom20 = nifty_context.get("^NSEI_mom_20", 0)
    ma20  = nifty_context.get("^NSEI_ma_dist_20", 0)
    vol20 = nifty_context.get("^NSEI_vol_20", 0)

    market_score = (
    (mom20 * 100)
    + (ma20 * 50)
)

    if market_score >= 3:
        regime = "🟢 Bullish"
        regime_color = "#4CAF50"

    elif market_score >= -3:
        regime = "🟡 Neutral"
        regime_color = "#FFC107"

    else:
        regime = "🔴 Bearish"
        regime_color = "#F44336"

# -----------------------------
# Hero Card
# -----------------------------

    st.markdown(
    f"""
    <div style="
        background:#1e1e2e;
        padding:25px;
        border-radius:15px;
        border-left:8px solid {regime_color};
        margin-bottom:20px;
    ">
        <h2 style="margin:0;color:white;">
            {regime}
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

    st.caption("Overall NIFTY Market Regime")

    # -----------------------------
    # Key Metrics
    # -----------------------------

    st.markdown("### Key Market Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            kpi_box_float(
                "20 Day Momentum",
                mom20
            ),
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            kpi_box_float(
                "20 Day Volatility",
                vol20,
                is_volatility=True
            ),
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            kpi_box_float(
                "20 Day MA Distance",
                ma20
            ),
            unsafe_allow_html=True
        )

    # -----------------------------
    # Detailed Metrics
    # -----------------------------

    with st.expander("📈 View Detailed NIFTY Metrics", expanded=False):

        momentum_keys = {
            k: v for k, v in nifty_context.items()
            if "_mom_" in k
        }

        volatility_keys = {
            k: v for k, v in nifty_context.items()
            if "_vol_" in k
        }

        ma_keys = {
            k: v for k, v in nifty_context.items()
            if "_ma_dist_" in k
        }

        mean_ret_keys = {
            k: v for k, v in nifty_context.items()
            if "_mean_ret_" in k
        }

        # --------------------------------
        # Momentum
        # --------------------------------

        st.markdown("#### Momentum")

        cols = st.columns(2)

        for idx, (key, value) in enumerate(momentum_keys.items()):

            label = label_map.get(
                key.replace("^NSEI_", ""),
                key
            )

            with cols[idx % 2]:

                st.markdown(
                    kpi_box_float(label, value),
                    unsafe_allow_html=True
                )

        # --------------------------------
        # Volatility
        # --------------------------------

        st.markdown("#### Volatility")

        cols = st.columns(2)

        for idx, (key, value) in enumerate(volatility_keys.items()):

            label = label_map.get(
                key.replace("^NSEI_", ""),
                key
            )

            with cols[idx % 2]:

                st.markdown(
                    kpi_box_float(
                        label,
                        value,
                        is_volatility=True
                    ),
                    unsafe_allow_html=True
                )

        # --------------------------------
        # MA Distance
        # --------------------------------

        st.markdown("#### Moving Average Distance")

        cols = st.columns(3)

        for idx, (key, value) in enumerate(ma_keys.items()):

            label = label_map.get(
                key.replace("^NSEI_", ""),
                key
            )

            with cols[idx]:

                st.markdown(
                    kpi_box_float(
                        label,
                        value
                    ),
                    unsafe_allow_html=True
                )

        # --------------------------------
        # Mean Return
        # --------------------------------

        st.markdown("#### Mean Return")

        cols = st.columns(2)

        for idx, (key, value) in enumerate(mean_ret_keys.items()):

            label = label_map.get(
                key.replace("^NSEI_", ""),
                key
            )

            with cols[idx % 2]:

                st.markdown(
                    kpi_box_float(
                        label,
                        value
                    ),
                    unsafe_allow_html=True
                )


    # Relative Performance
    st.subheader("📈 Relative Performance")
    st.caption("20 Day Momentum comparison across all stocks in the sector.")
    st.dataframe(
        relative_performance.style.format({"20D Momentum": "{:.2f}%"}),
        use_container_width=True
    )
    fig = px.bar(relative_performance, x='Stocks', y='20D Momentum', title='Sector Relative Performance')
    st.plotly_chart(fig,use_container_width=True)

# ── ENTRY RISK + RISK/REWARD ──────────────────────────────────────────────────
elif current_page == "⚠️ Entry Risk (with Risk/Reward prediction)":
    if st.session_state['result'] is None:
        st.warning("Please go to Home and run the analysis first")
        st.stop()

    result = st.session_state['result']
    ticker = st.session_state['ticker']
    sector = st.session_state['sector']

    st.title("⚠️ Entry Risk & Risk/Reward")
    st.markdown(f"**{ticker}** | {sector} Sector")
    st.markdown("---")

    latest_row  = get_latest_features(result['main_feature_matrix'])
    raw_data    = result['raw_data']
    drawdown    = get_drawdown_metrics(raw_data, ticker)
    buy_risk    = get_buy_risk_score(latest_row, ticker, drawdown)
    preds       = get_predictions(result['lr_model'], result['xgb_model'],result['lr_weight'], result['xgb_weight'], latest_row)
    consensus   = preds['consensus']
    risk_reward = get_risk_reward(buy_risk['score'], consensus)

    # ── Entry Risk Score ──────────────────────────────────────────────────────
    if buy_risk['label'] == "High Risk":
        risk_color = "#F44336"
    elif buy_risk['label'] == "Moderate Risk":
        risk_color = "#FF9800"
    else:
        risk_color = "#4CAF50"

    st.markdown(f"""
    <div style="background-color:#1e1e2e; padding:40px; border-radius:15px;
    border: 3px solid {risk_color}; text-align:center; margin-bottom:20px;">
        <p style="color:gray; margin:0; font-size:16px;">Entry Risk Score</p>
        <p style="color:{risk_color}; margin:0; font-size:72px; font-weight:bold;">{buy_risk['score']}/100</p>
        <p style="color:{risk_color}; margin:0; font-size:24px;">{buy_risk['label']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "Entry Risk measures how risky it is to buy this stock right now, based on volatility "
        "and drawdown from peak. Higher score = higher risk."
    )
    st.markdown("---")

    # ── Drawdown ──────────────────────────────────────────────────────────────
    st.subheader("📉 Drawdown Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(kpi_box_drawdown("Current Drawdown", drawdown['current_drawdown']), unsafe_allow_html=True)
    with col2:
        st.markdown(kpi_box_drawdown("Max Drawdown", drawdown['max_drawdown']), unsafe_allow_html=True)

    st.markdown("---")

    # ── Volume Shock ──────────────────────────────────────────────────────────
    st.subheader("📊 Volume Shock")
    st.caption("Current volume relative to 20-day average. Values above 1.5x signal unusual activity; above 3x suggests institutional selling.")

    vol_shock = buy_risk.get('vol_shock')
    if vol_shock is not None:
        if vol_shock >= 3.0:
            vs_color = "#F44336"
            vs_label = "Extreme — Likely Institutional Selling"
        elif vol_shock >= 2.0:
            vs_color = "#FF9800"
            vs_label = "High — Elevated Selling Pressure"
        elif vol_shock >= 1.5:
            vs_color = "#FFC107"
            vs_label = "Elevated — Watch Closely"
        else:
            vs_color = "#4CAF50"
            vs_label = "Normal"

        st.markdown(f"""
        <div style="background-color:#1e1e2e; padding:20px; border-radius:10px;
        border-left:4px solid {vs_color}; margin-bottom:15px;">
            <p style="color:gray; margin:0; font-size:13px;">Volume Shock (Vol / 20D Avg Vol)</p>
            <p style="color:{vs_color}; margin:0; font-size:36px; font-weight:bold;">{vol_shock:.2f}x</p>
            <p style="color:{vs_color}; margin:0; font-size:14px;">{vs_label}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Volume shock data not available.")


    # ── Risk / Reward ─────────────────────────────────────────────────────────
    st.subheader("⚖️ Risk / Reward Assessment")
    st.caption(
        "Derived from the Entry Risk Score and Consensus Probability. "
        "A favourable rating means the potential upside outweighs the current risk."
    )

    rr_label = risk_reward.get('label', 'N/A')

    rr_score = risk_reward.get('ratio', None)
    rr_desc  = f"Reward {risk_reward.get('estimated_reward', 0):.1f} vs Risk {risk_reward.get('estimated_risk', 0):.1f}"

    rr_color_map = {
        "Excellent": "#4CAF50",
        "Good":      "#8BC34A",
        "Neutral":   "#FF9800",
        "Poor":      "#F44336",
        "Very Poor": "#B71C1C"
    }
    rr_color = rr_color_map.get(rr_label, "#888888")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div style="background-color:#1e1e2e; padding:30px; border-radius:15px;
        border: 3px solid {rr_color}; text-align:center;">
            <p style="color:gray; margin:0; font-size:14px;">Risk / Reward Rating</p>
            <p style="color:{rr_color}; margin:0; font-size:48px; font-weight:bold;">{rr_label}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        if rr_score is not None:
            st.markdown(f"""
            <div style="background-color:#1e1e2e; padding:30px; border-radius:15px;
            border: 3px solid {rr_color}; text-align:center;">
                <p style="color:gray; margin:0; font-size:14px;">Risk / Reward Score</p>
                <p style="color:{rr_color}; margin:0; font-size:48px; font-weight:bold;">{rr_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

    if rr_desc:
        st.info(f"💡 {rr_desc}")


# ── CHARTS ────────────────────────────────────────────────────────────────────
elif current_page == "📉 Charts":
    if st.session_state['result'] is None:
        st.warning("Please go to Home and run the analysis first")
        st.stop()

    result   = st.session_state['result']
    ticker   = st.session_state['ticker']
    sector   = st.session_state['sector']
    raw_data = result['raw_data']

    st.title("📉 Charts")
    st.markdown(f"**{ticker}** | {sector} Sector")
    st.markdown("---")

    sector_tickers = sector_dict[sector]
    feature_matrix = result['main_feature_matrix']
    latest_row     = get_latest_features(feature_matrix)

    # ── 1. Normalised Sector Price Comparison ─────────────────────────────────
    st.subheader("🏭 Sector Price Comparison (Normalised)")
    st.caption("All stocks normalised to 100 at start of period so relative performance is directly comparable.")

    all_tickers = list(set([ticker] + sector_tickers))
    norm_dict   = {}
    for t in all_tickers:
        col_name = f'{t}_Close'
        if col_name not in raw_data.columns:
            continue
        series = raw_data[col_name].dropna().copy()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        norm_dict[t] = (series / series.iloc[0]) * 100

    if norm_dict:
        norm_df = pd.DataFrame(norm_dict)
        # Put target ticker first so it renders with a distinct colour
        cols_order = [ticker] + [c for c in norm_df.columns if c != ticker]
        fig = px.line(norm_df[cols_order], x=norm_df.index, y=cols_order, title=f'Sector Price Comparison (Normalised)')
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("No price data found for sector peers.")

    st.markdown("---")

    # ── 2. Momentum Trend ────────────────────────────────────────────────────
    st.subheader("📈 Momentum Trend")
    st.caption(f"Rolling momentum values over time for {ticker}.")

    mom_cols = [c for c in feature_matrix.columns if c.startswith(f'{ticker}_mom_')]
    if mom_cols:
        mom_df = feature_matrix[mom_cols].copy()
        mom_df.index = pd.to_datetime(mom_df.index).tz_localize(None)
        mom_df.columns = [c.replace(f'{ticker}_mom_', '') + 'D Mom' for c in mom_cols]
        fig = px.line(mom_df, x=mom_df.index, y=mom_df.columns, title=f'Momentum Trend')
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("Momentum columns not found in feature matrix.")

    st.markdown("---")

    # ── 3. Volatility Trend ──────────────────────────────────────────────────
    st.subheader("⚡ Volatility Trend")
    st.caption(f"Rolling volatility over time for {ticker}.")

    vol_cols = [c for c in feature_matrix.columns if c.startswith(f'{ticker}_vol_')]
    if vol_cols:
        vol_df = feature_matrix[vol_cols].copy()
        vol_df.index = pd.to_datetime(vol_df.index).tz_localize(None)
        vol_df.columns = [c.replace(f'{ticker}_vol_', '') + 'D Vol' for c in vol_cols]
        fig = px.area(vol_df, x=vol_df.index, y=vol_df.columns, title=f'Volatility Trend')
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.info("Volatility columns not found in feature matrix.")

    st.markdown("---")

    # ── 4. Drawdown from Peak ─────────────────────────────────────────────────
    st.subheader("📉 Drawdown from Peak")
    st.caption(f"How far {ticker} has fallen from its rolling all-time high at each point in time.")

    close_col = f'{ticker}_Close'
    if close_col in raw_data.columns:
        price = raw_data[close_col].dropna().copy()
        price.index = pd.to_datetime(price.index).tz_localize(None)
        drawdown_series = (price - price.cummax()) / price.cummax()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown_series.index, y=drawdown_series.values, mode='lines', name='Drawdown'))
        fig.update_layout(title=f'Drawdown from Peak', xaxis_title='Date', yaxis_title='Drawdown')
        st.plotly_chart(fig,use_container_width=True)
    else:
        st.info(f"Price data for {ticker} not found.")

    st.markdown("---")
