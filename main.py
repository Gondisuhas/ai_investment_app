import os
import time
import sqlite3
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Optional: Gemini
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ---------------------------
# Configuration & Secrets
# ---------------------------
# Safely access secrets with fallback
try:
    if "GOOGLE_API_KEY" in st.secrets:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    else:
        GOOGLE_API_KEY = ""
except (KeyError, FileNotFoundError, Exception):
    GOOGLE_API_KEY = ""

try:
    if "APP_PASSWORD" in st.secrets:
        APP_PASSWORD = st.secrets["APP_PASSWORD"]
    else:
        APP_PASSWORD = "password123"
except (KeyError, FileNotFoundError, Exception):
    APP_PASSWORD = "password123"

# Configure Gemini if available
if GENAI_AVAILABLE and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.sidebar.warning(f"Gemini configuration failed: {e}")
        GENAI_AVAILABLE = False

def pick_gemini_model():
    """Select the best available Gemini model"""
    if not GENAI_AVAILABLE or not GOOGLE_API_KEY:
        return None
    try:
        models = genai.list_models()
        # Priority order of models to try
        preferred_models = [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-pro-latest"
        ]
        
        available_models = []
        for m in models:
            name = getattr(m, "name", "")
            supported = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in supported:
                # Remove 'models/' prefix
                clean_name = name.replace("models/", "")
                available_models.append(clean_name)
        
        # Try to find preferred model
        for pref in preferred_models:
            if pref in available_models:
                return pref
        
        # Fallback: return first available model that supports generateContent
        if available_models:
            return available_models[0]
        
        return "gemini-2.5-flash"
    except Exception:
        return "gemini-2.5-flash"

MODEL_NAME = pick_gemini_model()

# ---------------------------
# Database (SQLite) for portfolio persistence
# ---------------------------
DB_PATH = "portfolio.db"

@st.cache_resource
def init_db():
    """Initialize database connection"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            qty REAL NOT NULL,
            avg_price REAL NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn

conn = init_db()

def add_position_db(ticker, qty, avg_price):
    """Add a position to the portfolio"""
    c = conn.cursor()
    c.execute("INSERT INTO portfolio (ticker, qty, avg_price) VALUES (?, ?, ?)",
              (ticker.upper(), float(qty), float(avg_price)))
    conn.commit()

def remove_position_db(row_id):
    """Remove a position from the portfolio"""
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE id = ?", (row_id,))
    conn.commit()

def list_positions_db():
    """List all portfolio positions"""
    c = conn.cursor()
    c.execute("SELECT id, ticker, qty, avg_price, added_at FROM portfolio ORDER BY added_at DESC")
    rows = c.fetchall()
    cols = ["id", "ticker", "qty", "avg_price", "added_at"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data(ttl=60)
def get_current_price(ticker):
    """Fetch current price for a ticker"""
    try:
        t = yf.Ticker(ticker)
        data = t.history(period="1d")
        if data is not None and not data.empty:
            return float(data["Close"].iloc[-1])
        return None
    except Exception as e:
        st.error(f"Error fetching price for {ticker}: {e}")
        return None

def compute_indicators(df):
    """Compute technical indicators"""
    df = df.copy()
    if "Close" not in df.columns or df.empty:
        return df
    
    # EMA
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    
    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    
    return df

def calculate_risk_score(df):
    """
    Calculate a quantitative risk score (0-100) based on technical indicators
    Higher score = Higher risk
    
    Factors:
    - Volatility (30%): Higher volatility = higher risk
    - RSI extremes (20%): Overbought/oversold = higher risk
    - Price momentum (20%): Declining trend = higher risk
    - MACD divergence (15%): Weak momentum = higher risk
    - Volume trend (15%): Declining volume = higher risk
    """
    try:
        if df is None or df.empty or len(df) < 30:
            return 50, "Insufficient data"
        
        risk_components = {}
        
        # 1. Volatility Risk (30 points)
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        vol_risk = min(volatility * 100, 30)  # Cap at 30
        risk_components['Volatility'] = round(vol_risk, 1)
        
        # 2. RSI Risk (20 points)
        current_rsi = df['RSI'].iloc[-1]
        if pd.isna(current_rsi):
            rsi_risk = 10
        elif current_rsi > 70:  # Overbought
            rsi_risk = 20 * (current_rsi - 70) / 30
        elif current_rsi < 30:  # Oversold
            rsi_risk = 20 * (30 - current_rsi) / 30
        else:  # Neutral zone
            rsi_risk = 5
        risk_components['RSI'] = round(rsi_risk, 1)
        
        # 3. Price Momentum Risk (20 points)
        # Compare recent prices to 30-day average
        recent_avg = df['Close'].tail(5).mean()
        month_avg = df['Close'].tail(30).mean()
        momentum_change = (recent_avg - month_avg) / month_avg
        
        if momentum_change < -0.05:  # Declining > 5%
            momentum_risk = 20
        elif momentum_change < 0:  # Slightly declining
            momentum_risk = 10
        elif momentum_change > 0.05:  # Rising > 5%
            momentum_risk = 5
        else:  # Slightly rising
            momentum_risk = 8
        risk_components['Momentum'] = round(momentum_risk, 1)
        
        # 4. MACD Risk (15 points)
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['Signal'].iloc[-1]
        
        if pd.isna(current_macd) or pd.isna(current_signal):
            macd_risk = 7.5
        else:
            macd_diff = current_macd - current_signal
            if macd_diff < 0:  # Bearish
                macd_risk = 15
            elif abs(macd_diff) < 0.5:  # Weak signal
                macd_risk = 10
            else:  # Bullish
                macd_risk = 5
        risk_components['MACD'] = round(macd_risk, 1)
        
        # 5. Volume Risk (15 points)
        if 'Volume' in df.columns:
            recent_vol = df['Volume'].tail(5).mean()
            avg_vol = df['Volume'].tail(30).mean()
            
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                if vol_ratio < 0.7:  # Low volume
                    volume_risk = 15
                elif vol_ratio < 0.9:
                    volume_risk = 10
                else:
                    volume_risk = 5
            else:
                volume_risk = 7.5
        else:
            volume_risk = 7.5
        risk_components['Volume'] = round(volume_risk, 1)
        
        # Total Risk Score
        total_risk = sum(risk_components.values())
        
        # Risk Level Classification
        if total_risk < 30:
            risk_level = "Low Risk 🟢"
        elif total_risk < 50:
            risk_level = "Moderate Risk 🟡"
        elif total_risk < 70:
            risk_level = "High Risk 🟠"
        else:
            risk_level = "Very High Risk 🔴"
        
        return round(total_risk, 1), risk_level, risk_components
        
    except Exception as e:
        return 50, "Calculation Error", {"Error": str(e)}
    """Generate trading signal based on EMA crossover"""
    try:
        if len(df) < 2:
            return "N/A"
        if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]:
            return "BUY"
        elif df["EMA20"].iloc[-1] < df["EMA50"].iloc[-1]:
            return "SELL"
        else:
            return "HOLD"
    except Exception:
        return "N/A"

def ask_gemini(prompt):
    """Query Gemini API"""
    if not GENAI_AVAILABLE:
        return "⚠️ Gemini AI is not available. Install google-generativeai package."
    if not GOOGLE_API_KEY:
        return "⚠️ Gemini not configured. Add GOOGLE_API_KEY to Streamlit secrets to enable AI features."
    try:
        # Try different model names if the current one fails
        model_names = [
            MODEL_NAME,
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
            "gemini-flash-latest",
            "gemini-pro-latest"
        ]
        
        for model_name in model_names:
            if not model_name:
                continue
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                return getattr(resp, "text", str(resp))
            except Exception as e:
                error_str = str(e)
                # If 404 or model not found, try next model
                if ("404" in error_str or "not found" in error_str.lower()) and model_name != model_names[-1]:
                    continue
                else:
                    raise e
        
        return "❌ Could not find a working Gemini model. Please check the Settings page."
    except Exception as e:
        return f"❌ Gemini error: {e}"

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Gemini Investment Terminal",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Authentication
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Gemini Investment Terminal")
    st.markdown("### Welcome, Captain Suhas")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pw = st.text_input("Enter app password", type="password", key="login_password")
        if st.button("🚀 Login", use_container_width=True):
            if pw == APP_PASSWORD:
                st.session_state.authenticated = True
                st.success("✅ Login successful!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Incorrect password. Please try again.")
    st.stop()

# ---------------------------
# Main Application
# ---------------------------
st.markdown('<p class="main-header">📊 Gemini Investment Terminal</p>', unsafe_allow_html=True)
st.markdown("**Captain Suhas Dashboard**")

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/rocket.png", width=80)
    st.markdown("### 🧭 Navigation")
    pages = ["🏠 Home", "⚡ Real-Time", "📈 Stock Analyzer", "₿ Crypto", 
             "💼 Portfolio", "📰 News & Sentiment", "🔮 Predictions", "⚙️ Settings"]
    page = st.selectbox("Select Page", pages, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### ⚡ Controls")
    refresh_auto = st.slider("Auto-refresh (sec)", 0, 60, 0, help="0 = disabled")
    
    st.markdown("---")
    st.markdown("### 🤖 AI Status")
    
    # Debug info for secrets
    if GOOGLE_API_KEY:
        st.success(f"✅ API Key: {GOOGLE_API_KEY[:8]}...{GOOGLE_API_KEY[-4:]}")
    else:
        st.error("❌ No API key found")
    
    if GENAI_AVAILABLE and GOOGLE_API_KEY and MODEL_NAME:
        st.success(f"✅ Model: `{MODEL_NAME}`")
    elif GENAI_AVAILABLE and GOOGLE_API_KEY:
        st.warning("⚠️ API key present but model not detected")
    elif GENAI_AVAILABLE:
        st.warning("⚠️ API key missing")
    else:
        st.error("❌ Gemini library not installed")
    
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# Initialize last refresh time
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0

# ---------------------------
# Page: Home
# ---------------------------
if page == "🏠 Home":
    st.header("🏠 Welcome, Captain Suhas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Quick Start Guide
        
        **Navigation:**
        - **⚡ Real-Time**: Live multi-ticker tracking with 1-minute charts
        - **📈 Stock Analyzer**: Deep dive into stocks with AI analysis
        - **₿ Crypto**: Cryptocurrency analysis and sentiment
        - **💼 Portfolio**: Manage your positions (persistent storage)
        - **📰 News & Sentiment**: Latest headlines with AI insights
        - **🔮 Predictions**: AI-powered price forecasts
        - **⚙️ Settings**: Diagnostics and configuration
        """)
    
    with col2:
        st.markdown("""
        ### 💡 Pro Tips
        
        - **Indian Stocks**: Use `.NS` suffix (e.g., `TCS.NS`, `INFY.NS`)
        - **US Stocks**: Direct ticker (e.g., `AAPL`, `MSFT`)
        - **Crypto**: Use format `BTC-USD`, `ETH-USD`
        - **Portfolio**: Data persists in SQLite database
        - **AI Features**: Requires Gemini API key in secrets
        """)
    
    st.info("📌 **Note**: Your portfolio is stored locally in `portfolio.db` and persists between sessions.")

# ---------------------------
# Page: Real-Time
# ---------------------------
elif page == "⚡ Real-Time":
    st.header("⚡ Real-Time Multi-Ticker Tracker")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        tickers_raw = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOGL", 
                                    help="Examples: AAPL,MSFT or TCS.NS,INFY.NS")
    with col2:
        st.write("")
        st.write("")
        refresh_now = st.button("🔄 Refresh Now", use_container_width=True)
    
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    
    # Auto-refresh logic
    should_refresh = refresh_now or (refresh_auto > 0 and (time.time() - st.session_state.last_refresh > refresh_auto))
    
    if tickers and should_refresh:
        st.session_state.last_refresh = time.time()
        
        for t in tickers:
            with st.expander(f"📊 {t}", expanded=True):
                try:
                    ticker_obj = yf.Ticker(t)
                    intraday = ticker_obj.history(period="1d", interval="1m")
                    
                    if intraday is None or intraday.empty:
                        st.warning(f"⚠️ No intraday data for {t}. Check ticker format.")
                        continue
                    
                    # Current price
                    latest = intraday["Close"].iloc[-1]
                    prev_close = intraday["Close"].iloc[0]
                    change = latest - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${latest:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
                    col2.metric("High", f"${intraday['High'].max():.2f}")
                    col3.metric("Low", f"${intraday['Low'].min():.2f}")
                    
                    # Chart with EMA
                    intraday["EMA20"] = intraday["Close"].ewm(span=20, adjust=False).mean()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=intraday.index,
                        open=intraday["Open"],
                        high=intraday["High"],
                        low=intraday["Low"],
                        close=intraday["Close"],
                        name="Price"
                    ))
                    fig.add_trace(go.Scatter(
                        x=intraday.index,
                        y=intraday["EMA20"],
                        mode="lines",
                        name="EMA20",
                        line=dict(color="orange", width=2)
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        height=400,
                        xaxis_title="Time",
                        yaxis_title="Price",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Error fetching {t}: {e}")

# ---------------------------
# Page: Stock Analyzer
# ---------------------------
elif page == "📈 Stock Analyzer":
    st.header("📈 Stock Analyzer")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker Symbol", "AAPL", help="Enter stock ticker").upper()
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    with col3:
        interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    
    if st.button("🔍 Analyze Stock", use_container_width=True):
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                t = yf.Ticker(ticker)
                hist = t.history(period=period, interval=interval)
                
                if hist is None or hist.empty:
                    st.error("❌ No data available for this ticker.")
                else:
                    # Price Chart
                    st.subheader("📊 Price Chart")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist["Close"],
                        mode="lines",
                        name="Close Price",
                        line=dict(color="#1f77b4", width=2)
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        height=450,
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Fundamentals
                    st.subheader("📋 Fundamentals")
                    info = getattr(t, "info", {}) or {}
                    
                    cols = st.columns(4)
                    metrics = [
                        ("Market Cap", info.get("marketCap", "N/A")),
                        ("PE Ratio", info.get("trailingPE", "N/A")),
                        ("52W High", info.get("fiftyTwoWeekHigh", "N/A")),
                        ("52W Low", info.get("fiftyTwoWeekLow", "N/A"))
                    ]
                    
                    for col, (label, value) in zip(cols, metrics):
                        if isinstance(value, (int, float)):
                            col.metric(label, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
                        else:
                            col.metric(label, value)
                    
                    # Technical Indicators
                    st.subheader("📊 Technical Analysis")
                    df = compute_indicators(hist)
                    
                    # Calculate Risk Score
                    risk_score, risk_level, risk_breakdown = calculate_risk_score(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Latest Metrics:**")
                        st.write(f"- Close: ${df['Close'].iloc[-1]:.2f}")
                        st.write(f"- EMA20: ${df['EMA20'].iloc[-1]:.2f}")
                        st.write(f"- EMA50: ${df['EMA50'].iloc[-1]:.2f}")
                        st.write(f"- RSI: {df['RSI'].iloc[-1]:.2f}")
                        st.write(f"- MACD: {df['MACD'].iloc[-1]:.4f}")
                    
                    with col2:
                        signal = generate_signal(df)
                        signal_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡", "N/A": "⚪"}
                        st.markdown(f"### {signal_color.get(signal, '⚪')} Signal: **{signal}**")
                        st.write("Based on EMA20/EMA50 crossover")
                        
                        # RSI interpretation
                        rsi_val = df['RSI'].iloc[-1]
                        if rsi_val > 70:
                            st.warning("⚠️ Overbought (RSI > 70)")
                        elif rsi_val < 30:
                            st.info("💡 Oversold (RSI < 30)")
                    
                    # Risk Score Display
                    st.markdown("---")
                    st.subheader("⚠️ Quantitative Risk Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Risk Score", f"{risk_score}/100")
                    col2.metric("Risk Level", risk_level)
                    col3.metric("Technical Signal", signal)
                    
                    # Risk Breakdown
                    st.write("**Risk Components:**")
                    risk_df = pd.DataFrame([
                        {"Factor": "Volatility (30%)", "Score": risk_breakdown.get('Volatility', 0), "Max": 30},
                        {"Factor": "RSI Extremes (20%)", "Score": risk_breakdown.get('RSI', 0), "Max": 20},
                        {"Factor": "Price Momentum (20%)", "Score": risk_breakdown.get('Momentum', 0), "Max": 20},
                        {"Factor": "MACD Signal (15%)", "Score": risk_breakdown.get('MACD', 0), "Max": 15},
                        {"Factor": "Volume Trend (15%)", "Score": risk_breakdown.get('Volume', 0), "Max": 15}
                    ])
                    
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)
                    
                    st.info("""
                    **How Risk is Calculated:**
                    - **Volatility**: Higher price swings = higher risk
                    - **RSI**: Overbought (>70) or oversold (<30) = higher risk
                    - **Momentum**: Declining price trend = higher risk
                    - **MACD**: Bearish signal or weak momentum = higher risk
                    - **Volume**: Declining volume = higher risk (less liquidity)
                    """)
                    
                    # Recent indicator data
                    st.write("**Recent Technical Data:**")
                    st.dataframe(
                        df.tail(5)[["Close", "EMA20", "EMA50", "RSI", "MACD", "Signal"]].round(4),
                        use_container_width=True
                    )
                    
                    # AI Analysis
                    st.subheader("🤖 AI Analysis")
                    prompt = f"""You are a senior market analyst. Analyze {ticker} based on its recent {period} performance.
                    
Latest data:
- Current Price: ${df['Close'].iloc[-1]:.2f}
- EMA20: ${df['EMA20'].iloc[-1]:.2f}
- EMA50: ${df['EMA50'].iloc[-1]:.2f}
- RSI: {df['RSI'].iloc[-1]:.2f}
- Technical Signal: {signal}
- Calculated Risk Score: {risk_score}/100 ({risk_level})

Risk Breakdown:
- Volatility Risk: {risk_breakdown.get('Volatility', 0)}/30
- RSI Risk: {risk_breakdown.get('RSI', 0)}/20
- Momentum Risk: {risk_breakdown.get('Momentum', 0)}/20
- MACD Risk: {risk_breakdown.get('MACD', 0)}/15
- Volume Risk: {risk_breakdown.get('Volume', 0)}/15

Provide:
1. Overall Sentiment (Bullish/Neutral/Bearish) - explain why based on the metrics
2. Commentary on the risk score - is it justified?
3. Recommendation (Buy/Hold/Sell) with conviction level
4. Key technical levels to watch (support/resistance)
5. Important considerations for investors

Keep response concise and actionable. Focus on interpreting the quantitative data provided."""
                    
                    with st.spinner("🤖 Consulting Gemini AI..."):
                        ai_response = ask_gemini(prompt)
                    
                    st.markdown(ai_response)
                    
            except Exception as e:
                st.error(f"❌ Error analyzing stock: {e}")

# ---------------------------
# Page: Crypto
# ---------------------------
elif page == "₿ Crypto":
    st.header("₿ Cryptocurrency Analyzer")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        crypto = st.text_input("Crypto Ticker", "BTC-USD", 
                               help="Format: BTC-USD, ETH-USD, etc.").upper()
    with col2:
        c_period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo"], index=2)
    
    if st.button("🔍 Analyze Crypto", use_container_width=True):
        with st.spinner(f"Analyzing {crypto}..."):
            try:
                crypto_obj = yf.Ticker(crypto)
                interval = "1m" if c_period == "1d" else ("15m" if c_period == "5d" else "1h")
                ch = crypto_obj.history(period=c_period, interval=interval)
                
                if ch is None or ch.empty:
                    st.error("❌ No data available for this crypto ticker.")
                else:
                    # Metrics
                    latest = ch["Close"].iloc[-1]
                    high = ch["High"].max()
                    low = ch["Low"].min()
                    vol = ch["Volume"].sum()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${latest:,.2f}")
                    col2.metric(f"{c_period} High", f"${high:,.2f}")
                    col3.metric(f"{c_period} Low", f"${low:,.2f}")
                    col4.metric("Total Volume", f"{vol:,.0f}")
                    
                    # Candlestick Chart
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=ch.index,
                        open=ch["Open"],
                        high=ch["High"],
                        low=ch["Low"],
                        close=ch["Close"],
                        name=crypto
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        height=500,
                        xaxis_title="Time",
                        yaxis_title="Price (USD)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume Chart
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=ch.index,
                        y=ch["Volume"],
                        name="Volume",
                        marker_color="lightblue"
                    ))
                    fig_vol.update_layout(
                        template="plotly_dark",
                        height=250,
                        xaxis_title="Time",
                        yaxis_title="Volume",
                        showlegend=False
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # AI Sentiment
                    st.subheader("🤖 AI Crypto Sentiment")
                    prompt = f"""Analyze {crypto} cryptocurrency for the last {c_period}.

Current data:
- Price: ${latest:,.2f}
- Period High: ${high:,.2f}
- Period Low: ${low:,.2f}
- Price range: {((high-low)/low*100):.2f}%

Provide:
1. Overall sentiment (Bullish/Bearish/Neutral)
2. Short-term outlook (next 7 days)
3. Key support and resistance levels
4. Risk assessment
5. Trading considerations

Be concise and specific."""
                    
                    with st.spinner("🤖 Analyzing with Gemini..."):
                        ai_response = ask_gemini(prompt)
                    
                    st.markdown(ai_response)
                    
            except Exception as e:
                st.error(f"❌ Error analyzing crypto: {e}")

# ---------------------------
# Page: Portfolio
# ---------------------------
elif page == "💼 Portfolio":
    st.header("💼 Portfolio Management")
    
    st.markdown("### ➕ Add New Position")
    
    with st.form("add_position", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ticker = st.text_input("Ticker", "AAPL", help="Stock ticker symbol")
        with col2:
            qty = st.number_input("Quantity", min_value=0.01, value=1.0, step=1.0)
        with col3:
            avg = st.number_input("Avg Price", min_value=0.0, value=0.0, step=0.01,
                                 help="0 = fetch current price")
        
        submitted = st.form_submit_button("➕ Add Position", use_container_width=True)
        
        if submitted:
            ticker = ticker.upper().strip()
            if not ticker:
                st.error("❌ Please enter a ticker symbol.")
            else:
                if avg == 0.0:
                    with st.spinner(f"Fetching current price for {ticker}..."):
                        cur = get_current_price(ticker)
                    if cur is None:
                        st.error("❌ Could not fetch current price. Please enter average price manually.")
                    else:
                        add_position_db(ticker, qty, cur)
                        st.success(f"✅ Added {qty} x {ticker} @ ${cur:.2f}")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    add_position_db(ticker, qty, avg)
                    st.success(f"✅ Added {qty} x {ticker} @ ${avg:.2f}")
                    time.sleep(0.5)
                    st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Current Holdings")
    
    df = list_positions_db()
    
    if df.empty:
        st.info("📭 No positions yet. Add your first position above!")
    else:
        # Compute live values
        rows = []
        total_value = 0
        total_pl = 0
        
        with st.spinner("Fetching live prices..."):
            for idx, r in df.iterrows():
                cur = get_current_price(r['ticker'])
                if cur is None:
                    cur = r['avg_price']  # Fallback to avg price
                
                val = cur * r['qty']
                pl = (cur - r['avg_price']) * r['qty']
                pl_pct = ((cur - r['avg_price']) / r['avg_price'] * 100) if r['avg_price'] > 0 else 0
                
                total_value += val
                total_pl += pl
                
                rows.append({
                    "ID": r['id'],
                    "Ticker": r['ticker'],
                    "Qty": r['qty'],
                    "Avg Price": f"${r['avg_price']:.2f}",
                    "Current": f"${cur:.2f}",
                    "Value": f"${val:.2f}",
                    "P&L": f"${pl:.2f}",
                    "P&L %": f"{pl_pct:.2f}%"
                })
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${total_value:,.2f}")
        col2.metric("Total P&L", f"${total_pl:,.2f}", 
                   delta=f"{(total_pl/total_value*100):.2f}%" if total_value > 0 else "0%")
        col3.metric("Positions", len(df))
        
        st.markdown("---")
        
        # Display portfolio
        pdf = pd.DataFrame(rows)
        st.dataframe(pdf, use_container_width=True, hide_index=True)
        
        # Remove position
        st.markdown("### 🗑️ Remove Position")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            rem_id = st.number_input("Enter Position ID", min_value=1, value=1, step=1)
        with col2:
            st.write("")
            st.write("")
            if st.button("🗑️ Remove", use_container_width=True):
                remove_position_db(rem_id)
                st.success(f"✅ Position {rem_id} removed.")
                time.sleep(0.5)
                st.rerun()

# ---------------------------
# Page: News & Sentiment
# ---------------------------
elif page == "📰 News & Sentiment":
    st.header("📰 News & Sentiment Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        nt = st.text_input("Ticker for News", "AAPL", help="Enter stock ticker").upper()
    with col2:
        ncount = st.slider("Headlines", 1, 10, 5)
    
    if st.button("📰 Fetch News", use_container_width=True):
        with st.spinner(f"Fetching news for {nt}..."):
            try:
                t = yf.Ticker(nt)
                raw_news = getattr(t, "news", []) or []
                
                if not raw_news:
                    st.warning("⚠️ No news available from yfinance. Coverage may vary by ticker.")
                else:
                    headlines = []
                    for item in raw_news[:ncount]:
                        title = item.get("title", "")
                        link = item.get("link", "")
                        publisher = item.get("publisher", "Unknown")
                        
                        if title:
                            headlines.append({"title": title, "link": link, "publisher": publisher})
                    
                    if headlines:
                        st.subheader(f"📰 Latest Headlines for {nt}")
                        
                        for i, h in enumerate(headlines, 1):
                            with st.expander(f"{i}. {h['title']}", expanded=(i <= 3)):
                                st.write(f"**Publisher:** {h['publisher']}")
                                if h['link']:
                                    st.markdown(f"[Read full article]({h['link']})")
                        
                        # AI Sentiment Analysis
                        st.markdown("---")
                        st.subheader("🤖 AI Sentiment Analysis")
                        
                        headline_text = "\n".join([f"{i+1}. {h['title']}" for i, h in enumerate(headlines)])
                        
                        prompt = f"""Analyze the sentiment and potential market impact of these recent headlines for {nt}:

{headline_text}

Provide:
1. Overall sentiment (Positive/Negative/Mixed/Neutral)
2. Key themes from the news
3. Potential impact on stock price (Short-term and Medium-term)
4. Investor considerations
5. Risk factors mentioned

Be specific and actionable."""
                        
                        with st.spinner("🤖 Analyzing sentiment with Gemini..."):
                            ai_response = ask_gemini(prompt)
                        
                        st.markdown(ai_response)
                    else:
                        st.warning("⚠️ Could not extract headlines from news data.")
                        
            except Exception as e:
                st.error(f"❌ Error fetching news: {e}")

# ---------------------------
# Page: Predictions
# ---------------------------
elif page == "🔮 Predictions":
    st.header("🔮 AI Price Predictions")
    
    st.info("⚠️ **Disclaimer**: These are AI-generated predictions based on historical data and should NOT be used as financial advice. Always do your own research.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        pt = st.text_input("Ticker", "AAPL", help="Enter stock ticker").upper()
    with col2:
        days = st.slider("Days Ahead", 1, 30, 7)
    
    if st.button("🔮 Generate Prediction", use_container_width=True):
        with st.spinner(f"Generating prediction for {pt}..."):
            try:
                ticker_obj = yf.Ticker(pt)
                hist = ticker_obj.history(period="6mo")
                
                if hist is None or hist.empty:
                    st.error("❌ Not enough historical data for prediction.")
                else:
                    # Display recent price action
                    st.subheader("📊 Recent Price History")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index,
                        y=hist["Close"],
                        mode="lines",
                        name="Close Price",
                        line=dict(color="#1f77b4", width=2)
                    ))
                    fig.update_layout(
                        template="plotly_dark",
                        height=350,
                        xaxis_title="Date",
                        yaxis_title="Price ($)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Get recent data for AI
                    recent_closes = hist["Close"].tail(30).tolist()
                    current_price = recent_closes[-1]
                    avg_30d = np.mean(recent_closes)
                    volatility = np.std(recent_closes)
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${current_price:.2f}")
                    col2.metric("30-Day Avg", f"${avg_30d:.2f}")
                    col3.metric("Volatility", f"${volatility:.2f}")
                    
                    # AI Prediction
                    st.markdown("---")
                    st.subheader("🤖 AI-Generated Forecast")
                    
                    prompt = f"""You are a quantitative analyst. Based on the recent 30-day closing prices for {pt}, provide a probabilistic forecast.

Recent closing prices (last 30 days): {recent_closes}

Current statistics:
- Current Price: ${current_price:.2f}
- 30-Day Average: ${avg_30d:.2f}
- Volatility (Std Dev): ${volatility:.2f}

Forecast for the next {days} days:

Provide:
1. Expected price range (Low-High)
2. Most likely price target
3. Probability of price increase vs. decrease
4. Confidence level (0-100)
5. Key factors that could affect the prediction
6. Risk considerations
7. Technical support/resistance levels

Be realistic and acknowledge uncertainty. Frame this as a probabilistic analysis, not a guarantee."""
                    
                    with st.spinner("🤖 Generating forecast with Gemini..."):
                        ai_response = ask_gemini(prompt)
                    
                    st.markdown(ai_response)
                    
                    st.warning("⚠️ **Important**: This prediction is based on historical patterns and AI analysis. Markets are unpredictable and many factors can affect prices. Never invest based solely on predictions.")
                    
            except Exception as e:
                st.error(f"❌ Error generating prediction: {e}")

# ---------------------------
# Page: Settings
# ---------------------------
elif page == "⚙️ Settings":
    st.header("⚙️ Settings & Diagnostics")
    
    # Gemini Configuration
    st.subheader("🤖 Gemini AI Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Status:**")
        if GENAI_AVAILABLE:
            st.success("✅ google-generativeai installed")
        else:
            st.error("❌ google-generativeai not installed")
            st.code("pip install google-generativeai")
        
        if GOOGLE_API_KEY:
            st.success(f"✅ API Key configured (length: {len(GOOGLE_API_KEY)})")
        else:
            st.warning("⚠️ No API key found in secrets")
    
    with col2:
        st.write("**Current Model:**")
        if MODEL_NAME:
            st.code(MODEL_NAME)
        else:
            st.error("No model detected")
    
    if st.button("🔍 List Available Models"):
        if not GENAI_AVAILABLE:
            st.error("❌ google-generativeai not installed")
        elif not GOOGLE_API_KEY:
            st.error("❌ API key not configured")
        else:
            try:
                with st.spinner("Fetching models..."):
                    models = genai.list_models()
                    model_list = [m.name for m in models]
                    st.write(f"Found {len(model_list)} models:")
                    st.json(model_list)
            except Exception as e:
                st.error(f"❌ Error listing models: {e}")
    
    st.markdown("---")
    
    # Database Management
    st.subheader("💾 Database Management")
    
    df = list_positions_db()
    st.write(f"**Current Positions:** {len(df)}")
    st.write(f"**Database Path:** `{DB_PATH}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Positions", type="primary"):
            c = conn.cursor()
            c.execute("DELETE FROM portfolio")
            conn.commit()
            st.success("✅ Portfolio cleared successfully!")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("📊 Show Database Stats"):
            c = conn.cursor()
            c.execute("SELECT COUNT(*) as count, SUM(qty) as total_qty FROM portfolio")
            stats = c.fetchone()
            st.write(f"- Total Positions: {stats[0]}")
            st.write(f"- Total Shares: {stats[1]}")
    
    st.markdown("---")
    
    # System Information
    st.subheader("ℹ️ System Information")
    
    info_data = {
        "Python Packages": ["streamlit", "yfinance", "pandas", "numpy", "plotly"],
        "Database": DB_PATH,
        "Authenticated": st.session_state.authenticated,
        "Auto-refresh": f"{refresh_auto}s" if refresh_auto > 0 else "Disabled"
    }
    
    for key, value in info_data.items():
        st.write(f"**{key}:** {value}")
    
    st.markdown("---")
    
    # API Testing
    st.subheader("🧪 API Testing")
    
    test_ticker = st.text_input("Test Ticker", "AAPL")
    
    if st.button("🧪 Test Yahoo Finance API"):
        with st.spinner(f"Testing {test_ticker}..."):
            try:
                t = yf.Ticker(test_ticker)
                data = t.history(period="1d")
                if data is not None and not data.empty:
                    st.success("✅ Yahoo Finance API working")
                    st.write("Latest data:")
                    st.dataframe(data.tail(1))
                else:
                    st.error("❌ No data returned")
            except Exception as e:
                st.error(f"❌ API Error: {e}")
    
    if st.button("🧪 Test Gemini API"):
        with st.spinner("Testing Gemini..."):
            response = ask_gemini("Respond with 'API working' if you receive this.")
            st.write("**Response:**")
            st.write(response)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Gemini Investment Terminal</strong> — Built for Captain Suhas</p>
    <p>🗄️ Local SQLite persistence enabled | 🤖 Powered by Gemini AI</p>
    <p style='font-size: 0.8em;'>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)