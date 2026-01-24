import os
import time
import sqlite3
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from collections import Counter

# ---------------------------
# Natural Language Processing Setup
# ---------------------------
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    st.warning("TextBlob not installed. Install with: pip install textblob")

# ---------------------------
# Configuration & Secrets
# ---------------------------
try:
    if "APP_PASSWORD" in st.secrets:
        APP_PASSWORD = st.secrets["APP_PASSWORD"]
    else:
        APP_PASSWORD = "password123"
except (KeyError, FileNotFoundError, Exception):
    APP_PASSWORD = "password123"

# ---------------------------
# Local Sentiment Analysis
# ---------------------------
def analyze_sentiment_local(text):
    """
    Perform sentiment analysis using TextBlob (no API needed)
    Returns: sentiment score (-1 to 1), polarity label, and confidence
    """
    if not TEXTBLOB_AVAILABLE or not text:
        return 0.0, "Neutral", 0.5
    
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        
        # Confidence based on subjectivity (inverse relationship)
        confidence = 1 - subjectivity
        
        return polarity, label, confidence
    except Exception as e:
        return 0.0, "Neutral", 0.5

def analyze_news_sentiment(headlines):
    """
    Analyze sentiment from multiple news headlines
    Returns aggregate sentiment analysis
    """
    if not headlines:
        return {
            "overall_sentiment": "Neutral",
            "sentiment_score": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "confidence": 0.0
        }
    
    sentiments = []
    labels_count = {"Positive": 0, "Negative": 0, "Neutral": 0}
    
    for headline in headlines:
        polarity, label, confidence = analyze_sentiment_local(headline)
        sentiments.append(polarity)
        labels_count[label] += 1
    
    avg_sentiment = np.mean(sentiments) if sentiments else 0.0
    
    # Determine overall sentiment
    if avg_sentiment > 0.15:
        overall = "Bullish"
    elif avg_sentiment < -0.15:
        overall = "Bearish"
    else:
        overall = "Neutral"
    
    return {
        "overall_sentiment": overall,
        "sentiment_score": avg_sentiment,
        "positive_count": labels_count["Positive"],
        "negative_count": labels_count["Negative"],
        "neutral_count": labels_count["Neutral"],
        "confidence": abs(avg_sentiment)
    }

# ---------------------------
# Local AI Analysis (Rule-Based)
# ---------------------------
def generate_local_analysis(ticker, data, analysis_type="stock"):
    """
    Generate analysis using rule-based logic (no API needed)
    """
    try:
        current_price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        ema20 = data['EMA20'].iloc[-1]
        ema50 = data['EMA50'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        signal = data['Signal'].iloc[-1]
        
        # Price momentum
        price_change_30d = ((current_price - data['Close'].iloc[-30]) / data['Close'].iloc[-30]) * 100
        
        analysis = f"""
## Technical Analysis for {ticker}

### Current Market Position
- **Price**: ${current_price:.2f}
- **30-Day Change**: {price_change_30d:+.2f}%

### Technical Indicators
- **RSI (14)**: {rsi:.2f}
  - {' Overbought territory - potential sell signal' if rsi > 70 else ' Oversold territory - potential buy opportunity' if rsi < 30 else 'Neutral range'}
  
- **Moving Averages**:
  - EMA(20): ${ema20:.2f}
  - EMA(50): ${ema50:.2f}
  - Status: {'Bullish crossover' if ema20 > ema50 else 'Bearish crossover'}

- **MACD**: {macd:.4f}
  - Signal Line: {signal:.4f}
  - Momentum: {'Positive' if macd > signal else 'Negative'}

### Trading Signal
"""
        
        # Generate signal
        if ema20 > ema50 and rsi < 70 and macd > signal:
            analysis += "**BUY Signal** - Multiple bullish indicators align\n"
            analysis += "- Uptrend confirmed by moving averages\n"
            analysis += "- MACD shows positive momentum\n"
            analysis += "- RSI not yet overbought\n"
        elif ema20 < ema50 and rsi > 30 and macd < signal:
            analysis += "**SELL Signal** - Bearish trend developing\n"
            analysis += "- Downtrend confirmed by moving averages\n"
            analysis += "- MACD shows negative momentum\n"
            analysis += "- Consider taking profits or setting stop losses\n"
        else:
            analysis += "**HOLD Signal** - Mixed signals, wait for clearer trend\n"
            analysis += "- Consolidation phase\n"
            analysis += "- Monitor for breakout signals\n"
        
        # Risk assessment
        volatility = data['Close'].pct_change().std() * np.sqrt(252)
        analysis += f"\n### Risk Assessment\n"
        analysis += f"- **Volatility**: {volatility*100:.2f}%\n"
        
        if volatility > 0.4:
            analysis += "- High volatility - suitable for risk-tolerant traders\n"
        elif volatility > 0.25:
            analysis += "- Moderate volatility - balanced risk profile\n"
        else:
            analysis += "- Low volatility - conservative investment\n"
        
        analysis += "\n### Key Levels to Watch\n"
        recent_high = data['High'].tail(30).max()
        recent_low = data['Low'].tail(30).min()
        analysis += f"- **Resistance**: ${recent_high:.2f}\n"
        analysis += f"- **Support**: ${recent_low:.2f}\n"
        
        analysis += "\n---\n*Analysis generated using technical indicators and historical data*"
        
        return analysis
        
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def generate_comparison_analysis(comparison_data):
    """Generate comparison analysis without API"""
    if not comparison_data:
        return "No data available for comparison"
    
    analysis = "## Stock Comparison Analysis\n\n"
    
    # Find best performer
    best_return = max(comparison_data, key=lambda x: float(x.get('Return', '0%').replace('%', '').replace('+', '')))
    lowest_risk = min(comparison_data, key=lambda x: int(x.get('Risk Score', '50/100').split('/')[0]))
    
    analysis += f"### Performance Leaders\n"
    analysis += f"- **Best Return**: {best_return['Ticker']}\n"
    analysis += f"- **Lowest Risk**: {lowest_risk['Ticker']}\n\n"
    
    analysis += "### Individual Analysis\n\n"
    
    for stock in comparison_data:
        ticker = stock['Ticker']
        signal = stock['Signal']
        risk = stock['Risk Score']
        
        analysis += f"**{ticker}**\n"
        analysis += f"- Signal: {signal}\n"
        analysis += f"- Risk Level: {risk}\n"
        
        if signal == "BUY" and int(risk.split('/')[0]) < 50:
            analysis += f"- Recommendation: Strong Buy candidate with favorable risk/reward\n"
        elif signal == "SELL":
            analysis += f"- Recommendation: Consider reducing exposure\n"
        else:
            analysis += f"- Recommendation: Hold and monitor\n"
        
        analysis += "\n"
    
    analysis += "---\n*Comparison based on technical analysis and risk metrics*"
    
    return analysis

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
# Helper Functions with Rate Limiting
# ---------------------------
@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_current_price(ticker):
    """Fetch current price for a ticker"""
    try:
        time.sleep(0.5)  # Rate limiting
        t = yf.Ticker(ticker)
        data = t.history(period="1d")
        if data is not None and not data.empty:
            return float(data["Close"].iloc[-1])
        return None
    except Exception as e:
        if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
            st.warning(f"Rate limited for {ticker}. Using cached data or try again in 1 minute.")
        else:
            st.error(f"Error fetching price for {ticker}: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_history(ticker, period="3mo", interval="1d"):
    """Fetch stock history with caching and rate limiting"""
    try:
        time.sleep(0.5)  # Rate limiting
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        return hist
    except Exception as e:
        if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
            st.error(" Yahoo Finance rate limit reached. Please wait 60 seconds and try again.")
            st.info("**Tip**: Yahoo Finance limits requests. Try:\n- Waiting 1-2 minutes between analyses\n- Using different tickers\n- Refreshing the page")
        else:
            st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_stock_info(ticker):
    """Fetch stock info with caching"""
    try:
        time.sleep(0.3)  # Rate limiting
        t = yf.Ticker(ticker)
        return getattr(t, "info", {}) or {}
    except Exception as e:
        return {}

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

def generate_signal(df):
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

def calculate_risk_score(df):
    """Calculate quantitative risk score (0-100)"""
    try:
        if df is None or df.empty or len(df) < 30:
            return 50, "Insufficient data", {}
        
        risk_components = {}
        
        # 1. Volatility Risk (30 points)
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        vol_risk = min(volatility * 100, 30)
        risk_components['Volatility'] = round(vol_risk, 1)
        
        # 2. RSI Risk (20 points)
        current_rsi = df['RSI'].iloc[-1]
        if pd.isna(current_rsi):
            rsi_risk = 10
        elif current_rsi > 70:
            rsi_risk = 20 * (current_rsi - 70) / 30
        elif current_rsi < 30:
            rsi_risk = 20 * (30 - current_rsi) / 30
        else:
            rsi_risk = 5
        risk_components['RSI'] = round(rsi_risk, 1)
        
        # 3. Price Momentum Risk (20 points)
        recent_avg = df['Close'].tail(5).mean()
        month_avg = df['Close'].tail(30).mean()
        momentum_change = (recent_avg - month_avg) / month_avg
        
        if momentum_change < -0.05:
            momentum_risk = 20
        elif momentum_change < 0:
            momentum_risk = 10
        elif momentum_change > 0.05:
            momentum_risk = 5
        else:
            momentum_risk = 8
        risk_components['Momentum'] = round(momentum_risk, 1)
        
        # 4. MACD Risk (15 points)
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['Signal'].iloc[-1]
        
        if pd.isna(current_macd) or pd.isna(current_signal):
            macd_risk = 7.5
        else:
            macd_diff = current_macd - current_signal
            if macd_diff < 0:
                macd_risk = 15
            elif abs(macd_diff) < 0.5:
                macd_risk = 10
            else:
                macd_risk = 5
        risk_components['MACD'] = round(macd_risk, 1)
        
        # 5. Volume Risk (15 points)
        if 'Volume' in df.columns:
            recent_vol = df['Volume'].tail(5).mean()
            avg_vol = df['Volume'].tail(30).mean()
            
            if avg_vol > 0:
                vol_ratio = recent_vol / avg_vol
                if vol_ratio < 0.7:
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
        
        total_risk = sum(risk_components.values())
        
        if total_risk < 30:
            risk_level = "Low Risk"
        elif total_risk < 50:
            risk_level = "Moderate Risk"
        elif total_risk < 70:
            risk_level = "High Risk"
        else:
            risk_level = "Very High Risk"
        
        return round(total_risk, 1), risk_level, risk_components
        
    except Exception as e:
        return 50, "Calculation Error", {"Error": str(e)}

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="Professional Investment Platform",
    layout="wide",
    page_icon="",
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
    st.title(" Professional Investment Platform")
    st.markdown("### Secure Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pw = st.text_input("Enter password", type="password", key="login_password")
        if st.button("Login", use_container_width=True):
            if pw == APP_PASSWORD:
                st.session_state.authenticated = True
                st.success("Login successful")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Incorrect password")
    st.stop()

# ---------------------------
# Main Application
# ---------------------------
st.markdown('<p class="main-header"> Professional Investment Platform</p>', unsafe_allow_html=True)
st.markdown("**Advanced Technical Analysis Dashboard**")

# Sidebar Navigation
with st.sidebar:
    st.markdown("### Navigation")
    pages = ["Home", "Real-Time Monitor", "Stock Analyzer", "Cryptocurrency", 
             "Portfolio Manager", "News & Sentiment", "Predictions", 
             "Research Assistant", "Market Screener", 
             "Compare Stocks", "Settings"]
    page = st.selectbox("Select Page", pages, label_visibility="collapsed")
    
    st.markdown("---")
    st.markdown("### Controls")
    refresh_auto = st.slider("Auto-refresh (sec)", 0, 60, 0, help="0 = disabled")
    
    st.markdown("---")
    st.markdown("### System Status")
    st.success("✓ Local Analysis Active")
    st.success("✓ Sentiment Analysis Ready" if TEXTBLOB_AVAILABLE else "⚠ Install TextBlob")
    st.info("No API Keys Required")
    
    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# Initialize last refresh time
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0

if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = 0

def check_rate_limit(min_seconds=5):
    """Check if enough time has passed since last request"""
    current_time = time.time()
    time_passed = current_time - st.session_state.last_analysis_time
    
    if time_passed < min_seconds:
        remaining = int(min_seconds - time_passed)
        return False, remaining
    
    st.session_state.last_analysis_time = current_time
    return True, 0

# ---------------------------
# Page: Home
# ---------------------------
if page == "Home":
    st.header("Welcome to Professional Investment Platform")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Quick Start Guide
        
        **Navigation:**
        - **Real-Time Monitor**: Live multi-ticker tracking with charts
        - **Stock Analyzer**: Deep technical analysis
        - **Cryptocurrency**: Crypto market analysis
        - **Portfolio Manager**: Track your investments
        - **News & Sentiment**: AI-powered sentiment analysis
        - **Predictions**: Technical forecasting
        - **Research Assistant**: Investment research tools
        - **Market Screener**: Find stocks by criteria
        - **Compare Stocks**: Side-by-side analysis
        - **Settings**: System configuration
        """)
    
    with col2:
        st.markdown("""
        ### Platform Features
        
        - **No API Keys Required**: All analysis runs locally
        - **Real-Time Data**: Live market data via yfinance
        - **Sentiment Analysis**: Natural language processing
        - **Technical Indicators**: RSI, MACD, EMA, and more
        - **Risk Assessment**: Quantitative risk scoring
        - **Portfolio Tracking**: Persistent local database
        - **Professional Grade**: Rule-based analysis engine
        
        ### Supported Markets
        - 🇺🇸 US Stocks (e.g., AAPL, MSFT)
        - 🇮🇳 Indian Stocks (e.g., TCS.NS, INFY.NS)
        -  Cryptocurrencies (e.g., BTC-USD, ETH-USD)
        """)
    
    st.info("**Note**: All data is processed locally. Your portfolio is stored in a local SQLite database.")

# ---------------------------
# Page: Real-Time
# ---------------------------
elif page == "Real-Time Monitor":
    st.header("Real-Time Multi-Ticker Monitor")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        tickers_raw = st.text_input("Enter tickers (comma-separated)", "AAPL,MSFT,GOOGL")
    with col2:
        st.write("")
        st.write("")
        refresh_now = st.button("Refresh Now", use_container_width=True)
    
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    
    should_refresh = refresh_now or (refresh_auto > 0 and (time.time() - st.session_state.last_refresh > refresh_auto))
    
    if tickers and should_refresh:
        st.session_state.last_refresh = time.time()
        
        for t in tickers:
            with st.expander(f" {t}", expanded=True):
                try:
                    ticker_obj = yf.Ticker(t)
                    intraday = ticker_obj.history(period="1d", interval="1m")
                    
                    if intraday is None or intraday.empty:
                        st.warning(f"No intraday data for {t}")
                        continue
                    
                    latest = intraday["Close"].iloc[-1]
                    prev_close = intraday["Close"].iloc[0]
                    change = latest - prev_close
                    change_pct = (change / prev_close) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${latest:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
                    col2.metric("High", f"${intraday['High'].max():.2f}")
                    col3.metric("Low", f"${intraday['Low'].min():.2f}")
                    
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
                    st.error(f"Error fetching {t}: {e}")

# ---------------------------
# Page: Stock Analyzer
# ---------------------------
elif page == "Stock Analyzer":
    st.header(" Stock Analyzer")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker Symbol", "AAPL").upper()
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    with col3:
        interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    
    if st.button("Analyze Stock", use_container_width=True):
        # Check rate limit
        can_proceed, wait_time = check_rate_limit(min_seconds=5)
        
        if not can_proceed:
            st.warning(f" Please wait {wait_time} seconds between analyses to avoid rate limits.")
            st.info("Yahoo Finance has strict rate limits. This cooldown helps prevent blocking.")
            st.stop()
        
        with st.spinner(f"Analyzing {ticker}..."):
            try:
                # Use cached function
                hist = get_stock_history(ticker, period=period, interval=interval)
                
                if hist is None:
                    st.error(" Could not fetch data. Please wait 60 seconds and try again.")
                    st.info("**Yahoo Finance Rate Limits**: Free tier has strict limits. Wait between requests.")
                    st.stop()
                
                if hist.empty:
                    st.error("No data available for this ticker")
                else:
                    # Price Chart
                    st.subheader(" Price Chart")
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
                    st.subheader(" Fundamentals")
                    info = get_stock_info(ticker)
                    
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
                    
                    # Technical Analysis
                    st.subheader(" Technical Analysis")
                    df = compute_indicators(hist)
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
                        st.markdown(f"### Signal: **{signal}**")
                        st.write("Based on EMA20/EMA50 crossover")
                        
                        rsi_val = df['RSI'].iloc[-1]
                        if rsi_val > 70:
                            st.warning("Overbought (RSI > 70)")
                        elif rsi_val < 30:
                            st.info("ℹ Oversold (RSI < 30)")
                    
                    # Risk Analysis
                    st.markdown("---")
                    st.subheader(" Risk Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Risk Score", f"{risk_score}/100")
                    col2.metric("Risk Level", risk_level)
                    col3.metric("Technical Signal", signal)
                    
                    st.write("**Risk Components:**")
                    risk_df = pd.DataFrame([
                        {"Factor": "Volatility (30%)", "Score": risk_breakdown.get('Volatility', 0), "Max": 30},
                        {"Factor": "RSI Extremes (20%)", "Score": risk_breakdown.get('RSI', 0), "Max": 20},
                        {"Factor": "Price Momentum (20%)", "Score": risk_breakdown.get('Momentum', 0), "Max": 20},
                        {"Factor": "MACD Signal (15%)", "Score": risk_breakdown.get('MACD', 0), "Max": 15},
                        {"Factor": "Volume Trend (15%)", "Score": risk_breakdown.get('Volume', 0), "Max": 15}
                    ])
                    
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)
                    
                    # Local AI Analysis
                    st.subheader(" AI Analysis")
                    analysis = generate_local_analysis(ticker, df)
                    st.markdown(analysis)
                    
            except Exception as e:
                error_msg = str(e)
                if "Too Many Requests" in error_msg or "rate limit" in error_msg.lower():
                    st.error(" **Rate Limit Exceeded**")
                    st.warning("""
                    Yahoo Finance has blocked too many requests. Please:
                    
                    1. **Wait 60-120 seconds** before trying again
                    2. **Refresh the page** to clear the session
                    3. **Try a different ticker** 
                    4. **Use cached data** by re-analyzing the same ticker
                    
                    **Why this happens**: Yahoo Finance free tier has strict rate limits to prevent abuse.
                    """)
                else:
                    st.error(f"Error analyzing stock: {e}")

# [Continue with remaining pages... Character limit reached. Would you like me to continue with the rest?]

# ---------------------------
# Page: News & Sentiment
# ---------------------------
elif page == "News & Sentiment":
    st.header(" News & Sentiment Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        nt = st.text_input("Ticker for News", "AAPL").upper()
    with col2:
        ncount = st.slider("Headlines", 1, 10, 5)
    
    if st.button("Fetch News", use_container_width=True):
        with st.spinner(f"Fetching news for {nt}..."):
            try:
                t = yf.Ticker(nt)
                raw_news = getattr(t, "news", []) or []
                
                if not raw_news:
                    st.warning(" Yahoo Finance did not return news for this ticker.")
                    st.info("""
                    **Why this happens:**
                    - Yahoo Finance news API is unreliable and often doesn't return data
                    - News availability varies by ticker and region
                    - Some tickers have limited news coverage
                    
                    **Try these alternatives:**
                    - Use popular tickers like: AAPL, MSFT, GOOGL, TSLA, AMZN
                    - For Indian stocks: RELIANCE.NS, TCS.NS, INFY.NS
                    - Check financial news websites directly
                    - Use the Stock Analyzer for technical analysis instead
                    """)
                    
                    # Provide sample sentiment analysis
                    st.markdown("---")
                    st.subheader(" Demo: Sentiment Analysis Capability")
                    st.write("Here's how sentiment analysis works with sample headlines:")
                    
                    sample_headlines = [
                        f"{nt} reports record quarterly earnings beating analyst expectations",
                        f"{nt} announces strategic partnership with major tech company",
                        f"Analysts remain bullish on {nt} stock prospects",
                        f"{nt} faces regulatory scrutiny in European markets",
                        f"Mixed signals for {nt} as market volatility increases"
                    ]
                    
                    sample_results = analyze_news_sentiment(sample_headlines)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Sample Sentiment", sample_results['overall_sentiment'])
                    col2.metric("Positive", sample_results['positive_count'])
                    col3.metric("Negative", sample_results['negative_count'])
                    col4.metric("Neutral", sample_results['neutral_count'])
                    
                    st.write("**Sample Headlines with Sentiment:**")
                    for headline in sample_headlines:
                        polarity, label, confidence = analyze_sentiment_local(headline)
                        if label == "Positive":
                            st.success(f"🟢 {headline} - **{label}** ({polarity:.2f})")
                        elif label == "Negative":
                            st.error(f"🔴 {headline} - **{label}** ({polarity:.2f})")
                        else:
                            st.info(f"🟡 {headline} - **{label}** ({polarity:.2f})")
                else:
                    headlines = []
                    for item in raw_news[:ncount]:
                        title = item.get("title", "")
                        link = item.get("link", "")
                        publisher = item.get("publisher", "Unknown")
                        
                        if title:
                            headlines.append({"title": title, "link": link, "publisher": publisher})
                    
                    if headlines:
                        st.subheader(f" Latest Headlines for {nt}")
                        
                        # Extract headline texts for sentiment analysis
                        headline_texts = [h['title'] for h in headlines]
                        
                        # Perform sentiment analysis
                        sentiment_results = analyze_news_sentiment(headline_texts)
                        
                        # Display sentiment summary
                        st.markdown("###  Sentiment Analysis Summary")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("Overall Sentiment", sentiment_results['overall_sentiment'])
                        col2.metric("Positive", sentiment_results['positive_count'])
                        col3.metric("Negative", sentiment_results['negative_count'])
                        col4.metric("Neutral", sentiment_results['neutral_count'])
                        
                        # Sentiment score visualization
                        sentiment_score = sentiment_results['sentiment_score']
                        st.progress(min(max((sentiment_score + 1) / 2, 0), 1))
                        
                        if sentiment_score > 0.15:
                            st.success(" Bullish sentiment detected - Positive news flow")
                        elif sentiment_score < -0.15:
                            st.error(" Bearish sentiment detected - Negative news flow")
                        else:
                            st.info(" Neutral sentiment - Mixed signals")
                        
                        st.markdown("---")
                        st.markdown("###  Individual Headlines")
                        
                        for i, h in enumerate(headlines, 1):
                            polarity, label, confidence = analyze_sentiment_local(h['title'])
                            
                            # Color code based on sentiment
                            if label == "Positive":
                                sentiment_color = "🟢"
                            elif label == "Negative":
                                sentiment_color = "🔴"
                            else:
                                sentiment_color = "🟡"
                            
                            with st.expander(f"{sentiment_color} {h['title']}", expanded=(i <= 3)):
                                st.write(f"**Publisher:** {h['publisher']}")
                                st.write(f"**Sentiment:** {label} (Score: {polarity:.2f})")
                                st.write(f"**Confidence:** {confidence:.2%}")
                                if h['link']:
                                    st.markdown(f"[Read full article]({h['link']})")
                        
                        # Detailed sentiment analysis
                        st.markdown("---")
                        st.subheader("🔍 Detailed Sentiment Analysis")
                        
                        analysis_text = f"""
## News Sentiment Report for {nt}

### Sentiment Breakdown
- **Overall Sentiment**: {sentiment_results['overall_sentiment']}
- **Sentiment Score**: {sentiment_score:.3f} (Range: -1 to +1)
- **Positive Headlines**: {sentiment_results['positive_count']}
- **Negative Headlines**: {sentiment_results['negative_count']}
- **Neutral Headlines**: {sentiment_results['neutral_count']}

### Market Implications

"""
                        
                        if sentiment_results['overall_sentiment'] == "Bullish":
                            analysis_text += """
**Bullish Outlook:**
- Positive news flow suggests favorable market perception
- Increased investor confidence likely
- Potential for upward price movement
- Consider monitoring for entry opportunities

**Key Considerations:**
- Verify if positive sentiment is backed by fundamentals
- Watch for profit-taking if sentiment becomes too optimistic
- Maintain risk management protocols
"""
                        elif sentiment_results['overall_sentiment'] == "Bearish":
                            analysis_text += """
**Bearish Outlook:**
- Negative news flow indicates potential concerns
- Investor sentiment may be declining
- Downward price pressure possible
- Exercise caution with new positions

**Key Considerations:**
- Determine if negative sentiment is temporary or fundamental
- Look for potential oversold opportunities
- Consider protective strategies for existing positions
"""
                        else:
                            analysis_text += """
**Neutral Outlook:**
- Mixed news signals indicate uncertainty
- Market may be in consolidation phase
- Wait for clearer directional signals
- Focus on other technical indicators

**Key Considerations:**
- Monitor for sentiment shifts
- Maintain balanced approach
- Consider diversification strategies
"""
                        
                        analysis_text += f"""

### Sentiment Trend
Based on the {len(headlines)} most recent headlines, the aggregate sentiment score of {sentiment_score:.3f} suggests:
"""
                        
                        if abs(sentiment_score) > 0.3:
                            analysis_text += "- **Strong conviction** in the prevailing sentiment\n"
                        elif abs(sentiment_score) > 0.15:
                            analysis_text += "- **Moderate conviction** with room for reversal\n"
                        else:
                            analysis_text += "- **Low conviction** - sentiment could shift quickly\n"
                        
                        analysis_text += "\n### Recommended Actions\n"
                        
                        if sentiment_results['positive_count'] > sentiment_results['negative_count'] * 2:
                            analysis_text += "- Consider the stock favorably for new positions\n"
                            analysis_text += "- Monitor for profit-taking opportunities\n"
                        elif sentiment_results['negative_count'] > sentiment_results['positive_count'] * 2:
                            analysis_text += "- Exercise caution with new investments\n"
                            analysis_text += "- Review existing positions for risk\n"
                        else:
                            analysis_text += "- Maintain watchlist status\n"
                            analysis_text += "- Wait for clearer signals before acting\n"
                        
                        analysis_text += "\n---\n*Analysis based on natural language processing of news headlines*"
                        
                        st.markdown(analysis_text)
                    else:
                        st.warning("Could not extract headlines from news data")
                        
            except Exception as e:
                st.error(f"Error fetching news: {e}")

# ---------------------------
# Page: Cryptocurrency
# ---------------------------
elif page == "Cryptocurrency":
    st.header(" Cryptocurrency Analyzer")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        crypto = st.text_input("Crypto Ticker", "BTC-USD").upper()
    with col2:
        c_period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo"], index=2)
    
    if st.button("Analyze Crypto", use_container_width=True):
        with st.spinner(f"Analyzing {crypto}..."):
            try:
                crypto_obj = yf.Ticker(crypto)
                interval = "1m" if c_period == "1d" else ("15m" if c_period == "5d" else "1h")
                ch = crypto_obj.history(period=c_period, interval=interval)
                
                if ch is None or ch.empty:
                    st.error("No data available for this crypto ticker")
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
                    
                    # Technical Analysis
                    st.subheader(" Technical Analysis")
                    ch_indicators = compute_indicators(ch)
                    
                    if len(ch_indicators) > 50:
                        crypto_analysis = f"""
## Cryptocurrency Analysis: {crypto}

### Current Market Status
- **Price**: ${latest:,.2f}
- **Period Range**: ${low:,.2f} - ${high:,.2f}
- **Volatility**: {((high-low)/low*100):.2f}%

### Technical Indicators
"""
                        
                        if 'RSI' in ch_indicators.columns:
                            rsi_val = ch_indicators['RSI'].iloc[-1]
                            crypto_analysis += f"- **RSI**: {rsi_val:.2f}\n"
                            if rsi_val > 70:
                                crypto_analysis += "  - Overbought condition - potential correction ahead\n"
                            elif rsi_val < 30:
                                crypto_analysis += "  - Oversold condition - potential bounce opportunity\n"
                            else:
                                crypto_analysis += "  - Neutral range\n"
                        
                        # Price momentum
                        returns = ch['Close'].pct_change().dropna()
                        volatility = returns.std() * np.sqrt(24)  # Hourly volatility
                        
                        crypto_analysis += f"\n### Risk Assessment\n"
                        crypto_analysis += f"- **Volatility**: {volatility*100:.2f}%\n"
                        
                        if volatility > 0.5:
                            crypto_analysis += "- **Risk Level**: Very High - Extreme price swings expected\n"
                            crypto_analysis += "- **Recommendation**: Only for experienced traders with high risk tolerance\n"
                        elif volatility > 0.3:
                            crypto_analysis += "- **Risk Level**: High - Significant price movements likely\n"
                            crypto_analysis += "- **Recommendation**: Use tight stop losses and position sizing\n"
                        else:
                            crypto_analysis += "- **Risk Level**: Moderate - Standard crypto volatility\n"
                            crypto_analysis += "- **Recommendation**: Normal trading strategies apply\n"
                        
                        # Support and resistance
                        recent_high = ch['High'].tail(20).max()
                        recent_low = ch['Low'].tail(20).min()
                        
                        crypto_analysis += f"\n### Key Levels\n"
                        crypto_analysis += f"- **Resistance**: ${recent_high:,.2f}\n"
                        crypto_analysis += f"- **Support**: ${recent_low:,.2f}\n"
                        crypto_analysis += f"- **Current Position**: {((latest-recent_low)/(recent_high-recent_low)*100):.1f}% of range\n"
                        
                        crypto_analysis += "\n### Trading Considerations\n"
                        crypto_analysis += "- Monitor Bitcoin dominance for market correlation\n"
                        crypto_analysis += "- Watch for regulatory news affecting crypto markets\n"
                        crypto_analysis += "- Consider market sentiment and social media trends\n"
                        crypto_analysis += "- Use appropriate position sizing for volatile assets\n"
                        
                        crypto_analysis += "\n---\n*Crypto analysis based on technical indicators and price action*"
                        
                        st.markdown(crypto_analysis)
                    else:
                        st.info("Insufficient data for detailed technical analysis")
                    
            except Exception as e:
                st.error(f"Error analyzing crypto: {e}")

# ---------------------------
# Page: Portfolio Manager
# ---------------------------
elif page == "Portfolio Manager":
    st.header(" Portfolio Management")
    
    st.markdown("### Add New Position")
    
    with st.form("add_position", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ticker = st.text_input("Ticker", "AAPL")
        with col2:
            qty = st.number_input("Quantity", min_value=0.01, value=1.0, step=1.0)
        with col3:
            avg = st.number_input("Avg Price", min_value=0.0, value=0.0, step=0.01,
                                 help="0 = fetch current price")
        
        submitted = st.form_submit_button("Add Position", use_container_width=True)
        
        if submitted:
            ticker = ticker.upper().strip()
            if not ticker:
                st.error("Please enter a ticker symbol")
            else:
                if avg == 0.0:
                    with st.spinner(f"Fetching current price for {ticker}..."):
                        cur = get_current_price(ticker)
                    if cur is None:
                        st.error("Could not fetch current price. Please enter average price manually")
                    else:
                        add_position_db(ticker, qty, cur)
                        st.success(f"Added {qty} x {ticker} @ ${cur:.2f}")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    add_position_db(ticker, qty, avg)
                    st.success(f"Added {qty} x {ticker} @ ${avg:.2f}")
                    time.sleep(0.5)
                    st.rerun()
    
    st.markdown("---")
    st.markdown("### Current Holdings")
    
    df = list_positions_db()
    
    if df.empty:
        st.info("No positions yet. Add your first position above")
    else:
        rows = []
        total_value = 0
        total_pl = 0
        
        with st.spinner("Fetching live prices..."):
            for idx, r in df.iterrows():
                cur = get_current_price(r['ticker'])
                if cur is None:
                    cur = r['avg_price']
                
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
        st.markdown("### Remove Position")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            rem_id = st.number_input("Enter Position ID", min_value=1, value=1, step=1)
        with col2:
            st.write("")
            st.write("")
            if st.button("Remove", use_container_width=True):
                remove_position_db(rem_id)
                st.success(f"Position {rem_id} removed")
                time.sleep(0.5)
                st.rerun()

# ---------------------------
# Page: Predictions
# ---------------------------
elif page == "Predictions":
    st.header(" Technical Price Predictions")
    
    st.info("**Disclaimer**: These predictions are based on technical analysis and historical patterns. Markets are unpredictable. Always do your own research.")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        pt = st.text_input("Ticker", "AAPL").upper()
    with col2:
        days = st.slider("Days Ahead", 1, 30, 7)
    
    if st.button("Generate Prediction", use_container_width=True):
        with st.spinner(f"Generating prediction for {pt}..."):
            try:
                ticker_obj = yf.Ticker(pt)
                hist = ticker_obj.history(period="6mo")
                
                if hist is None or hist.empty:
                    st.error("Not enough historical data for prediction")
                else:
                    # Display recent price action
                    st.subheader(" Recent Price History")
                    
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
                    
                    # Calculate statistics
                    recent_closes = hist["Close"].tail(30).tolist()
                    current_price = recent_closes[-1]
                    avg_30d = np.mean(recent_closes)
                    volatility = np.std(recent_closes)
                    returns = hist["Close"].pct_change().dropna()
                    avg_daily_return = returns.mean()
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${current_price:.2f}")
                    col2.metric("30-Day Avg", f"${avg_30d:.2f}")
                    col3.metric("Volatility", f"${volatility:.2f}")
                    
                    # Technical prediction
                    st.markdown("---")
                    st.subheader(" Technical Forecast")
                    
                    # Simple momentum-based prediction
                    predicted_price = current_price * (1 + avg_daily_return * days)
                    upper_bound = predicted_price + (volatility * np.sqrt(days) * 2)
                    lower_bound = predicted_price - (volatility * np.sqrt(days) * 2)
                    
                    # Compute technical indicators
                    df_pred = compute_indicators(hist)
                    signal = generate_signal(df_pred)
                    rsi = df_pred['RSI'].iloc[-1]
                    
                    prediction_text = f"""
## {days}-Day Price Forecast for {pt}

### Predicted Price Range
- **Most Likely Price**: ${predicted_price:.2f}
- **Upper Bound (95% confidence)**: ${upper_bound:.2f}
- **Lower Bound (95% confidence)**: ${lower_bound:.2f}
- **Expected Change**: {((predicted_price - current_price) / current_price * 100):+.2f}%

### Prediction Methodology
This forecast uses:
- Historical volatility: ${volatility:.2f}
- Average daily return: {avg_daily_return*100:.3f}%
- Current momentum indicators
- Technical signal: {signal}

### Probability Analysis
"""
                    
                    # Probability assessment
                    if signal == "BUY" and rsi < 70:
                        prediction_text += "- **Upside Probability**: 65-70%\n"
                        prediction_text += "- **Reasoning**: Bullish technical setup with room to run\n"
                    elif signal == "SELL" and rsi > 30:
                        prediction_text += "- **Downside Probability**: 65-70%\n"
                        prediction_text += "- **Reasoning**: Bearish technical setup suggests weakness\n"
                    else:
                        prediction_text += "- **Probability**: 50-55% (Neutral)\n"
                        prediction_text += "- **Reasoning**: Mixed signals, direction unclear\n"
                    
                    prediction_text += f"\n### Key Factors to Monitor\n"
                    
                    # Support/Resistance
                    resistance = hist['High'].tail(30).max()
                    support = hist['Low'].tail(30).min()
                    
                    prediction_text += f"- **Resistance Level**: ${resistance:.2f}\n"
                    prediction_text += f"- **Support Level**: ${support:.2f}\n"
                    
                    if predicted_price > resistance:
                        prediction_text += "- Price prediction above resistance - breakout scenario\n"
                    elif predicted_price < support:
                        prediction_text += "- Price prediction below support - breakdown scenario\n"
                    
                    prediction_text += "\n### Risk Considerations\n"
                    prediction_text += f"- Historical volatility suggests {volatility/current_price*100:.1f}% daily price swings\n"
                    prediction_text += "- Predictions become less reliable beyond 7-10 days\n"
                    prediction_text += "- External factors (news, earnings, macro events) can invalidate technical predictions\n"
                    prediction_text += "- Always use stop-loss orders to manage risk\n"
                    
                    prediction_text += "\n### Recommended Strategy\n"
                    
                    if signal == "BUY":
                        prediction_text += "- **Entry**: Consider positions near support levels\n"
                        prediction_text += f"- **Stop Loss**: ${support * 0.97:.2f} (3% below support)\n"
                        prediction_text += f"- **Target**: ${predicted_price:.2f}\n"
                    elif signal == "SELL":
                        prediction_text += "- **Action**: Consider reducing exposure or hedging\n"
                        prediction_text += "- **Watch**: Breakdown below support could accelerate losses\n"
                    else:
                        prediction_text += "- **Action**: Wait for clearer directional signals\n"
                        prediction_text += "- **Monitor**: Key technical levels for breakout\n"
                    
                    prediction_text += "\n---\n"
                    prediction_text += "**Important**: This is a probabilistic forecast based on technical analysis. "
                    prediction_text += "Markets can move differently due to unexpected events, news, or changes in fundamentals. "
                    prediction_text += "Never invest based solely on predictions. Always conduct comprehensive research."
                    
                    st.markdown(prediction_text)
                    
                    # Visualization
                    st.subheader(" Price Projection Visualization")
                    
                    future_dates = pd.date_range(start=hist.index[-1], periods=days+1, freq='D')[1:]
                    
                    fig_pred = go.Figure()
                    
                    # Historical prices
                    fig_pred.add_trace(go.Scatter(
                        x=hist.index[-30:],
                        y=hist["Close"].tail(30),
                        mode="lines",
                        name="Historical",
                        line=dict(color="blue", width=2)
                    ))
                    
                    # Predicted price
                    pred_line = [current_price] + [predicted_price] * days
                    dates_pred = [hist.index[-1]] + list(future_dates)
                    
                    fig_pred.add_trace(go.Scatter(
                        x=dates_pred,
                        y=pred_line,
                        mode="lines",
                        name="Prediction",
                        line=dict(color="green", width=2, dash="dash")
                    ))
                    
                    # Confidence intervals
                    upper_line = [current_price] + [upper_bound] * days
                    lower_line = [current_price] + [lower_bound] * days
                    
                    fig_pred.add_trace(go.Scatter(
                        x=dates_pred,
                        y=upper_line,
                        mode="lines",
                        name="Upper Bound",
                        line=dict(color="red", width=1, dash="dot")
                    ))
                    
                    fig_pred.add_trace(go.Scatter(
                        x=dates_pred,
                        y=lower_line,
                        mode="lines",
                        name="Lower Bound",
                        line=dict(color="red", width=1, dash="dot"),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)'
                    ))
                    
                    fig_pred.update_layout(
                        template="plotly_dark",
                        height=450,
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    st.warning("**Risk Warning**: Past performance does not guarantee future results. This prediction is for educational purposes only.")
                    
            except Exception as e:
                st.error(f"Error generating prediction: {e}")

# ---------------------------
# Page: Research Assistant
# ---------------------------
elif page == "Research Assistant":
    st.header(" Investment Research Assistant")
    st.markdown("Analyze stocks with comprehensive technical and fundamental research")
    
    # Quick action buttons
    st.markdown("### Quick Analysis Templates")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Top Momentum Stocks", use_container_width=True):
            st.session_state.research_query = "Analyze stocks showing strong momentum with RSI between 50-70"
    with col2:
        if st.button("Oversold Opportunities", use_container_width=True):
            st.session_state.research_query = "Find oversold stocks with RSI < 30 that might rebound"
    with col3:
        if st.button("Low Risk Stocks", use_container_width=True):
            st.session_state.research_query = "Identify low volatility stocks with stable trends"
    
    st.markdown("---")
    
    # Research query
    if "research_query" not in st.session_state:
        st.session_state.research_query = ""
    
    tickers_to_research = st.text_input(
        "Enter tickers to research (comma-separated)",
        placeholder="e.g., AAPL,MSFT,GOOGL",
        help="Enter specific tickers for detailed analysis"
    )
    
    if st.button("Conduct Research", type="primary", use_container_width=True):
        if not tickers_to_research.strip():
            st.error("Please enter at least one ticker")
        else:
            tickers_list = [t.strip().upper() for t in tickers_to_research.split(",") if t.strip()]
            
            st.info(f"Researching {len(tickers_list)} stock(s)...")
            
            research_results = []
            
            for ticker in tickers_list:
                with st.expander(f" Analysis: {ticker}", expanded=True):
                    try:
                        hist = get_stock_history(ticker, period="6mo")
                        
                        if hist is not None and not hist.empty and len(hist) > 50:
                            df = compute_indicators(hist)
                            risk_score, risk_level, risk_breakdown = calculate_risk_score(df)
                            signal = generate_signal(df)
                            
                            # Display metrics
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
                            col2.metric("Signal", signal)
                            col3.metric("Risk", f"{risk_score}/100")
                            col4.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                            
                            # Generate analysis
                            analysis = generate_local_analysis(ticker, df)
                            st.markdown(analysis)
                            
                            research_results.append({
                                "ticker": ticker,
                                "signal": signal,
                                "risk": risk_score,
                                "price": df['Close'].iloc[-1]
                            })
                        else:
                            st.warning(f"Insufficient data for {ticker}")
                    except Exception as e:
                        st.error(f"Error researching {ticker}: {e}")
            
            # Summary
            if research_results:
                st.markdown("---")
                st.subheader(" Research Summary")
                
                buy_signals = [r for r in research_results if r['signal'] == 'BUY']
                low_risk = [r for r in research_results if r['risk'] < 40]
                
                summary_text = f"""
### Portfolio Recommendations

**Total Stocks Analyzed**: {len(research_results)}

**Buy Signals**: {len(buy_signals)} stocks
"""
                
                if buy_signals:
                    summary_text += "- " + ", ".join([r['ticker'] for r in buy_signals]) + "\n"
                
                summary_text += f"\n**Low Risk Opportunities**: {len(low_risk)} stocks\n"
                
                if low_risk:
                    summary_text += "- " + ", ".join([r['ticker'] for r in low_risk]) + "\n"
                
                summary_text += "\n### Investment Strategy Recommendations\n\n"
                
                if buy_signals and low_risk:
                    overlap = [r['ticker'] for r in buy_signals if r in low_risk]
                    if overlap:
                        summary_text += f"**Priority Targets**: {', '.join(overlap)} - Both low risk and buy signals\n\n"
                
                summary_text += "### Next Steps\n"
                summary_text += "1. Review individual stock analysis above\n"
                summary_text += "2. Verify technical signals with fundamental research\n"
                summary_text += "3. Consider position sizing based on risk scores\n"
                summary_text += "4. Set appropriate stop-loss levels\n"
                summary_text += "5. Monitor regularly for signal changes\n"
                
                st.markdown(summary_text)

# ---------------------------
# Page: Market Screener
# ---------------------------
elif page == "Market Screener":
    st.header("🔎 Market Screener")
    st.markdown("Find stocks matching your investment criteria")
    
    st.markdown("### Screening Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Technical Filters**")
        rsi_min = st.slider("Min RSI", 0, 100, 30)
        rsi_max = st.slider("Max RSI", 0, 100, 70)
        
        signal_filter = st.multiselect(
            "Trading Signal",
            ["BUY", "SELL", "HOLD"],
            default=["BUY"]
        )
    
    with col2:
        st.markdown("**Risk Filters**")
        risk_max = st.slider("Max Risk Score", 0, 100, 60)
    
    # Stock universe
    st.markdown("### Stock Universe")
    
    preset = st.radio(
        "Choose preset or custom:",
        ["US Top 30", "Indian Nifty 20", "Custom List"],
        horizontal=True
    )
    
    if preset == "US Top 30":
        stock_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "JPM", "JNJ",
                         "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "NFLX", "ADBE", "CRM",
                         "INTC", "CSCO", "PFE", "KO", "PEP", "ABT", "TMO", "COST", "AVGO", "ACN"]
    elif preset == "Indian Nifty 20":
        stock_universe = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
                         "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
                         "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
                         "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "NESTLEIND.NS"]
    else:
        custom_input = st.text_area(
            "Enter tickers (comma-separated)",
            "AAPL,MSFT,GOOGL,TSLA"
        )
        stock_universe = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    
    st.info(f"Will screen {len(stock_universe)} stocks")
    
    if st.button("Run Screener", type="primary", use_container_width=True):
        with st.spinner(f"Screening {len(stock_universe)} stocks..."):
            results = []
            progress_bar = st.progress(0)
            
            for idx, ticker in enumerate(stock_universe):
                try:
                    hist = get_stock_history(ticker, period="3mo")
                    
                    if hist is not None and not hist.empty and len(hist) > 50:
                        df = compute_indicators(hist)
                        risk_score, risk_level, _ = calculate_risk_score(df)
                        signal = generate_signal(df)
                        
                        current_rsi = df['RSI'].iloc[-1]
                        current_price = df['Close'].iloc[-1]
                        
                        if (rsi_min <= current_rsi <= rsi_max and 
                            signal in signal_filter and 
                            risk_score <= risk_max):
                            
                            info = get_stock_info(ticker)
                            
                            results.append({
                                "Ticker": ticker,
                                "Price": f"${current_price:.2f}",
                                "Signal": signal,
                                "RSI": f"{current_rsi:.1f}",
                                "Risk": f"{risk_score:.0f}",
                                "Risk Level": risk_level,
                                "Sector": info.get("sector", "N/A")
                            })
                    
                    progress_bar.progress((idx + 1) / len(stock_universe))
                    time.sleep(0.3)  # Rate limiting
                    
                except Exception:
                    continue
            
            progress_bar.empty()
            
            if results:
                st.success(f"✓ Found {len(results)} stocks matching criteria")
                
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values("Risk", ascending=True)
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # Summary analysis
                st.markdown("---")
                st.subheader(" Screening Results Analysis")
                
                summary = f"""
### Screening Summary

**Total Matches**: {len(results)} stocks out of {len(stock_universe)} screened

**Filter Criteria Applied**:
- RSI Range: {rsi_min} - {rsi_max}
- Signals: {', '.join(signal_filter)}
- Maximum Risk Score: {risk_max}

**Top Picks** (Lowest Risk):
"""
                
                top_5 = results_df.head(5)
                for idx, row in top_5.iterrows():
                    summary += f"\n{idx+1}. **{row['Ticker']}** - Price: {row['Price']}, Risk: {row['Risk']}, Signal: {row['Signal']}"
                
                summary += "\n\n### Sector Distribution\n"
                sector_counts = results_df['Sector'].value_counts()
                for sector, count in sector_counts.items():
                    summary += f"- {sector}: {count} stocks\n"
                
                summary += "\n### Investment Considerations\n"
                summary += "- Diversify across multiple sectors\n"
                summary += "- Consider position sizing based on risk scores\n"
                summary += "- Verify fundamentals before investing\n"
                summary += "- Set stop-losses according to volatility\n"
                
                st.markdown(summary)
                
            else:
                st.warning("No stocks found matching your criteria. Try adjusting filters.")

# ---------------------------
# Page: Compare Stocks
# ---------------------------
elif page == "Compare Stocks":
    st.header(" Side-by-Side Stock Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ticker1 = st.text_input("Stock 1", "AAPL", key="comp1").upper()
    with col2:
        ticker2 = st.text_input("Stock 2", "MSFT", key="comp2").upper()
    
    col3, col4 = st.columns(2)
    with col3:
        ticker3 = st.text_input("Stock 3 (optional)", "", key="comp3").upper()
    with col4:
        ticker4 = st.text_input("Stock 4 (optional)", "", key="comp4").upper()
    
    compare_period = st.selectbox("Comparison Period", ["1mo", "3mo", "6mo", "1y"], index=1)
    
    if st.button("Compare Stocks", type="primary", use_container_width=True):
        tickers = [t for t in [ticker1, ticker2, ticker3, ticker4] if t]
        
        if len(tickers) < 2:
            st.error("Please enter at least 2 tickers to compare")
        else:
            comparison_data = []
            price_history = {}
            
            with st.spinner(f"Analyzing {len(tickers)} stocks..."):
                for ticker in tickers:
                    try:
                        hist = get_stock_history(ticker, period=compare_period)
                        info = get_stock_info(ticker)
                        
                        if hist is not None and not hist.empty:
                            df = compute_indicators(hist)
                            risk_score, risk_level, _ = calculate_risk_score(df)
                            signal = generate_signal(df)
                            
                            start_price = hist['Close'].iloc[0]
                            end_price = hist['Close'].iloc[-1]
                            returns = ((end_price - start_price) / start_price) * 100
                            
                            price_history[ticker] = hist['Close']
                            
                            comparison_data.append({
                                "Ticker": ticker,
                                "Price": f"${end_price:.2f}",
                                "Return": f"{returns:+.2f}%",
                                "Signal": signal,
                                "RSI": f"{df['RSI'].iloc[-1]:.1f}",
                                "Risk Score": f"{risk_score:.0f}/100",
                                "Risk Level": risk_level,
                                "P/E Ratio": info.get("trailingPE", "N/A"),
                                "Sector": info.get("sector", "N/A")
                            })
                        time.sleep(0.5)  # Rate limiting
                    except Exception as e:
                        st.error(f"Error analyzing {ticker}: {e}")
            
            if comparison_data:
                # Comparison table
                st.subheader(" Comparison Table")
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Price comparison chart
                st.subheader(" Performance Comparison")
                
                if price_history:
                    fig = go.Figure()
                    
                    for ticker, prices in price_history.items():
                        normalized = (prices / prices.iloc[0]) * 100
                        fig.add_trace(go.Scatter(
                            x=normalized.index,
                            y=normalized,
                            mode="lines",
                            name=ticker,
                            line=dict(width=3)
                        ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=450,
                        xaxis_title="Date",
                        yaxis_title="Normalized Price (Base = 100)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Comparison analysis
                st.markdown("---")
                st.subheader(" Comparison Analysis")
                
                analysis = generate_comparison_analysis(comparison_data)
                st.markdown(analysis)

# ---------------------------
# Page: Settings
# ---------------------------
elif page == "Settings":
    st.header("Settings & System Information")
    
    # System Status
    st.subheader(" System Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Analysis Engine:**")
        st.success(" Local analysis active")
        st.success(" Technical indicators operational")
        st.success(" Risk scoring enabled")
        
        if TEXTBLOB_AVAILABLE:
            st.success("Sentiment analysis ready")
        else:
            st.warning(" TextBlob not installed")
            st.code("pip install textblob")
    
    with col2:
        st.write("**Data Sources:**")
        st.info(" Yahoo Finance API")
        st.info(" Local SQLite database")
        st.info(" No external API keys required")
    
    st.markdown("---")
    
    # Database Management
    st.subheader(" Database Management")
    
    df = list_positions_db()
    st.write(f"**Current Positions:** {len(df)}")
    st.write(f"**Database Path:** `{DB_PATH}`")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear All Positions", type="primary"):
            c = conn.cursor()
            c.execute("DELETE FROM portfolio")
            conn.commit()
            st.success(" Portfolio cleared successfully")
            time.sleep(1)
            st.rerun()
    
    with col2:
        if st.button("Show Database Stats"):
            c = conn.cursor()
            c.execute("SELECT COUNT(*) as count, SUM(qty) as total_qty FROM portfolio")
            stats = c.fetchone()
            st.write(f"- Total Positions: {stats[0]}")
            st.write(f"- Total Shares: {stats[1]}")
    
    st.markdown("---")
    
    # System Information
    st.subheader(" System Information")
    
    st.write("**Required Packages:**")
    st.code("streamlit, yfinance, pandas, numpy, plotly, textblob")
    
    st.write(f"**Database:** {DB_PATH}")
    st.write(f"**Authenticated:** {st.session_state.authenticated}")
    st.write(f"**Auto-refresh:** {'Disabled' if refresh_auto == 0 else f'{refresh_auto}s'}")
    
    st.markdown("---")
    
    # API Testing
    st.subheader(" System Testing")
    
    test_ticker = st.text_input("Test Ticker", "AAPL")
    
    if st.button("Test Data Fetch"):
        with st.spinner(f"Testing {test_ticker}..."):
            try:
                hist = get_stock_history(test_ticker, period="1d")
                if hist is not None and not hist.empty:
                    st.success(" Data fetch successful")
                    st.write("Latest data:")
                    st.dataframe(hist.tail(1))
                else:
                    st.error(" No data returned")
            except Exception as e:
                st.error(f" Error: {e}")
    
    if st.button("Test Sentiment Analysis"):
        if TEXTBLOB_AVAILABLE:
            test_text = "Apple announces record quarterly earnings with strong iPhone sales"
            polarity, label, confidence = analyze_sentiment_local(test_text)
            st.success(" Sentiment analysis operational")
            st.write(f"Test Text: {test_text}")
            st.write(f"Sentiment: {label} (Score: {polarity:.2f}, Confidence: {confidence:.2%})")
        else:
            st.error(" TextBlob not available")  