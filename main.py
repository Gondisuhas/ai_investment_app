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
    if not TEXTBLOB_AVAILABLE or not text:
        return 0.0, "Neutral", 0.5
    try:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
        confidence = 1 - subjectivity
        return polarity, label, confidence
    except Exception:
        return 0.0, "Neutral", 0.5

def analyze_news_sentiment(headlines):
    if not headlines:
        return {"overall_sentiment": "Neutral", "sentiment_score": 0.0,
                "positive_count": 0, "negative_count": 0, "neutral_count": 0, "confidence": 0.0}
    sentiments = []
    labels_count = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for headline in headlines:
        polarity, label, confidence = analyze_sentiment_local(headline)
        sentiments.append(polarity)
        labels_count[label] += 1
    avg_sentiment = np.mean(sentiments) if sentiments else 0.0
    if avg_sentiment > 0.15:
        overall = "Bullish"
    elif avg_sentiment < -0.15:
        overall = "Bearish"
    else:
        overall = "Neutral"
    return {"overall_sentiment": overall, "sentiment_score": avg_sentiment,
            "positive_count": labels_count["Positive"], "negative_count": labels_count["Negative"],
            "neutral_count": labels_count["Neutral"], "confidence": abs(avg_sentiment)}

# ---------------------------
# Local AI Analysis (Rule-Based)
# ---------------------------
def generate_local_analysis(ticker, data, analysis_type="stock"):
    try:
        current_price = data['Close'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        ema20 = data['EMA20'].iloc[-1]
        ema50 = data['EMA50'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        signal = data['Signal'].iloc[-1]
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
    if not comparison_data:
        return "No data available for comparison"
    analysis = "## Stock Comparison Analysis\n\n"
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
# Database (SQLite)
# ---------------------------
DB_PATH = "portfolio.db"

@st.cache_resource
def init_db():
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
    c = conn.cursor()
    c.execute("INSERT INTO portfolio (ticker, qty, avg_price) VALUES (?, ?, ?)",
              (ticker.upper(), float(qty), float(avg_price)))
    conn.commit()

def remove_position_db(row_id):
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE id = ?", (row_id,))
    conn.commit()

def list_positions_db():
    c = conn.cursor()
    c.execute("SELECT id, ticker, qty, avg_price, added_at FROM portfolio ORDER BY added_at DESC")
    rows = c.fetchall()
    cols = ["id", "ticker", "qty", "avg_price", "added_at"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data(ttl=300)
def get_current_price(ticker):
    try:
        time.sleep(0.5)
        t = yf.Ticker(ticker)
        data = t.history(period="1d")
        if data is not None and not data.empty:
            return float(data["Close"].iloc[-1])
        return None
    except Exception:
        return None

@st.cache_data(ttl=600)
def get_stock_history(ticker, period="3mo", interval="1d"):
    try:
        time.sleep(0.5)
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        return hist
    except Exception as e:
        if "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
            st.error("Yahoo Finance rate limit reached. Please wait 60 seconds.")
        else:
            st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=600)
def get_stock_info(ticker):
    try:
        time.sleep(0.3)
        t = yf.Ticker(ticker)
        return getattr(t, "info", {}) or {}
    except Exception:
        return {}

def compute_indicators(df):
    df = df.copy()
    if "Close" not in df.columns or df.empty:
        return df
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def generate_signal(df):
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
    try:
        if df is None or df.empty or len(df) < 30:
            return 50, "Insufficient data", {}
        risk_components = {}
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        vol_risk = min(volatility * 100, 30)
        risk_components['Volatility'] = round(vol_risk, 1)
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
    page_title="Apex Trading Terminal",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ---------------------------
# PROFESSIONAL DARK TERMINAL CSS
# ---------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600&family=Syne:wght@400;600;700;800&display=swap');

    /* ── Base Reset ── */
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace;
        background-color: #080c14;
        color: #c8d8f0;
    }

    .stApp {
        background: #080c14;
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ── Main container ── */
    .block-container {
        padding: 1.5rem 2.5rem;
        max-width: 1400px;
    }

    /* ── Terminal Header ── */
    .terminal-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #00d4ff 0%, #0088cc 50%, #00ff88 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0;
        line-height: 1.1;
    }

    .terminal-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #3a5a7a;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: 0.2rem;
        margin-bottom: 1.5rem;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #050810 !important;
        border-right: 1px solid #0d1f33;
    }

    [data-testid="stSidebar"] .block-container {
        padding: 1.5rem 1rem;
    }

    .sidebar-logo {
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }

    .sidebar-tagline {
        font-size: 0.6rem;
        color: #2a4a6a;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #0d1f33;
        padding-bottom: 1rem;
    }

    .sidebar-section {
        font-size: 0.6rem;
        color: #2a4a6a;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin: 1rem 0 0.5rem 0;
    }

    /* ── Selectbox & Inputs ── */
    .stSelectbox > div > div {
        background: #0a1525 !important;
        border: 1px solid #0d2040 !important;
        border-radius: 6px !important;
        color: #c8d8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    .stTextInput > div > div > input {
        background: #0a1525 !important;
        border: 1px solid #0d2040 !important;
        border-radius: 6px !important;
        color: #00d4ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 0.9rem !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.1) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #2a4a6a !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #001833, #002244) !important;
        color: #00d4ff !important;
        border: 1px solid #00d4ff !important;
        border-radius: 6px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        padding: 0.55rem 1rem !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #00d4ff, #0088cc) !important;
        color: #080c14 !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        transform: translateY(-1px) !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00d4ff, #0077bb) !important;
        color: #080c14 !important;
        font-weight: 600 !important;
    }

    /* ── Metric Cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #0a1525 0%, #080f1e 100%);
        border: 1px solid #0d2040;
        border-radius: 10px;
        padding: 1rem 1.2rem !important;
        position: relative;
        overflow: hidden;
    }

    [data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #00d4ff, #00ff88);
    }

    [data-testid="stMetric"] label {
        color: #3a6a8a !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.12em !important;
        text-transform: uppercase !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stMetricValue"] {
        color: #e8f4ff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
    }

    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.75rem !important;
    }

    /* ── Dividers ── */
    hr {
        border: none !important;
        border-top: 1px solid #0d2040 !important;
        margin: 1.5rem 0 !important;
    }

    /* ── Headers ── */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif !important;
        color: #c8d8f0 !important;
        letter-spacing: -0.01em !important;
    }

    h1 { font-size: 1.6rem !important; font-weight: 700 !important; }
    h2 { font-size: 1.25rem !important; font-weight: 600 !important; }
    h3 {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #3a8aaa !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }

    /* ── Page Section Header ── */
    .page-header {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: #e8f4ff;
        border-left: 3px solid #00d4ff;
        padding-left: 0.8rem;
        margin-bottom: 1.2rem;
        letter-spacing: -0.01em;
    }

    .page-sub {
        font-size: 0.7rem;
        color: #2a5a7a;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-top: -0.9rem;
        margin-bottom: 1.5rem;
        padding-left: 1.1rem;
    }

    /* ── Signal Badge ── */
    .signal-buy {
        display: inline-block;
        background: rgba(0, 255, 100, 0.12);
        color: #00ff64;
        border: 1px solid #00ff64;
        border-radius: 4px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
    }

    .signal-sell {
        display: inline-block;
        background: rgba(255, 60, 60, 0.12);
        color: #ff4444;
        border: 1px solid #ff4444;
        border-radius: 4px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
    }

    .signal-hold {
        display: inline-block;
        background: rgba(255, 180, 0, 0.12);
        color: #ffb400;
        border: 1px solid #ffb400;
        border-radius: 4px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Info/Warning/Success boxes ── */
    .stAlert {
        border-radius: 8px !important;
        border-left: 3px solid !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    [data-testid="stNotification"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Dataframe ── */
    .stDataFrame {
        border: 1px solid #0d2040 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    [data-testid="stDataFrame"] th {
        background: #050810 !important;
        color: #3a8aaa !important;
        font-size: 0.65rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-bottom: 1px solid #0d2040 !important;
    }

    [data-testid="stDataFrame"] td {
        background: #0a1525 !important;
        color: #c8d8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        border-bottom: 1px solid #0a1a2a !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: #0a1525 !important;
        border: 1px solid #0d2040 !important;
        border-radius: 8px !important;
        color: #c8d8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    .streamlit-expanderContent {
        background: #080c14 !important;
        border: 1px solid #0d2040 !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div > div {
        background: #00d4ff !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88) !important;
    }

    /* ── Ticker/Tag Pill ── */
    .ticker-tag {
        display: inline-block;
        background: rgba(0, 212, 255, 0.08);
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        font-family: 'JetBrains Mono', monospace;
        margin: 0 0.15rem;
    }

    /* ── Status Dots ── */
    .status-dot {
        display: inline-block;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }

    .status-dot.green { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
    .status-dot.blue  { background: #00d4ff; box-shadow: 0 0 6px #00d4ff; }
    .status-dot.red   { background: #ff4444; box-shadow: 0 0 6px #ff4444; }

    @keyframes pulse {
        0%   { opacity: 1; }
        50%  { opacity: 0.4; }
        100% { opacity: 1; }
    }

    /* ── Dashboard stat card ── */
    .stat-card {
        background: linear-gradient(135deg, #0a1525, #050d1a);
        border: 1px solid #0d2040;
        border-radius: 10px;
        padding: 1.2rem;
        position: relative;
        overflow: hidden;
    }

    .stat-card::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00d4ff33, transparent);
    }

    .stat-label {
        font-size: 0.62rem;
        color: #2a5a7a;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .stat-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #e8f4ff;
        letter-spacing: -0.02em;
        font-family: 'JetBrains Mono', monospace;
    }

    .stat-change.positive { color: #00ff88; font-size: 0.75rem; }
    .stat-change.negative { color: #ff4444; font-size: 0.75rem; }

    /* ── Form styling ── */
    [data-testid="stForm"] {
        background: #0a1525;
        border: 1px solid #0d2040;
        border-radius: 10px;
        padding: 1.2rem;
    }

    /* ── Radio buttons ── */
    .stRadio > div {
        gap: 1rem;
    }

    .stRadio > div > label {
        background: #0a1525 !important;
        border: 1px solid #0d2040 !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.8rem !important;
        color: #8ab0c8 !important;
        font-size: 0.8rem !important;
        cursor: pointer !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── Multiselect ── */
    .stMultiSelect > div > div {
        background: #0a1525 !important;
        border: 1px solid #0d2040 !important;
        border-radius: 6px !important;
    }

    /* ── Number input ── */
    .stNumberInput > div > div > input {
        background: #0a1525 !important;
        border: 1px solid #0d2040 !important;
        color: #c8d8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        border-radius: 6px !important;
    }

    /* ── Checkbox ── */
    .stCheckbox > label {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        color: #8ab0c8 !important;
    }

    /* ── Login screen ── */
    .login-wrapper {
        max-width: 380px;
        margin: 6rem auto 0;
        background: linear-gradient(135deg, #0a1525, #050d1a);
        border: 1px solid #0d2040;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
    }

    .login-logo {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    .login-tagline {
        font-size: 0.65rem;
        color: #2a5a7a;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    /* ── Spinner ── */
    .stSpinner > div {
        border-color: #00d4ff !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #050810; }
    ::-webkit-scrollbar-thumb { background: #0d2040; border-radius: 2px; }
    ::-webkit-scrollbar-thumb:hover { background: #00d4ff44; }

    /* ── Tab bar ── */
    .stTabs [data-baseweb="tab-list"] {
        background: #050810;
        border-bottom: 1px solid #0d2040;
        gap: 0;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #3a6a8a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border-bottom: 2px solid transparent !important;
        padding: 0.6rem 1.2rem !important;
    }

    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom: 2px solid #00d4ff !important;
        background: rgba(0, 212, 255, 0.05) !important;
    }

    /* ── Textarea ── */
    .stTextArea > div > div > textarea {
        background: #0a1525 !important;
        border: 1px solid #0d2040 !important;
        color: #c8d8f0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        border-radius: 6px !important;
    }

    /* ── Markdown text ── */
    .stMarkdown p {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #8ab0c8;
        line-height: 1.7;
    }

    .stMarkdown strong {
        color: #c8d8f0;
    }

    .stMarkdown code {
        background: #0a1525;
        border: 1px solid #0d2040;
        color: #00d4ff;
        border-radius: 4px;
        font-size: 0.8rem;
        padding: 0.1rem 0.4rem;
    }

    /* ── Responsive padding ── */
    @media (max-width: 768px) {
        .block-container { padding: 1rem; }
        .terminal-header { font-size: 1.4rem; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Authentication
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div class="login-wrapper">
        <div class="login-logo">APEX</div>
        <div class="login-tagline">Trading Terminal · v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        pw = st.text_input("", type="password", placeholder="Enter access password", key="login_password",
                           label_visibility="collapsed")
        if st.button("AUTHENTICATE", use_container_width=True):
            if pw == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Access denied — incorrect credentials")
    st.stop()

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">APEX</div>
    <div class="sidebar-tagline">Trading Terminal</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    pages = ["Home", "Real-Time Monitor", "Stock Analyzer", "Cryptocurrency",
             "Portfolio Manager", "News & Sentiment", "Predictions",
             "Research Assistant", "Market Screener", "Compare Stocks", "Settings"]
    page = st.selectbox("", pages, label_visibility="collapsed")

    st.markdown('<div class="sidebar-section">Controls</div>', unsafe_allow_html=True)
    refresh_auto = st.slider("Auto-refresh (sec)", 0, 60, 0, help="0 = disabled")

    st.markdown('<div class="sidebar-section">System</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="status-dot green"></span><span style="font-size:0.72rem;color:#3a6a8a">Local Analysis Active</span>', unsafe_allow_html=True)
    st.markdown(f'<br><span class="status-dot {"green" if TEXTBLOB_AVAILABLE else "red"}"></span><span style="font-size:0.72rem;color:#3a6a8a">{"Sentiment Ready" if TEXTBLOB_AVAILABLE else "TextBlob Missing"}</span>', unsafe_allow_html=True)
    st.markdown(f'<br><span class="status-dot blue"></span><span style="font-size:0.72rem;color:#3a6a8a">No API Keys Required</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("LOGOUT", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# Initialize session state
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = 0
if "last_analysis_time" not in st.session_state:
    st.session_state.last_analysis_time = 0

def check_rate_limit(min_seconds=5):
    current_time = time.time()
    time_passed = current_time - st.session_state.last_analysis_time
    if time_passed < min_seconds:
        remaining = int(min_seconds - time_passed)
        return False, remaining
    st.session_state.last_analysis_time = current_time
    return True, 0

# ── Plotly dark theme shared config ──
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#080c14",
    plot_bgcolor="#080c14",
    font=dict(family="JetBrains Mono, monospace", color="#8ab0c8", size=11),
    xaxis=dict(gridcolor="#0d2040", linecolor="#0d2040", zerolinecolor="#0d2040"),
    yaxis=dict(gridcolor="#0d2040", linecolor="#0d2040", zerolinecolor="#0d2040"),
    hovermode="x unified",
)

def styled_chart(fig, height=420):
    fig.update_layout(height=height, **PLOTLY_THEME)
    st.plotly_chart(fig, use_container_width=True)

def signal_badge(signal):
    cls = {"BUY": "signal-buy", "SELL": "signal-sell"}.get(signal, "signal-hold")
    return f'<span class="{cls}">{signal}</span>'

# ---------------------------
# Page: Home
# ---------------------------
if page == "Home":
    st.markdown('<div class="terminal-header">APEX Trading Terminal</div>', unsafe_allow_html=True)
    st.markdown('<div class="terminal-subtitle">Advanced Technical Analysis Platform · No API Keys Required</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Analysis Engine</div>
            <div class="stat-value" style="font-size:1rem;color:#00d4ff">Rule-Based AI</div>
            <div style="font-size:0.72rem;color:#2a5a7a;margin-top:0.3rem">RSI · MACD · EMA · Risk Scoring</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Data Source</div>
            <div class="stat-value" style="font-size:1rem;color:#00ff88">Yahoo Finance</div>
            <div style="font-size:0.72rem;color:#2a5a7a;margin-top:0.3rem">Live market data · 5m cache</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-card">
            <div class="stat-label">Coverage</div>
            <div class="stat-value" style="font-size:1rem;color:#ffb400">Global Markets</div>
            <div style="font-size:0.72rem;color:#2a5a7a;margin-top:0.3rem">US · India (NSE) · Crypto</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Navigation Guide")
        nav_items = [
            ("📡", "Real-Time Monitor", "Live multi-ticker tracking with candlestick charts"),
            ("🔬", "Stock Analyzer", "Deep dive: indicators, risk score, AI analysis"),
            ("₿", "Cryptocurrency", "Crypto market analysis with volume"),
            ("💼", "Portfolio Manager", "Track holdings with live P&L"),
            ("📰", "News & Sentiment", "NLP-powered news sentiment scoring"),
            ("📈", "Predictions", "Technical price forecasting with confidence intervals"),
            ("🔍", "Research Assistant", "Multi-stock research with summaries"),
            ("🎯", "Market Screener", "Filter stocks by RSI, signal, risk"),
            ("⚖️", "Compare Stocks", "Normalized performance comparison"),
        ]
        for icon, name, desc in nav_items:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:0.8rem;padding:0.5rem 0;border-bottom:1px solid #0a1a2a">
                <span style="font-size:1rem;width:20px;flex-shrink:0">{icon}</span>
                <div>
                    <span style="font-size:0.8rem;color:#c8d8f0;font-family:JetBrains Mono,monospace">{name}</span>
                    <br><span style="font-size:0.7rem;color:#2a5a7a;font-family:JetBrains Mono,monospace">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Supported Tickers")
        st.markdown("""
        <div class="stat-card" style="margin-bottom:0.8rem">
            <div class="stat-label">🇺🇸 US Equities</div>
            <div style="margin-top:0.5rem">
                <span class="ticker-tag">AAPL</span>
                <span class="ticker-tag">MSFT</span>
                <span class="ticker-tag">GOOGL</span>
                <span class="ticker-tag">NVDA</span>
                <span class="ticker-tag">TSLA</span>
            </div>
        </div>
        <div class="stat-card" style="margin-bottom:0.8rem">
            <div class="stat-label">🇮🇳 Indian (NSE)</div>
            <div style="margin-top:0.5rem">
                <span class="ticker-tag">TCS.NS</span>
                <span class="ticker-tag">INFY.NS</span>
                <span class="ticker-tag">RELIANCE.NS</span>
            </div>
        </div>
        <div class="stat-card">
            <div class="stat-label">₿ Crypto</div>
            <div style="margin-top:0.5rem">
                <span class="ticker-tag">BTC-USD</span>
                <span class="ticker-tag">ETH-USD</span>
                <span class="ticker-tag">SOL-USD</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.info("💡 All analysis runs locally. Portfolio stored in SQLite. No external API keys needed.")

# ---------------------------
# Page: Real-Time Monitor
# ---------------------------
elif page == "Real-Time Monitor":
    st.markdown('<div class="page-header">📡 Real-Time Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Live intraday candlestick tracking · 1-minute intervals</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        tickers_raw = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL")
    with col2:
        st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
        refresh_now = st.button("⟳ REFRESH", use_container_width=True)

    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    should_refresh = refresh_now or (refresh_auto > 0 and (time.time() - st.session_state.last_refresh > refresh_auto))

    if tickers and should_refresh:
        st.session_state.last_refresh = time.time()
        for t in tickers:
            with st.expander(f"▶  {t}", expanded=True):
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
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${latest:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
                    col2.metric("High", f"${intraday['High'].max():.2f}")
                    col3.metric("Low", f"${intraday['Low'].min():.2f}")
                    col4.metric("Volume", f"{intraday['Volume'].sum():,.0f}")
                    intraday["EMA20"] = intraday["Close"].ewm(span=20, adjust=False).mean()
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=intraday.index, open=intraday["Open"], high=intraday["High"],
                        low=intraday["Low"], close=intraday["Close"], name="Price",
                        increasing_line_color="#00ff88", decreasing_line_color="#ff4444"
                    ))
                    fig.add_trace(go.Scatter(
                        x=intraday.index, y=intraday["EMA20"],
                        mode="lines", name="EMA20",
                        line=dict(color="#ffb400", width=1.5)
                    ))
                    styled_chart(fig)
                except Exception as e:
                    st.error(f"Error fetching {t}: {e}")
    elif not tickers:
        st.info("Enter tickers above and click Refresh to load live data.")

# ---------------------------
# Page: Stock Analyzer
# ---------------------------
elif page == "Stock Analyzer":
    st.markdown('<div class="page-header">🔬 Stock Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Technical indicators · Risk scoring · AI signal generation</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.text_input("Ticker Symbol", "AAPL").upper()
    with col2:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
    with col3:
        interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

    if st.button("▶ ANALYZE STOCK", use_container_width=True):
        can_proceed, wait_time = check_rate_limit(min_seconds=5)
        if not can_proceed:
            st.warning(f"⏳ Rate limit — please wait {wait_time}s before next request.")
            st.stop()

        with st.spinner(f"Analyzing {ticker}..."):
            try:
                hist = get_stock_history(ticker, period=period, interval=interval)
                if hist is None:
                    st.error("Could not fetch data. Wait 60s and try again.")
                    st.stop()
                if hist.empty:
                    st.error("No data for this ticker.")
                else:
                    # Price Chart
                    st.markdown("#### Price History")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index, y=hist["Close"], mode="lines", name="Close",
                        line=dict(color="#00d4ff", width=2),
                        fill='tozeroy', fillcolor="rgba(0,212,255,0.04)"
                    ))
                    styled_chart(fig)

                    # Fundamentals
                    st.markdown("#### Fundamentals")
                    info = get_stock_info(ticker)
                    cols = st.columns(4)
                    metrics = [
                        ("Market Cap", info.get("marketCap", "N/A")),
                        ("P/E Ratio", info.get("trailingPE", "N/A")),
                        ("52W High", info.get("fiftyTwoWeekHigh", "N/A")),
                        ("52W Low", info.get("fiftyTwoWeekLow", "N/A"))
                    ]
                    for col, (label, value) in zip(cols, metrics):
                        if isinstance(value, (int, float)):
                            col.metric(label, f"{value:,.2f}" if isinstance(value, float) else f"{value:,}")
                        else:
                            col.metric(label, str(value))

                    # Technical Analysis
                    st.markdown("---")
                    st.markdown("#### Technical Analysis")
                    df = compute_indicators(hist)
                    risk_score, risk_level, risk_breakdown = calculate_risk_score(df)
                    signal = generate_signal(df)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Close", f"${df['Close'].iloc[-1]:.2f}")
                    col2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.1f}")
                    col3.metric("MACD", f"{df['MACD'].iloc[-1]:.4f}")
                    col4.metric("Signal", signal)

                    col1, col2 = st.columns(2)
                    with col1:
                        rsi_val = df['RSI'].iloc[-1]
                        if rsi_val > 70:
                            st.warning(f"⚠ Overbought — RSI {rsi_val:.1f} > 70")
                        elif rsi_val < 30:
                            st.info(f"ℹ Oversold — RSI {rsi_val:.1f} < 30")
                        else:
                            st.success(f"✓ RSI neutral at {rsi_val:.1f}")
                        ema_status = "Bullish crossover ↑" if df['EMA20'].iloc[-1] > df['EMA50'].iloc[-1] else "Bearish crossover ↓"
                        st.markdown(f"**EMA Status:** {ema_status}")
                        st.markdown(f"**EMA20:** ${df['EMA20'].iloc[-1]:.2f}  |  **EMA50:** ${df['EMA50'].iloc[-1]:.2f}")
                    with col2:
                        st.markdown(f"**Trading Signal:** {signal_badge(signal)}", unsafe_allow_html=True)
                        st.markdown(f"<div style='margin-top:0.8rem'><b>Risk Score:</b> {risk_score}/100 — {risk_level}</div>", unsafe_allow_html=True)
                        risk_color = "#00ff88" if risk_score < 40 else "#ffb400" if risk_score < 65 else "#ff4444"
                        st.markdown(f"<div style='margin-top:0.5rem;height:6px;background:#0d2040;border-radius:3px'><div style='width:{risk_score}%;height:100%;background:{risk_color};border-radius:3px'></div></div>", unsafe_allow_html=True)

                    # Risk breakdown
                    st.markdown("---")
                    st.markdown("#### Risk Components")
                    risk_df = pd.DataFrame([
                        {"Factor": "Volatility", "Score": risk_breakdown.get('Volatility', 0), "Max": 30},
                        {"Factor": "RSI Extremes", "Score": risk_breakdown.get('RSI', 0), "Max": 20},
                        {"Factor": "Price Momentum", "Score": risk_breakdown.get('Momentum', 0), "Max": 20},
                        {"Factor": "MACD Signal", "Score": risk_breakdown.get('MACD', 0), "Max": 15},
                        {"Factor": "Volume Trend", "Score": risk_breakdown.get('Volume', 0), "Max": 15},
                    ])
                    st.dataframe(risk_df, use_container_width=True, hide_index=True)

                    # AI Analysis
                    st.markdown("---")
                    st.markdown("#### AI Analysis Report")
                    analysis = generate_local_analysis(ticker, df)
                    st.markdown(analysis)

            except Exception as e:
                if "Too Many Requests" in str(e):
                    st.error("Rate limit hit. Wait 60–120s and retry.")
                else:
                    st.error(f"Error: {e}")

# ---------------------------
# Page: News & Sentiment
# ---------------------------
elif page == "News & Sentiment":
    st.markdown('<div class="page-header">📰 News & Sentiment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">NLP-powered headline sentiment analysis via TextBlob</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        nt = st.text_input("Ticker for News", "AAPL").upper()
    with col2:
        ncount = st.slider("Headlines", 1, 10, 5)

    if st.button("▶ FETCH NEWS", use_container_width=True):
        with st.spinner(f"Fetching news for {nt}..."):
            try:
                t = yf.Ticker(nt)
                raw_news = getattr(t, "news", []) or []

                if not raw_news:
                    st.warning("No live news returned by Yahoo Finance for this ticker.")
                    st.markdown("---")
                    st.markdown("#### Demo: Sentiment Analysis")
                    sample_headlines = [
                        f"{nt} reports record quarterly earnings beating analyst expectations",
                        f"{nt} announces strategic partnership with major tech company",
                        f"Analysts remain bullish on {nt} stock prospects",
                        f"{nt} faces regulatory scrutiny in European markets",
                        f"Mixed signals for {nt} as market volatility increases"
                    ]
                    sample_results = analyze_news_sentiment(sample_headlines)
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Sentiment", sample_results['overall_sentiment'])
                    col2.metric("Positive", sample_results['positive_count'])
                    col3.metric("Negative", sample_results['negative_count'])
                    col4.metric("Neutral", sample_results['neutral_count'])
                    for headline in sample_headlines:
                        polarity, label, _ = analyze_sentiment_local(headline)
                        icon = "🟢" if label == "Positive" else "🔴" if label == "Negative" else "🟡"
                        st.markdown(f"`{icon}` **{label}** ({polarity:+.2f}) — {headline}")
                else:
                    headlines = []
                    for item in raw_news[:ncount]:
                        title = item.get("title", "")
                        if title:
                            headlines.append({"title": title, "link": item.get("link", ""), "publisher": item.get("publisher", "Unknown")})

                    if headlines:
                        headline_texts = [h['title'] for h in headlines]
                        sentiment_results = analyze_news_sentiment(headline_texts)
                        score = sentiment_results['sentiment_score']

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Overall", sentiment_results['overall_sentiment'])
                        col2.metric("Positive", sentiment_results['positive_count'])
                        col3.metric("Negative", sentiment_results['negative_count'])
                        col4.metric("Neutral", sentiment_results['neutral_count'])

                        st.progress(min(max((score + 1) / 2, 0), 1))
                        if score > 0.15:
                            st.success("📈 Bullish sentiment — positive news flow detected")
                        elif score < -0.15:
                            st.error("📉 Bearish sentiment — negative news flow detected")
                        else:
                            st.info("⚖ Neutral sentiment — mixed signals")

                        st.markdown("---")
                        st.markdown("#### Individual Headlines")
                        for i, h in enumerate(headlines, 1):
                            polarity, label, confidence = analyze_sentiment_local(h['title'])
                            icon = "🟢" if label == "Positive" else "🔴" if label == "Negative" else "🟡"
                            with st.expander(f"{icon} {h['title']}", expanded=(i <= 2)):
                                st.markdown(f"**Publisher:** {h['publisher']}")
                                st.markdown(f"**Sentiment:** {label}  |  **Score:** {polarity:+.3f}  |  **Confidence:** {confidence:.1%}")
                                if h['link']:
                                    st.markdown(f"[Read full article ↗]({h['link']})")
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------
# Page: Cryptocurrency
# ---------------------------
elif page == "Cryptocurrency":
    st.markdown('<div class="page-header">₿ Cryptocurrency Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time crypto analysis with volume and technical indicators</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        crypto = st.text_input("Crypto Ticker", "BTC-USD").upper()
    with col2:
        c_period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo"], index=2)

    if st.button("▶ ANALYZE CRYPTO", use_container_width=True):
        with st.spinner(f"Analyzing {crypto}..."):
            try:
                crypto_obj = yf.Ticker(crypto)
                interval = "1m" if c_period == "1d" else ("15m" if c_period == "5d" else "1h")
                ch = crypto_obj.history(period=c_period, interval=interval)
                if ch is None or ch.empty:
                    st.error("No data for this crypto ticker.")
                else:
                    latest = ch["Close"].iloc[-1]
                    high = ch["High"].max()
                    low = ch["Low"].min()
                    vol = ch["Volume"].sum()
                    price_change = ((latest - ch["Close"].iloc[0]) / ch["Close"].iloc[0]) * 100

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Price", f"${latest:,.2f}", f"{price_change:+.2f}%")
                    col2.metric(f"{c_period} High", f"${high:,.2f}")
                    col3.metric(f"{c_period} Low", f"${low:,.2f}")
                    col4.metric("Volume", f"{vol:,.0f}")

                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(
                        x=ch.index, open=ch["Open"], high=ch["High"],
                        low=ch["Low"], close=ch["Close"], name=crypto,
                        increasing_line_color="#00ff88", decreasing_line_color="#ff4444"
                    ))
                    styled_chart(fig, height=480)

                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=ch.index, y=ch["Volume"], name="Volume",
                        marker_color="rgba(0,212,255,0.4)"
                    ))
                    styled_chart(fig_vol, height=200)

                    st.markdown("---")
                    st.markdown("#### Technical Analysis")
                    ch_indicators = compute_indicators(ch)
                    if len(ch_indicators) > 50:
                        volatility = ch['Close'].pct_change().dropna().std() * np.sqrt(24)
                        recent_high = ch['High'].tail(20).max()
                        recent_low = ch['Low'].tail(20).min()
                        rsi_val = ch_indicators['RSI'].iloc[-1] if 'RSI' in ch_indicators.columns else None

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Volatility", f"{volatility*100:.2f}%")
                        col2.metric("Resistance", f"${recent_high:,.2f}")
                        col3.metric("Support", f"${recent_low:,.2f}")

                        if rsi_val is not None:
                            if rsi_val > 70:
                                st.warning(f"⚠ Overbought — RSI {rsi_val:.1f}")
                            elif rsi_val < 30:
                                st.info(f"ℹ Oversold — RSI {rsi_val:.1f}")

                        range_pct = ((latest - recent_low) / (recent_high - recent_low) * 100) if (recent_high - recent_low) > 0 else 50
                        st.markdown(f"**Price in range:** {range_pct:.1f}% (0% = support, 100% = resistance)")
                        st.markdown(f"<div style='height:8px;background:#0d2040;border-radius:4px'><div style='width:{range_pct}%;height:100%;background:linear-gradient(90deg,#00ff88,#00d4ff);border-radius:4px'></div></div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------
# Page: Portfolio Manager
# ---------------------------
elif page == "Portfolio Manager":
    st.markdown('<div class="page-header">💼 Portfolio Manager</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Track holdings with live P&L · Persistent SQLite storage</div>', unsafe_allow_html=True)

    st.markdown("#### Add Position")
    with st.form("add_position", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            ticker = st.text_input("Ticker", "AAPL")
        with col2:
            qty = st.number_input("Quantity", min_value=0.01, value=1.0, step=1.0)
        with col3:
            avg = st.number_input("Avg Price ($)", min_value=0.0, value=0.0, step=0.01,
                                  help="Leave 0 to auto-fetch current price")
        submitted = st.form_submit_button("+ ADD POSITION", use_container_width=True)
        if submitted:
            ticker = ticker.upper().strip()
            if not ticker:
                st.error("Please enter a ticker.")
            else:
                if avg == 0.0:
                    with st.spinner(f"Fetching price for {ticker}..."):
                        cur = get_current_price(ticker)
                    if cur is None:
                        st.error("Could not fetch price. Enter avg price manually.")
                    else:
                        add_position_db(ticker, qty, cur)
                        st.success(f"Added {qty}x {ticker} @ ${cur:.2f}")
                        time.sleep(0.5)
                        st.rerun()
                else:
                    add_position_db(ticker, qty, avg)
                    st.success(f"Added {qty}x {ticker} @ ${avg:.2f}")
                    time.sleep(0.5)
                    st.rerun()

    st.markdown("---")
    st.markdown("#### Holdings")
    df = list_positions_db()

    if df.empty:
        st.info("No positions yet. Add your first position above.")
    else:
        rows = []
        total_value = 0
        total_pl = 0
        with st.spinner("Fetching live prices..."):
            for _, r in df.iterrows():
                cur = get_current_price(r['ticker']) or r['avg_price']
                val = cur * r['qty']
                pl = (cur - r['avg_price']) * r['qty']
                pl_pct = ((cur - r['avg_price']) / r['avg_price'] * 100) if r['avg_price'] > 0 else 0
                total_value += val
                total_pl += pl
                rows.append({
                    "ID": r['id'], "Ticker": r['ticker'], "Qty": r['qty'],
                    "Avg $": f"${r['avg_price']:.2f}", "Current $": f"${cur:.2f}",
                    "Value": f"${val:.2f}",
                    "P&L": f"${pl:+.2f}",
                    "P&L %": f"{pl_pct:+.2f}%"
                })

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Value", f"${total_value:,.2f}")
        col2.metric("Total P&L", f"${total_pl:+,.2f}",
                    delta=f"{(total_pl/total_value*100):+.2f}%" if total_value > 0 else "0%")
        col3.metric("Positions", len(df))

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("#### Remove Position")
        col1, col2 = st.columns([2, 1])
        with col1:
            rem_id = st.number_input("Position ID", min_value=1, value=1, step=1)
        with col2:
            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            if st.button("REMOVE", use_container_width=True):
                remove_position_db(rem_id)
                st.success(f"Position {rem_id} removed.")
                time.sleep(0.5)
                st.rerun()

# ---------------------------
# Page: Predictions
# ---------------------------
elif page == "Predictions":
    st.markdown('<div class="page-header">📈 Price Predictions</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Momentum-based technical forecasting with confidence intervals</div>', unsafe_allow_html=True)

    st.warning("⚠ Predictions are based on technical analysis only. Markets are unpredictable. Always do your own research.")

    col1, col2 = st.columns([2, 1])
    with col1:
        pt = st.text_input("Ticker", "AAPL").upper()
    with col2:
        days = st.slider("Days Ahead", 1, 30, 7)

    if st.button("▶ GENERATE FORECAST", use_container_width=True):
        with st.spinner(f"Forecasting {pt}..."):
            try:
                hist = yf.Ticker(pt).history(period="6mo")
                if hist is None or hist.empty:
                    st.error("Insufficient data.")
                else:
                    recent_closes = hist["Close"].tail(30).tolist()
                    current_price = recent_closes[-1]
                    avg_30d = np.mean(recent_closes)
                    volatility = np.std(recent_closes)
                    returns = hist["Close"].pct_change().dropna()
                    avg_daily_return = returns.mean()

                    predicted_price = current_price * (1 + avg_daily_return * days)
                    upper_bound = predicted_price + (volatility * np.sqrt(days) * 2)
                    lower_bound = predicted_price - (volatility * np.sqrt(days) * 2)

                    df_pred = compute_indicators(hist)
                    signal = generate_signal(df_pred)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current", f"${current_price:.2f}")
                    col2.metric("Forecast", f"${predicted_price:.2f}",
                                f"{((predicted_price-current_price)/current_price*100):+.2f}%")
                    col3.metric("Upper Bound", f"${upper_bound:.2f}")
                    col4.metric("Lower Bound", f"${lower_bound:.2f}")

                    future_dates = pd.date_range(start=hist.index[-1], periods=days+1, freq='D')[1:]
                    dates_pred = [hist.index[-1]] + list(future_dates)

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=hist.index[-30:], y=hist["Close"].tail(30),
                        mode="lines", name="Historical",
                        line=dict(color="#00d4ff", width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=dates_pred, y=[current_price] + [upper_bound]*days,
                        mode="lines", name="Upper Bound",
                        line=dict(color="#ff4444", width=1, dash="dot")
                    ))
                    fig.add_trace(go.Scatter(
                        x=dates_pred, y=[current_price] + [lower_bound]*days,
                        mode="lines", name="Lower Bound",
                        line=dict(color="#ff4444", width=1, dash="dot"),
                        fill='tonexty', fillcolor='rgba(255,68,68,0.07)'
                    ))
                    fig.add_trace(go.Scatter(
                        x=dates_pred, y=[current_price] + [predicted_price]*days,
                        mode="lines", name="Forecast",
                        line=dict(color="#00ff88", width=2, dash="dash")
                    ))
                    styled_chart(fig, height=450)

                    st.markdown("---")
                    analysis = generate_local_analysis(pt, df_pred)
                    st.markdown(analysis)
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------
# Page: Research Assistant
# ---------------------------
elif page == "Research Assistant":
    st.markdown('<div class="page-header">🔍 Research Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Multi-stock deep analysis with buy/sell recommendations</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📈 Top Momentum", use_container_width=True):
            st.session_state.ra_tickers = "AAPL,NVDA,MSFT"
    with col2:
        if st.button("📉 Oversold Picks", use_container_width=True):
            st.session_state.ra_tickers = "META,AMZN,NFLX"
    with col3:
        if st.button("🛡 Low Risk", use_container_width=True):
            st.session_state.ra_tickers = "JNJ,KO,PG"

    if "ra_tickers" not in st.session_state:
        st.session_state.ra_tickers = ""

    tickers_to_research = st.text_input("Tickers to research (comma-separated)",
                                         st.session_state.ra_tickers,
                                         placeholder="e.g. AAPL,MSFT,GOOGL")

    if st.button("▶ CONDUCT RESEARCH", type="primary", use_container_width=True):
        if not tickers_to_research.strip():
            st.error("Enter at least one ticker.")
        else:
            tickers_list = [t.strip().upper() for t in tickers_to_research.split(",") if t.strip()]
            research_results = []
            for ticker in tickers_list:
                with st.expander(f"▶ {ticker}", expanded=True):
                    try:
                        hist = get_stock_history(ticker, period="6mo")
                        if hist is not None and not hist.empty and len(hist) > 50:
                            df = compute_indicators(hist)
                            risk_score, risk_level, _ = calculate_risk_score(df)
                            signal = generate_signal(df)
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
                            col2.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                            col3.metric("Risk", f"{risk_score}/100")
                            col4.metric("Signal", signal)
                            st.markdown(generate_local_analysis(ticker, df))
                            research_results.append({"ticker": ticker, "signal": signal, "risk": risk_score})
                        else:
                            st.warning(f"Insufficient data for {ticker}")
                    except Exception as e:
                        st.error(f"Error: {e}")

            if research_results:
                st.markdown("---")
                st.markdown("#### Research Summary")
                buy_signals = [r for r in research_results if r['signal'] == 'BUY']
                low_risk = [r for r in research_results if r['risk'] < 40]
                col1, col2, col3 = st.columns(3)
                col1.metric("Analyzed", len(research_results))
                col2.metric("Buy Signals", len(buy_signals))
                col3.metric("Low Risk", len(low_risk))
                if buy_signals:
                    st.success("BUY signals: " + ", ".join([r['ticker'] for r in buy_signals]))
                if low_risk:
                    st.info("Low risk: " + ", ".join([r['ticker'] for r in low_risk]))

# ---------------------------
# Page: Market Screener
# ---------------------------
elif page == "Market Screener":
    st.markdown('<div class="page-header">🎯 Market Screener</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Filter stocks by RSI, signal, and risk score</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Technical Filters")
        rsi_min = st.slider("Min RSI", 0, 100, 30)
        rsi_max = st.slider("Max RSI", 0, 100, 70)
        signal_filter = st.multiselect("Signal", ["BUY", "SELL", "HOLD"], default=["BUY"])
    with col2:
        st.markdown("#### Risk Filter")
        risk_max = st.slider("Max Risk Score", 0, 100, 60)

    preset = st.radio("Stock Universe", ["US Top 30", "Indian Nifty 20", "Custom"], horizontal=True)

    if preset == "US Top 30":
        stock_universe = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TSLA","META","BRK.B","JPM","JNJ",
                          "V","PG","UNH","HD","MA","DIS","PYPL","NFLX","ADBE","CRM",
                          "INTC","CSCO","PFE","KO","PEP","ABT","TMO","COST","AVGO","ACN"]
    elif preset == "Indian Nifty 20":
        stock_universe = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HINDUNILVR.NS",
                          "ICICIBANK.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
                          "LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
                          "WIPRO.NS","HCLTECH.NS","ULTRACEMCO.NS","BAJFINANCE.NS","NESTLEIND.NS"]
    else:
        custom_input = st.text_area("Custom tickers (comma-separated)", "AAPL,MSFT,GOOGL,TSLA")
        stock_universe = [t.strip().upper() for t in custom_input.split(",") if t.strip()]

    st.info(f"Universe: {len(stock_universe)} stocks")

    if st.button("▶ RUN SCREENER", type="primary", use_container_width=True):
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
                        rsi_val = df['RSI'].iloc[-1]
                        if (rsi_min <= rsi_val <= rsi_max and signal in signal_filter and risk_score <= risk_max):
                            info = get_stock_info(ticker)
                            results.append({
                                "Ticker": ticker, "Price": f"${df['Close'].iloc[-1]:.2f}",
                                "Signal": signal, "RSI": f"{rsi_val:.1f}",
                                "Risk": f"{risk_score:.0f}",
                                "Risk Level": risk_level, "Sector": info.get("sector", "N/A")
                            })
                    progress_bar.progress((idx + 1) / len(stock_universe))
                    time.sleep(0.3)
                except Exception:
                    continue
            progress_bar.empty()

            if results:
                st.success(f"✓ {len(results)} stocks matched your criteria")
                results_df = pd.DataFrame(results).sort_values("Risk")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No stocks matched. Try adjusting filters.")

# ---------------------------
# Page: Compare Stocks
# ---------------------------
elif page == "Compare Stocks":
    st.markdown('<div class="page-header">⚖️ Stock Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Side-by-side normalized performance comparison</div>', unsafe_allow_html=True)

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

    compare_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)

    if st.button("▶ COMPARE", type="primary", use_container_width=True):
        tickers = [t for t in [ticker1, ticker2, ticker3, ticker4] if t]
        if len(tickers) < 2:
            st.error("Enter at least 2 tickers.")
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
                            returns = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                            price_history[ticker] = hist['Close']
                            comparison_data.append({
                                "Ticker": ticker, "Price": f"${hist['Close'].iloc[-1]:.2f}",
                                "Return": f"{returns:+.2f}%", "Signal": signal,
                                "RSI": f"{df['RSI'].iloc[-1]:.1f}",
                                "Risk Score": f"{risk_score:.0f}/100",
                                "Risk Level": risk_level,
                                "P/E": info.get("trailingPE", "N/A"),
                                "Sector": info.get("sector", "N/A")
                            })
                        time.sleep(0.5)
                    except Exception as e:
                        st.error(f"Error: {e}")

            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                st.markdown("---")
                st.markdown("#### Normalized Performance (Base = 100)")
                if price_history:
                    colors = ["#00d4ff", "#00ff88", "#ffb400", "#ff4444"]
                    fig = go.Figure()
                    for i, (t, prices) in enumerate(price_history.items()):
                        normalized = (prices / prices.iloc[0]) * 100
                        fig.add_trace(go.Scatter(
                            x=normalized.index, y=normalized, mode="lines", name=t,
                            line=dict(color=colors[i % len(colors)], width=2.5)
                        ))
                    styled_chart(fig, height=450)
                st.markdown("---")
                st.markdown(generate_comparison_analysis(comparison_data))

# ---------------------------
# Page: Settings
# ---------------------------
elif page == "Settings":
    st.markdown('<div class="page-header">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">System status · Database management · Configuration</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### System Status")
        st.markdown(f'<span class="status-dot green"></span><span style="font-size:0.8rem;color:#8ab0c8"> Local Analysis Engine</span>', unsafe_allow_html=True)
        st.markdown(f'<br><span class="status-dot green"></span><span style="font-size:0.8rem;color:#8ab0c8"> Technical Indicators</span>', unsafe_allow_html=True)
        st.markdown(f'<br><span class="status-dot {"green" if TEXTBLOB_AVAILABLE else "red"}"></span><span style="font-size:0.8rem;color:#8ab0c8"> Sentiment Analysis {"(ready)" if TEXTBLOB_AVAILABLE else "(install textblob)"}</span>', unsafe_allow_html=True)
        if not TEXTBLOB_AVAILABLE:
            st.code("pip install textblob")
    with col2:
        st.markdown("#### Data Sources")
        st.markdown('<span class="status-dot blue"></span><span style="font-size:0.8rem;color:#8ab0c8"> Yahoo Finance API</span>', unsafe_allow_html=True)
        st.markdown('<br><span class="status-dot blue"></span><span style="font-size:0.8rem;color:#8ab0c8"> Local SQLite Database</span>', unsafe_allow_html=True)
        st.markdown('<br><span class="status-dot green"></span><span style="font-size:0.8rem;color:#8ab0c8"> No External API Keys Required</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Database Management")
    df_db = list_positions_db()
    col1, col2 = st.columns(2)
    col1.metric("Current Positions", len(df_db))
    col2.metric("Database Path", DB_PATH)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 CLEAR ALL POSITIONS", use_container_width=True):
            c = conn.cursor()
            c.execute("DELETE FROM portfolio")
            conn.commit()
            st.success("Portfolio cleared.")
            time.sleep(1)
            st.rerun()
    with col2:
        if st.button("📊 DB STATS", use_container_width=True):
            c = conn.cursor()
            c.execute("SELECT COUNT(*) as count, SUM(qty) as total_qty FROM portfolio")
            stats = c.fetchone()
            st.markdown(f"Positions: **{stats[0]}** · Total shares: **{stats[1]}**")

    st.markdown("---")
    st.markdown("#### Test System")
    test_ticker = st.text_input("Test Ticker", "AAPL")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("TEST DATA FETCH", use_container_width=True):
            with st.spinner(f"Testing {test_ticker}..."):
                hist = get_stock_history(test_ticker, period="1d")
                if hist is not None and not hist.empty:
                    st.success("Data fetch successful")
                    st.dataframe(hist.tail(1))
                else:
                    st.error("No data returned")
    with col2:
        if st.button("TEST SENTIMENT", use_container_width=True):
            if TEXTBLOB_AVAILABLE:
                test_text = "Apple announces record quarterly earnings with strong iPhone sales"
                polarity, label, confidence = analyze_sentiment_local(test_text)
                st.success(f"Sentiment: {label} ({polarity:+.2f}, confidence: {confidence:.1%})")
            else:
                st.error("TextBlob not installed")

    st.markdown("---")
    st.markdown("#### Dependencies")
    st.code("pip install streamlit yfinance pandas numpy plotly textblob")