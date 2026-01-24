import os
import time
import sqlite3
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from textblob import TextBlob

# ---------------------------
# 1. UI INJECTION (Stitch Integration)
# ---------------------------
def apply_stitch_ui():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;900&display=swap');
        
        .stApp {
            background-color: #0f1323;
            font-family: 'Inter', sans-serif;
            color: #ffffff;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #0f1323 !important;
            border-right: 1px solid #2C3642;
        }

        /* Card Styling (Glassmorphism) */
        div[data-testid="metric-container"] {
            background-color: #1A232E;
            border: 1px solid #2C3642;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }

        /* Custom Button (InvestPro Blue) */
        .stButton>button {
            background: linear-gradient(135deg, #607AFB 0%, #3b82f6 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.6rem 1.2rem !important;
        }

        /* Table and Inputs */
        .stDataFrame, .stTable {
            background-color: #1A232E;
            border-radius: 12px;
            border: 1px solid #2C3642;
        }
        
        .stTextInput>div>div>input, .stSelectbox>div>div>div {
            background-color: #1A232E !important;
            color: white !important;
            border: 1px solid #2C3642 !important;
        }

        h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }
        .stMarkdown p { color: #94a3b8 !important; }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# 2. CORE LOGIC (From Your Original Code)
# ---------------------------
@st.cache_resource
def init_db():
    conn = sqlite3.connect("portfolio.db", check_same_thread=False)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                ticker TEXT NOT NULL, qty REAL NOT NULL, 
                avg_price REAL NOT NULL, added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    return conn

conn = init_db()

def compute_indicators(df):
    df = df.copy()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA50"] = df["Close"].ewm(span=50, adjust=False).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))
    df["MACD"] = df["Close"].ewm(span=12, adjust=False).mean() - df["Close"].ewm(span=26, adjust=False).mean()
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df

def generate_local_analysis(ticker, data):
    current_price = data['Close'].iloc[-1]
    rsi = data['RSI'].iloc[-1]
    ema20 = data['EMA20'].iloc[-1]
    ema50 = data['EMA50'].iloc[-1]
    return f"**{ticker} Analysis**: Price at ${current_price:.2f}. RSI is {rsi:.2f}. {'Bullish' if ema20 > ema50 else 'Bearish'} trend."

# ---------------------------
# 3. PAGE CONFIG & AUTH
# ---------------------------
st.set_page_config(page_title="InvestPro", layout="wide")
apply_stitch_ui()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>InvestPro Login</h1>", unsafe_allow_html=True)
        pw = st.text_input("Enter Access Key", type="password")
        if st.button("Access Dashboard", use_container_width=True):
            if pw == "password123":
                st.session_state.authenticated = True
                st.rerun()
    st.stop()

# ---------------------------
# 4. MAIN NAVIGATION
# ---------------------------
with st.sidebar:
    st.markdown("### 💎 InvestPro")
    page = st.selectbox("Menu", ["Real-Time Monitor", "Stock Analyzer", "Portfolio Manager", "News & Sentiment", "Predictions", "Market Screener"])
    st.divider()
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

# ---------------------------
# 5. PAGE CONTENT (Integrated)
# ---------------------------
st.markdown(f"<h2>{page}</h2>", unsafe_allow_html=True)

if page == "Real-Time Monitor":
    tickers = st.text_input("Enter Tickers", "AAPL,MSFT,GOOGL").upper().split(",")
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            price = yf.Ticker(t).history(period="1d")["Close"].iloc[-1]
            st.metric(t, f"${price:.2f}")

elif page == "Stock Analyzer":
    ticker = st.text_input("Symbol", "AAPL").upper()
    if st.button("Run Deep Analysis"):
        hist = yf.Ticker(ticker).history(period="6mo")
        df = compute_indicators(hist)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}")
        m2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        m3.metric("MACD", f"{df['MACD'].iloc[-1]:.4f}")
        
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(generate_local_analysis(ticker, df))

elif page == "Portfolio Manager":
    with st.form("add_pos"):
        c1, c2, c3 = st.columns(3)
        t = c1.text_input("Ticker")
        q = c2.number_input("Qty", min_value=0.1)
        p = c3.number_input("Avg Price")
        if st.form_submit_button("Add to Portfolio"):
            cur = conn.cursor()
            cur.execute("INSERT INTO portfolio (ticker, qty, avg_price) VALUES (?,?,?)", (t.upper(), q, p))
            conn.commit()
            st.rerun()
    
    data = pd.read_sql_query("SELECT id, ticker, qty, avg_price FROM portfolio", conn)
    st.dataframe(data, use_container_width=True)

elif page == "News & Sentiment":
    nt = st.text_input("Ticker", "TSLA").upper()
    if st.button("Analyze News"):
        news = yf.Ticker(nt).news
        for item in news[:5]:
            blob = TextBlob(item['title'])
            sentiment = "🟢 Positive" if blob.sentiment.polarity > 0 else "🔴 Negative"
            st.write(f"{sentiment} | {item['title']}")

elif page == "Predictions":
    pt = st.text_input("Ticker for Forecast", "NVDA").upper()
    if st.button("Generate Forecast"):
        hist = yf.Ticker(pt).history(period="1mo")["Close"]
        future_price = hist.iloc[-1] * (1 + (hist.pct_change().mean() * 7))
        st.markdown(f"### Estimated 7-Day Target: **${future_price:.2f}**")
        st.line_chart(hist)

elif page == "Market Screener":
    st.write("Screening Top 10 S&P 500 Stocks...")
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "V", "JPM"]
    results = []
    for s in universe:
        h = yf.Ticker(s).history(period="1mo")
        d = compute_indicators(h)
        results.append({"Ticker": s, "RSI": d['RSI'].iloc[-1], "EMA20": d['EMA20'].iloc[-1]})
    st.table(pd.DataFrame(results))