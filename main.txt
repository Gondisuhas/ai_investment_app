# main.py
import os
import time
import sqlite3
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Optional: Gemini
try:
    import google.generativeai as genai
except Exception:
    genai = None

# ---------------------------
# Load env
# ---------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "password123")  # default if not set

# Configure Gemini if available
if genai and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
    except Exception:
        pass

def pick_gemini_model():
    if not genai:
        return None
    try:
        models = genai.list_models()
        for m in models:
            name = getattr(m, "name", "")
            supported = getattr(m, "supported_generation_methods", []) or []
            if "gemini-1.5" in name and "generateContent" in supported:
                return name
        # fallback:
        return "gemini-1.5-flash"
    except Exception:
        return "gemini-1.5-flash"

MODEL_NAME = pick_gemini_model()

# ---------------------------
# Database (SQLite) for portfolio persistence
# ---------------------------
DB_PATH = "portfolio.db"

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
# Helpers: indicators, signals, Gemini wrapper
# ---------------------------
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
        if df["EMA20"].iloc[-1] > df["EMA50"].iloc[-1]:
            return "BUY"
        elif df["EMA20"].iloc[-1] < df["EMA50"].iloc[-1]:
            return "SELL"
        else:
            return "HOLD"
    except Exception:
        return "N/A"

def ask_gemini(prompt):
    if not genai or not GOOGLE_API_KEY:
        return "Gemini not configured. Add GOOGLE_API_KEY to .env to enable AI features."
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        resp = model.generate_content(prompt)
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"[Gemini error] {e}"

# ---------------------------
# UI & Auth
# ---------------------------
st.set_page_config(page_title="Gemini Investment Terminal", layout="wide", page_icon="📊")
st.title("Gemini Investment Terminal — Captain Suhas")

# Simple password auth
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    pw = st.text_input("Enter app password", type="password")
    if st.button("Login"):
        if pw == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

# After login: navigation
pages = ["Home", "Real-Time", "Stock Analyzer", "Crypto", "Portfolio", "News & Sentiment", "Predictions", "Settings"]
page = st.sidebar.selectbox("Navigate", pages)

# Optional local logo path (your uploaded file). If you don't want it, ignore.
LOGO_PATH = "/mnt/data/ee3e67f9-a255-4b1b-af18-56f1a632a319.png"  # provided file path from uploads

# show small header info
st.sidebar.markdown("**Controls**")
refresh_auto = st.sidebar.slider("Auto-refresh interval (sec) — 0 = off", 0, 60, 15)
st.sidebar.markdown("---")
st.sidebar.markdown(f"Gemini model: `{MODEL_NAME}`")
st.sidebar.markdown("API key configured." if GOOGLE_API_KEY else "No Gemini API key found.")

# ---------- HOME ----------
if page == "Home":
    st.header("Welcome, Captain Suhas")
    st.write("Use the sidebar to navigate. App stores your portfolio in a local SQLite DB (`portfolio.db`).")
    st.info("Tip: Indian tickers use `.NS` suffix (e.g., TCS.NS, INFY.NS). Crypto: BTC-USD, ETH-USD.")

# ---------- REAL-TIME ----------
elif page == "Real-Time":
    st.header("Real-Time Multi-Ticker Tracker")
    tickers_raw = st.text_input("Tickers (comma-separated)", "AAPL")
    refresh_now = st.button("Refresh now")
    tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]

    if tickers and (refresh_now or (refresh_auto > 0 and (time.time() - st.session_state.get("last_refresh", 0) > refresh_auto))):
        st.session_state.last_refresh = time.time()
        failed = []
        for t in tickers:
            st.markdown(f"### {t}")
            try:
                intraday = yf.Ticker(t).history(period="1d", interval="1m")
                if intraday is None or intraday.empty:
                    st.warning(f"No intraday data for {t}. Try a single ticker or use correct market suffix (e.g., INFY.NS).")
                    failed.append(t)
                    continue
                latest = intraday["Close"].iloc[-1]
                st.metric(f"{t} Live Price", f"${latest:.2f}")
                # indicators & EMA overlay
                intraday["EMA20"] = intraday["Close"].ewm(span=20).mean()
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=intraday.index, open=intraday["Open"], high=intraday["High"],
                                             low=intraday["Low"], close=intraday["Close"], name="Candle"))
                fig.add_trace(go.Scatter(x=intraday.index, y=intraday["EMA20"], mode="lines", name="EMA20"))
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error fetching {t}: {e}")
                failed.append(t)
        if failed:
            st.error("Failed: " + ", ".join(failed))

# ---------- STOCK ANALYZER ----------
elif page == "Stock Analyzer":
    st.header("Stock Analyzer: historical & AI")
    ticker = st.text_input("Ticker (single)", "AAPL").upper()
    period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=0)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
    if st.button("Analyze"):
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval=interval)
        if hist is None or hist.empty:
            st.error("No data.")
        else:
            st.subheader("Price Chart")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines+markers", name="Close"))
            fig.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Fundamentals")
            info = getattr(t, "info", {}) or {}
            cols = st.columns(4)
            cols[0].metric("Market Cap", info.get("marketCap", "N/A"))
            cols[1].metric("PE Ratio", info.get("trailingPE", "N/A"))
            cols[2].metric("52w High", info.get("fiftyTwoWeekHigh", "N/A"))
            cols[3].metric("52w Low", info.get("fiftyTwoWeekLow", "N/A"))
            st.subheader("Technicals & AI Analysis")
            df = compute_indicators(hist)
            st.write("Latest Close:", df["Close"].iloc[-1])
            st.write("SMA/EMA snapshot:")
            st.dataframe(df.tail(3)[["EMA20", "EMA50", "RSI", "MACD"]].round(4))
            signal = generate_signal(df)
            st.info(f"Signal based on EMA20/50: **{signal}**")
            # AI summary
            prompt = (f"You are a senior market analyst. Summarize {ticker} with last {period} of closes. "
                      "Give sentiment (Bullish/Neutral/Bearish), risk score 0-100, short action (Buy/Hold/Sell), and reason.")
            with st.spinner("Asking Gemini..."):
                ai = ask_gemini(prompt)
            st.subheader("Gemini Analysis")
            st.write(ai)

# ---------- CRYPTO ----------
elif page == "Crypto":
    st.header("Crypto Analyzer")
    crypto = st.text_input("Crypto ticker (Yahoo format)", "BTC-USD")
    c_period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo"], index=0)
    if st.button("Analyze Crypto"):
        ch = yf.Ticker(crypto).history(period=c_period, interval="1m" if c_period=="1d" else "1h")
        if ch is None or ch.empty:
            st.error("No data for this crypto.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=ch.index, open=ch["Open"], high=ch["High"], low=ch["Low"], close=ch["Close"]))
            fig.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("AI Crypto Sentiment")
            prompt = f"Analyze {crypto} for the last {c_period}: sentiment, short-term outlook, and risk."
            st.write(ask_gemini(prompt))

# ---------- PORTFOLIO ----------
elif page == "Portfolio":
    st.header("Portfolio (persistent)")
    with st.form("add_pos", clear_on_submit=True):
        pt, pq, pp = st.columns(3)
        with pt: ticker = st.text_input("Ticker", "AAPL")
        with pq: qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=1.0)
        with pp: avg = st.number_input("Avg Price (0 -> fetch current)", min_value=0.0, value=0.0, step=0.01)
        submitted = st.form_submit_button("Add")
        if submitted:
            if avg == 0.0:
                cur = get_current_price(ticker)
                if cur is None:
                    st.error("Could not fetch current price; enter avg.")
                else:
                    add_position_db(ticker, qty, cur)
                    st.success(f"Added {qty} x {ticker} @ {cur:.2f}")
            else:
                add_position_db(ticker, qty, avg)
                st.success(f"Added {qty} x {ticker} @ {avg:.2f}")

    df = list_positions_db()
    if df.empty:
        st.info("No positions yet.")
    else:
        # compute live values
        rows=[]
        for idx, r in df.iterrows():
            cur = get_current_price(r['ticker']) or 0.0
            val = cur * r['qty']
            pl = (cur - r['avg_price']) * r['qty']
            rows.append({"id": r['id'], "ticker": r['ticker'], "qty": r['qty'], "avg": r['avg_price'], "current": round(cur,2), "value": round(val,2), "pl": round(pl,2)})
        pdf = pd.DataFrame(rows)
        st.dataframe(pdf)
        rem = st.number_input("Enter ID to remove", min_value=0, value=0, step=1)
        if st.button("Remove position"):
            remove_position_db(rem)
            st.success("Removed.")

# ---------- NEWS & SENTIMENT ----------
elif page == "News & Sentiment":
    st.header("News & Sentiment")
    nt = st.text_input("Ticker for headlines", "AAPL")
    ncount = st.slider("How many headlines", 1, 10, 5)
    if st.button("Fetch Headlines"):
        t = yf.Ticker(nt)
        raw = getattr(t, "news", []) or []
        headlines = [i.get("title") or i.get("link") or str(i) for i in raw][:ncount]
        if not headlines:
            st.warning("No headlines from yfinance; coverage varies.")
        else:
            st.subheader("Headlines")
            for h in headlines:
                st.write("- " + h)
            prompt = "Analyze sentiment and investor impact for these headlines:\n\n" + "\n".join(headlines)
            with st.spinner("Asking Gemini"):
                ai = ask_gemini(prompt)
            st.markdown("### Gemini News Sentiment")
            st.write(ai)

# ---------- PREDICTIONS ----------
elif page == "Predictions":
    st.header("AI Prediction (textual)")
    pt = st.text_input("Ticker", "AAPL").upper()
    days = st.slider("Days ahead", 1, 14, 7)
    if st.button("Predict"):
        hist = yf.Ticker(pt).history(period="3mo")
        if hist is None or hist.empty:
            st.error("Not enough history.")
        else:
            snippet = hist["Close"].tail(30).tolist()
            prompt = f"Given recent closes {snippet} for {pt}, provide a probabilistic outlook for next {days} days: expected low/high, probability of increase, confidence 0-100, and rationale."
            with st.spinner("Asking Gemini"):
                out = ask_gemini(prompt)
            st.markdown("### Prediction")
            st.write(out)

# ---------- SETTINGS ----------
elif page == "Settings":
    st.header("Settings & Diagnostics")
    st.write("Gemini model detected:")
    st.code(MODEL_NAME if MODEL_NAME else "None")
    if st.button("List available models"):
        if not genai:
            st.error("google-generativeai not installed.")
        else:
            try:
                models = genai.list_models()
                st.write([m.name for m in models])
            except Exception as e:
                st.error(f"Error listing models: {e}")
    if st.button("Clear database (remove all positions)"):
        c = conn.cursor()
        c.execute("DELETE FROM portfolio")
        conn.commit()
        st.success("Portfolio cleared.")

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Gemini Investment Terminal — built for Captain Suhas. Local SQLite persistence enabled.")
