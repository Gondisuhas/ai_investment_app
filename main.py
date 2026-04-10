st.info("💡 **Tip**: Use this comparison to make informed decisions based on your investment goals and risk tolerance.")

# ---------------------------
# Page: Watchlist & Alerts (NEW - Feature #4)
# ---------------------------
elif page == "🔔 Watchlist & Alerts":
    st.header("🔔 Watchlist & Price Alerts")
    
    tab1, tab2 = st.tabs(["📋 Watchlists", "🔔 Price Alerts"])
    
    # Tab 1: Watchlists
    with tab1:
        st.subheader("📋 Manage Watchlists")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### ➕ Add to Watchlist")
            with st.form("add_watchlist"):
                wcol1, wcol2, wcol3 = st.columns([2, 2, 3])
                with wcol1:
                    new_list = st.text_input("Watchlist Name", "My Watchlist")
                with wcol2:
                    new_ticker = st.text_input("Ticker", "AAPL")
                with wcol3:
                    new_notes = st.text_input("Notes (optional)", "")
                
                if st.form_submit_button("➕ Add to Watchlist", use_container_width=True):
                    if new_ticker.strip():
                        add_to_watchlist(new_list, new_ticker, new_notes)
                        st.success(f"✅ Added {new_ticker.upper()} to {new_list}")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("Please enter a ticker")
        
        with col2:
            st.markdown("### 📊 Quick Stats")
            watchlist_names = get_watchlist_names()
            all_watchlist = get_watchlist()
            st.metric("Total Watchlists", len(watchlist_names))
            st.metric("Total Tickers", len(all_watchlist))
        
        st.markdown("---")
        
        # Display watchlists
        watchlist_names = get_watchlist_names()
        
        if not watchlist_names:
            st.info("📭 No watchlists yet. Create your first one above!")
        else:
            selected_list = st.selectbox("Select Watchlist", ["All"] + watchlist_names)
            
            if selected_list == "All":
                watchlist_df = get_watchlist()
            else:
                watchlist_df = get_watchlist(selected_list)
            
            if not watchlist_df.empty:
                st.markdown(f"### 📊 {selected_list} ({len(watchlist_df)} tickers)")
                
                # Get live prices
                watchlist_data = []
                
                with st.spinner("Fetching live prices..."):
                    for _, row in watchlist_df.iterrows():
                        try:
                            current = get_current_price(row['ticker'])
                            if current:
                                # Get basic info
                                t = yf.Ticker(row['ticker'])
                                hist = t.history(period="5d")
                                
                                if not hist.empty:
                                    prev_close = hist['Close'].iloc[-2] if len(hist) > 1 else hist['Close'].iloc[-1]
                                    change = current - prev_close
                                    change_pct = (change / prev_close * 100) if prev_close > 0 else 0
                                    
                                    watchlist_data.append({
                                        "ID": row['id'],
                                        "List": row['list_name'],
                                        "Ticker": row['ticker'],
                                        "Price": f"${current:.2f}",
                                        "Change": f"${change:+.2f}",
                                        "Change %": f"{change_pct:+.2f}%",
                                        "Notes": row['notes'],
                                        "Added": row['added_at']
                                    })
                        except:
                            watchlist_data.append({
                                "ID": row['id'],
                                "List": row['list_name'],
                                "Ticker": row['ticker'],
                                "Price": "N/A",
                                "Change": "N/A",
                                "Change %": "N/A",
                                "Notes": row['notes'],
                                "Added": row['added_at']
                            })
                
                if watchlist_data:
                    watch_display_df = pd.DataFrame(watchlist_data)
                    st.dataframe(watch_display_df, use_container_width=True, hide_index=True)
                    
                    # Remove from watchlist
                    st.markdown("### 🗑️ Remove from Watchlist")
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        remove_id = st.number_input("Enter ID to remove", min_value=1, value=1, step=1)
                    with col2:
                        st.write("")
                        st.write("")
                        if st.button("🗑️ Remove", use_container_width=True):
                            remove_from_watchlist(remove_id)
                            st.success("✅ Removed from watchlist")
                            time.sleep(0.5)
                            st.rerun()
    
    # Tab 2: Price Alerts
    with tab2:
        st.subheader("🔔 Price Alerts")
        
        # Check alerts
        triggered = check_alerts()
        if triggered:
            st.warning(f"⚠️ {len(triggered)} alert(s) triggered!")
            for alert_id, ticker, current, target, alert_type in triggered:
                st.error(f"🔔 **{ticker}** is now ${current:.2f} ({alert_type} target ${target:.2f})")
                if st.button(f"Dismiss alert for {ticker}", key=f"dismiss_{alert_id}"):
                    deactivate_alert(alert_id)
                    st.rerun()
        
        st.markdown("### ➕ Create Price Alert")
        
        with st.form("add_alert"):
            acol1, acol2, acol3 = st.columns(3)
            
            with acol1:
                alert_ticker = st.text_input("Ticker", "AAPL")
            with acol2:
                alert_price = st.number_input("Target Price", min_value=0.01, value=100.0, step=0.01)
            with acol3:
                alert_type = st.selectbox("Alert When", ["above", "below"])
            
            if st.form_submit_button("🔔 Create Alert", use_container_width=True):
                if alert_ticker.strip():
                    add_price_alert(alert_ticker, alert_price, alert_type)
                    st.success(f"✅ Alert created: {alert_ticker.upper()} {alert_type} ${alert_price:.2f}")
                    time.sleep(0.5)
                    st.rerun()
        
        st.markdown("---")
        
        # Display active alerts
        alerts_df = get_active_alerts()
        
        if alerts_df.empty:
            st.info("📭 No active alerts. Create one above!")
        else:
            st.markdown(f"### 📊 Active Alerts ({len(alerts_df)})")
            
            # Add current prices
            alert_display = []
            for _, row in alerts_df.iterrows():
                current = get_current_price(row['ticker'])
                if current:
                    distance = current - row['target_price']
                    distance_pct = (distance / row['target_price'] * 100) if row['target_price'] > 0 else 0
                    
                    alert_display.append({
                        "ID": row['id'],
                        "Ticker": row['ticker'],
                        "Current": f"${current:.2f}",
                        "Target": f"${row['target_price']:.2f}",
                        "Type": row['alert_type'].upper(),
                        "Distance": f"${distance:+.2f} ({distance_pct:+.2f}%)",
                        "Created": row['created_at']
                    })
                else:
                    alert_display.append({
                        "ID": row['id'],
                        "Ticker": row['ticker'],
                        "Current": "N/A",
                        "Target": f"${row['target_price']:.2f}",
                        "Type": row['alert_type'].upper(),
                        "Distance": "N/A",
                        "Created": row['created_at']
                    })
            
            if alert_display:
                alert_df = pd.DataFrame(alert_display)
                st.dataframe(alert_df, use_container_width=True, hide_index=True)
                
                # Deactivate alert
                st.markdown("### 🗑️ Deactivate Alert")
                col1, col2 = st.columns([2, 1])
                with col1:
                    deact_id = st.number_input("Enter Alert ID", min_value=1, value=1, step=1, key="deact")
                with col2:
                    st.write("")
                    st.write("")
                    if st.button("🗑️ Deactivate", use_container_width=True):
                        deactivate_alert(deact_id)
                        st.success("✅ Alert deactivated")
                        time.sleep(0.5)
                        st.rerun()

# ---------------------------
# Page: Sector Analysis (NEW - Feature #7)
# ---------------------------
elif page == "📊 Sector Analysis":
    st.header("📊 Sector Analysis & Comparison")
    
    # Define major sectors and representative stocks
    SECTORS = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA", "ADBE", "CRM", "INTC", "AMD"],
        "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "TMO", "ABT", "DHR", "MRK", "LLY", "BMY"],
        "Finance": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
        "Consumer": ["AMZN", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "DG", "COST"],
        "Industrial": ["BA", "CAT", "HON", "UPS", "GE", "MMM", "LMT", "RTX", "DE", "EMR"],
        "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "DLR", "O", "WELL", "AVB"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG", "ED"]
    }
    
    st.markdown("### 🎯 Select Sectors to Analyze")
    
    selected_sectors = st.multiselect(
        "Choose sectors (select 2-4 for best comparison)",
        list(SECTORS.keys()),
        default=["Technology", "Healthcare", "Finance"]
    )
    
    analysis_depth = st.slider("Stocks per sector", 3, 10, 5)
    
    if st.button("📊 Analyze Sectors", type="primary", use_container_width=True):
        if not selected_sectors:
            st.error("Please select at least one sector")
        else:
            sector_data = {}
            
            with st.spinner(f"Analyzing {len(selected_sectors)} sectors..."):
                for sector in selected_sectors:
                    stocks = SECTORS[sector][:analysis_depth]
                    sector_results = []
                    
                    for ticker in stocks:
                        try:
                            t = yf.Ticker(ticker)
                            hist = t.history(period="3mo")
                            
                            if hist is not None and not hist.empty:
                                df = compute_indicators(hist)
                                risk_score, risk_level, _ = calculate_risk_score(df)
                                signal = generate_signal(df)
                                
                                # Calculate returns
                                start_price = hist['Close'].iloc[0]
                                end_price = hist['Close'].iloc[-1]
                                returns_3m = ((end_price - start_price) / start_price) * 100
                                
                                sector_results.append({
                                    "ticker": ticker,
                                    "price": end_price,
                                    "returns": returns_3m,
                                    "risk": risk_score,
                                    "signal": signal,
                                    "rsi": df['RSI'].iloc[-1]
                                })
                        except:
                            continue
                    
                    if sector_results:
                        sector_data[sector] = sector_results
                
                # Display results
                if sector_data:
                    st.success(f"✅ Analysis complete for {len(sector_data)} sectors!")
                    
                    # Sector Performance Summary
                    st.subheader("📊 Sector Performance Summary")
                    
                    summary_data = []
                    for sector, results in sector_data.items():
                        avg_return = np.mean([r['returns'] for r in results])
                        avg_risk = np.mean([r['risk'] for r in results])
                        buy_signals = len([r for r in results if r['signal'] == 'BUY'])
                        
                        summary_data.append({
                            "Sector": sector,
                            "Avg 3M Return": f"{avg_return:+.2f}%",
                            "Avg Risk": f"{avg_risk:.1f}/100",
                            "BUY Signals": f"{buy_signals}/{len(results)}",
                            "Stocks Analyzed": len(results)
                        })
                    
                    summary_df = pd.DataFrame(summary_data)
                    summary_df = summary_df.sort_values("Avg 3M Return", ascending=False)
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    
                    # Sector Comparison Chart
                    st.subheader("📈 Sector Returns Comparison")
                    
                    fig = go.Figure()
                    
                    for sector, results in sector_data.items():
                        returns = [r['returns'] for r in results]
                        tickers = [r['ticker'] for r in results]
                        
                        fig.add_trace(go.Bar(
                            name=sector,
                            x=tickers,
                            y=returns,
                            text=[f"{r:+.1f}%" for r in returns],
                            textposition='outside'
                        ))
                    
                    fig.update_layout(
                        template="plotly_dark",
                        height=500,
                        barmode='group',
                        xaxis_title="Stocks",
                        yaxis_title="3-Month Returns (%)",
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk vs Return Scatter
                    st.subheader("⚖️ Risk vs Return Analysis")
                    
                    scatter_fig = go.Figure()
                    
                    for sector, results in sector_data.items():
                        risks = [r['risk'] for r in results]
                        returns = [r['returns'] for r in results]
                        tickers = [r['ticker'] for r in results]
                        
                        scatter_fig.add_trace(go.Scatter(
                            name=sector,
                            x=risks,
                            y=returns,
                            mode='markers+text',
                            text=tickers,
                            textposition='top center',
                            marker=dict(size=12)
                        ))
                    
                    scatter_fig.update_layout(
                        template="plotly_dark",
                        height=500,
                        xaxis_title="Risk Score",
                        yaxis_title="3-Month Returns (%)",
                        hovermode="closest"
                    )
                    
                    st.plotly_chart(scatter_fig, use_container_width=True)
                    
                    # AI Sector Analysis
                    st.markdown("---")
                    st.subheader("🤖 AI Sector Insights")
                    
                    # Prepare summary for AI
                    sector_summary = "\n".join([
                        f"**{sector}:** Avg Return: {np.mean([r['returns'] for r in results]):.2f}%, "
                        f"Avg Risk: {np.mean([r['risk'] for r in results]):.1f}, "
                        f"BUY signals: {len([r for r in results if r['signal'] == 'BUY'])}/{len(results)}"
                        for sector, results in sector_data.items()
                    ])
                    
                    prompt = f"""Analyze these sector performances:

{sector_summary}

Provide:
1. **Best Performing Sector** and why
2. **Sector Rotation Strategy** - which sectors to favor now
3. **Risk-Adjusted Winners** - best risk/return ratio
4. **Sector Outlook** - future expectations for each
5. **Diversification Advice** - optimal sector allocation
6. **Key Catalysts** to watch for each sector

Be specific and actionable for investors."""
                    
                    with st.spinner("🤖 Generating sector insights..."):
                        ai_analysis = ask_gemini(prompt)
                    
                    st.markdown(ai_analysis)
                else:
                    st.error("❌ Could not fetch data for selected sectors")

# ---------------------------
# Page: Export Reports (NEW - Feature #10)
# ---------------------------
elif page == "📱 Export Reports":
    st.header("📱 Export Reports & Analysis")
    
    st.markdown("""
    Generate and export comprehensive reports of your analysis, portfolio, and market research.
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Portfolio Report", "📈 Stock Analysis Report", "🔍 Custom Report"])
    
    # Tab 1: Portfolio Report
    with tab1:
        st.subheader("📊 Portfolio Performance Report")
        
        if st.button("📄 Generate Portfolio Report", use_container_width=True):
            df = list_positions_db()
            
            if df.empty:
                st.warning("No portfolio positions to report")
            else:
                report = []
                report.append("# 📊 PORTFOLIO PERFORMANCE REPORT")
                report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report.append("=" * 80)
                report.append("")
                
                # Fetch live data
                total_value = 0
                total_pl = 0
                
                report.append("## Holdings Summary")
                report.append("")
                
                for _, row in df.iterrows():
                    current = get_current_price(row['ticker'])
                    if current:
                        value = current * row['qty']
                        pl = (current - row['avg_price']) * row['qty']
                        pl_pct = ((current - row['avg_price']) / row['avg_price'] * 100) if row['avg_price'] > 0 else 0
                        
                        total_value += value
                        total_pl += pl
                        
                        report.append(f"### {row['ticker']}")
                        report.append(f"- Quantity: {row['qty']}")
                        report.append(f"- Average Price: ${row['avg_price']:.2f}")
                        report.append(f"- Current Price: ${current:.2f}")
                        report.append(f"- Position Value: ${value:.2f}")
                        report.append(f"- P&L: ${pl:+.2f} ({pl_pct:+.2f}%)")
                        report.append("")
                
                report.append("=" * 80)
                report.append("## Portfolio Totals")
                report.append(f"- **Total Value:** ${total_value:,.2f}")
                report.append(f"- **Total P&L:** ${total_pl:+,.2f}")
                report.append(f"- **Total Return:** {(total_pl/total_value*100):+.2f}%" if total_value > 0 else "N/A")
                report.append(f"- **Number of Positions:** {len(df)}")
                
                report_text = "\n".join(report)
                
                st.text_area("Portfolio Report", report_text, height=400)
                
                # Download button
                st.download_button(
                    label="💾 Download as TXT",
                    data=report_text,
                    file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
    
    # Tab 2: Stock Analysis Report
    with tab2:
        st.subheader("📈 Stock Analysis Report")
        
        report_ticker = st.text_input("Enter ticker for detailed report", "AAPL").upper()
        report_period = st.selectbox("Analysis Period", ["1mo", "3mo", "6mo", "1y"], index=1)
        
        if st.button("📄 Generate Stock Report", use_container_width=True):
            with st.spinner(f"Generating report for {report_ticker}..."):
                try:
                    t = yf.Ticker(report_ticker)
                    hist = t.history(period=report_period)
                    info = getattr(t, "info", {}) or {}
                    
                    if hist is not None and not hist.empty:
                        df = compute_indicators(hist)
                        risk_score, risk_level, risk_breakdown = calculate_risk_score(df)
                        signal = generate_signal(df)
                        
                        # Build report
                        report = []
                        report.append(f"# 📈 STOCK ANALYSIS REPORT: {report_ticker}")
                        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        report.append(f"Analysis Period: {report_period}")
                        report.append("=" * 80)
                        report.append("")
                        
                        report.append("## Company Information")
                        report.append(f"- **Name:** {info.get('longName', 'N/A')}")
                        report.append(f"- **Sector:** {info.get('sector', 'N/A')}")
                        report.append(f"- **Industry:** {info.get('industry', 'N/A')}")
                        report.append(f"- **Market Cap:** ${info.get('marketCap', 0):,}")
                        report.append("")
                        
                        report.append("## Price Information")
                        report.append(f"- **Current Price:** ${df['Close'].iloc[-1]:.2f}")
                        report.append(f"- **52-Week High:** ${info.get('fiftyTwoWeekHigh', 'N/A')}")
                        report.append(f"- **52-Week Low:** ${info.get('fiftyTwoWeekLow', 'N/A')}")
                        report.append(f"- **P/E Ratio:** {info.get('trailingPE', 'N/A')}")
                        report.append("")
                        
                        report.append("## Technical Analysis")
                        report.append(f"- **EMA20:** ${df['EMA20'].iloc[-1]:.2f}")
                        report.append(f"- **EMA50:** ${df['EMA50'].iloc[-1]:.2f}")
                        report.append(f"- **RSI:** {df['RSI'].iloc[-1]:.2f}")
                        report.append(f"- **MACD:** {df['MACD'].iloc[-1]:.4f}")
                        report.append(f"- **Signal:** {signal}")
                        report.append("")
                        
                        report.append("## Risk Assessment")
                        report.append(f"- **Risk Score:** {risk_score}/100")
                        report.append(f"- **Risk Level:** {risk_level}")
                        report.append(f"- **Volatility Risk:** {risk_breakdown.get('Volatility', 0)}/30")
                        report.append(f"- **RSI Risk:** {risk_breakdown.get('RSI', 0)}/20")
                        report.append(f"- **Momentum Risk:** {risk_breakdown.get('Momentum', 0)}/20")
                        report.append("")
                        
                        # Calculate returns
                        start_price = hist['Close'].iloc[0]
                        end_price = hist['Close'].iloc[-1]
                        returns = ((end_price - start_price) / start_price) * 100
                        
                        report.append("## Performance")
                        report.append(f"- **Period Return:** {returns:+.2f}%")
                        report.append(f"- **Start Price:** ${start_price:.2f}")
                        report.append(f"- **End Price:** ${end_price:.2f}")
                        report.append("")
                        
                        report.append("=" * 80)
                        report.append("## Disclaimer")
                        report.append("This report is for informational purposes only and should not be")
                        report.append("considered financial advice. Always conduct your own research and")
                        report.append("consult with a licensed financial advisor.")
                        
                        report_text = "\n".join(report)
                        
                        st.text_area("Stock Analysis Report", report_text, height=500)
                        
                        # Download button
                        st.download_button(
                            label="💾 Download as TXT",
                            data=report_text,
                            file_name=f"{report_ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    else:
                        st.error("No data available for this ticker")
                        
                except Exception as e:
                    st.error(f"Error generating report: {e}")
    
    # Tab 3: Custom Report
    with tab3:
        st.subheader("🔍 Custom Analysis Report")
        
        st.markdown("Generate a custom report with AI analysis on any investment topic")
        
        custom_query = st.text_area(
            "What would you like to analyze?",
            "Analyze the current market conditions and provide investment recommendations for a conservative investor",
            height=100
        )
        
        include_tickers = st.text_input("Include specific tickers (optional, comma-separated)", "")
        
        if st.button("📄 Generate Custom Report", use_container_width=True):
            with st.spinner("Generating custom report..."):
                tickers_list = [t.strip().upper() for t in include_tickers.split(",") if t.strip()]
                
                ticker_data = ""
                if tickers_list:
                    ticker_data = "\n\nIncluded Tickers Data:\n"
                    for ticker in tickers_list:
                        try:
                            current = get_current_price(ticker)
                            if current:
                                ticker_data += f"- {ticker}: ${current:.2f}\n"
                        except:
                            continue
                
                prompt = f"""Generate a comprehensive investment analysis report based on this query:

"{custom_query}"
{ticker_data}

Structure the report with:
1. Executive Summary
2. Market Analysis
3. Specific Recommendations
4. Risk Assessment
5. Action Plan
6. Conclusion

Make it detailed, professional, and actionable."""
                
                ai_report = ask_gemini(prompt)
                
                # Format report
                report = []
                report.append("# 🔍 CUSTOM INVESTMENT ANALYSIS REPORT")
                report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                report.append("=" * 80)
                report.append("")
                report.append(f"## Query: {custom_query}")
                report.append("")
                report.append("=" * 80)
                report.append("")
                report.append(ai_report)
                report.append("")
                report.append("=" * 80)
                report.append("## Disclaimer")
                report.append("This AI-generated report is for informational purposes only.")
                report.append("Always consult with a licensed financial advisor before making investment decisions.")
                
                report_text = "\n".join(report)
                
                st.text_area("Custom Report", report_text, height=500)
                
                st.download_button(
                    label="💾 Download as TXT",
                    data=report_text,
                    file_name=f"custom_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

# ---------------------------
# Page: Global Markets (NEW - Feature #9)
# ---------------------------
elif page == "🌍 Global Markets":
    st.header("🌍 Global Markets Overview")
    
    st.markdown("""
    Track major global indices and international markets in real-time.
    """)
    
    # Major Global Indices
    GLOBAL_INDICES = {
        "🇺🇸 United States": {
            "S&P 500": "^GSPC",
            "Dow Jones": "^DJI",
            "NASDAQ": "^IXIC",
            "Russell 2000": "^RUT"
        },
        "🇮🇳 India": {
            "Nifty 50": "^NSEI",
            "Sensex": "^BSESN",
            "Nifty Bank": "^NSEBANK",
            "Nifty IT": "^CNXIT"
        },
        "🇪🇺 Europe": {
            "FTSE 100 (UK)": "^FTSE",
            "DAX (Germany)": "^GDAXI",
            "CAC 40 (France)": "^FCHI",
            "STOXX 50": "^STOXX50E"
        },
        "🇯🇵 Asia-Pacific": {
            "Nikkei 225": "^N225",
            "Hang Seng": "^HSI",
            "Shanghai Composite": "000001.SS",
            "ASX 200": "^AXJO"
        }
    }
    
    # Currency Pairs
    CURRENCIES = {
        "USD/INR": "INR=X",
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CNY": "CNY=X"
    }
    
    tab1, tab2, tab3 = st.tabs(["import os
import time
import sqlite3
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import hashlib
import secrets

# Optional: Gemini
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

# ---------------------------
# Security Configuration
# ---------------------------
# Rate limiting
if "request_count" not in st.session_state:
    st.session_state.request_count = 0
    st.session_state.last_reset = time.time()

if "failed_login_attempts" not in st.session_state:
    st.session_state.failed_login_attempts = 0
    st.session_state.lockout_until = 0

# Session timeout (30 minutes)
SESSION_TIMEOUT = 1800  # 30 minutes in seconds

if "last_activity" not in st.session_state:
    st.session_state.last_activity = time.time()

# Check session timeout
if "authenticated" in st.session_state and st.session_state.authenticated:
    if time.time() - st.session_state.last_activity > SESSION_TIMEOUT:
        st.session_state.authenticated = False
        st.warning("⏱️ Session expired due to inactivity. Please login again.")
        st.stop()
    else:
        st.session_state.last_activity = time.time()

# Rate limit check
def check_rate_limit():
    current_time = time.time()
    if current_time - st.session_state.last_reset > 60:  # Reset every minute
        st.session_state.request_count = 0
        st.session_state.last_reset = current_time
    
    st.session_state.request_count += 1
    if st.session_state.request_count > 30:  # Max 30 requests per minute
        st.error("⚠️ Rate limit exceeded. Please wait a minute.")
        time.sleep(60)
        return False
    return True

# Check login lockout
def check_lockout():
    if time.time() < st.session_state.lockout_until:
        remaining = int(st.session_state.lockout_until - time.time())
        st.error(f"🔒 Too many failed attempts. Locked out for {remaining} seconds.")
        return True
    return False

# Hash password for comparison
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

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

# Admin password for Settings access
try:
    if "ADMIN_PASSWORD" in st.secrets:
        ADMIN_PASSWORD = st.secrets["ADMIN_PASSWORD"]
    else:
        ADMIN_PASSWORD = "admin123"
except (KeyError, FileNotFoundError, Exception):
    ADMIN_PASSWORD = "admin123"

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
    
    # Portfolio table
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            qty REAL NOT NULL,
            avg_price REAL NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Watchlist table
    c.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            list_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            notes TEXT
        )
    """)
    
    # Price alerts table
    c.execute("""
        CREATE TABLE IF NOT EXISTS price_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            target_price REAL NOT NULL,
            alert_type TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            triggered_at TIMESTAMP
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
# Watchlist & Alerts Functions
# ---------------------------
def add_to_watchlist(list_name, ticker, notes=""):
    """Add ticker to watchlist"""
    c = conn.cursor()
    c.execute("INSERT INTO watchlist (list_name, ticker, notes) VALUES (?, ?, ?)",
              (list_name, ticker.upper(), notes))
    conn.commit()

def remove_from_watchlist(watch_id):
    """Remove from watchlist"""
    c = conn.cursor()
    c.execute("DELETE FROM watchlist WHERE id = ?", (watch_id,))
    conn.commit()

def get_watchlist(list_name=None):
    """Get watchlist items"""
    c = conn.cursor()
    if list_name:
        c.execute("SELECT id, list_name, ticker, notes, added_at FROM watchlist WHERE list_name = ? ORDER BY added_at DESC", (list_name,))
    else:
        c.execute("SELECT id, list_name, ticker, notes, added_at FROM watchlist ORDER BY list_name, added_at DESC")
    rows = c.fetchall()
    cols = ["id", "list_name", "ticker", "notes", "added_at"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

def get_watchlist_names():
    """Get unique watchlist names"""
    c = conn.cursor()
    c.execute("SELECT DISTINCT list_name FROM watchlist ORDER BY list_name")
    return [row[0] for row in c.fetchall()]

def add_price_alert(ticker, target_price, alert_type):
    """Add price alert"""
    c = conn.cursor()
    c.execute("INSERT INTO price_alerts (ticker, target_price, alert_type) VALUES (?, ?, ?)",
              (ticker.upper(), float(target_price), alert_type))
    conn.commit()

def get_active_alerts():
    """Get active price alerts"""
    c = conn.cursor()
    c.execute("SELECT id, ticker, target_price, alert_type, created_at FROM price_alerts WHERE is_active = 1 ORDER BY created_at DESC")
    rows = c.fetchall()
    cols = ["id", "ticker", "target_price", "alert_type", "created_at"]
    return pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)

def check_alerts():
    """Check if any alerts should be triggered"""
    c = conn.cursor()
    c.execute("SELECT id, ticker, target_price, alert_type FROM price_alerts WHERE is_active = 1")
    alerts = c.fetchall()
    
    triggered = []
    for alert_id, ticker, target_price, alert_type in alerts:
        try:
            current_price = get_current_price(ticker)
            if current_price:
                if alert_type == "above" and current_price >= target_price:
                    triggered.append((alert_id, ticker, current_price, target_price, "above"))
                elif alert_type == "below" and current_price <= target_price:
                    triggered.append((alert_id, ticker, current_price, target_price, "below"))
        except:
            continue
    
    return triggered

def deactivate_alert(alert_id):
    """Deactivate an alert"""
    c = conn.cursor()
    c.execute("UPDATE price_alerts SET is_active = 0, triggered_at = CURRENT_TIMESTAMP WHERE id = ?", (alert_id,))
    conn.commit()

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
    page_title="Investment Terminal",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Hide Streamlit branding and add security notice
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
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
    .security-notice {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background: rgba(0,0,0,0.7);
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 10px;
        z-index: 999;
    }
    
    /* Disable right-click */
    body {
        -webkit-user-select: none;
        -moz-user-select: none;
        -ms-user-select: none;
        user-select: none;
    }
</style>
<div class="security-notice">🔒 Secure Session</div>
<script>
    // Disable right-click
    document.addEventListener('contextmenu', event => event.preventDefault());
    
    // Disable F12, Ctrl+Shift+I, Ctrl+Shift+J, Ctrl+U
    document.onkeydown = function(e) {
        if(e.keyCode == 123 || 
           (e.ctrlKey && e.shiftKey && e.keyCode == 'I'.charCodeAt(0)) ||
           (e.ctrlKey && e.shiftKey && e.keyCode == 'J'.charCodeAt(0)) ||
           (e.ctrlKey && e.keyCode == 'U'.charCodeAt(0))) {
            return false;
        }
    }
    
    // Clear console
    console.clear();
    
    // Prevent console access
    setInterval(function() {
        console.clear();
    }, 1000);
</script>
""", unsafe_allow_html=True)

# ---------------------------
# Authentication
# ---------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Secure Investment Terminal")
    st.markdown("### Welcome, Captain Suhas")
    
    # Check lockout
    if check_lockout():
        st.stop()
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pw = st.text_input("Enter app password", type="password", key="login_password")
        
        # Show attempts remaining
        attempts_remaining = 5 - st.session_state.failed_login_attempts
        if st.session_state.failed_login_attempts > 0:
            st.warning(f"⚠️ {attempts_remaining} attempts remaining")
        
        if st.button("🚀 Login", use_container_width=True):
            if pw == APP_PASSWORD:
                st.session_state.authenticated = True
                st.session_state.failed_login_attempts = 0
                st.session_state.last_activity = time.time()
                st.success("✅ Login successful!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.session_state.failed_login_attempts += 1
                
                # Lock out after 5 failed attempts
                if st.session_state.failed_login_attempts >= 5:
                    st.session_state.lockout_until = time.time() + 300  # 5 minutes
                    st.error("🔒 Too many failed attempts! Locked out for 5 minutes.")
                else:
                    st.error(f"❌ Incorrect password. {attempts_remaining - 1} attempts remaining.")
    
    # Security notice
    st.markdown("---")
    st.info("🔒 **Security Features Active:**\n- Session timeout: 30 minutes\n- Rate limiting enabled\n- Failed login protection\n- Code obfuscation")
    st.stop()

# ---------------------------
# Main Application
# ---------------------------
st.markdown('<p class="main-header">📊 Investment Terminal</p>', unsafe_allow_html=True)

# Session info in corner
col1, col2 = st.columns([6, 1])
with col1:
    st.markdown("**Captain Suhas Dashboard**")
with col2:
    time_left = int(SESSION_TIMEOUT - (time.time() - st.session_state.last_activity))
    st.caption(f"⏱️ {time_left//60}m")

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/rocket.png", width=80)
    st.markdown("### 🧭 Navigation")
    pages = ["🏠 Home", "⚡ Real-Time", "📈 Stock Analyzer", "₿ Crypto", 
             "💼 Portfolio", "📰 News & Sentiment", "🔮 Predictions", 
             "🤖 AI Research Assistant", "📊 Market Screener", 
             "📚 Compare Stocks", "🔔 Watchlist & Alerts", "📊 Sector Analysis",
             "📱 Export Reports", "🌍 Global Markets", "⚙️ Settings"]
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
# Page: AI Research Assistant (NEW)
# ---------------------------
elif page == "🤖 AI Research Assistant":
    st.header("🤖 AI Research Assistant")
    st.markdown("Ask any investment question and get AI-powered insights with real market data!")
    
    # Quick action buttons
    st.markdown("### 💡 Quick Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Hot Stocks Today", use_container_width=True):
            st.session_state.research_query = "Which stocks are showing the strongest momentum today? Analyze top gainers."
    with col2:
        if st.button("📉 Oversold Opportunities", use_container_width=True):
            st.session_state.research_query = "Find oversold stocks with RSI < 30 that might bounce back soon."
    with col3:
        if st.button("🏆 Top Tech Stocks", use_container_width=True):
            st.session_state.research_query = "What are the best technology stocks to invest in right now?"
    
    col4, col5, col6 = st.columns(3)
    with col4:
        if st.button("💰 Best Dividend Stocks", use_container_width=True):
            st.session_state.research_query = "What are the highest dividend-paying stocks with stable fundamentals?"
    with col5:
        if st.button("🌟 Growth Stocks", use_container_width=True):
            st.session_state.research_query = "Which growth stocks have the highest potential for 2025?"
    with col6:
        if st.button("🛡️ Safe Investments", use_container_width=True):
            st.session_state.research_query = "What are the safest, low-risk stocks for conservative investors?"
    
    st.markdown("---")
    
    # Custom query input
    if "research_query" not in st.session_state:
        st.session_state.research_query = ""
    
    query = st.text_area(
        "🔍 Ask your investment question:",
        value=st.session_state.research_query,
        height=100,
        placeholder="Examples:\n- Which stocks will rise high this month and why?\n- Compare AAPL vs MSFT for long-term investment\n- What are the best Indian stocks in the EV sector?\n- Should I buy Tesla now? Analyze risks and opportunities"
    )
    
    # Analysis depth
    col1, col2 = st.columns([2, 1])
    with col1:
        tickers_to_analyze = st.text_input(
            "📊 Specific tickers to analyze (optional, comma-separated)",
            placeholder="e.g., AAPL,MSFT,GOOGL or INFY.NS,TCS.NS",
            help="Leave empty for general analysis"
        )
    with col2:
        depth = st.selectbox("Analysis Depth", ["Quick", "Detailed", "Deep Dive"], index=1)
    
    if st.button("🔍 Research & Analyze", type="primary", use_container_width=True):
        if not query.strip():
            st.error("❌ Please enter a question!")
        else:
            with st.spinner("🤖 AI is researching your question..."):
                try:
                    # Parse tickers if provided
                    tickers_list = []
                    if tickers_to_analyze.strip():
                        tickers_list = [t.strip().upper() for t in tickers_to_analyze.split(",") if t.strip()]
                    
                    # Build comprehensive analysis
                    analysis_data = {}
                    
                    if tickers_list:
                        st.info(f"📊 Analyzing {len(tickers_list)} ticker(s)...")
                        
                        for ticker in tickers_list:
                            with st.expander(f"📈 Data for {ticker}", expanded=False):
                                try:
                                    t = yf.Ticker(ticker)
                                    hist = t.history(period="3mo")
                                    info = getattr(t, "info", {}) or {}
                                    
                                    if hist is not None and not hist.empty:
                                        df = compute_indicators(hist)
                                        risk_score, risk_level, risk_breakdown = calculate_risk_score(df)
                                        signal = generate_signal(df)
                                        
                                        # Display quick metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        col1.metric("Price", f"${df['Close'].iloc[-1]:.2f}")
                                        col2.metric("Signal", signal)
                                        col3.metric("Risk", f"{risk_score}/100")
                                        col4.metric("RSI", f"{df['RSI'].iloc[-1]:.1f}")
                                        
                                        # Store data for AI
                                        analysis_data[ticker] = {
                                            "current_price": df['Close'].iloc[-1],
                                            "signal": signal,
                                            "risk_score": risk_score,
                                            "risk_level": risk_level,
                                            "rsi": df['RSI'].iloc[-1],
                                            "ema20": df['EMA20'].iloc[-1],
                                            "ema50": df['EMA50'].iloc[-1],
                                            "macd": df['MACD'].iloc[-1],
                                            "market_cap": info.get("marketCap", "N/A"),
                                            "pe_ratio": info.get("trailingPE", "N/A"),
                                            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
                                            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
                                            "sector": info.get("sector", "N/A"),
                                            "industry": info.get("industry", "N/A")
                                        }
                                    else:
                                        st.warning(f"No data for {ticker}")
                                except Exception as e:
                                    st.error(f"Error fetching {ticker}: {e}")
                    
                    # Build AI prompt
                    st.markdown("---")
                    st.subheader("🤖 AI Analysis")
                    
                    if depth == "Quick":
                        detail_instruction = "Provide a concise 3-4 paragraph analysis."
                    elif depth == "Detailed":
                        detail_instruction = "Provide a comprehensive analysis with clear sections and actionable insights."
                    else:
                        detail_instruction = "Provide an in-depth, research-grade analysis with detailed reasoning, multiple perspectives, and risk considerations."
                    
                    if analysis_data:
                        data_summary = "\n\n".join([
                            f"**{ticker}:**\n" + 
                            f"- Price: ${data['current_price']:.2f}\n" +
                            f"- Technical Signal: {data['signal']}\n" +
                            f"- Risk Score: {data['risk_score']}/100 ({data['risk_level']})\n" +
                            f"- RSI: {data['rsi']:.1f}\n" +
                            f"- EMA20: ${data['ema20']:.2f}, EMA50: ${data['ema50']:.2f}\n" +
                            f"- Market Cap: {data['market_cap']}\n" +
                            f"- P/E Ratio: {data['pe_ratio']}\n" +
                            f"- Sector: {data['sector']}, Industry: {data['industry']}\n"
                            for ticker, data in analysis_data.items()
                        ])
                        
                        prompt = f"""You are a senior investment analyst and financial advisor. A client has asked: "{query}"

I've gathered real-time market data for the following stocks:

{data_summary}

{detail_instruction}

Please provide:
1. **Direct Answer** to the question
2. **Stock Recommendations** with specific tickers and reasoning
3. **Risk Assessment** for each recommendation
4. **Entry/Exit Strategy** if applicable
5. **Key Catalysts** to watch
6. **Alternative Options** if relevant
7. **Important Disclaimers** and risks

Use the actual data provided above. Be specific, actionable, and honest about uncertainties."""
                    else:
                        prompt = f"""You are a senior investment analyst and financial advisor. A client has asked: "{query}"

{detail_instruction}

Please provide:
1. **Direct Answer** to the question
2. **Stock Recommendations** with specific reasoning (suggest 3-5 tickers if applicable)
3. **Analysis Framework** - what factors to consider
4. **Risk Considerations** 
5. **Market Context** - current market conditions relevant to this question
6. **Action Plan** - concrete steps the investor should take
7. **Important Disclaimers**

Be specific, provide ticker symbols where relevant, and explain your reasoning clearly."""
                    
                    with st.spinner("🧠 Generating comprehensive analysis..."):
                        ai_response = ask_gemini(prompt)
                    
                    st.markdown(ai_response)
                    
                    # Add disclaimer
                    st.warning("⚠️ **Disclaimer**: This is AI-generated analysis based on current data. Always conduct your own research and consult with a licensed financial advisor before making investment decisions.")
                    
                    # Save query option
                    if st.button("💾 Save this analysis"):
                        st.success("✅ Analysis saved! (Feature coming soon - will save to local database)")
                    
                except Exception as e:
                    st.error(f"❌ Error during research: {e}")

# ---------------------------
# Page: Market Screener (NEW)
# ---------------------------
elif page == "📊 Market Screener":
    st.header("📊 Market Screener")
    st.markdown("Find stocks matching your criteria with real-time technical analysis")
    
    st.markdown("### 🎯 Screening Criteria")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Technical Filters**")
        rsi_min = st.slider("Min RSI", 0, 100, 30, help="Find oversold (30) or overbought (70) stocks")
        rsi_max = st.slider("Max RSI", 0, 100, 70)
        
        signal_filter = st.multiselect(
            "Trading Signal",
            ["BUY", "SELL", "HOLD"],
            default=["BUY"],
            help="Filter by EMA crossover signal"
        )
    
    with col2:
        st.markdown("**⚠️ Risk Filters**")
        risk_max = st.slider("Max Risk Score", 0, 100, 60, help="Filter out high-risk stocks")
        
        sectors = st.multiselect(
            "Sectors (optional)",
            ["Technology", "Healthcare", "Finance", "Energy", "Consumer", "Industrial"],
            help="Leave empty for all sectors"
        )
    
    # Stock universe
    st.markdown("### 📋 Stock Universe")
    
    preset = st.radio(
        "Choose preset or custom:",
        ["🇺🇸 US Top 50", "🇮🇳 Indian Nifty 50", "💎 Custom List"],
        horizontal=True
    )
    
    if preset == "🇺🇸 US Top 50":
        stock_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "JPM", "JNJ",
                         "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "NFLX", "ADBE", "CRM",
                         "INTC", "CSCO", "PFE", "KO", "PEP", "ABT", "TMO", "COST", "AVGO", "ACN"]
    elif preset == "🇮🇳 Indian Nifty 50":
        stock_universe = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS", 
                         "ICICIBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
                         "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "TITAN.NS",
                         "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS", "NESTLEIND.NS"]
    else:
        custom_input = st.text_area(
            "Enter tickers (comma-separated)",
            "AAPL,MSFT,GOOGL,TSLA",
            help="Enter tickers with proper suffix (e.g., .NS for India)"
        )
        stock_universe = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    
    st.info(f"📊 Will screen {len(stock_universe)} stocks")
    
    if st.button("🔍 Run Screener", type="primary", use_container_width=True):
        with st.spinner(f"Screening {len(stock_universe)} stocks... This may take a minute..."):
            results = []
            progress_bar = st.progress(0)
            
            for idx, ticker in enumerate(stock_universe):
                try:
                    t = yf.Ticker(ticker)
                    hist = t.history(period="3mo")
                    
                    if hist is not None and not hist.empty and len(hist) > 50:
                        df = compute_indicators(hist)
                        risk_score, risk_level, _ = calculate_risk_score(df)
                        signal = generate_signal(df)
                        
                        current_rsi = df['RSI'].iloc[-1]
                        current_price = df['Close'].iloc[-1]
                        
                        # Apply filters
                        if (rsi_min <= current_rsi <= rsi_max and 
                            signal in signal_filter and 
                            risk_score <= risk_max):
                            
                            info = getattr(t, "info", {}) or {}
                            
                            results.append({
                                "Ticker": ticker,
                                "Price": f"${current_price:.2f}",
                                "Signal": signal,
                                "RSI": f"{current_rsi:.1f}",
                                "Risk": f"{risk_score:.0f}",
                                "Risk Level": risk_level,
                                "Sector": info.get("sector", "N/A"),
                                "Market Cap": info.get("marketCap", 0)
                            })
                    
                    progress_bar.progress((idx + 1) / len(stock_universe))
                    
                except Exception:
                    continue
            
            progress_bar.empty()
            
            if results:
                st.success(f"✅ Found {len(results)} stocks matching your criteria!")
                
                results_df = pd.DataFrame(results)
                
                # Sort by Risk Score (ascending)
                results_df = results_df.sort_values("Risk", ascending=True)
                
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                
                # AI Summary
                st.markdown("---")
                st.subheader("🤖 AI Screening Summary")
                
                top_picks = results_df.head(5)['Ticker'].tolist()
                
                prompt = f"""Based on a technical screening of stocks, these {len(results)} stocks passed the following criteria:
- RSI between {rsi_min} and {rsi_max}
- Trading signals: {', '.join(signal_filter)}
- Maximum risk score: {risk_max}

Top 5 picks from screening: {', '.join(top_picks)}

Please provide:
1. **Overview** of what these criteria mean
2. **Analysis of top 3 picks** - why they stand out
3. **Investment strategy** for these stocks
4. **Risks to consider**
5. **Next steps** for investors

Be concise but insightful."""
                
                with st.spinner("Getting AI insights..."):
                    ai_summary = ask_gemini(prompt)
                
                st.markdown(ai_summary)
                
            else:
                st.warning("⚠️ No stocks found matching your criteria. Try adjusting the filters.")

# ---------------------------
# Page: Compare Stocks (NEW)
# ---------------------------
elif page == "📚 Compare Stocks":
    st.header("📚 Side-by-Side Stock Comparison")
    st.markdown("Compare multiple stocks across all key metrics")
    
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
    
    if st.button("📊 Compare", type="primary", use_container_width=True):
        tickers = [t for t in [ticker1, ticker2, ticker3, ticker4] if t]
        
        if len(tickers) < 2:
            st.error("❌ Please enter at least 2 tickers to compare")
        else:
            comparison_data = []
            price_history = {}
            
            with st.spinner(f"Analyzing {len(tickers)} stocks..."):
                for ticker in tickers:
                    try:
                        t = yf.Ticker(ticker)
                        hist = t.history(period=compare_period)
                        info = getattr(t, "info", {}) or {}
                        
                        if hist is not None and not hist.empty:
                            df = compute_indicators(hist)
                            risk_score, risk_level, _ = calculate_risk_score(df)
                            signal = generate_signal(df)
                            
                            # Calculate returns
                            start_price = hist['Close'].iloc[0]
                            end_price = hist['Close'].iloc[-1]
                            returns = ((end_price - start_price) / start_price) * 100
                            
                            price_history[ticker] = hist['Close']
                            
                            comparison_data.append({
                                "Ticker": ticker,
                                "Price": f"${end_price:.2f}",
                                f"{compare_period} Return": f"{returns:+.2f}%",
                                "Signal": signal,
                                "RSI": f"{df['RSI'].iloc[-1]:.1f}",
                                "Risk Score": f"{risk_score:.0f}/100",
                                "Risk Level": risk_level,
                                "Market Cap": info.get("marketCap", "N/A"),
                                "P/E Ratio": info.get("trailingPE", "N/A"),
                                "Sector": info.get("sector", "N/A"),
                                "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
                                "52W Low": info.get("fiftyTwoWeekLow", "N/A")
                            })
                    except Exception as e:
                        st.error(f"Error analyzing {ticker}: {e}")
            
            if comparison_data:
                # Comparison table
                st.subheader("📊 Comparison Table")
                comp_df = pd.DataFrame(comparison_data)
                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                
                # Price comparison chart
                st.subheader("📈 Price Performance Comparison")
                
                if price_history:
                    fig = go.Figure()
                    
                    for ticker, prices in price_history.items():
                        # Normalize to 100 for fair comparison
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
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # AI Comparison
                st.markdown("---")
                st.subheader("🤖 AI Comparison Analysis")
                
                comparison_summary = "\n".join([
                    f"**{row['Ticker']}:** Price ${row['Price']}, {compare_period} Return: {row[f'{compare_period} Return']}, "
                    f"Signal: {row['Signal']}, Risk: {row['Risk Score']}, P/E: {row['P/E Ratio']}, Sector: {row['Sector']}"
                    for row in comparison_data
                ])
                
                prompt = f"""Compare these stocks for investment decision:

{comparison_summary}

Provide a comprehensive comparison including:
1. **Winner Analysis** - Which stock is the best choice and why?
2. **Strengths & Weaknesses** of each stock
3. **Risk Comparison** - Which is safest/riskiest?
4. **Investment Scenarios** - Which stock for which type of investor?
5. **Key Differentiators** - What sets each apart?
6. **Final Recommendation** with reasoning

Be specific and actionable. Consider both technical and fundamental factors."""
                
                with st.spinner("🤖 Generating comparison insights..."):
                    ai_comparison = ask_gemini(prompt)
                
                st.markdown(ai_comparison)
                
                st.info("💡 **Tip**: Use this comparison to make informed decisions based on your investment goals and risk tolerance.")

# ---------------------------
# Page: Settings
# ---------------------------
elif page == "⚙️ Settings":
    st.header("⚙️ Settings & Diagnostics")
    
    # Admin authentication for sensitive settings
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if not st.session_state.admin_authenticated:
        st.warning("🔒 This page contains sensitive information. Admin authentication required.")
        admin_pw = st.text_input("Enter admin password", type="password", key="admin_pw")
        if st.button("🔓 Authenticate as Admin"):
            if admin_pw == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.success("✅ Admin authenticated!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("❌ Incorrect admin password")
        st.stop()
    
    # Security Settings
    st.subheader("🔒 Security Status")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Session Timeout", "30 min")
    col2.metric("Rate Limit", "30/min")
    col3.metric("Failed Attempts", st.session_state.failed_login_attempts)
    
    st.markdown("**Active Security Features:**")
    st.write("✅ Session timeout (30 minutes)")
    st.write("✅ Rate limiting (30 requests/minute)")
    st.write("✅ Failed login protection (5 attempts → 5 min lockout)")
    st.write("✅ Code obfuscation (right-click disabled)")
    st.write("✅ Developer tools blocked")
    st.write("✅ Admin password for sensitive settings")
    
    if st.button("🔄 Reset Failed Login Counter"):
        st.session_state.failed_login_attempts = 0
        st.session_state.lockout_until = 0
        st.success("✅ Login counter reset")
    
    st.markdown("---")
    
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
            # Only show partial key
            masked_key = GOOGLE_API_KEY[:8] + "..." + GOOGLE_API_KEY[-4:]
            st.success(f"✅ API Key: {masked_key}")
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
                    model_list = [m.name for m in models if "generateContent" in getattr(m, "supported_generation_methods", [])]
                    st.write(f"Found {len(model_list)} compatible models:")
                    st.json(model_list[:10])  # Show first 10
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
            confirm = st.checkbox("⚠️ Confirm deletion")
            if confirm:
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
    
    # System Information (Limited)
    st.subheader("ℹ️ System Information")
    
    info_data = {
        "Status": "Active",
        "Database": "Connected",
        "Authenticated": "Yes",
        "Auto-refresh": f"{refresh_auto}s" if refresh_auto > 0 else "Disabled"
    }
    
    for key, value in info_data.items():
        st.write(f"**{key}:** {value}")
    
    st.markdown("---")
    
    # Change Passwords
    st.subheader("🔑 Change Passwords")
    
    with st.expander("Change App Password"):
        st.warning("⚠️ Note: Password changes require updating Streamlit secrets")
        st.code("""
# Add to .streamlit/secrets.toml:
APP_PASSWORD = "your-new-password"
ADMIN_PASSWORD = "your-admin-password"
        """)
    
    # API Testing (Limited)
    st.markdown("---")
    st.subheader("🧪 API Testing")
    
    test_ticker = st.text_input("Test Ticker", "AAPL")
    
    if st.button("🧪 Test Yahoo Finance API"):
        if not check_rate_limit():
            st.stop()
        
        with st.spinner(f"Testing {test_ticker}..."):
            try:
                t = yf.Ticker(test_ticker)
                data = t.history(period="1d")
                if data is not None and not data.empty:
                    st.success("✅ Yahoo Finance API working")
                    st.write("Latest Close:", f"${data['Close'].iloc[-1]:.2f}")
                else:
                    st.error("❌ No data returned")
            except Exception as e:
                st.error(f"❌ API Error: {e}")
    
    if st.button("🧪 Test Gemini API"):
        if not check_rate_limit():
            st.stop()
        
        with st.spinner("Testing Gemini..."):
            response = ask_gemini("Respond with 'API working' if you receive this message.")
            if "API working" in response or "working" in response.lower():
                st.success("✅ Gemini API is working!")
            st.write("**Response:**")
            st.write(response)
    
    # Logout admin
    st.markdown("---")
    if st.button("🚪 Logout from Admin", use_container_width=True):
        st.session_state.admin_authenticated = False
        st.rerun()

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")

# Check for triggered alerts in footer
if "authenticated" in st.session_state and st.session_state.authenticated:
    triggered = check_alerts()
    if triggered:
        alert_msg = f"🔔 {len(triggered)} alert(s) triggered! Check Watchlist & Alerts page."
        st.warning(alert_msg)

st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Investment Terminal</strong> — Built for Captain Suhas</p>
    <p>🗄️ Local SQLite persistence enabled | 🤖 Powered by Gemini AI</p>
    <p style='font-size: 0.8em;'>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)