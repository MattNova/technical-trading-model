import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

# NOTE: These imports assume the file structure is correct, as provided in the project summary.
# If these imports fail, it means the directory structure is incorrect.
from data_providers.fmp_provider import FMPProvider
from analyzers.financial_analyzer import FinancialAnalyzer
from utils.plotting import create_analysis_chart

# --- CONFIGURATION ---
st.set_page_config(layout="wide", page_title="Trading Model")

# --- API Key Management (Secure) ---
try:
    # Use Streamlit Secrets (recommended)
    FMP_API_KEY = st.secrets["fmp_api_key"]
except KeyError:
    # Fallback to Environment Variable
    FMP_API_KEY = os.environ.get("FMP_API_KEY")
    if not FMP_API_KEY:
        st.error("FMP API Key not configured. Please add 'fmp_api_key' to your secrets or environment.")
        st.stop()

# --- CACHED DATA LOAD (Calculations for all 500+ tickers) ---
@st.cache_data(ttl=3600, show_spinner="Running full analysis on 500+ stocks (cached for 1 hour)...")
def run_full_analysis(api_key):
    """
    Fetches S&P 500 list, gets historical data, runs technical analysis,
    and runs fundamental analysis for all available tickers.
    """
    print("--- RUNNING FULL ANALYSIS (from Streamlit) ---")
    provider = FMPProvider(api_key=api_key)
    analyzer = FinancialAnalyzer()

    # Get Tickers: SPY, QQQ + S&P 500 constituents
    default_watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN"]
    watchlist = default_watchlist
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
        response = requests.get(url, timeout=10); response.raise_for_status()
        data = response.json()
        if data:
            tickers = [stock['symbol'] for stock in data]
            if tickers and len(tickers) > 10:
                # Combine defaults and S&P 500, removing duplicates
                watchlist = list(dict.fromkeys(default_watchlist + tickers))[:502]
    except Exception as e:
        print(f"Could not fetch S&P 500 list: {e}. Falling back to default.")
    
    all_tech_data, all_fund_data, all_fund_ranks = {}, {}, {}
    progress_bar = st.progress(0, "Analyzing stocks...")
    
    for i, ticker in enumerate(watchlist):
        progress_bar.progress((i + 1) / len(watchlist), f"Analyzing {ticker} ({i+1}/{len(watchlist)})...")
        # Throttle to respect FMP limits (5 reqs/sec for free tier, adjusted here)
        if i > 0 and i % 5 == 0:
            time.sleep(1.5) 
            
        tech_df = provider.get_daily_stock_data(ticker, '1990-01-01', pd.to_datetime('today').strftime('%Y-%m-%d'))
        
        if not tech_df.empty and len(tech_df) > 200:
            data_with_indicators, _ = analyzer.run_full_analysis(tech_df.copy())
            all_tech_data[ticker] = data_with_indicators
        
        # Only fetch fundamentals for non-index tickers
        if ticker not in ["SPY", "QQQ"] and not tech_df.empty:
            fund_df = provider.get_daily_fundamental_ratios(ticker, daily_prices=tech_df)
            
            if not fund_df.empty:
                for metric in ['P/E', 'P/S', 'PEG']:
                    if metric in fund_df.columns:
                        # Pre-calculate rank for plotting in Fundamental Explorer
                        fund_df[f'{metric}_Rank_Plot'] = fund_df[metric].expanding(min_periods=20).apply(
                            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 1 else np.nan, raw=False)
                all_fund_data[ticker] = fund_df
                _, ranks = analyzer.run_full_fundamental_analysis(fund_df)
                all_fund_ranks[ticker] = ranks

    progress_bar.empty()
    return all_tech_data, all_fund_data, all_fund_ranks

def get_quick_prices(api_key, tickers):
    """Fetches the latest real-time prices for a list of tickers."""
    provider = FMPProvider(api_key=api_key)
    prices = {}
    with st.spinner(f"Fetching latest prices for {len(tickers)} stocks..."):
        for i, ticker in enumerate(tickers):
            if i > 0 and i % 5 == 0:
                time.sleep(1.5)
            price_data = provider.get_latest_price(ticker)
            if price_data:
                prices[ticker] = price_data
    st.toast("Latest prices refreshed!")
    return prices

# --- Fundamental Plotting Function (Unchanged) ---
def create_fundamental_chart(df: pd.DataFrame, metric: str, title: str):
    if df.empty or metric not in df.columns: return None
    rank_col = f'{metric}_Rank_Plot'
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df.index, y=df[metric], name=f'{metric} Value', line=dict(color='blue')), secondary_y=False)
    if rank_col in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[rank_col], name='Percentile Rank', line=dict(color='red', dash='dot')), secondary_y=True)
        fig.add_hline(y=25, line_dash="dash", line_color="green", opacity=0.5, secondary_y=True)
        fig.add_hline(y=75, line_dash="dash", line_color="red", opacity=0.5, secondary_y=True)
    fig.update_layout(title_text=title, hovermode="x unified", showlegend=False)
    fig.update_yaxes(title_text=f"<b>{metric} Value</b>", secondary_y=False, showgrid=False)
    fig.update_yaxes(title_text="<b>Percentile Rank (%)</b>", secondary_y=True, range=[0, 100], showgrid=True, gridcolor='#D3D3D3')
    if metric == 'P/E':
        pe_min = df['P/E'][df['P/E'] > -100].quantile(0.01) if not df['P/E'][df['P/E'] > -100].empty else 0
        pe_max = df['P/E'][df['P/E'] < 200].quantile(0.99) if not df['P/E'][df['P/E'] < 200].empty else 50
        fig.update_yaxes(range=[pe_min, pe_max * 1.1], secondary_y=False)
    elif metric == 'P/S' and 'P/S' in df.columns and not df['P/S'].empty: 
        fig.update_yaxes(range=[0, df['P/S'].quantile(0.99) * 1.1], secondary_y=False)
    elif metric == 'PEG': 
        fig.update_yaxes(range=[-5, 5], secondary_y=False)
    return fig

# --- App Layout ---
st.sidebar.title("App Controls")
page = st.sidebar.radio("Select a Page", ["Technical Dashboard", "Fundamental Explorer"])

st.sidebar.markdown("---")
st.sidebar.subheader("Manual Data Control")

if 'show_password_prompt' not in st.session_state:
    st.session_state.show_password_prompt = False

if st.sidebar.button("Run FULL Analysis (Clear Cache)"):
    st.session_state.show_password_prompt = True

if st.session_state.show_password_prompt:
    with st.sidebar.form("password_form"):
        password = st.text_input("Enter Admin Password", type="password")
        submitted = st.form_submit_button("Submit")
        if submitted:
            try:
                correct_password = st.secrets["admin_password"]
                if password == correct_password:
                    st.session_state.show_password_prompt = False
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Incorrect password")
            except KeyError:
                st.error("Admin password is not set in secrets.")

# Load the core data (cached or newly run)
try:
    tech_data, fund_data, fund_ranks = run_full_analysis(FMP_API_KEY)
except Exception as e:
    st.error(f"A critical error occurred during analysis: {e}")
    st.stop()

if not tech_data:
    st.error("The main analysis returned 0 stocks. Check your FMP API Key and limits.")
    st.stop()
    
# Quick Price Refresh State
if 'quick_prices' not in st.session_state:
    st.session_state.quick_prices = {}

if st.sidebar.button("Quick Price Refresh"):
    st.session_state.quick_prices = get_quick_prices(FMP_API_KEY, list(tech_data.keys()))
    st.rerun() 

st.title("📈 Trading Model Dashboard")
st.success(f"Analysis complete for {len(tech_data)} stocks.")

if st.session_state.quick_prices:
    st.markdown("### Current Price Snapshot")
    price_comparison = []
    for ticker, latest_data in st.session_state.quick_prices.items():
        if ticker in tech_data and not tech_data[ticker].empty:
            last_close = tech_data[ticker]['close'].iloc[-1]
            latest_price = latest_data.get('price')
            if latest_price:
                change = latest_price - last_close
                change_pct = (change / last_close) * 100
                price_comparison.append({ 'Ticker': ticker, 'Analysis Close': f"${last_close:.2f}", 'Latest Price': f"${latest_price:.2f}", 'Change ($)': f"{change:.2f}", 'Change (%)': f"{change_pct:.2f}%" })
    if price_comparison:
        comparison_df = pd.DataFrame(price_comparison).set_index('Ticker')
        st.dataframe(comparison_df.head(25), use_container_width=True)

# --- TECHNICAL DASHBOARD PAGE ---
if page == "Technical Dashboard":
    st.header("Technical Signals and Rankings")
    
    # 1. Build the full Tear Sheet DataFrame
    tear_sheet_data = []
    for ticker, df in tech_data.items():
        if df.empty: continue
        latest = df.iloc[-1]
        row = { 
            'Ticker': ticker, 
            'Trend_Score': latest.get('Trend_Score'), 
            'Reversion_Score': latest.get('Reversion_Score'), 
            'Close': f"${latest.get('close'):.2f}", 
            'RSI': f"{latest.get('RSI_14'):.2f}", 
            'Stoch': f"{latest.get('STOCHk_14_3_3'):.2f}"
        }
        if ticker in fund_ranks:
            # We must instantiate the analyzer here to get the summary from the existing data
            summary, _ = FinancialAnalyzer().run_full_fundamental_analysis(fund_data.get(ticker, pd.DataFrame()))
            for metric in ['P/E', 'P/S', 'PEG']:
                row[f'{metric}_Short_Term_Rank'] = summary.get(f'{metric}_Short_Term_Rank')
                row[f'{metric}_Long_Term_Rank'] = summary.get(f'{metric}_Long_Term_Rank')
        tear_sheet_data.append(row)
    
    if tear_sheet_data:
        tear_sheet_df = pd.DataFrame(tear_sheet_data).set_index('Ticker')

        # --- TREND-FOLLOWING SECTION (UPDATED) ---
        st.header("📈 Trend-Following Signals")
        col1, col2 = st.columns(2)
        
        # Long Trend (Highest Positive Trend_Score)
        with col1: 
            st.subheader("Strong Long Trends (Top 25)")
            long_trends = tear_sheet_df.sort_values(by="Trend_Score", ascending=False).head(25)
            st.dataframe(long_trends, use_container_width=True)
            
        # Short Trend (Lowest Negative Trend_Score)
        with col2: 
            st.subheader("Strong Short Trends (Bottom 25)")
            # Sorting ASC will put the most negative scores (e.g., -3) at the top
            short_trends = tear_sheet_df.sort_values(by="Trend_Score", ascending=True).head(25)
            st.dataframe(short_trends, use_container_width=True)

        # --- MEAN REVERSION SECTION (UNCHNAGED) ---
        st.header("📉 Mean Reversion Signals") 
        col1, col2 = st.columns(2)
        
        with col1: 
            st.subheader("Overbought (Bearish)")
            st.dataframe(tear_sheet_df[tear_sheet_df['Reversion_Score'] < 0].sort_values(by="Reversion_Score").head(25), use_container_width=True)
        with col2: 
            st.subheader("Oversold (Bullish)")
            st.dataframe(tear_sheet_df[tear_sheet_df['Reversion_Score'] > 0].sort_values(by="Reversion_Score", ascending=False).head(25), use_container_width=True)

        # --- FUNDAMENTAL RANKINGS (UNCHNAGED) ---
        st.header("💎 Fundamental Valuation Rankings (Long-Term)") 
        fund_rank_cols = [c for c in tear_sheet_df.columns if 'Long_Term_Rank' in c]
        if fund_rank_cols:
            fund_summary_df = tear_sheet_df[fund_rank_cols].copy().dropna(how='all')
            if not fund_summary_df.empty:
                fund_summary_df['Overall_Rank'] = fund_summary_df.mean(axis=1, skipna=True)
                # Lower rank is better (cheaper relative to history)
                st.dataframe(fund_summary_df.sort_values(by='Overall_Rank').dropna(subset=['Overall_Rank']).head(25), use_container_width=True)

        # --- ON-DEMAND TICKER LOOKUP (NEW SECTION) ---
        st.header("🔍 On-Demand Ticker Lookup")
        
        # Use st.selectbox with a search feature, listing all available tickers
        all_tickers = list(tech_data.keys())
        selected_ticker = st.selectbox(
            "Select or type any analyzed stock ticker:", 
            options=all_tickers, 
            index=all_tickers.index("SPY") if "SPY" in all_tickers else 0 # Default to SPY or first ticker
        )
        
        if selected_ticker and selected_ticker in tech_data:
            st.plotly_chart(create_analysis_chart(selected_ticker, tech_data[selected_ticker]), use_container_width=True)
        else:
            st.warning("Please select a ticker to view its detailed technical analysis chart.")


# --- FUNDAMENTAL EXPLORER PAGE (UNCHNAGED) ---
elif page == "Fundamental Explorer":
    st.title("💎 Fundamental Valuation Explorer")
    ticker_list = list(fund_data.keys())
    if not ticker_list: 
        st.warning("No stocks with fundamental data were found (SPY/QQQ excluded)."); 
        st.stop()
        
    selected_ticker = st.selectbox("Select a stock for detailed analysis:", ticker_list)
    
    st.header(f"Valuation Ranking Summary for {selected_ticker}")
    ranks = fund_ranks.get(selected_ticker, {})
    raw_data = fund_data.get(selected_ticker)
    
    summary_rows = []
    for metric in ['P/E', 'P/S', 'PEG']:
        metric_ranks = ranks.get(metric, {})
        row = {'Metric': metric}
        if raw_data is not None and metric in raw_data.columns and not raw_data.empty:
            row['Current'] = f"{raw_data[metric].iloc[-1]:.2f}"
        
        timeframes = ['30 days', '90 days', '120 days', '1 year', '3 years', '5 years', '10 years', 'Full History']
        for timeframe in timeframes:
            rank_val = metric_ranks.get(timeframe)
            row[timeframe] = f"{rank_val:.1f}%" if pd.notna(rank_val) else "N/A"
        summary_rows.append(row)
        
    st.dataframe(pd.DataFrame(summary_rows).set_index('Metric'), use_container_width=True)
    
    st.header(f"Historical Ratio Charts for {selected_ticker} (Daily)")
    if raw_data is not None:
        tab1, tab2, tab3 = st.tabs(["P/E Ratio", "P/S Ratio", "PEG Ratio"])
        with tab1:
            fig = create_fundamental_chart(raw_data, 'P/E', f"{selected_ticker} Daily P/E Ratio & Rank")
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.warning("No P/E data to plot.")
        with tab2:
            fig = create_fundamental_chart(raw_data, 'P/S', f"{selected_ticker} Daily P/S Ratio & Rank")
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.warning("No P/S data to plot.")
        with tab3:
            fig = create_fundamental_chart(raw_data, 'PEG', f"{selected_ticker} Daily PEG Ratio & Rank")
            if fig: st.plotly_chart(fig, use_container_width=True)
            else: st.warning("No PEG data to plot.")