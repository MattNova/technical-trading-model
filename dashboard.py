import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time

from data_providers.fmp_provider import FMPProvider
from analyzers.financial_analyzer import FinancialAnalyzer
from utils.plotting import create_analysis_chart

st.set_page_config(layout="wide", page_title="Trading Model")

# --- API Key Management (Secure) ---
try:
    FMP_API_KEY = st.secrets["fmp_api_key"]
except KeyError:
    FMP_API_KEY = os.environ.get("FMP_API_KEY")
    if not FMP_API_KEY:
        st.error("API Key not configured.")
        st.stop()

@st.cache_data(ttl=3600, show_spinner="Running full analysis on 500+ stocks (cached for 1 hour)...")
def run_full_analysis(api_key):
    print("--- RUNNING FULL ANALYSIS (from Streamlit) ---")
    provider = FMPProvider(api_key=api_key)
    analyzer = FinancialAnalyzer()

    default_watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN"]
    watchlist = default_watchlist
    
    try:
        url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
        response = requests.get(url, timeout=10); response.raise_for_status()
        data = response.json()
        if data:
            tickers = [stock['symbol'] for stock in data]
            if tickers and len(tickers) > 10:
                watchlist = list(dict.fromkeys(default_watchlist + tickers))[:502]
    except Exception as e:
        print(f"Could not fetch S&P 500 list: {e}. Falling back to default.")
    
    all_tech_data, all_fund_data, all_fund_ranks = {}, {}, {}
    for i, ticker in enumerate(watchlist):
        if i > 0 and i % 5 == 0:
            time.sleep(1)
            
        tech_df = provider.get_daily_stock_data(ticker, '1990-01-01', pd.to_datetime('today').strftime('%Y-%m-%d'))
        if not tech_df.empty and len(tech_df) > 200:
            data_with_indicators, _ = analyzer.run_full_analysis(tech_df.copy())
            all_tech_data[ticker] = data_with_indicators
        
        if ticker not in ["SPY", "QQQ"]:
            fund_df = provider.get_daily_fundamental_ratios(ticker)
            if not fund_df.empty:
                for metric in ['P/E', 'P/S', 'PEG']:
                    if metric in fund_df.columns:
                        fund_df[f'{metric}_Rank_Plot'] = fund_df[metric].expanding(min_periods=20).apply(
                            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 1 else np.nan, raw=False)
                all_fund_data[ticker] = fund_df
                _, ranks = analyzer.run_full_fundamental_analysis(fund_df)
                all_fund_ranks[ticker] = ranks
    return all_tech_data, all_fund_data, all_fund_ranks

def get_quick_prices(api_key, tickers):
    provider = FMPProvider(api_key=api_key)
    prices = {}
    with st.spinner(f"Fetching latest prices for {len(tickers)} stocks..."):
        for i, ticker in enumerate(tickers):
            if i > 0 and i % 5 == 0:
                time.sleep(1)
            price_data = provider.get_latest_price(ticker)
            if price_data:
                prices[ticker] = price_data
    st.toast("Latest prices refreshed!")
    return prices

def create_fundamental_chart(df: pd.DataFrame, metric: str, title: str):
    # This function is now fixed
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
    
    # --- CRITICAL FIX: Use if/elif to prevent overwriting the y-axis range ---
    if metric == 'P/E':
        # Make the P/E range dynamic to handle different stocks
        pe_min = df['P/E'][df['P/E'] > -100].quantile(0.01)
        pe_max = df['P/E'][df['P/E'] < 200].quantile(0.99)
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

# --- PASSWORD PROTECTION LOGIC ---
if 'show_password_prompt' not in st.session_state:
    st.session_state.show_password_prompt = False

if st.sidebar.button("Run FULL Analysis (Clear Cache)"):
    st.session_state.show_password_prompt = True

if st.session_state.show_password_prompt:
    with st.sidebar.form("password_form"):
        password = st.text_input("Enter Admin Password", type="password")
        submitted = st.form_submit_button("Submit")
        if submitted:
            # Check password against the one stored in secrets
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

# --- Data Loading ---
try:
    tech_data, fund_data, fund_ranks = run_full_analysis(FMP_API_KEY)
except Exception as e:
    st.error(f"A critical error occurred during the main data analysis: {e}")
    st.stop()

if not tech_data:
    st.error("The main analysis returned 0 stocks. Check your FMP API Key and limits.")
    st.stop()
    
if 'quick_prices' not in st.session_state:
    st.session_state.quick_prices = {}

if st.sidebar.button("Quick Price Refresh"):
    st.session_state.quick_prices = get_quick_prices(FMP_API_KEY, list(tech_data.keys()))
    st.rerun() 

# --- Main App Display ---
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

if page == "Technical Dashboard":
    st.header("Technical Signals and Rankings")
    tear_sheet_data = []
    for ticker, df in tech_data.items():
        if df.empty: continue
        latest = df.iloc[-1]
        row = { 'Ticker': ticker, 'Trend_Score': latest.get('Trend_Score'), 'Reversion_Score': latest.get('Reversion_Score'), 'Close': f"${latest.get('close'):.2f}", 'RSI': f"{latest.get('RSI_14'):.2f}", 'Stoch': f"{latest.get('STOCHk_14_3_3'):.2f}"}
        if ticker in fund_ranks:
            summary, _ = FinancialAnalyzer().run_full_fundamental_analysis(fund_data.get(ticker, pd.DataFrame()))
            for metric in ['P/E', 'P/S', 'PEG']:
                row[f'{metric}_Short_Term_Rank'] = summary.get(f'{metric}_Short_Term_Rank')
                row[f'{metric}_Long_Term_Rank'] = summary.get(f'{metric}_Long_Term_Rank')
        tear_sheet_data.append(row)
    if tear_sheet_data:
        tear_sheet_df = pd.DataFrame(tear_sheet_data).set_index('Ticker')
        st.header("📈 Trend-Following Signals"); st.dataframe(tear_sheet_df.sort_values(by="Trend_Score", ascending=False).head(25), use_container_width=True)
        st.header("📉 Mean Reversion Signals"); col1, col2 = st.columns(2)
        with col1: st.subheader("Overbought (Bearish)"); st.dataframe(tear_sheet_df[tear_sheet_df['Reversion_Score'] < 0].sort_values(by="Reversion_Score").head(25), use_container_width=True)
        with col2: st.subheader("Oversold (Bullish)"); st.dataframe(tear_sheet_df[tear_sheet_df['Reversion_Score'] > 0].sort_values(by="Reversion_Score", ascending=False).head(25), use_container_width=True)
        st.header("💎 Fundamental Valuation Rankings (Long-Term)"); fund_rank_cols = [c for c in tear_sheet_df.columns if 'Long_Term_Rank' in c]
        if fund_rank_cols:
            fund_summary_df = tear_sheet_df[fund_rank_cols].copy().dropna(how='all')
            if not fund_summary_df.empty:
                fund_summary_df['Overall_Rank'] = fund_summary_df.mean(axis=1, skipna=True)
                st.dataframe(fund_summary_df.sort_values(by='Overall_Rank').dropna(subset=['Overall_Rank']).head(25), use_container_width=True)
        st.header("🔬 Chart Explorer"); selected_ticker = st.selectbox("Select a stock to chart:", list(tech_data.keys()))
        if selected_ticker and selected_ticker in tech_data:
            st.plotly_chart(create_analysis_chart(selected_ticker, tech_data[selected_ticker]), use_container_width=True)

elif page == "Fundamental Explorer":
    st.title("💎 Fundamental Valuation Explorer")
    ticker_list = list(fund_data.keys())
    if not ticker_list: st.warning("No stocks with fundamental data were found."); st.stop()
    selected_ticker = st.selectbox("Select a stock for detailed analysis:", ticker_list)
    st.header(f"Valuation Ranking Summary for {selected_ticker}"); ranks = fund_ranks.get(selected_ticker, {}); raw_data = fund_data.get(selected_ticker)
    summary_rows = []
    for metric in ['P/E', 'P/S', 'PEG']:
        metric_ranks = ranks.get(metric, {}); row = {'Metric': metric}
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