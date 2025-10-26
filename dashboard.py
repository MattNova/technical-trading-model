import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_providers.fmp_provider import FMPProvider
from analyzers.financial_analyzer import FinancialAnalyzer
from utils.plotting import create_analysis_chart

st.set_page_config(layout="wide", page_title="Trading Model")

FMP_API_KEY = "KsgRir2v2TpyStK6KaerT6cQbNw2Av32"

@st.cache_data(ttl=3600)
def run_full_analysis(api_key):
    print("--- RUNNING FULL ANALYSIS (from Streamlit) ---")
    provider = FMPProvider(api_key=api_key)
    analyzer = FinancialAnalyzer()

    # Get Watchlist (S&P 500)
    watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"]
    try:
        url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
        response = requests.get(url); response.raise_for_status()
        tickers = [stock['symbol'] for stock in response.json()]
        watchlist = list(dict.fromkeys(["SPY", "QQQ"] + tickers))[:502]
    except Exception as e:
        print(f"Could not fetch S&P 500 list: {e}")

    all_tech_data, all_fund_data, all_fund_ranks = {}, {}, {}
    progress_bar = st.progress(0, "Analyzing stocks...")
    
    for i, ticker in enumerate(watchlist):
        progress_bar.progress((i + 1) / len(watchlist), f"Analyzing {ticker}...")
        
        # 1. Technical Analysis
        tech_df = provider.get_daily_stock_data(ticker, '1990-01-01', pd.to_datetime('today').strftime('%Y-%m-%d'))
        if not tech_df.empty and len(tech_df) > 200:
            all_tech_data[ticker], _ = analyzer.run_full_analysis(tech_df.copy())
        
        # 2. Fundamental Analysis
        if ticker not in ["SPY", "QQQ"]:
            fund_df = provider.get_daily_fundamental_ratios(ticker)
            if not fund_df.empty:
                # *** CRITICAL FIX: Calculate and add the percentile rank column for plotting ***
                for metric in ['P/E', 'P/S', 'PEG']:
                    if metric in fund_df.columns:
                        # Calculate an expanding percentile rank for each day vs its own history
                        fund_df[f'{metric}_Rank_Plot'] = fund_df[metric].expanding(min_periods=20).apply(
                            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 1 else np.nan,
                            raw=False
                        )
                
                all_fund_data[ticker] = fund_df
                _, ranks = analyzer.run_full_fundamental_analysis(fund_df)
                all_fund_ranks[ticker] = ranks

    progress_bar.empty()
    return all_tech_data, all_fund_data, all_fund_ranks

# --- Charting Function for Fundamental Data ---
def create_fundamental_chart(df: pd.DataFrame, metric: str, title: str):
    if df.empty or metric not in df.columns:
        return None
        
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
    
    if metric == 'P/E': fig.update_yaxes(range=[-50, 100], secondary_y=False)
    if metric == 'P/S' and not df['P/S'].empty: fig.update_yaxes(range=[0, df['P/S'].quantile(0.99) * 1.1], secondary_y=False)
    if metric == 'PEG': fig.update_yaxes(range=[-5, 5], secondary_y=False)
    
    return fig

# --- Main App Logic ---
st.sidebar.title("App Controls")
if st.sidebar.button("Refresh All Data (Clear Cache)"):
    st.cache_data.clear()
    st.rerun()

page = st.sidebar.radio("Select a Page", ["Technical Dashboard", "Fundamental Explorer"])

try:
    tech_data, fund_data, fund_ranks = run_full_analysis(FMP_API_KEY)
    st.success(f"Analysis complete for {len(tech_data)} stocks.")
except Exception as e:
    st.error(f"A critical error occurred during analysis startup: {e}")
    st.stop()

st.title("📈 Trading Model Dashboard")

# ==============================================================================
# PAGE 1: TECHNICAL DASHBOARD
# ==============================================================================
if page == "Technical Dashboard":
    st.header("Technical Signals and Rankings")
    
    tear_sheet_data = []
    for ticker, df in tech_data.items():
        if df.empty: continue
        latest = df.iloc[-1]
        row = {'Ticker': ticker, 'Trend_Score': latest.get('Trend_Score'), 'Reversion_Score': latest.get('Reversion_Score'),
               'Close': latest.get('close'), 'RSI': latest.get('RSI_14'), 'Stoch': latest.get('STOCHk_14_3_3')}
        
        if ticker in fund_ranks:
            summary, _ = FinancialAnalyzer().run_full_fundamental_analysis(fund_data.get(ticker, pd.DataFrame()))
            for metric in ['P/E', 'P/S', 'PEG']:
                row[f'{metric}_Short_Term_Rank'] = summary.get(f'{metric}_Short_Term_Rank')
                row[f'{metric}_Long_Term_Rank'] = summary.get(f'{metric}_Long_Term_Rank')
        tear_sheet_data.append(row)

    tear_sheet_df = pd.DataFrame(tear_sheet_data).set_index('Ticker')

    st.header("📈 Trend-Following Signals")
    st.dataframe(tear_sheet_df.sort_values(by="Trend_Score", ascending=False).head(25))
    st.header("📉 Mean Reversion Signals")
    col1, col2 = st.columns(2)
    with col1: st.subheader("Overbought (Bearish)"); st.dataframe(tear_sheet_df[tear_sheet_df['Reversion_Score'] < 0].sort_values(by="Reversion_Score").head(25))
    with col2: st.subheader("Oversold (Bullish)"); st.dataframe(tear_sheet_df[tear_sheet_df['Reversion_Score'] > 0].sort_values(by="Reversion_Score", ascending=False).head(25))
    
    st.header("💎 Fundamental Valuation Rankings")
    fund_rank_cols = [col for col in tear_sheet_df.columns if 'Long_Term_Rank' in col]
    if fund_rank_cols:
        fund_summary_df = tear_sheet_df[fund_rank_cols].copy().dropna(how='all')
        if not fund_summary_df.empty:
            fund_summary_df['Overall_Rank'] = fund_summary_df[fund_rank_cols].mean(axis=1, skipna=True)
            st.dataframe(fund_summary_df.sort_values(by='Overall_Rank').dropna(subset=['Overall_Rank']).head(25))

    st.header("🔬 Chart Explorer")
    selected_ticker = st.selectbox("Select a stock to chart:", list(tech_data.keys()))
    if selected_ticker and selected_ticker in tech_data:
        fig = create_analysis_chart(selected_ticker, tech_data[selected_ticker])
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 2: FUNDAMENTAL EXPLORER
# ==============================================================================
elif page == "Fundamental Explorer":
    st.title("💎 Fundamental Valuation Explorer")

    ticker_list = list(fund_data.keys())
    if not ticker_list:
        st.warning("No stocks with fundamental data were found.")
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
        
        for timeframe in ['30 days', '90 days', '120 days', '1 year', '3 years', '5 years', '10 years', 'Full History']:
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