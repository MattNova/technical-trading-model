import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_analysis_chart(ticker: str, df: pd.DataFrame):
    if df.empty: return None
    plot_df = df.copy()
    # Ensure a clean DatetimeIndex for plotting and comparisons
    try:
        plot_df.index = pd.to_datetime(plot_df.index, errors='coerce')
        plot_df = plot_df[plot_df.index.notna()].sort_index()
    except Exception:
        pass
    # Ensure numeric OHLC and drop invalid rows
    for col in ['open','high','low','close','volume']:
        if col in plot_df.columns:
            plot_df[col] = pd.to_numeric(plot_df[col], errors='coerce')
    plot_df = plot_df.dropna(subset=['open','high','low','close'])
    if plot_df.empty: return None
    
    # Guard against pathologic min dates (e.g., epoch 1970) by clipping to first valid date
    first_ts = plot_df.index[0]
    if first_ts.year < 1980:
        plot_df = plot_df[plot_df.index >= plot_df.index[plot_df.index.get_loc(first_ts)]]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=('', 'RSI & Stochastics (Reversion)', 'MACD (Trend)'),
                        row_heights=[0.80, 0.10, 0.10])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], 
                                 increasing=dict(line=dict(color='#00CC96')), decreasing=dict(line=dict(color='#EF553B')), name='Price'), row=1, col=1)
    
    # Reversion Regime Change Visualization
    if 'Reversion_Score' in plot_df.columns and len(plot_df.index) > 0:
        plot_df['reversion_state'] = np.where(plot_df['Reversion_Score'] > 0, 1, np.where(plot_df['Reversion_Score'] < 0, -1, 0))
        plot_df['regime_change'] = plot_df['reversion_state'].diff().fillna(0)
        regime_starts = plot_df.index[plot_df['regime_change'] != 0]
        # Ensure the first index is included as a start bound
        if len(regime_starts) == 0 or regime_starts[0] != plot_df.index[0]:
            regime_starts = pd.Index([plot_df.index[0]]).append(regime_starts)
        for i, start in enumerate(regime_starts):
            state = plot_df.loc[start, 'reversion_state']
            end = regime_starts[i+1] if i+1 < len(regime_starts) else plot_df.index[-1]
            if state != 0: 
                fig.add_vrect(x0=start, x1=end, fillcolor="green" if state > 0 else "red", opacity=0.15, layer="below", row=1, col=1)

    # --- MA RIBBON TRACES (5, 10, 15, 20) ---
    ma_colors = {'SMA_5': 'lime', 'SMA_10': 'cyan', 'SMA_15': 'yellow', 'SMA_20': 'orange'}
    for k, c in ma_colors.items():
        if k in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[k], line=dict(color=c, width=1), name=k.replace('SMA_','')), row=1, col=1)
            
    # Include 50 & 200 for reference
    if 'SMA_50' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], line=dict(color='gray', width=1, dash='dot'), name='50 SMA'), row=1, col=1)
    if 'SMA_200' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], line=dict(color='red', width=1, dash='dash'), name='200 SMA'), row=1, col=1)

    # Bollinger Bands and Subplots
    if 'BBU_10_1.0' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBU_10_1.0'], line=dict(color='gray', dash='dash', width=1), name='BB U (1σ)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBL_10_1.0'], line=dict(color='gray', dash='dash', width=1), name='BB L (1σ)'), row=1, col=1)
    if 'RSI_14' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI_14'], line=dict(color='blue'), name='RSI'), row=2, col=1)
    if 'STOCHk_14_3_3' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['STOCHk_14_3_3'], line=dict(color='orange', dash='dot'), name='Stoch %K'), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1); fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
    if 'MACD_12_26_9' in plot_df.columns:
        hist_col = 'MACDh_12_26_9'
        if hist_col not in plot_df.columns:
            plot_df[hist_col] = plot_df['MACD_12_26_9'] - plot_df['MACDs_12_26_9']
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_12_26_9'], line=dict(color='purple'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACDs_12_26_9'], line=dict(color='orange', dash='dot'), name='Signal'), row=3, col=1)
        colors = ['green' if val >= 0 else 'red' for val in plot_df[hist_col]]; fig.add_trace(go.Bar(x=plot_df.index, y=plot_df[hist_col], marker_color=colors, name='Hist'), row=3, col=1)
    
    # --- TEAR SHEET/ANNOTATION UPDATE ---
    latest = plot_df.iloc[-1]; 
    def colorize(s): 
        if pd.isna(s): return "<span style='color:gray;font-weight:bold;'>N/A</span>"
        s=int(s); c='green' if s > 0 else 'red' if s < 0 else 'gray'; return f"<span style='color:{c};font-weight:bold;'>{s}</span>"
    def fv(v): return f"{v:.2f}" if pd.notna(v) else "N/A"
    
    trend_score=latest.get('Trend_Score', np.nan); rev_score=latest.get('Reversion_Score', np.nan)
    tear_sheet = (f"<b>TREND (Max +/-4):</b> Total={colorize(trend_score)} | Ribbon={colorize(latest.get('score_ma_ribbon',np.nan))} | 200SMA={colorize(latest.get('score_200sma',np.nan))} | MACD={colorize(latest.get('score_macd',np.nan))} | Cross={colorize(latest.get('score_cross',np.nan))}<br>"
                  f"<b>REVERSION (Max +/-4):</b> Total={colorize(rev_score)} | RSI={colorize(latest.get('score_rsi',np.nan))} | Stoch={colorize(latest.get('score_stoch',np.nan))} | Pattern={colorize(latest.get('score_patterns',np.nan))} | BBands={colorize(latest.get('score_bbands',np.nan))} | Conf={fv(latest.get('Reversion_Confidence',np.nan))}<br>"
                  f"<b>VALUES:</b> RSI={fv(latest.get('RSI_14',np.nan))} | Stoch=%K={fv(latest.get('STOCHk_14_3_3',np.nan))} | MACD: {fv(latest.get('MACD_12_26_9',np.nan))}/{fv(latest.get('MACDs_12_26_9',np.nan))}")
    fig.add_annotation(text=tear_sheet, align='left', showarrow=False, xref='paper', yref='paper', x=0.01, y=1.0, bgcolor="rgba(50,50,50,0.95)", bordercolor="black", borderwidth=1, xanchor='left', yanchor='bottom', font=dict(size=11, color='white'))
    
    ma_ann = [{'n': '5 SMA', 'k': 'SMA_5', 'c': 'lime'}, {'n': '20 SMA', 'k': 'SMA_20', 'c': 'orange'}, {'n': '200 SMA', 'k': 'SMA_200', 'c': 'red'}]
    for i, ma in enumerate(ma_ann):
        val = latest.get(ma['k']); text = f"<span style='color:{ma['c']};'>{ma['n']}: {fv(val)}</span>" if pd.notna(val) else ""
        if pd.notna(val): fig.add_annotation(text=text, xref='paper', yref='y', x=1.0, y=val, showarrow=False, xanchor='left', yanchor='middle', font=dict(size=10,color='white'), bgcolor="rgba(0,0,0,0.5)", yshift=-i*12)

    # Overlay simple candlestick pattern markers (last 200)
    last_n = plot_df.tail(200)
    y_base = last_n['high'] * 1.01
    if 'pattern_hammer' in last_n.columns:
        idx = last_n.index[last_n['pattern_hammer'] == 1]
        if len(idx):
            fig.add_trace(go.Scatter(x=idx, y=last_n.loc[idx, 'low']*0.995, mode='markers', marker=dict(symbol='triangle-up', color='green', size=8), name='Hammer'), row=1, col=1)
    if 'pattern_engulfing_bull' in last_n.columns:
        idx = last_n.index[last_n['pattern_engulfing_bull'] == 1]
        if len(idx):
            fig.add_trace(go.Scatter(x=idx, y=y_base.loc[idx], mode='markers', marker=dict(symbol='circle', color='green', size=6), name='Bull Engulf'), row=1, col=1)
    if 'pattern_engulfing_bear' in last_n.columns:
        idx = last_n.index[last_n['pattern_engulfing_bear'] == 1]
        if len(idx):
            fig.add_trace(go.Scatter(x=idx, y=y_base.loc[idx], mode='markers', marker=dict(symbol='x', color='red', size=6), name='Bear Engulf'), row=1, col=1)
    if 'pattern_morning_star' in last_n.columns:
        idx = last_n.index[last_n['pattern_morning_star'] == 1]
        if len(idx):
            fig.add_trace(go.Scatter(x=idx, y=y_base.loc[idx], mode='markers', marker=dict(symbol='star', color='gold', size=8), name='Morning Star'), row=1, col=1)

    fig.update_layout(title_text=f"<b>{ticker} Technicals</b>", title_x=0.5, xaxis_rangeslider_visible=False, template='plotly', font=dict(color='black'), showlegend=False, margin=dict(r=120,t=160,b=80))
    fig.update_xaxes(showgrid=True, gridcolor='#E0E0E0', zerolinecolor='#C0C0C0'); fig.update_yaxes(showgrid=True, gridcolor='#E0E0E0', zerolinecolor='#C0C0C0')
    return fig