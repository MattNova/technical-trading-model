# utils/plotting.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_analysis_chart(ticker: str, df: pd.DataFrame, period: str = '1y'):
    if df.empty: return None

    end_date = df.index[-1]; start_date = end_date - pd.DateOffset(years=1)
    if period == '5y': start_date = end_date - pd.DateOffset(years=5)
    elif period == '3m': start_date = end_date - pd.DateOffset(months=3)
    plot_df = df.loc[start_date:end_date]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=('', 'RSI & Stochastics (Reversion)', 'MACD (Trend)'),
                        row_heights=[0.80, 0.10, 0.10])

    # Candlestick (Colors are explicit and work in both light/dark mode)
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], 
                                 increasing=dict(line=dict(color='#00CC96')), decreasing=dict(line=dict(color='#EF553B')), name='Price'), row=1, col=1)
    
    # Shading based on Reversion_Score (Logic remains the same)
    if 'Reversion_Score' in plot_df.columns:
        plot_df['reversion_state'] = np.where(plot_df['Reversion_Score'] > 0, 1, np.where(plot_df['Reversion_Score'] < 0, -1, 0))
        plot_df['regime_change'] = plot_df['reversion_state'].diff().fillna(plot_df['reversion_state'].iloc[0])
        regime_starts = plot_df[plot_df['regime_change'] != 0].index
        if plot_df['reversion_state'].iloc[0] != 0 and plot_df.index[0] not in regime_starts:
            regime_starts = plot_df.index[[0]].union(regime_starts)

        for i, start_regime in enumerate(regime_starts):
            current_state = plot_df.loc[start_regime, 'reversion_state']
            if current_state == 0: continue
            end_regime = regime_starts[i+1] if i+1 < len(regime_starts) else plot_df.index[-1]
            fill_color = "green" if current_state > 0 else "red"
            fig.add_vrect(x0=start_regime, x1=end_regime, fillcolor=fill_color, opacity=0.15, layer="below", row=1, col=1)

    # Indicator Traces - SMAs
    if 'SMA_50' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], line=dict(color='lime', width=1), name='50 SMA'), row=1, col=1)
    if 'SMA_100' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_100'], line=dict(color='orange', width=1), name='100 SMA'), row=1, col=1)
    if 'SMA_200' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], line=dict(color='red', width=2), name='200 SMA'), row=1, col=1)
    
    # NEW: Add Bollinger Bands Traces (Using 10-day, 1.5 StdDev)
    if 'BBU_10_1.5' in plot_df.columns:
        # The Middle Band (BBM) for BBands 10, 1.5 is BBL_10_1.5 + BBU_10_1.5 / 2 - we'll just draw the upper and lower.
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBU_10_1.5'], line=dict(color='gray', dash='dash', width=1), name='BBands U (1.5)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBL_10_1.5'], line=dict(color='gray', dash='dash', width=1), name='BBands L (1.5)'), row=1, col=1)
    
    # Indicator Traces - RSI & Stochastics (unchanged)
    if 'RSI_14' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['RSI_14'], line=dict(color='blue'), name='RSI'), row=2, col=1)
    if 'STOCHk_14_3_3' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['STOCHk_14_3_3'], line=dict(color='orange', dash='dot'), name='Stoch %K'), row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_color="red", row=2, col=1); fig.add_hline(y=20, line_dash="dash", line_color="green", row=2, col=1)
    
    # Indicator Traces - MACD (unchanged)
    if 'MACD_12_26_9' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACD_12_26_9'], line=dict(color='purple'), name='MACD'), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['MACDs_12_26_9'], line=dict(color='orange', dash='dot'), name='Signal Line'), row=3, col=1)
        colors = ['green' if val >= 0 else 'red' for val in plot_df['MACDh_12_26_9']]
        fig.add_trace(go.Bar(x=plot_df.index, y=plot_df['MACDh_12_26_9'], marker_color=colors, name='Histogram'), row=3, col=1)

    latest = plot_df.iloc[-1]
    
    # Helper functions and Score Banner (Unchanged, BBands Score is still calculated)
    def colorize_score(score): 
        score = int(score); color = 'green' if score > 0 else 'red' if score < 0 else 'gray'
        return f"<span style='color:{color}; font-weight:bold;'>{score}</span>"
    def format_value(value): return f"{value:.2f}"
    
    trend_score = int(latest.get('Trend_Score', 0)); reversion_score = int(latest.get('Reversion_Score', 0))

    tear_sheet_text = (
        f"<b>TREND:</b> Total={colorize_score(trend_score)} | 200 SMA Score={colorize_score(latest.get('score_200sma', 0))} | MACD Score={colorize_score(latest.get('score_macd', 0))}"
        f"<br>" 
        f"<b>REVERSION:</b> Total={colorize_score(reversion_score)} | RSI Score={colorize_score(latest.get('score_rsi', 0))} | Stoch Score={colorize_score(latest.get('score_stoch', 0))} | BBands Score={colorize_score(latest.get('score_bbands', 0))}"
        f"<br>" 
        f"<b>REVERSION VALUES:</b> RSI={format_value(latest.get('RSI_14', 0))} | Stoch=%K={format_value(latest.get('STOCHk_14_3_3', 0))} | MACD: Val={format_value(latest.get('MACD_12_26_9', 0))} / Sig={format_value(latest.get('MACDs_12_26_9', 0))}"
    )
    
    fig.add_annotation(text=tear_sheet_text, align='left', showarrow=False, xref='paper', yref='paper',
                       x=0.01, y=1.0, bgcolor="rgba(50, 50, 50, 0.95)", bordercolor="black", borderwidth=1,
                       xanchor='left', yanchor='bottom', font=dict(size=11, color='white')) 
    
    # MA Annotations
    ma_annotations = [
        {'name': '50 SMA', 'key': 'SMA_50', 'color': 'lime'}, 
        {'name': '100 SMA', 'key': 'SMA_100', 'color': 'orange'},
        {'name': '200 SMA', 'key': 'SMA_200', 'color': 'red'},
    ]
    
    for i, ma in enumerate(ma_annotations):
        value = latest.get(ma['key'], None)
        if value is not None:
            text = f"<span style='color:{ma['color']};'>{ma['name']}: {format_value(value)}</span>"
            fig.add_annotation(
                text=text, 
                xref='paper', yref='y',
                x=1.0, y=value,
                showarrow=False, 
                xanchor='left', yanchor='middle',
                font=dict(size=10, color='white'), 
                bgcolor="rgba(0,0,0,0.5)", 
                yshift=-i * 12
            )

    # ... (Rest of styling is unchanged) ...
    fig.update_layout(title_text=f"<b>{ticker} Technicals</b>", title_x=0.5, xaxis_rangeslider_visible=False,
                      template='plotly', 
                      font=dict(color='black'), 
                      showlegend=False,
                      margin=dict(r=120, t=140, b=80)) 
    
    fig.update_traces(showlegend=False)

    return fig