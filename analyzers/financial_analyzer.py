import pandas as pd
import pandas_ta as ta
import numpy as np

class FinancialAnalyzer:
    def run_full_analysis(self, stock_data: pd.DataFrame):
        if stock_data.empty or len(stock_data) < 200: return None, None
        data_with_indicators = stock_data.copy()
        
        # Volatility/Reversion Indicators
        bbands = data_with_indicators.ta.bbands(length=10, std=1.5)
        stoch = data_with_indicators.ta.stoch(k=14, d=3)
        if bbands is not None: data_with_indicators = pd.concat([data_with_indicators, bbands], axis=1)
        if stoch is not None: data_with_indicators = pd.concat([data_with_indicators, stoch], axis=1)
        
        # Trend Indicators
        macd = data_with_indicators.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None: data_with_indicators = pd.concat([data_with_indicators, macd], axis=1)

        # --- NEW: Short-Term MA Ribbon Indicators ---
        data_with_indicators['SMA_5'] = data_with_indicators.ta.sma(length=5)
        data_with_indicators['SMA_10'] = data_with_indicators.ta.sma(length=10)
        data_with_indicators['SMA_15'] = data_with_indicators.ta.sma(length=15)
        data_with_indicators['SMA_20'] = data_with_indicators.ta.sma(length=20)
        # --- END NEW MAs ---
        
        # Keep 50 and 200 for other scoring components
        data_with_indicators['SMA_50'] = data_with_indicators.ta.sma(length=50) 
        data_with_indicators['SMA_200'] = data_with_indicators.ta.sma(length=200) 
        data_with_indicators['RSI_14'] = data_with_indicators.ta.rsi(length=14)
        
        self._calculate_scores(data_with_indicators)
        latest_indicators = data_with_indicators.iloc[-1]
        return data_with_indicators, latest_indicators

    def _calculate_percentile_rank(self, metric_series: pd.Series) -> float:
        history = metric_series.dropna()
        if len(history) < 2: return np.nan
        latest_value = history.iloc[-1]
        rank_score = (history < latest_value).sum() / len(history)
        return rank_score * 100

    def run_full_fundamental_analysis(self, daily_fund_data: pd.DataFrame):
        all_metrics = ['P/E', 'P/S', 'PEG']
        def get_default_summary():
            s = {}; 
            for metric in all_metrics: s[f'{metric}_Short_Term_Rank'] = np.nan; s[f'{metric}_Long_Term_Rank'] = np.nan
            return s
        if daily_fund_data.empty: return get_default_summary(), {}

        timeframes = {'30 days': 21, '90 days': 63, '120 days': 84, '1 year': 252, '2 years': 504,
                      '3 years': 756, '5 years': 1260, '7 years': 1764, '10 years': 2520, 'Full History': len(daily_fund_data)}
        full_rank_data = {}
        for metric in all_metrics:
            if metric not in daily_fund_data.columns: continue
            metric_ranks = {}; series = daily_fund_data[metric].dropna()
            for label, days in timeframes.items():
                if len(series) >= days: metric_ranks[label] = self._calculate_percentile_rank(series.iloc[-days:])
            full_rank_data[metric] = metric_ranks
        
        summary = get_default_summary()
        for metric, ranks in full_rank_data.items():
            short_labels = ['30 days', '90 days', '120 days']; long_labels = ['1 year', '2 years', '3 years', '5 years', '7 years', '10 years', 'Full History']
            short_ranks = [ranks.get(l) for l in short_labels if ranks.get(l) is not None]; long_ranks = [ranks.get(l) for l in long_labels if ranks.get(l) is not None]
            summary[f'{metric}_Short_Term_Rank'] = np.mean(short_ranks) if short_ranks else np.nan
            summary[f'{metric}_Long_Term_Rank'] = np.mean(long_ranks) if long_ranks else np.nan
        return summary, full_rank_data

    def _calculate_scores(self, df: pd.DataFrame):
        # --- NEW MA RIBBON SCORING LOGIC (5/10/15/20) ---
        df['score_ma_ribbon'] = 0
        
        # Condition 1: Check for Bullish Alignment (5 > 10 > 15 > 20)
        bullish_ribbon = (df.get('SMA_5') > df.get('SMA_10')) & \
                         (df.get('SMA_10') > df.get('SMA_15')) & \
                         (df.get('SMA_15') > df.get('SMA_20'))
        df.loc[bullish_ribbon, 'score_ma_ribbon'] = 1
        
        # Condition 2: Check for Bearish Alignment (5 < 10 < 15 < 20)
        bearish_ribbon = (df.get('SMA_5') < df.get('SMA_10')) & \
                         (df.get('SMA_10') < df.get('SMA_15')) & \
                         (df.get('SMA_15') < df.get('SMA_20'))
        df.loc[bearish_ribbon, 'score_ma_ribbon'] = -1
        # --- END NEW MA RIBBON SCORING ---
        
        # Ensure we use the 50SMA and 200SMA scores for the other points
        df['score_50sma'] = np.where(df.get('close') > df.get('SMA_50'), 1, -1) 
        df['score_200sma'] = np.where(df.get('close') > df.get('SMA_200'), 1, -1) 
        df['score_macd'] = np.where(df.get('MACD_12_26_9') > df.get('MACDs_12_26_9'), 1, -1)
        
        # TREND SCORE (Max +4 / -4) - NOW includes MA Ribbon and uses 50/200/MACD for +3
        df['Trend_Score'] = df[['score_ma_ribbon', 'score_50sma', 'score_200sma', 'score_macd']].sum(axis=1)
        
        # REVERSION SCORE (Max +4 / -4 - Logic is unchanged)
        df['score_rsi'] = np.where(df.get('RSI_14') < 30, 2, np.where(df.get('RSI_14') < 40, 1, np.where(df.get('RSI_14') > 70, -2, np.where(df.get('RSI_14') > 60, -1, 0))))
        df['score_stoch'] = np.where(df.get('STOCHk_14_3_3') < 20, 1, np.where(df.get('STOCHk_14_3_3') > 80, -1, 0))
        df['score_bbands'] = np.where(df.get('close') < df.get('BBL_10_1.5'), 1, np.where(df.get('close') > df.get('BBU_10_1.5'), -1, 0))
        df['Reversion_Score'] = df[['score_rsi', 'score_stoch', 'score_bbands']].sum(axis=1)```

---

### **File 2: `utils/plotting.py` (Full Replacement)**

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def create_analysis_chart(ticker: str, df: pd.DataFrame):
    if df.empty: return None
    plot_df = df.copy()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=('', 'RSI & Stochastics (Reversion)', 'MACD (Trend)'),
                        row_heights=[0.80, 0.10, 0.10])
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df['open'], high=plot_df['high'], low=plot_df['low'], close=plot_df['close'], 
                                 increasing=dict(line=dict(color='#00CC96')), decreasing=dict(line=dict(color='#EF553B')), name='Price'), row=1, col=1)
    
    # Reversion Regime Change Visualization (remains the same)
    if 'Reversion_Score' in plot_df.columns:
        plot_df['reversion_state'] = np.where(plot_df['Reversion_Score'] > 0, 1, np.where(plot_df['Reversion_Score'] < 0, -1, 0))
        plot_df['regime_change'] = plot_df['reversion_state'].diff().fillna(0)
        regime_starts = plot_df[plot_df['regime_change'] != 0].index
        if plot_df.index.min() not in regime_starts: regime_starts = pd.Index([plot_df.index.min()]).union(regime_starts)
        
        for i, start in enumerate(regime_starts):
            state = plot_df.loc[start, 'reversion_state']; 
            end = regime_starts[i+1] if i+1 < len(regime_starts) else plot_df.index[-1]
            if state != 0: 
                fig.add_vrect(x0=start, x1=end, fillcolor="green" if state > 0 else "red", opacity=0.15, layer="below", row=1, col=1)

    # --- NEW MA RIBBON TRACES (5, 10, 15, 20) ---
    ma_colors = {'SMA_5': 'lime', 'SMA_10': 'cyan', 'SMA_15': 'yellow', 'SMA_20': 'orange'}
    for k, c in ma_colors.items():
        if k in plot_df.columns:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[k], line=dict(color=c, width=1), name=k.replace('SMA_','')), row=1, col=1)
            
    # Include 50 & 200 for reference (less prominent)
    if 'SMA_50' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_50'], line=dict(color='gray', width=1, dash='dot'), name='50 SMA'), row=1, col=1)
    if 'SMA_200' in plot_df.columns: fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['SMA_200'], line=dict(color='red', width=1, dash='dash'), name='200 SMA'), row=1, col=1)
    # --- END NEW MA RIBBON TRACES ---


    # Bollinger Bands and Subplots (remain the same)
    if 'BBU_10_1.5' in plot_df.columns:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBU_10_1.5'], line=dict(color='gray', dash='dash', width=1), name='BB U'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['BBL_10_1.5'], line=dict(color='gray', dash='dash', width=1), name='BB L'), row=1, col=1)
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
    
    # NOTE: The dashboard logic is updated to track score_50sma, score_200sma, and score_ma_ribbon
    trend_score=latest.get('Trend_Score', np.nan); rev_score=latest.get('Reversion_Score', np.nan)
    tear_sheet = (f"<b>TREND (Max +/-4):</b> Total={colorize(trend_score)} | Ribbon={colorize(latest.get('score_ma_ribbon',np.nan))} | 50SMA={colorize(latest.get('score_50sma',np.nan))} | 200SMA={colorize(latest.get('score_200sma',np.nan))} | MACD={colorize(latest.get('score_macd',np.nan))}<br>"
                  f"<b>REVERSION (Max +/-4):</b> Total={colorize(rev_score)} | RSI={colorize(latest.get('score_rsi',np.nan))} | Stoch={colorize(latest.get('score_stoch',np.nan))} | BBands={colorize(latest.get('score_bbands',np.nan))}<br>"
                  f"<b>VALUES:</b> RSI={fv(latest.get('RSI_14',np.nan))} | Stoch=%K={fv(latest.get('STOCHk_14_3_3',np.nan))} | MACD: {fv(latest.get('MACD_12_26_9',np.nan))}/{fv(latest.get('MACDs_12_26_9',np.nan))}")
    fig.add_annotation(text=tear_sheet, align='left', showarrow=False, xref='paper', yref='paper', x=0.01, y=1.0, bgcolor="rgba(50,50,50,0.95)", bordercolor="black", borderwidth=1, xanchor='left', yanchor='bottom', font=dict(size=11, color='white'))
    
    # MA Annotations - Only showing 5, 20, and 200 for simplicity on the side
    ma_ann = [{'n': '5 SMA', 'k': 'SMA_5', 'c': 'lime'}, {'n': '20 SMA', 'k': 'SMA_20', 'c': 'orange'}, {'n': '200 SMA', 'k': 'SMA_200', 'c': 'red'}]
    for i, ma in enumerate(ma_ann):
        val = latest.get(ma['k']); text = f"<span style='color:{ma['c']};'>{ma['n']}: {fv(val)}</span>" if pd.notna(val) else ""
        if pd.notna(val): fig.add_annotation(text=text, xref='paper', yref='y', x=1.0, y=val, showarrow=False, xanchor='left', yanchor='middle', font=dict(size=10,color='white'), bgcolor="rgba(0,0,0,0.5)", yshift=-i*12)

    fig.update_layout(title_text=f"<b>{ticker} Technicals</b>", title_x=0.5, xaxis_rangeslider_visible=False, template='plotly', font=dict(color='black'), showlegend=False, margin=dict(r=120,t=140,b=80))
    fig.update_xaxes(showgrid=True, gridcolor='#E0E0E0', zerolinecolor='#C0C0C0', title_font=dict(color='black'), tickfont=dict(color='black'))
    fig.update_yaxes(showgrid=True, gridcolor='#E0E0E0', zerolinecolor='#C0C0C0', title_font=dict(color='black'), tickfont=dict(color='black'))
    for ann in fig['layout']['annotations']:
        if 'xref' in ann and ann['xref'] == 'x domain': ann['font']['color'] = 'black'
    fig.update_traces(showlegend=False)
    return fig