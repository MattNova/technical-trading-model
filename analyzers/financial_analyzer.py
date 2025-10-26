import pandas as pd
import pandas_ta as ta
import numpy as np

class FinancialAnalyzer:
    def run_full_analysis(self, stock_data: pd.DataFrame):
        if stock_data.empty or len(stock_data) < 200: return None, None
        data_with_indicators = stock_data.copy()
        
        bbands = data_with_indicators.ta.bbands(length=10, std=1.5); macd = data_with_indicators.ta.macd(fast=12, slow=26, signal=9)
        stoch = data_with_indicators.ta.stoch(k=14, d=3)
        if bbands is not None: data_with_indicators = pd.concat([data_with_indicators, bbands], axis=1)
        if macd is not None: data_with_indicators = pd.concat([data_with_indicators, macd], axis=1)
        if stoch is not None: data_with_indicators = pd.concat([data_with_indicators, stoch], axis=1)
        data_with_indicators['SMA_50'] = data_with_indicators.ta.sma(length=50); data_with_indicators['SMA_100'] = data_with_indicators.ta.sma(length=100)
        data_with_indicators['SMA_200'] = data_with_indicators.ta.sma(length=200); data_with_indicators['RSI_14'] = data_with_indicators.ta.rsi(length=14)
        
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
            s = {}
            for metric in all_metrics:
                s[f'{metric}_Short_Term_Rank'] = np.nan
                s[f'{metric}_Long_Term_Rank'] = np.nan
            return s

        if daily_fund_data.empty: return get_default_summary(), {}

        timeframes = {'30 days': 21, '90 days': 63, '120 days': 84, '1 year': 252, '2 years': 504,
                      '3 years': 756, '5 years': 1260, '7 years': 1764, '10 years': 2520, 'Full History': len(daily_fund_data)}

        full_rank_data = {}
        for metric in all_metrics:
            if metric not in daily_fund_data.columns: continue
            
            metric_ranks = {}
            series = daily_fund_data[metric].dropna()
            
            for label, days in timeframes.items():
                if len(series) >= days:
                    subset = series.iloc[-days:]
                    metric_ranks[label] = self._calculate_percentile_rank(subset)
            full_rank_data[metric] = metric_ranks
        
        summary = get_default_summary()
        for metric, ranks in full_rank_data.items():
            short_term_labels = ['30 days', '90 days', '120 days']
            long_term_labels = ['1 year', '2 years', '3 years', '5 years', '7 years', '10 years', 'Full History']
            
            short_term_ranks = [ranks.get(l) for l in short_term_labels if ranks.get(l) is not None]
            long_term_ranks = [ranks.get(l) for l in long_term_labels if ranks.get(l) is not None]

            summary[f'{metric}_Short_Term_Rank'] = np.mean(short_term_ranks) if short_term_ranks else np.nan
            summary[f'{metric}_Long_Term_Rank'] = np.mean(long_term_ranks) if long_term_ranks else np.nan
        
        return summary, full_rank_data

    def _calculate_scores(self, df: pd.DataFrame):
        df['score_ma_ribbon'] = 0; df['score_200sma'] = np.where(df.get('close') > df.get('SMA_200'), 1, -1)
        df['score_macd'] = np.where(df.get('MACD_12_26_9') > df.get('MACDs_12_26_9'), 1, -1)
        df['Trend_Score'] = df[['score_ma_ribbon', 'score_200sma', 'score_macd']].sum(axis=1)
        df['score_rsi'] = np.where(df.get('RSI_14') < 30, 2, np.where(df.get('RSI_14') < 40, 1, np.where(df.get('RSI_14') > 70, -2, np.where(df.get('RSI_14') > 60, -1, 0))))
        df['score_stoch'] = np.where(df.get('STOCHk_14_3_3') < 30, 1, np.where(df.get('STOCHk_14_3_3') > 70, -1, 0))
        df['score_bbands'] = np.where(df.get('close') < df.get('BBL_10_1.5'), 1, np.where(df.get('close') > df.get('BBU_10_1.5'), -1, 0))
        df['Reversion_Score'] = df[['score_rsi', 'score_stoch', 'score_bbands']].sum(axis=1)