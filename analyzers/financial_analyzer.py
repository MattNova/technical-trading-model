import os
import tempfile
from pathlib import Path

# Ensure Numba can run without caching issues when pandas_ta imports njit utilities
os.environ.setdefault("NUMBA_DISABLE_CACHING", "1")
if "NUMBA_CACHE_DIR" not in os.environ:
    try:
        project_cache = Path(__file__).resolve().parent.parent / ".numba_cache"
        project_cache.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(project_cache)
    except Exception:
        os.environ["NUMBA_CACHE_DIR"] = tempfile.mkdtemp(prefix="numba_cache_")

import pandas as pd

# Force numba to avoid disk caching to bypass locator errors from third-party libs
try:
    import numba as _numba  # type: ignore
    _orig_njit = _numba.njit
    def _njit_no_cache(*args, **kwargs):
        kwargs["cache"] = False
        return _orig_njit(*args, **kwargs)
    _numba.njit = _njit_no_cache  # type: ignore
except Exception:
    pass

import pandas_ta as ta
import numpy as np

class FinancialAnalyzer:
    def run_full_analysis(self, stock_data: pd.DataFrame, volatility_lookback: int = None, model_selections: dict = None, spy_data: pd.DataFrame = None):
        """
        Run full technical analysis on stock data.
        
        Args:
            stock_data: DataFrame with OHLCV data
            volatility_lookback: Optional lookback period for volatility calculation (for backtest engine)
            model_selections: Optional dict to enable/disable scoring components (for backtest engine)
            spy_data: Optional SPY data for market regime detection (for backtest engine)
        
        Returns:
            If volatility_lookback is provided (backtest mode): returns DataFrame only
            Otherwise (dashboard mode): returns tuple (DataFrame, latest_indicators)
        """
        if stock_data.empty or len(stock_data) < 200:
            if volatility_lookback is not None:
                return None
            return None, None
        
        data_with_indicators = stock_data.copy()
        is_backtest_mode = volatility_lookback is not None
        
        # Use std=1.0 for both dashboard and backtest modes (consistent across all modes)
        bb_std = 1.0
        
        # Volatility/Reversion Indicators
        bbands = data_with_indicators.ta.bbands(length=20, std=bb_std)
        stoch = data_with_indicators.ta.stoch(k=14, d=3)
        if bbands is not None: data_with_indicators = pd.concat([data_with_indicators, bbands], axis=1)
        if stoch is not None: data_with_indicators = pd.concat([data_with_indicators, stoch], axis=1)
        
        # Trend Indicators
        macd = data_with_indicators.ta.macd(fast=12, slow=26, signal=9)
        if macd is not None: data_with_indicators = pd.concat([data_with_indicators, macd], axis=1)

        # --- MA Ribbon Indicators ---
        data_with_indicators['SMA_5'] = data_with_indicators.ta.sma(length=5)
        data_with_indicators['SMA_10'] = data_with_indicators.ta.sma(length=10)
        data_with_indicators['SMA_15'] = data_with_indicators.ta.sma(length=15)
        data_with_indicators['SMA_20'] = data_with_indicators.ta.sma(length=20)
        
        # Keep 50 and 200 for other scoring components
        data_with_indicators['SMA_50'] = data_with_indicators.ta.sma(length=50) 
        data_with_indicators['SMA_200'] = data_with_indicators.ta.sma(length=200) 
        data_with_indicators['RSI_14'] = data_with_indicators.ta.rsi(length=14)
        
        # Calculate scores with optional model_selections support
        # Set backtest RSI thresholds flag if in backtest mode
        if is_backtest_mode and model_selections is not None:
            model_selections['rsi_backtest_thresholds'] = True
        self._calculate_scores(data_with_indicators, model_selections)
        
        # Detect candlestick patterns for both dashboard and backtest modes
        self._detect_candlestick_patterns(data_with_indicators)
        
        # Add volatility calculation for backtest mode
        if is_backtest_mode:
            data_with_indicators['log_return'] = np.log(data_with_indicators['close'] / data_with_indicators['close'].shift(1))
            data_with_indicators['historical_volatility'] = data_with_indicators['log_return'].rolling(window=volatility_lookback).std() * np.sqrt(252)
        
        # Add market regime detection (for both dashboard and backtest modes)
        # CRITICAL: Validate spy_data is a valid DataFrame before using it
        if (spy_data is not None and 
            isinstance(spy_data, pd.DataFrame) and 
            not spy_data.empty and 
            hasattr(spy_data, 'index') and 
            'close' in spy_data.columns and
            len(spy_data) >= 200):  # Need at least 200 days for SMA_200
            try:
                spy_copy = spy_data.copy()
                # Validate the copy succeeded
                if spy_copy is not None and isinstance(spy_copy, pd.DataFrame) and not spy_copy.empty:
                    spy_copy['SMA_200'] = spy_copy['close'].rolling(window=200).mean()
                    spy_copy['IsBullMarket'] = (spy_copy['close'] > spy_copy['SMA_200']).astype(int)
                    data_with_indicators = data_with_indicators.join(spy_copy[['IsBullMarket']], how='left')
                    data_with_indicators['IsBullMarket'] = data_with_indicators['IsBullMarket'].ffill().bfill().infer_objects(copy=False)
                    data_with_indicators['MarketRegime'] = data_with_indicators['IsBullMarket'].apply(lambda x: 'Bull' if x == 1 else 'Bear')
                else:
                    # Copy failed or is invalid
                    data_with_indicators['MarketRegime'] = 'Unknown'
                    data_with_indicators['IsBullMarket'] = 0
            except Exception as e:
                # Any error in processing - set to Unknown
                data_with_indicators['MarketRegime'] = 'Unknown'
                data_with_indicators['IsBullMarket'] = 0
        else:
            data_with_indicators['MarketRegime'] = 'Unknown'
            data_with_indicators['IsBullMarket'] = 0
        
        # Return format depends on mode
        if is_backtest_mode:
            return data_with_indicators.dropna()
        else:
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
            metric_ranks = {}
            # Get the full series (with forward-filled and backward-filled values)
            series = daily_fund_data[metric]
            
            # After forward-fill and backward-fill in get_daily_fundamental_ratios,
            # we should have data for all dates in the dataframe
            # However, there might still be NaN from division by zero (e.g., PEG when eps_growth is 0)
            # So we drop NaN only for percentile calculation, but use full index for day counting
            
            # Get series without NaN for calculation
            series_no_nan = series.dropna()
            
            # Check if we have enough data points for each timeframe
            for label, days in timeframes.items():
                if label == 'Full History':
                    # For full history, use all available non-NaN data
                    if len(series_no_nan) >= 2:
                        metric_ranks[label] = self._calculate_percentile_rank(series_no_nan)
                else:
                    # For fixed timeframes, check if we have enough data points
                    # After forward/backward fill, we should have enough if the dataframe has enough days
                    if len(series) >= days:
                        # Use the last N days from the full series
                        window_series = series.iloc[-days:]
                        # Drop NaN only for calculation (should be minimal after forward/backward fill)
                        window_no_nan = window_series.dropna()
                        if len(window_no_nan) >= 2:  # Need at least 2 values to calculate percentile
                            metric_ranks[label] = self._calculate_percentile_rank(window_no_nan)
            full_rank_data[metric] = metric_ranks
        
        summary = get_default_summary()
        for metric, ranks in full_rank_data.items():
            short_labels = ['30 days', '90 days', '120 days']; long_labels = ['1 year', '2 years', '3 years', '5 years', '7 years', '10 years', 'Full History']
            short_ranks = [ranks.get(l) for l in short_labels if ranks.get(l) is not None]; long_ranks = [ranks.get(l) for l in long_labels if ranks.get(l) is not None]
            summary[f'{metric}_Short_Term_Rank'] = np.mean(short_ranks) if short_ranks else np.nan
            summary[f'{metric}_Long_Term_Rank'] = np.mean(long_ranks) if long_ranks else np.nan
        return summary, full_rank_data

    def _calculate_scores(self, df: pd.DataFrame, model_selections: dict = None):
        """
        Calculate trend and reversion scores.
        
        Args:
            df: DataFrame with indicators
            model_selections: Optional dict to enable/disable components. 
                             If provided, only enabled components are included in final scores.
                             Keys: 'score_ma_ribbon', 'score_50sma', 'score_200sma', 'score_macd',
                                   'score_rsi', 'score_stoch', 'score_bbands'
        """
        # --- MA RIBBON SCORING LOGIC (5/10/15/20) ---
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
        
        # Add score_50sma for backtest compatibility
        df['score_50sma'] = np.where(df.get('close') > df.get('SMA_50'), 1, -1)
        
        # Other scoring components
        df['score_200sma'] = np.where(df.get('close') > df.get('SMA_200'), 1, -1)
        df['score_macd'] = np.where(df.get('MACD_12_26_9') > df.get('MACDs_12_26_9'), 1, -1)

        # Golden/Death Cross: +1 on golden cross (50 crosses above 200), -1 on death cross
        sma50 = df.get('SMA_50')
        sma200 = df.get('SMA_200')
        prev_sma50 = sma50.shift(1)
        prev_sma200 = sma200.shift(1)
        cross_up = (sma50 > sma200) & (prev_sma50 <= prev_sma200)
        cross_down = (sma50 < sma200) & (prev_sma50 >= prev_sma200)
        df['score_cross'] = 0
        df.loc[cross_up, 'score_cross'] = 1
        df.loc[cross_down, 'score_cross'] = -1
        
        # REVERSION SCORE COMPONENTS
        # RSI scoring: RSI <= 20 for +2, RSI <= 30 for +1 (downside/bullish signals)
        # RSI >= 70 for -1, RSI >= 80 for -2 (upside/bearish signals)
        rsi = df.get('RSI_14')
        # Apply same thresholds for both dashboard and backtest
        df['score_rsi'] = np.where(rsi <= 20, 2,
                            np.where(rsi <= 30, 1,
                            np.where(rsi >= 80, -2,
                            np.where(rsi >= 70, -1, 0))))
        
        df['score_stoch'] = np.where(df.get('STOCHk_14_3_3') < 20, 1, np.where(df.get('STOCHk_14_3_3') > 80, -1, 0))
        
        # BBands scoring - handle column names for length=20, std=1.0
        bbl_col = None
        bbu_col = None
        if 'BBL_20_1.0' in df.columns:
            bbl_col = 'BBL_20_1.0'
            bbu_col = 'BBU_20_1.0'
        elif 'BBL_20_1.5' in df.columns:
            bbl_col = 'BBL_20_1.5'
            bbu_col = 'BBU_20_1.5'
        elif 'BBL_20_1' in df.columns:  # Handle case where pandas_ta might use 1 instead of 1.0
            bbl_col = 'BBL_20_1'
            bbu_col = 'BBU_20_1'
        elif 'BBL_10_1.0' in df.columns:  # Fallback for legacy data with length=10
            bbl_col = 'BBL_10_1.0'
            bbu_col = 'BBU_10_1.0'
        elif 'BBL_10_1.5' in df.columns:
            bbl_col = 'BBL_10_1.5'
            bbu_col = 'BBU_10_1.5'
        elif 'BBL_10_1' in df.columns:
            bbl_col = 'BBL_10_1'
            bbu_col = 'BBU_10_1'
        
        if bbl_col and bbu_col:
            # Use previous day's bands to compare with current close (avoid look-ahead bias)
            # Shift bands by 1 period so we compare today's close to yesterday's bands
            prev_bbl = df.get(bbl_col).shift(1)
            prev_bbu = df.get(bbu_col).shift(1)
            df['score_bbands'] = np.where(df.get('close') < prev_bbl, 1, 
                                    np.where(df.get('close') > prev_bbu, -1, 0))
        else:
            df['score_bbands'] = 0
        
        # Candlestick pattern scoring (only in dashboard mode, when patterns exist)
        if 'pattern_hammer' in df.columns:
            bull_pattern = (df.get('pattern_hammer', 0) == 1) | (df.get('pattern_engulfing_bull', 0) == 1) | (df.get('pattern_morning_star', 0) == 1)
            bear_pattern = (df.get('pattern_engulfing_bear', 0) == 1)
            df['score_patterns'] = 0
            df.loc[bull_pattern & ~bear_pattern, 'score_patterns'] = 1
            df.loc[bear_pattern & ~bull_pattern, 'score_patterns'] = -1
        else:
            df['score_patterns'] = 0
        
        # Calculate final scores with optional model_selections filtering
        if model_selections is not None:
            # Filter trend components based on model_selections
            active_trend_scores = []
            for score_col in ['score_ma_ribbon', 'score_50sma', 'score_200sma', 'score_macd']:
                if model_selections.get(score_col, True):
                    active_trend_scores.append(score_col)
            
            # Note: score_cross is not in model_selections in backtest, so exclude it for backtest compatibility
            # Dashboard version uses score_cross, backtest doesn't
            if len(active_trend_scores) > 0:
                df['Trend_Score'] = df[active_trend_scores].sum(axis=1)
            else:
                df['Trend_Score'] = 0
            
            # Filter reversion components based on model_selections
            active_reversion_scores = []
            for score_col in ['score_rsi', 'score_stoch', 'score_bbands']:
                if model_selections.get(score_col, True):
                    active_reversion_scores.append(score_col)
            
            # score_patterns is not in model_selections, so exclude it for backtest
            if len(active_reversion_scores) > 0:
                df['Reversion_Score'] = df[active_reversion_scores].sum(axis=1)
            else:
                df['Reversion_Score'] = 0
        else:
            # Dashboard mode: use all components including score_cross and score_patterns
            # TREND SCORE (Max +4 / -4): ribbon, price vs 200, MACD, and cross signal
            df['Trend_Score'] = df[['score_ma_ribbon', 'score_200sma', 'score_macd', 'score_cross']].sum(axis=1)
            
            # REVERSION SCORE (Max +4 / -4): RSI, BBands, Stoch, and Patterns
            reversion_components = df[['score_rsi', 'score_bbands', 'score_stoch', 'score_patterns']].sum(axis=1)
            df['Reversion_Score'] = np.clip(reversion_components, -4, 4)
            
            # Optional: simple cluster/confidence = count of active reversion signals on the bar (0..4)
            active = (df[['score_rsi', 'score_bbands', 'score_stoch', 'score_patterns']] != 0).sum(axis=1)
            df['Reversion_Confidence'] = active

    def _detect_candlestick_patterns(self, df: pd.DataFrame):
        # Basic, fast pattern flags for last N bars; no TA-lib dependency
        o, h, l, c = df.get('open'), df.get('high'), df.get('low'), df.get('close')
        if any(x is None for x in [o, h, l, c]):
            return
        body = (c - o).abs()
        range_ = (h - l).replace(0, np.nan)
        upper_wick = (h - c).where(c >= o, h - o)
        lower_wick = (o - l).where(c >= o, c - l)

        # Hammer: small body near top of range, long lower wick
        hammer = (
            (body / range_ < 0.3) &
            (lower_wick / range_ > 0.5) &
            (upper_wick / range_ < 0.2)
        )

        # Engulfing
        prev_o, prev_c = o.shift(1), c.shift(1)
        bull_engulf = (c > o) & (prev_c < prev_o) & (o <= prev_c) & (c >= prev_o)
        bear_engulf = (c < o) & (prev_c > prev_o) & (o >= prev_c) & (c <= prev_o)

        # Morning star (simplified 3-candle pattern)
        c1_red = prev_c < prev_o
        c2_small = (body.shift(0) / range_.shift(0) < 0.25) & (c.shift(0) < prev_c)
        c3_green = (c.shift(-1) > o.shift(-1)) & (c.shift(-1) > ((prev_o + prev_c) / 2))
        morning_star = c1_red & c2_small & c3_green.shift(1).fillna(False)

        df['pattern_hammer'] = hammer.astype(int)
        df['pattern_engulfing_bull'] = bull_engulf.astype(int)
        df['pattern_engulfing_bear'] = bear_engulf.astype(int)
        df['pattern_morning_star'] = morning_star.astype(int)