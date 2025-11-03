# backtest_engine.py

import pandas as pd
import numpy as np
from scipy.stats import norm
from analyzers.financial_analyzer import FinancialAnalyzer

# Safe print function that handles BrokenPipeError in Streamlit
def safe_print(*args, **kwargs):
    """Print function that gracefully handles BrokenPipeError"""
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        # Streamlit redirects stdout/stderr, so pipes can break
        # Just ignore the error
        pass
    except (IOError, OSError):
        # Handle other pipe/IO errors
        pass

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Calculate Black-Scholes option price, ensuring non-negative result"""
    if T <= 0 or sigma <= 0: return 0.0
    if S <= 0 or K <= 0: return 0.0  # Invalid inputs
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        # CRITICAL: Options can't have negative values (floor at 0)
        return max(0.0, price)
    except:
        return 0.0  # Fallback for any calculation errors

def round_strike_to_nearest(price: float, increment: float = 1.0) -> float:
    return round(price / increment) * increment

def run_portfolio_backtest(data: pd.DataFrame, strategy_params: dict, initial_portfolio_value: float):
    """
    Run backtest simulation on stock data with given strategy parameters.
    
    Returns dict with:
    - portfolio_history: DataFrame with date index and value column
    - Trades: list of trade dicts
    - metrics: dict with performance metrics
    """
    # CRITICAL: Validate input data before processing
    if data is None or not isinstance(data, pd.DataFrame) or data.empty:
        safe_print("ERROR: backtest_engine received None or invalid data")
        return {
            'portfolio_history': pd.DataFrame(),
            'Trades': [],
            'metrics': {
                'Initial Portfolio Value': initial_portfolio_value,
                'Final Portfolio Value': initial_portfolio_value,
                'Total Return': 0.0,
                'Annualized Return (CAGR)': 0.0,
                'Max Drawdown': 0.0,
                'Win Rate': 0.0,
                'Total Trades': 0
            },
            'Final Portfolio Value': initial_portfolio_value,
            'Total Return': 0.0,
            'Annualized Return (CAGR)': 0.0,
            'Max Drawdown': 0.0,
            'Win Rate': 0.0,
            'Total Trades': 0
        }
    
    p = strategy_params
    portfolio_history = []
    cash = max(0.0, initial_portfolio_value)
    portfolio_value = max(0.0, initial_portfolio_value)
    peak_portfolio_value = max(0.0, portfolio_value)
    max_drawdown = 0.0
    trades = []
    in_position = False
    entry_data = {}
    
    # Required columns check
    required_cols = ['close', 'Trend_Score', 'Reversion_Score', 'historical_volatility']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        safe_print(f"ERROR: Missing required columns: {missing_cols}")
        return {
            'portfolio_history': pd.DataFrame(),
            'Trades': [],
            'metrics': {'Initial Portfolio Value': initial_portfolio_value, 'Final Portfolio Value': initial_portfolio_value, 'Total Return': 0.0, 'Annualized Return (CAGR)': 0.0, 'Max Drawdown': 0.0, 'Win Rate': 0.0, 'Total Trades': 0},
            'Final Portfolio Value': initial_portfolio_value,
            'Total Return': 0.0,
            'Annualized Return (CAGR)': 0.0,
            'Max Drawdown': 0.0,
            'Win Rate': 0.0,
            'Total Trades': 0
        }
    
    # Record initial state
    if len(data) > 0:
        portfolio_history.append({'date': data.index[0], 'value': portfolio_value})
    
    min_trend = p.get('min_trend_score', -np.inf)
    max_trend = p.get('max_trend_score', np.inf)
    min_reversion = p.get('min_reversion_score', -np.inf)
    max_reversion = p.get('max_reversion_score', np.inf)
    signal_type = p.get('signal_type', 'reversion')
    trade_direction = p.get('trade_direction', 'calls')
    
    for i in range(len(data)):
        row = data.iloc[i]
        if row is None or not isinstance(row, pd.Series):
            continue
        
        # Update position value if in position
        current_pos_value = 0.0
        if in_position:
            try:
                days_in = (row.name - entry_data['date']).days
                time_left_val = max(0.001, (p['time_to_expiration_days'] - days_in) / 365.0)
                current_opt_price = black_scholes_price(
                    row['close'], entry_data['strike_price'], time_left_val, 
                    p['risk_free_rate'], row['historical_volatility'], 
                    option_type=entry_data['direction']
                )
                current_pos_value = max(0.0, current_opt_price * entry_data['num_lots'] * 100)
            except:
                current_pos_value = 0.0
        
        portfolio_value = max(0.0, cash + current_pos_value)
        
        # Exit logic
        if in_position:
            try:
                unrealized_pnl_pct = (current_pos_value - entry_data['total_cost']) / entry_data['total_cost'] if entry_data.get('total_cost', 0) > 0 else 0.0
                entry_data['peak_pnl_pct'] = max(entry_data.get('peak_pnl_pct', 0.0), unrealized_pnl_pct)
                
                if not entry_data.get('trailing_stop_active', False) and unrealized_pnl_pct >= p.get('profit_target_pct', 0.5):
                    entry_data['trailing_stop_active'] = True
                
                exit_trade = False
                reason = ''
                if entry_data.get('trailing_stop_active', False):
                    trailing_stop_level = entry_data['peak_pnl_pct'] - p.get('trailing_stop_pct', 0.25)
                    if unrealized_pnl_pct < trailing_stop_level:
                        reason, exit_trade = 'Trailing Stop', True
                elif unrealized_pnl_pct <= p.get('stop_loss_pct', -0.5):
                    reason, exit_trade = 'Stop-Loss', True
                
                if not exit_trade and (p.get('time_to_expiration_days', 30) - (row.name - entry_data['date']).days) <= 0:
                    reason, exit_trade = 'Expiration', True
                
                if exit_trade:
                    pnl_dollars = current_pos_value - entry_data['total_cost']
                    cash = max(0.0, cash + current_pos_value)
                    trades.append({
                        'entry_date': entry_data['date'], 'exit_date': row.name, 'pnl_dollars': pnl_dollars, 'reason': reason,
                        'direction': entry_data['direction'].capitalize(), 'signal_type': entry_data.get('signal_type', 'unknown'),
                        'market_regime': entry_data.get('market_regime', 'Unknown'), 'entry_trend_score': entry_data.get('entry_trend_score', 0),
                        'entry_reversion_score': entry_data.get('entry_reversion_score', 0), 'entry_stock_price': entry_data.get('entry_stock_price', 0),
                        'exit_stock_price': row['close'], 'strike_price': entry_data.get('strike_price', 0),
                        'initial_num_lots': entry_data.get('initial_num_lots', 0), 'final_num_lots': entry_data.get('num_lots', 0)
                    })
                    in_position, entry_data, current_pos_value = False, {}, 0.0
            except Exception as e:
                safe_print(f"Error in exit logic: {e}")
        
        # Double down logic - check if we should add to position when it's down
        if in_position:
            try:
                # Check if double down is enabled and we haven't exceeded max position size
                double_down_levels = p.get('double_down_levels', {})
                max_position_pct = p.get('max_position_pct', 0.5)
                
                if double_down_levels and current_pos_value > 0:
                    # Calculate current unrealized P/L percentage
                    unrealized_pnl_pct = (current_pos_value - entry_data['total_cost']) / entry_data['total_cost'] if entry_data.get('total_cost', 0) > 0 else 0.0
                    
                    # Check each double down level
                    for dd_level_pct, dd_multiplier in double_down_levels.items():
                        # Check if we've hit this double down level and haven't already doubled down at this level
                        dd_triggered_key = f'dd_triggered_{dd_level_pct}'
                        if (unrealized_pnl_pct <= dd_level_pct and 
                            not entry_data.get(dd_triggered_key, False)):
                            
                            # Check if adding more would exceed max position size
                            current_position_pct = (entry_data['total_cost'] / initial_portfolio_value) if initial_portfolio_value > 0 else 0.0
                            
                            # Calculate new position size after double down
                            additional_allocation = entry_data['total_cost'] * (dd_multiplier - 1.0)
                            new_total_position_pct = (entry_data['total_cost'] + additional_allocation) / initial_portfolio_value if initial_portfolio_value > 0 else 0.0
                            
                            if new_total_position_pct <= max_position_pct:
                                # Calculate new option price and additional lots
                                days_in = (row.name - entry_data['date']).days
                                time_left_val = max(0.001, (p['time_to_expiration_days'] - days_in) / 365.0)
                                current_opt_price = black_scholes_price(
                                    row['close'], entry_data['strike_price'], time_left_val,
                                    p['risk_free_rate'], row['historical_volatility'],
                                    option_type=entry_data['direction']
                                )
                                
                                if current_opt_price > 0:
                                    # Calculate how many additional lots to add
                                    additional_cost = entry_data['total_cost'] * (dd_multiplier - 1.0)
                                    additional_lots = max(1, int(additional_cost / (current_opt_price * 100)))
                                    actual_additional_cost = current_opt_price * additional_lots * 100
                                    
                                    # Only proceed if we have enough cash
                                    if actual_additional_cost <= cash:
                                        cash -= actual_additional_cost
                                        entry_data['num_lots'] += additional_lots
                                        entry_data['total_cost'] += actual_additional_cost
                                        entry_data[dd_triggered_key] = True
                                        
                                        # Update current position value
                                        current_pos_value = max(0.0, current_opt_price * entry_data['num_lots'] * 100)
            except Exception as e:
                safe_print(f"Error in double down logic: {e}")
        
        # Entry logic
        if not in_position:
            try:
                trend_score = row.get('Trend_Score', 0) if 'Trend_Score' in row.index else 0
                reversion_score = row.get('Reversion_Score', 0) if 'Reversion_Score' in row.index else 0
                actionable_direction = None
                entry_signal_type = None
                
                # Get score thresholds - use min/max ranges primarily, entry_threshold as secondary
                min_trend = p.get('min_trend_score', -np.inf)
                max_trend = p.get('max_trend_score', np.inf)
                min_reversion = p.get('min_reversion_score', -np.inf)
                max_reversion = p.get('max_reversion_score', np.inf)
                entry_trend_threshold = p.get('entry_trend_threshold', None)
                entry_reversion_threshold = p.get('entry_reversion_threshold', None)
                
                # Signal detection based on signal_type parameter
                if signal_type == 'trend':
                    # Primary: check if score is within min/max range
                    # Secondary: if entry_threshold is set, also require it
                    trend_in_range_call = min_trend <= trend_score <= max_trend and trend_score > 0
                    trend_in_range_put = min_trend <= trend_score <= max_trend and trend_score < 0
                    
                    if entry_trend_threshold:
                        trend_ok_call = trend_in_range_call and trend_score >= entry_trend_threshold
                        trend_ok_put = trend_in_range_put and trend_score <= -entry_trend_threshold
                    else:
                        trend_ok_call = trend_in_range_call
                        trend_ok_put = trend_in_range_put
                    
                    if trend_ok_call and trade_direction in ['calls', 'both']:
                        actionable_direction, entry_signal_type = 'call', 'trend'
                    elif trend_ok_put and trade_direction in ['puts', 'both']:
                        actionable_direction, entry_signal_type = 'put', 'trend'
                        
                elif signal_type == 'reversion':
                    # Primary: check if score is within min/max range
                    # Secondary: if entry_threshold is set, also require it
                    reversion_in_range_call = min_reversion <= reversion_score <= max_reversion and reversion_score > 0
                    reversion_in_range_put = min_reversion <= reversion_score <= max_reversion and reversion_score < 0
                    
                    if entry_reversion_threshold:
                        reversion_ok_call = reversion_in_range_call and reversion_score >= entry_reversion_threshold
                        reversion_ok_put = reversion_in_range_put and reversion_score <= -entry_reversion_threshold
                    else:
                        reversion_ok_call = reversion_in_range_call
                        reversion_ok_put = reversion_in_range_put
                    
                    if reversion_ok_call and trade_direction in ['calls', 'both']:
                        actionable_direction, entry_signal_type = 'call', 'reversion'
                    elif reversion_ok_put and trade_direction in ['puts', 'both']:
                        actionable_direction, entry_signal_type = 'put', 'reversion'
                        
                elif signal_type in ['both', 'both_no_confirmation']:
                    # Check ranges first
                    trend_in_range = min_trend <= trend_score <= max_trend
                    reversion_in_range = min_reversion <= reversion_score <= max_reversion
                    
                    # For 'both' mode: use confirmation threshold logic
                    # For 'both_no_confirmation': just check ranges (less strict)
                    if signal_type == 'both':
                        # CONFIRMATION LOGIC: Threshold defines how much trend can deviate from reversion score
                        # Example: Reversion -2, confirmation=2 → Trend can be -2, -1, 0, +1, +2 (within 2 of -2)
                        # Example: Reversion -2, confirmation=0 → Trend must be 0 or negative
                        # Example: Reversion -3, confirmation=2 → Trend can be up to +2 (reversion -3 + threshold 2 = allows up to +2)
                        confirmation_threshold = p.get('confirmation_threshold', 0)
                        
                        # Check if reversion signal meets entry threshold
                        reversion_meets_threshold = False
                        if entry_reversion_threshold:
                            reversion_meets_threshold = abs(reversion_score) >= abs(entry_reversion_threshold)
                        else:
                            reversion_meets_threshold = abs(reversion_score) > 0
                        
                        if reversion_in_range and reversion_meets_threshold:
                            # Primary signal is reversion - determine direction from reversion score
                            if reversion_score > 0:  # Oversold = CALL signal
                                # Confirmation threshold = maximum SPREAD between reversion and trend
                                # Reversion +2, confirmation=2 → trend can be +2±2 = 0 to +4
                                # Reversion +2, confirmation=0 → trend must be +2 (exact match)
                                trend_min_allowed = reversion_score - confirmation_threshold  # Can be less positive
                                trend_max_allowed = reversion_score + confirmation_threshold  # Can be more positive
                                # Clamp to valid score range (-4 to +4)
                                trend_min_allowed = max(-4, trend_min_allowed)
                                trend_max_allowed = min(4, trend_max_allowed)
                                trend_ok = trend_in_range and (trend_min_allowed <= trend_score <= trend_max_allowed)
                                
                                if trend_ok and trade_direction in ['calls', 'both']:
                                    actionable_direction, entry_signal_type = 'call', 'both'
                                    
                            elif reversion_score < 0:  # Overbought = PUT signal
                                # Confirmation threshold = maximum SPREAD between reversion and trend (trend can go UP from reversion)
                                # For PUTs, trend can always go DOWN (more negative) to -4
                                # Reversion -2, confirmation=2 → trend can be -4 to -2+2=0 (so -4, -3, -2, -1, 0)
                                # Reversion -2, confirmation=0 → trend can be -4 to -2+0=-2 (so -4, -3, -2)
                                # Reversion -3, confirmation=2 → trend can be -4 to -3+2=-1 (so -4, -3, -2, -1)
                                trend_min_allowed = -4  # Can always go to -4 for PUTs
                                trend_max_allowed = reversion_score + confirmation_threshold  # Can go UP from reversion by confirmation amount
                                # Clamp to valid score range (-4 to +4)
                                trend_min_allowed = max(-4, trend_min_allowed)
                                trend_max_allowed = min(4, trend_max_allowed)
                                trend_ok = trend_in_range and (trend_min_allowed <= trend_score <= trend_max_allowed)
                                
                                if trend_ok and trade_direction in ['puts', 'both']:
                                    actionable_direction, entry_signal_type = 'put', 'both'
                    else:  # both_no_confirmation - use ranges only
                        trend_ok = trend_in_range
                        reversion_ok = reversion_in_range
                        
                        # Determine direction based on scores
                        if trend_ok and reversion_ok:
                            if trend_score > 0 and reversion_score > 0 and trade_direction in ['calls', 'both']:
                                actionable_direction, entry_signal_type = 'call', 'both'
                            elif trend_score < 0 and reversion_score < 0 and trade_direction in ['puts', 'both']:
                                actionable_direction, entry_signal_type = 'put', 'both'
                
                if actionable_direction:
                    strike = round_strike_to_nearest(row['close'], p.get('strike_increment', 5.0))
                    time_to_exp = p.get('time_to_expiration_days', 30) / 365.0
                    opt_price = black_scholes_price(row['close'], strike, time_to_exp, p.get('risk_free_rate', 0.05), row['historical_volatility'], actionable_direction)
                    
                    if opt_price > 0:
                        allocation = portfolio_value * p.get('position_size_pct', 0.1)
                        num_lots = max(1, int(allocation / (opt_price * 100)))
                        initial_cost = opt_price * num_lots * 100
                        
                        if initial_cost <= cash:
                            cash -= initial_cost
                            in_position = True
                            entry_data = {
                                'date': row.name,
                                'strike_price': strike,
                                'direction': actionable_direction,
                                'num_lots': num_lots,
                                'initial_num_lots': num_lots,
                                'total_cost': initial_cost,
                                'signal_type': entry_signal_type,
                                'market_regime': row.get('MarketRegime', 'Unknown'),
                                'entry_trend_score': trend_score,
                                'entry_reversion_score': reversion_score,
                                'entry_stock_price': row['close'],
                            }
                            current_pos_value = initial_cost
            except Exception as e:
                safe_print(f"Error in entry logic: {e}")
        
        # End of day reconciliation
        portfolio_value = max(0.0, cash + current_pos_value)
        peak_portfolio_value = max(peak_portfolio_value, portfolio_value)
        drawdown = (portfolio_value - peak_portfolio_value) / peak_portfolio_value if peak_portfolio_value > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)
        
        portfolio_history.append({'date': row.name, 'value': portfolio_value})
    
    # Calculate final metrics
    final_portfolio_value = portfolio_history[-1]['value'] if portfolio_history else initial_portfolio_value
    num_years = (data.index.max() - data.index.min()).days / 365.25 if len(data) > 1 else 0
    
    total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value if initial_portfolio_value > 0 else 0.0
    cagr = ((final_portfolio_value / initial_portfolio_value) ** (1.0 / num_years) - 1.0) if num_years > 0 and initial_portfolio_value > 0 else 0.0
    
    if portfolio_history:
        portfolio_df = pd.DataFrame(portfolio_history)
        if 'date' in portfolio_df.columns:
            portfolio_df = portfolio_df.set_index('date')
        portfolio_df['peak'] = portfolio_df['value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['value'] - portfolio_df['peak']) / portfolio_df['peak']
        max_drawdown = abs(portfolio_df['drawdown'].min()) if not portfolio_df['drawdown'].empty else 0.0
        portfolio_history_df = portfolio_df[['value']]
    else:
        portfolio_history_df = pd.DataFrame()
        max_drawdown = 0.0
    
    if trades:
        winning_trades = [t for t in trades if t.get('pnl_dollars', 0) > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0
    else:
        win_rate = 0.0
    
    # Return complete result
    return {
        'portfolio_history': portfolio_history_df,
        'Trades': trades,
        'metrics': {
            'Initial Portfolio Value': initial_portfolio_value,
            'Final Portfolio Value': final_portfolio_value,
            'Total Return': total_return,
            'Annualized Return (CAGR)': cagr,
            'Max Drawdown': max_drawdown,
            'Win Rate': win_rate,
            'Total Trades': len(trades),
        },
        'Final Portfolio Value': final_portfolio_value,
        'Total Return': total_return,
        'Annualized Return (CAGR)': cagr,
        'Max Drawdown': max_drawdown,
        'Win Rate': win_rate,
        'Total Trades': len(trades),
    }
