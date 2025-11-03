import sys
import os
import time
import hashlib
import pickle
import json
import re
from datetime import date, datetime
from pathlib import Path

import streamlit as st

# PERFORMANCE: Heavy imports deferred until after authentication to prevent login blocking
# These will be imported after successful login
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
# import requests

# --- PATH SETUP ---
# PERFORMANCE: Only modify path if needed (avoid on every rerun)
# Add project root to Python path - but only if not already added
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.abspath(os.path.join(_current_dir, '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# --- LAZY CACHE DIR ---
# Initialize as None - will be created only when first needed (after auth)
CACHE_DIR = None

# Safe print function that handles BrokenPipeError in Streamlit
def safe_print(*args, **kwargs):
    """Print function that gracefully handles BrokenPipeError"""
    try:
        print(*args, **kwargs)
    except BrokenPipeError:
        pass # Ignore Streamlit pipe errors
    except (IOError, OSError):
        pass # Handle other pipe/IO errors

# --- PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Trading Model")

# --- SIMPLE LOGIN - NOTHING BEFORE THIS ---
# CRITICAL: Login check MUST be first and simplest possible
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'auth_time' not in st.session_state:
    st.session_state.auth_time = None

# Check timeout first (only if authenticated)
AUTH_TIMEOUT = 24 * 60 * 60  # 24 hours
if st.session_state.authenticated and st.session_state.auth_time:
    if (time.time() - st.session_state.auth_time) > AUTH_TIMEOUT:
        st.session_state.authenticated = False
        st.session_state.auth_time = None

# Show login if not authenticated
if not st.session_state.authenticated:
    st.title("🔒 Trading Model Login")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        try:
            # Get passwords from secrets and strip whitespace
            admin_pw = str(st.secrets.get("admin_password", "")).strip()
            viewer_pw = str(st.secrets.get("viewer_password", "")).strip()
            password_input = str(password).strip() if password else ""
            
            # Compare passwords
            if password_input == admin_pw:
                st.session_state.authenticated = True
                st.session_state.user_role = 'admin'
                st.session_state.auth_time = time.time()
                st.rerun()
            elif password_input == viewer_pw:
                st.session_state.authenticated = True
                st.session_state.user_role = 'viewer'
                st.session_state.auth_time = time.time()
                st.rerun()
            else:
                st.error("❌ Incorrect password")
        except Exception as e:
            st.error(f"Error: {e}")
    st.stop()

# --- SESSION STATE INITIALIZATION ---
# Initialize all session state keys here, at the top
def initialize_session_state():
    defaults = {
        'authenticated': st.session_state.get('authenticated', False),
        'user_role': st.session_state.get('user_role', 'viewer'),
        'auth_time': st.session_state.get('auth_time'),
        'saved_watchlist': "SPY, QQQ, AAPL, MSFT",
        'quick_prices': {},
        'show_password_prompt': False,
        'last_optimization_target': 'Risk-Adjusted Return (CAGR / Max Drawdown)',
        'optimizer_results': None,
        'all_optimizer_results': None,
        'backtest_results': None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

initialize_session_state()


# --- APP EXECUTION STARTS HERE (ONLY IF AUTHENTICATED) ---

# PERFORMANCE: Import heavy libraries AFTER authentication (prevents login blocking)
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# Now that we are authenticated, we can safely perform other operations
# Get FMP API Key
try:
    FMP_API_KEY = st.secrets["fmp_api_key"]
except KeyError:
    st.error("FMP API Key not configured in `.streamlit/secrets.toml`.")
    st.stop()

# Import heavy modules (lazy - only when needed after login)
try:
    from data_providers.fmp_provider import FMPProvider
    from analyzers.financial_analyzer import FinancialAnalyzer
    from utils.plotting import create_analysis_chart
    from backtest_engine import run_portfolio_backtest
except ImportError as e:
    st.error(f"Failed to import required modules: {e}")
    st.stop()

# Apply theme (minimal - set default first, then allow change)
# PERFORMANCE: Move theme selector to sidebar AFTER page selection to avoid blocking
if "theme_choice" not in st.session_state:
    st.session_state.theme_choice = "Dark"

if st.session_state.theme_choice == "Dark":
    st.markdown("""
        <style>
        :root { --bg: #0f172a; --bg2:#111827; --text:#e5e7eb; --accent:#10b981; }
        .stApp { background: var(--bg); color: var(--text); }
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        div[data-testid="stMarkdownContainer"] h1, h2, h3 { color: var(--text); }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        :root { --bg: #ffffff; --bg2:#f8fafc; --text:#111827; --accent:#2563eb; }
        .stApp { background: var(--bg); color: var(--text); }
        .block-container { padding-top: 1rem; padding-bottom: 2rem; }
        div[data-testid="stMarkdownContainer"] h1, h2, h3 { color: var(--text); }
        </style>
    """, unsafe_allow_html=True)

# --- DATA & CACHE LOGIC ---
def normalize_portfolio_history(portfolio_history):
    """Helper function to normalize portfolio history from various formats to list of dicts"""
    if portfolio_history is None:
        return []
    
    if isinstance(portfolio_history, pd.DataFrame):
        hist_df = portfolio_history.copy()
        if isinstance(hist_df.index, pd.DatetimeIndex):
            hist_df = hist_df.reset_index()
            if 'index' in hist_df.columns:
                hist_df = hist_df.rename(columns={'index': 'date'})
        if 'date' in hist_df.columns and 'value' in hist_df.columns:
            return hist_df[['date', 'value']].to_dict('records')
        elif len(hist_df.columns) >= 2:
            hist_df = hist_df.rename(columns={hist_df.columns[0]: 'date', hist_df.columns[1]: 'value'})
            return hist_df[['date', 'value']].to_dict('records')
    elif isinstance(portfolio_history, list):
        return portfolio_history
    
    return []

def get_model_hash():
    """
    Generate a hash of calculation code, ignoring comments and whitespace.
    Cache only invalidates when actual calculation logic changes, not:
    - Comments or docstrings
    - Whitespace changes
    - UI code in dashboard.py
    
    PERFORMANCE: Caches hash in session state to avoid re-reading files on every page load.
    """
    # Check session state cache first (fast - avoids file I/O on tab switches)
    if 'model_hash_cache' in st.session_state:
        return st.session_state.model_hash_cache
    
    import re
    
    def normalize_code(content: str) -> str:
        """Remove comments and normalize whitespace to make hash stable"""
        lines = []
        in_multiline_string = False
        multiline_char = None
        
        for line in content.split('\n'):
            # Handle multiline strings (don't remove # inside strings)
            stripped = line.strip()
            
            # Skip pure comment lines
            if stripped.startswith('#'):
                continue
            
            # Skip empty lines
            if not stripped:
                continue
            
            # Remove inline comments (but preserve strings)
            # Simple approach: if # appears, check if it's in a string
            if '#' in line:
                # Very simple: if # is before any quote, it's likely a comment
                quote_positions = [line.find('"'), line.find("'"), line.find('"""'), line.find("'''")]
                quote_positions = [p for p in quote_positions if p >= 0]
                hash_pos = line.find('#')
                
                if hash_pos >= 0:
                    # If # comes before any quote, it's a comment
                    if not quote_positions or hash_pos < min(quote_positions):
                        line = line[:hash_pos]
            
            # Normalize whitespace (collapse multiple spaces to one)
            normalized = ' '.join(line.split())
            if normalized:
                lines.append(normalized)
        
        return '\n'.join(lines)
    
    try:
        model_files = [
            Path(__file__).parent / "analyzers" / "financial_analyzer.py",
            Path(__file__).parent / "backtest_engine.py",
        ]
        
        combined_hash = ""
        for file_path in model_files:
            if file_path.exists():
                try:
                    # Add timeout protection for slow file reads
                    content = file_path.read_text(encoding='utf-8')
                    # Normalize: remove comments, normalize whitespace
                    normalized = normalize_code(content)
                    if normalized:
                        file_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()[:8]
                        combined_hash += f"{file_path.name}:{file_hash}_"
                except Exception as e:
                    safe_print(f"Warning: Could not hash {file_path}: {e}")
                    # Fallback: hash file anyway
                    try:
                        with open(file_path, 'rb') as f:
                            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
                            combined_hash += f"{file_path.name}:{file_hash}_"
                    except Exception:
                        # If we can't read the file at all, skip it
                        pass
        
        result = combined_hash or "default"
        # Cache in session state for fast access on tab switches
        st.session_state.model_hash_cache = result
        return result
    except Exception as e:
        safe_print(f"Error in get_model_hash(): {e}")
        result = "default"
        st.session_state.model_hash_cache = result
        return result

def get_daily_update_key():
    now = datetime.now()
    if now.weekday() >= 5:
        last_friday = now - pd.offsets.BDay(1)
        date_key = last_friday.strftime('%Y-%m-%d') + "_WEEKEND_HOLD"
    else:
        date_key = now.strftime('%Y-%m-%d') + "_DAILY_UPDATE"
    model_hash = get_model_hash()
    return f"{date_key}_MODEL_{model_hash}"

def get_cache_file_path(cache_key):
    """Generates the file path for a given cache key."""
    global CACHE_DIR
    if CACHE_DIR is None:
        CACHE_DIR = "local_cache"
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
    return os.path.join(CACHE_DIR, f"analysis_{cache_key}.pkl")

def save_local_cache(cache_key, tech_data, fund_data, fund_ranks):
    """Save analysis data to a local pickle file."""
    if not tech_data:
        safe_print("CACHE: No data to save.")
        return
    
    # Package the data into a dictionary with cache_key for validation
    data_to_cache = {
        'tech_data': tech_data,
        'fund_data': fund_data,
        'fund_ranks': fund_ranks,
        'cache_key': cache_key
    }
    
    file_path = get_cache_file_path(cache_key)
    try:
        with open(file_path, "wb") as f:
            pickle.dump(data_to_cache, f)
        safe_print(f"CACHE: Successfully saved analysis to {file_path}")
    except Exception as e:
        safe_print(f"CACHE ERROR: Failed to save cache file {file_path}: {e}")

def _load_cache_file_internal(file_path, cache_key):
    """Internal function to load pickle file."""
    try:
        with open(file_path, "rb") as f:
            cached_data = pickle.load(f)
        # Verify cache key inside the file
        if cached_data.get('cache_key') == cache_key:
            return cached_data
        else:
            safe_print(f"CACHE WARNING: Cache key mismatch in {file_path}. Ignoring.")
            return None
    except Exception as e:
        safe_print(f"CACHE ERROR: Failed to load cache file {file_path}: {e}")
        return None

def load_local_cache(cache_key):
    """Load analysis data from local pickle file - finds compatible cache files.
    
    PERFORMANCE: Caches file listing to avoid slow os.listdir() on refresh.
    """
    global CACHE_DIR
    if CACHE_DIR is None:
        CACHE_DIR = "local_cache"
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            return None
    
    # PERFORMANCE: Cache the list of available cache files to avoid slow os.listdir() on refresh
    cache_files_cache_key = 'cached_cache_files_list'
    if cache_files_cache_key not in st.session_state:
        st.session_state[cache_files_cache_key] = None
    
    # Try exact match first (fast - single file check)
    exact_file_path = get_cache_file_path(cache_key)
    if os.path.exists(exact_file_path):
        try:
            with open(exact_file_path, "rb") as f:
                cached_data = pickle.load(f)
            if (isinstance(cached_data, dict) and 
                'tech_data' in cached_data and 
                cached_data.get('tech_data') and 
                len(cached_data.get('tech_data', {})) > 0):
                safe_print(f"CACHE HIT: Found exact match")
                return cached_data
        except Exception as e:
            safe_print(f"CACHE ERROR: Failed to load {exact_file_path}: {e}")
    
    # Fallback: Find most recent FULL_SP500 cache (if we're in FULL mode)
    # PERFORMANCE: Only scan files if we don't have a cached list AND we're in FULL mode
    if "_FULL_SP500" in cache_key:
        try:
            # Use cached file list if available and fresh (< 5 minutes old)
            cache_list_timestamp_key = 'cached_cache_files_timestamp'
            current_time = time.time()
            use_cached_list = (
                cache_files_cache_key in st.session_state and 
                st.session_state[cache_files_cache_key] is not None and
                cache_list_timestamp_key in st.session_state and
                (current_time - st.session_state.get(cache_list_timestamp_key, 0)) < 300  # 5 minutes
            )
            
            if use_cached_list:
                all_files = st.session_state[cache_files_cache_key]
            else:
                # Only do slow file listing if we don't have cached results
                all_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("analysis_") and "_FULL_SP500" in f and f.endswith(".pkl")]
                # Cache the list for future use
                st.session_state[cache_files_cache_key] = all_files
                st.session_state[cache_list_timestamp_key] = current_time
            
            if all_files:
                # Sort by modification time, most recent first
                # PERFORMANCE: Cache file mtimes if possible
                all_files.sort(key=lambda f: os.path.getmtime(os.path.join(CACHE_DIR, f)), reverse=True)
                # Try the most recent one
                recent_file = os.path.join(CACHE_DIR, all_files[0])
                with open(recent_file, "rb") as f:
                    cached_data = pickle.load(f)
                if (isinstance(cached_data, dict) and 
                    'tech_data' in cached_data and 
                    cached_data.get('tech_data') and 
                    len(cached_data.get('tech_data', {})) > 0):
                    safe_print(f"CACHE HIT: Using most recent FULL_SP500 cache")
                    return cached_data
        except Exception as e:
            safe_print(f"CACHE WARNING: Could not load fallback cache: {e}")
    
    return None

@st.cache_data(show_spinner=False, max_entries=10, ttl=None)
def run_full_analysis(api_key, cache_trigger_key, watchlist_input, quick_only): 
    # CRITICAL: This function only executes on cache miss
    # If cache hit, Streamlit returns cached data WITHOUT executing this function
    # If you see the progress bar, it means this function IS running = API calls being made
    
    # IMPORTANT: Only show warnings if function actually executes (cache miss)
    # If cache hit, this code never runs
    # NOTE: The warning is now shown in the main layout section above, not here
    # Set session state flag so main layout knows analysis is running
    if 'analysis_running_flag' not in st.session_state:
        st.session_state.analysis_running_flag = True
    
    # Terminal logging
    safe_print(f"⚠️ FULL ANALYSIS RUNNING - API CALLS BEING MADE! (Cache Key: {cache_trigger_key})")
    safe_print(f"⚠️ This should ONLY happen when: date changes OR model code changes OR cache cleared")
    safe_print(f"⚠️ NOTE: Hard refresh (Cmd+Shift+R) clears session state but @st.cache_data should still work!")
    provider = FMPProvider(api_key=api_key)
    analyzer = FinancialAnalyzer()
    default_watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "AMZN"]
    # Build watchlist from user input or default
    # Handle special marker for full S&P 500 mode (used for stable cache key)
    if watchlist_input and len(watchlist_input) == 1 and watchlist_input[0] == "_FULL_SP500_MODE":
        watchlist = default_watchlist  # Will be expanded to S&P 500 below
    else:
        watchlist = [t.strip().upper() for t in (watchlist_input or []) if t.strip()]
    if not watchlist:
        watchlist = default_watchlist
    
    # CRITICAL FIX: When quick_only=False, we always expand to S&P 500
    # So the cache key should be stable regardless of input watchlist
    # The cache_trigger_key already includes this information
    if not quick_only:
        try:
            url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
            response = requests.get(url, timeout=10); response.raise_for_status()
            data = response.json()
            if data:
                tickers = [stock['symbol'] for stock in data]
                if tickers and len(tickers) > 10:
                    # Always use S&P 500 for full mode - this ensures cache key stability
                    watchlist = list(dict.fromkeys(watchlist + tickers))[:502]
        except Exception as e:
            safe_print(f"Could not fetch S&P 500 list: {e}. Using watchlist only.")
    
    all_tech_data, all_fund_data, all_fund_ranks = {}, {}, {}
    # CRITICAL: Progress bar will only show if function actually runs (cache miss)
    # If @st.cache_data returns cached data, this function doesn't execute, so no progress bar
    # If you see this progress bar, it means API calls are being made (cache miss or first run)
    progress_bar = st.progress(0, "Analyzing stocks...")
    end_date_str = pd.to_datetime('today').strftime('%Y-%m-%d')
    
    # Load SPY data first for market regime calculation
    # Also ensure SPY and QQQ are always in tech_data (even if not in watchlist)
    spy_data = None
    essential_tickers_data = {}  # Track SPY and QQQ data if loaded separately
    
    # Load SPY (for market regime calculation)
    if 'SPY' in watchlist:
        spy_df = provider.get_daily_stock_data('SPY', '1990-01-01', end_date_str)
        if not spy_df.empty and len(spy_df) > 200:
            spy_data = spy_df.copy()
    else:
        # Try to load SPY even if not in watchlist (needed for market regime)
        try:
            spy_df = provider.get_daily_stock_data('SPY', '1990-01-01', end_date_str)
            if not spy_df.empty and len(spy_df) > 200:
                spy_data = spy_df.copy()
                essential_tickers_data['SPY'] = spy_df.copy()
        except:
            pass
    
    # Load QQQ (always include in tech_data)
    if 'QQQ' not in watchlist:
        try:
            qqq_df = provider.get_daily_stock_data('QQQ', '1990-01-01', end_date_str)
            if not qqq_df.empty and len(qqq_df) > 200:
                essential_tickers_data['QQQ'] = qqq_df.copy()
        except:
            pass
    
    for i, ticker in enumerate(watchlist):
        progress_bar.progress((i + 1) / len(watchlist), f"Analyzing {ticker} ({i+1}/{len(watchlist)})...")
        if not quick_only and i > 0 and i % 5 == 0:
            time.sleep(1.5)
        
        tech_df = provider.get_daily_stock_data(ticker, '1990-01-01', end_date_str)
        if not tech_df.empty and len(tech_df) > 200:
            # Pass SPY data for market regime calculation (for all tickers, not just backtest)
            data_with_indicators, _ = analyzer.run_full_analysis(tech_df.copy(), spy_data=spy_data)
            # CRITICAL: Only add to tech_data if analysis returned valid data (prevents None values in cache)
            if (data_with_indicators is not None and 
                isinstance(data_with_indicators, pd.DataFrame) and 
                not data_with_indicators.empty):
                all_tech_data[ticker] = data_with_indicators
            else:
                safe_print(f"⚠️ Skipping {ticker}: Analysis returned None or empty data")
        
        if ticker not in ["SPY", "QQQ"] and not tech_df.empty:
            fund_df = provider.get_daily_fundamental_ratios(ticker, daily_prices=tech_df)
            if not fund_df.empty:
                for metric in ['P/E', 'P/S', 'PEG']:
                    if metric in fund_df.columns:
                        fund_df[f'{metric}_Rank_Plot'] = fund_df[metric].expanding(min_periods=20).apply(
                            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100 if len(x) > 1 else np.nan, raw=False)
                all_fund_data[ticker] = fund_df
                _, ranks = analyzer.run_full_fundamental_analysis(fund_df)
                all_fund_ranks[ticker] = ranks
    
    # Always ensure SPY and QQQ are in tech_data (add them if not already there)
    # CRITICAL: Add SPY first, then QQQ (QQQ needs SPY for market regime)
    essential_tickers = ['SPY', 'QQQ']
    
    # First, ensure SPY is loaded and available
    spy_data_for_others = None
    if 'SPY' not in all_tech_data:
        spy_ticker_data = None
        if 'SPY' in essential_tickers_data:
            spy_ticker_data = essential_tickers_data['SPY']
        else:
            try:
                spy_df = provider.get_daily_stock_data('SPY', '1990-01-01', end_date_str)
                if not spy_df.empty and len(spy_df) > 200:
                    spy_ticker_data = spy_df.copy()
            except:
                pass
        
        if spy_ticker_data is not None and not spy_ticker_data.empty:
            try:
                # SPY uses its own data for market regime
                analyzed_spy, _ = analyzer.run_full_analysis(spy_ticker_data.copy(), spy_data=spy_ticker_data.copy())
                if (analyzed_spy is not None and 
                    isinstance(analyzed_spy, pd.DataFrame) and 
                    not analyzed_spy.empty):
                    all_tech_data['SPY'] = analyzed_spy
                    spy_data_for_others = spy_ticker_data.copy()  # Store for QQQ to use
                    safe_print("✅ Added SPY to tech_data (loaded separately)")
                else:
                    safe_print("⚠️ Skipping SPY: Analysis returned None or empty data")
            except Exception as e:
                safe_print(f"⚠️ Could not add SPY to tech_data: {e}")
    
    # Now add QQQ (which needs SPY for market regime)
    if 'QQQ' not in all_tech_data:
        qqq_ticker_data = None
        if 'QQQ' in essential_tickers_data:
            qqq_ticker_data = essential_tickers_data['QQQ']
        else:
            try:
                qqq_df = provider.get_daily_stock_data('QQQ', '1990-01-01', end_date_str)
                if not qqq_df.empty and len(qqq_df) > 200:
                    qqq_ticker_data = qqq_df.copy()
            except:
                pass
        
        if qqq_ticker_data is not None and not qqq_ticker_data.empty:
            try:
                # QQQ needs SPY data for market regime - use spy_data_for_others or get from all_tech_data
                spy_for_qqq = spy_data_for_others
                if spy_for_qqq is None and 'SPY' in all_tech_data:
                    # Try to get original SPY data if we have analyzed SPY
                    spy_analyzed = all_tech_data.get('SPY')
                    if spy_analyzed is not None and isinstance(spy_analyzed, pd.DataFrame) and 'close' in spy_analyzed.columns:
                        spy_for_qqq = spy_analyzed[['close', 'open', 'high', 'low', 'volume']].copy()
                
                analyzed_qqq, _ = analyzer.run_full_analysis(qqq_ticker_data.copy(), spy_data=spy_for_qqq)
                if (analyzed_qqq is not None and 
                    isinstance(analyzed_qqq, pd.DataFrame) and 
                    not analyzed_qqq.empty):
                    all_tech_data['QQQ'] = analyzed_qqq
                    safe_print("✅ Added QQQ to tech_data (loaded separately)")
                else:
                    safe_print("⚠️ Skipping QQQ: Analysis returned None or empty data")
            except Exception as e:
                safe_print(f"⚠️ Could not add QQQ to tech_data: {e}")

    progress_bar.empty()
    return all_tech_data, all_fund_data, all_fund_ranks

def get_quick_prices(api_key, tickers):
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
    fig.update_yaxes(title_text="<b>Percentile Rank (%)</b>", secondary_y=True, range=[0, 100], gridcolor='#D3D3D3')
    
    if metric == 'P/E':
        df['P/E'] = pd.to_numeric(df['P/E'], errors='coerce') 
        df_filtered = df['P/E'][df['P/E'] > -100]
        if not df_filtered.empty:
            pe_min = df_filtered.quantile(0.01)
            pe_max_filtered = df_filtered[df_filtered < 200]
            pe_max = pe_max_filtered.quantile(0.99) if not pe_max_filtered.empty else 50
            fig.update_yaxes(range=[pe_min, pe_max * 1.1], secondary_y=False)
        else:
            fig.update_yaxes(range=[0, 50], secondary_y=False) 
    
    elif metric == 'P/S' and 'P/S' in df.columns and not df['P/S'].empty: 
        fig.update_yaxes(range=[0, df['P/S'].quantile(0.99) * 1.1], secondary_y=False)
    elif metric == 'PEG': 
        fig.update_yaxes(range=[-5, 5], secondary_y=False)
    return fig

# Don't add loading placeholder - it might cause blocking
# Streamlit will show loading automatically

# PERFORMANCE: Select page FIRST - before loading any data
# This allows Model Explanation and Backtest to skip data loading if not needed
st.sidebar.title("App Controls")
# PERFORMANCE: Theme selector moved here (early in sidebar, after page selector)
try:
    st.session_state.theme_choice = st.sidebar.selectbox("Theme", ["Dark", "Light"], index=0 if st.session_state.theme_choice == "Dark" else 1, key="theme_selector")
except Exception:
    pass  # Keep default if error

page = st.sidebar.radio("Select a Page", ["Technical Dashboard", "Fundamental Explorer", "Model Explanation", "Backtest"])
st.sidebar.markdown("---")

# PERFORMANCE: Only load data if needed for the selected page
# Model Explanation and Backtest can load data lazily
needs_data_loading = page in ["Technical Dashboard", "Fundamental Explorer", "Backtest"]

st.sidebar.subheader("Watchlist (fast mode)")
wl_input = st.sidebar.text_input("Tickers (comma-separated)", value=st.session_state.get('saved_watchlist', ''), key="watchlist_input")
# Auto-save watchlist to session state (persists across reruns within the same session)
st.session_state.saved_watchlist = wl_input
watchlist = [t.strip().upper() for t in wl_input.split(',') if t.strip()][:50]
quick_only = st.sidebar.checkbox("Analyze only watchlist (faster)", value=False, help="When checked, only analyzes the tickers in your watchlist. When unchecked, analyzes all S&P 500 stocks (502 stocks, takes several minutes)")

# Initialize data variables (will be loaded only if needed)
cache_status_type = None  # 'session', 'local_file', 'streamlit', or 'running'
cache_date_key = None
num_stocks = None
tech_data = {}  # Initialize to prevent errors
fund_data = {}
fund_ranks = {}

# PERFORMANCE: Only load data if the current page needs it
# Model Explanation doesn't need tech_data, so skip loading for that page
if needs_data_loading:
    # Check session state first (fast, no API calls if already loaded in this session)
    # Session state persists during the browser session, but is cleared on hard refresh (Cmd+Shift+R)
    # NOTE: Regular refresh (Cmd+R) should NOT clear session state, but Streamlit sometimes does
    # IMPORTANT: Always check session state FIRST to avoid loading large cache files on tab switches

    # CRITICAL: Always use session state if it exists - don't reload on tab switches
    # This prevents blocking when switching tabs
    # PERFORMANCE: Skip cache key calculation if we have session state data (faster tab switching)
    if ('tech_data' in st.session_state and 
        st.session_state.tech_data is not None and
        isinstance(st.session_state.tech_data, dict) and
        len(st.session_state.tech_data) > 0):
        # Session state has data - USE IT IMMEDIATELY (don't check cache keys, don't reload)
        tech_data = st.session_state.tech_data
        fund_data = st.session_state.fund_data if 'fund_data' in st.session_state else {}
        fund_ranks = st.session_state.fund_ranks if 'fund_ranks' in st.session_state else {}
        cache_status_type = 'session'
        # Lazy: Only calculate cache key for display if needed (use cached date from previous calculation)
        if 'last_cache_date_key' in st.session_state:
            cache_date_key = st.session_state.last_cache_date_key
        else:
            # Only calculate if we need it for display (defers expensive hash calculation)
            daily_cache_key = get_daily_update_key()
            cache_date_key = daily_cache_key.split('_MODEL_')[0] if daily_cache_key else None
            st.session_state.last_cache_date_key = cache_date_key
        num_stocks = len(tech_data)
        # Cache ticker list for fast tab switching (avoid expensive dict.keys() on every tab switch)
        if 'available_tickers_cache' not in st.session_state or not st.session_state.available_tickers_cache:
            st.session_state.available_tickers_cache = list(tech_data.keys())
        safe_print(f"✅ SESSION CACHE HIT: Using {num_stocks} stocks from session state (fast - no reload)")

    # Only load from cache/API if we don't have valid session state data
    # CRITICAL: This should NOT run on tab switches - only when truly needed
    # PERFORMANCE: Only calculate cache key here when we actually need it (when data is not in session state)
    if not tech_data or len(tech_data) == 0:
        # Now calculate cache key (only when we actually need to load data)
        daily_cache_key = get_daily_update_key()
        
        # CRITICAL: The cache key must match what the function will actually use
        # When quick_only=False, the function expands to S&P 500 (502 stocks)
        # So we need to use a stable key that reflects this
        if quick_only:
            # For quick mode, use the actual watchlist
            effective_cache_key = f"{daily_cache_key}_QUICK_{tuple(sorted(watchlist))}"
        else:
            # For full mode, it will expand to S&P 500, so use a stable key for that
            effective_cache_key = f"{daily_cache_key}_FULL_SP500"
        
        cache_key_for_session = effective_cache_key
        # Store date key for fast access later
        st.session_state.last_cache_date_key = daily_cache_key.split('_MODEL_')[0] if daily_cache_key else None
        if quick_only:
            cache_watchlist_input = tuple(sorted(watchlist))
        else:
            cache_watchlist_input = ("_FULL_SP500_MODE",)
        
        # Try local cache (fast check - only exact match, no scanning)
        local_cache_result = load_local_cache(cache_key_for_session)
        
        if local_cache_result and isinstance(local_cache_result, dict):
            tech_data = local_cache_result.get('tech_data', {})
            fund_data = local_cache_result.get('fund_data', {})
            fund_ranks = local_cache_result.get('fund_ranks', {})
            
            # CRITICAL: Clean any None values from cache (fixes corrupted cache files)
            if tech_data and isinstance(tech_data, dict):
                none_count = 0
                valid_tech_data = {}
                for ticker, data in tech_data.items():
                    if (data is not None and 
                        isinstance(data, pd.DataFrame) and 
                        not data.empty):
                        valid_tech_data[ticker] = data
                    else:
                        none_count += 1
                if none_count > 0:
                    safe_print(f"⚠️ Removed {none_count} corrupted None entries from cache file")
                    tech_data = valid_tech_data
            
            if tech_data and len(tech_data) > 0:
                st.session_state.tech_data = tech_data
                st.session_state.fund_data = fund_data
                st.session_state.fund_ranks = fund_ranks
                st.session_state.cached_analysis_key = cache_key_for_session
                cache_status_type = 'local_file'
                cache_date_key = daily_cache_key.split('_MODEL_')[0]
                num_stocks = len(tech_data)
            else:
                # Cache exists but empty - use Streamlit cache
                tech_data, fund_data, fund_ranks = run_full_analysis(FMP_API_KEY, daily_cache_key, cache_watchlist_input, quick_only)
                if tech_data and len(tech_data) > 0:
                    save_local_cache(cache_key_for_session, tech_data, fund_data, fund_ranks)
                    st.session_state.tech_data = tech_data
                    st.session_state.fund_data = fund_data
                    st.session_state.fund_ranks = fund_ranks
                    st.session_state.cached_analysis_key = cache_key_for_session
                    cache_status_type = 'running'
                    cache_date_key = daily_cache_key.split('_MODEL_')[0]
                    num_stocks = len(tech_data)
        else:
            # No local cache - use Streamlit cache (which may trigger API calls)
            tech_data, fund_data, fund_ranks = run_full_analysis(FMP_API_KEY, daily_cache_key, cache_watchlist_input, quick_only)
            if tech_data and len(tech_data) > 0:
                save_local_cache(cache_key_for_session, tech_data, fund_data, fund_ranks)
                st.session_state.tech_data = tech_data
                st.session_state.fund_data = fund_data
                st.session_state.fund_ranks = fund_ranks
                st.session_state.cached_analysis_key = cache_key_for_session
                cache_status_type = 'running'
                cache_date_key = daily_cache_key.split('_MODEL_')[0]
                num_stocks = len(tech_data)

    if not tech_data or len(tech_data) == 0:
        if page != "Model Explanation":  # Model Explanation doesn't need data
            st.error("The main analysis returned 0 stocks.")
            st.stop()

# --- UI & CONTENT LAYOUT ---
# (Page selection already done earlier for performance)
# PERFORMANCE: Only show manual override form if we have tech_data (skip for Model Explanation)
if tech_data and len(tech_data) > 0:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Data Override (All Users)")
    with st.sidebar.form("manual_price_form"):
        tickers_to_override = st.selectbox(
            "Select Ticker for Manual Price:", 
            options=list(tech_data.keys()),
            key='manual_ticker_select'
        )
        override_date = st.date_input("Date of Price:", pd.to_datetime('today') - pd.offsets.BDay(1), key='manual_date_input')
        selected_ticker_for_label = st.session_state.get('manual_ticker_select', 'TICKER') 
        override_price = st.number_input(f"Closing Price for {selected_ticker_for_label}:", min_value=0.01, key='manual_price_input')
        submitted = st.form_submit_button("SAVE MANUAL PRICE")
        if submitted:
            provider = FMPProvider(api_key=FMP_API_KEY)
            manual_df = provider.get_manual_prices()
            date_ts = pd.to_datetime(override_date)
            if manual_df.empty:
                manual_df = pd.DataFrame(columns=[st.session_state.manual_ticker_select], index=pd.to_datetime([date_ts]))
            if st.session_state.manual_ticker_select not in manual_df.columns:
                 manual_df[st.session_state.manual_ticker_select] = np.nan
            manual_df.loc[date_ts, st.session_state.manual_ticker_select] = override_price
            provider.save_manual_prices(manual_df.dropna(axis=1, how='all'))
            st.success(f"Saved manual price for {st.session_state.manual_ticker_select}.")
            st.warning("Data will be used in the next scheduled analysis.")

# PERFORMANCE: Only show admin controls if we have tech_data (skip for Model Explanation)
if tech_data and len(tech_data) > 0:
    st.sidebar.markdown("---")
    if st.session_state.user_role == 'admin':
        st.sidebar.subheader("Admin Controls")
        if st.sidebar.button("Run FULL Analysis (Clear Cache)"):
            st.session_state.show_password_prompt = True
        if st.session_state.show_password_prompt:
            with st.sidebar.form("password_form_manual"):
                password_manual = st.text_input("Enter Admin Password", type="password", key="manual_password_input")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    try:
                        if password_manual == st.secrets["admin_password"]:
                            st.session_state.show_password_prompt = False
                            st.cache_data.clear()
                            st.rerun()
                        else: st.error("Incorrect password")
                    except KeyError: st.error("Admin password is not set.")
        if st.sidebar.button("Quick Price Refresh"):
            # Only fetch prices for watchlist stocks to keep it fast
            if quick_only:
                tickers_to_refresh = watchlist
            else:
                # Limit to first 100 tickers for S&P 500 mode (otherwise takes ~4-5 minutes)
                tickers_to_refresh = list(tech_data.keys())[:100]
                st.info(f"⚠️ Refreshing prices for first 100 stocks only (out of {len(tech_data)}). Use watchlist mode for specific tickers.")
            st.session_state.quick_prices = get_quick_prices(FMP_API_KEY, tickers_to_refresh)
            st.rerun() 
    else:
        st.sidebar.markdown("*(Admin controls hidden)*")

# Only show dashboard header on non-Backtest pages
# PERFORMANCE: Skip header completely for Model Explanation to avoid any blocking operations
if page != "Backtest" and page != "Model Explanation":
    # Header with title and date
    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.title("📈 Trading Model Dashboard")
    with col_date:
        if cache_date_key:
            # Extract just the date part (before _DAILY_UPDATE or _WEEKEND_HOLD)
            date_only = cache_date_key.split('_')[0]
            st.write("")  # Spacing
            st.caption(f"Last Model Data Run Date: {date_only}")
    
    # Show cache/analysis status - ALWAYS show an expander
    # Check if analysis is running (set by run_full_analysis function)
    is_analysis_running = st.session_state.get('analysis_running_flag', False)
    
    # If flag is set, we're running analysis (API calls happening)
    # OR if we just finished but haven't set cache_status_type yet
    if is_analysis_running or cache_status_type == 'running':
        if cache_status_type != 'running':
            cache_status_type = 'running'
        if not cache_date_key:
            # Use cached date key from session state if available, otherwise calculate
            if 'last_cache_date_key' in st.session_state and st.session_state.last_cache_date_key:
                cache_date_key = st.session_state.last_cache_date_key
            else:
                daily_cache_key = get_daily_update_key()
                cache_date_key = daily_cache_key.split('_MODEL_')[0] if daily_cache_key else None
    
    # Ensure cache_status_type is always set (default to unknown if not set)
    if not cache_status_type:
        cache_status_type = 'unknown'
    
    if not cache_date_key:
        # Use cached date key from session state if available, otherwise calculate
        if 'last_cache_date_key' in st.session_state and st.session_state.last_cache_date_key:
            cache_date_key = st.session_state.last_cache_date_key
        else:
            daily_cache_key = get_daily_update_key()
            cache_date_key = daily_cache_key.split('_MODEL_')[0] if daily_cache_key else None
    
    if not num_stocks:
        num_stocks = len(tech_data) if tech_data else 0
    
    # Always show expander with appropriate status
    if cache_status_type == 'running':
        # Analysis is running - show warning expander
        with st.expander("🚨 API CALLS IN PROGRESS - Analysis Running", expanded=True):
            st.warning("⚠️ **Making API calls - this should only happen when:**\n- First run\n- Date changed\n- Model code changed\n- Cache cleared\n\n**Let this complete once, then cache will be used.**")
    elif cache_status_type in ['session', 'local_file', 'streamlit']:
        # Using cache - show success expander
        cache_type_name = {
            'session': 'Session Cache',
            'local_file': 'Local File Cache',
            'streamlit': 'Streamlit Cache'
        }.get(cache_status_type, 'Cache')
        
        with st.expander(f"✅ Using {cache_type_name} - Analysis Complete", expanded=False):
            st.success(f"✅ **Analysis complete for {num_stocks} stocks. (No API calls, instant load)**")
    else:
        # Unknown status - show info
        with st.expander("ℹ️ Cache Status", expanded=False):
            st.info(f"Loaded {num_stocks} stocks. Cache status: {cache_status_type}")
    
    if st.session_state.user_role == 'admin':
        st.markdown("*(Logged in as Admin)*")
    else:
        st.markdown("*(Logged in as Viewer)*")

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
        st.dataframe(pd.DataFrame(price_comparison).set_index('Ticker').head(25), use_container_width=True)

if page == "Technical Dashboard":
    tear_sheet_data = []
    for ticker, df in tech_data.items():
        if df.empty: continue
        latest = df.iloc[-1]
        # Helper function to safely extract values
        def safe_get(key, default=None):
            if key in latest and pd.notna(latest.get(key, np.nan)):
                return latest.get(key)
            return default
        
        # Extract all scores and values
        trend_score = safe_get('Trend_Score')
        rev_score = safe_get('Reversion_Score')
        close_val = safe_get('close')
        # Trend components
        ma_ribbon = safe_get('score_ma_ribbon', 0)
        score_50sma = safe_get('score_50sma', 0)
        score_200sma = safe_get('score_200sma', 0)
        score_macd = safe_get('score_macd', 0)
        score_cross = safe_get('score_cross', 0)
        
        # Reversion components
        score_rsi = safe_get('score_rsi', 0)
        score_stoch = safe_get('score_stoch', 0)
        score_bbands = safe_get('score_bbands', 0)
        score_patterns = safe_get('score_patterns', 0)
        
        # Actual indicator values
        rsi_val = safe_get('RSI_14')
        stoch_val = safe_get('STOCHk_14_3_3')
        macd_val = safe_get('MACD_12_26_9')
        macd_signal_val = safe_get('MACDs_12_26_9')
        
        # Moving Average values
        sma_5 = safe_get('SMA_5')
        sma_10 = safe_get('SMA_10')
        sma_15 = safe_get('SMA_15')
        sma_20 = safe_get('SMA_20')
        sma_50 = safe_get('SMA_50')
        sma_200 = safe_get('SMA_200')
        
        # Convert scores to integers (remove decimals)
        def to_int(val):
            if pd.isna(val) or val is None:
                return None
            try:
                return int(float(val))
            except (ValueError, TypeError):
                return None
        
        row = {
            'Ticker': ticker,
            'Close': f"${close_val:.2f}" if pd.notna(close_val) else "N/A",
            'Trend_Score': to_int(trend_score),
            'Reversion_Score': to_int(rev_score),
            # Individual scores (for expandable section)
            'MA_Ribbon_Score': to_int(ma_ribbon),
            '50_SMA_Score': to_int(score_50sma),
            '200_SMA_Score': to_int(score_200sma),
            'MACD_Score': to_int(score_macd),
            'Cross_Score': to_int(score_cross),
            'RSI_Score': to_int(score_rsi),
            'Stoch_Score': to_int(score_stoch),
            'BBands_Score': to_int(score_bbands),
            'Patterns_Score': to_int(score_patterns),
            # Actual metric values
            'RSI': f"{rsi_val:.2f}" if pd.notna(rsi_val) else "N/A",
            'Stoch_%K': f"{stoch_val:.2f}" if pd.notna(stoch_val) else "N/A",
            'MACD': f"{macd_val:.2f}" if pd.notna(macd_val) else "N/A",
            'MACD_Signal': f"{macd_signal_val:.2f}" if pd.notna(macd_signal_val) else "N/A",
            'SMA_5': f"${sma_5:.2f}" if pd.notna(sma_5) else "N/A",
            'SMA_10': f"${sma_10:.2f}" if pd.notna(sma_10) else "N/A",
            'SMA_15': f"${sma_15:.2f}" if pd.notna(sma_15) else "N/A",
            'SMA_20': f"${sma_20:.2f}" if pd.notna(sma_20) else "N/A",
            'SMA_50': f"${sma_50:.2f}" if pd.notna(sma_50) else "N/A",
            'SMA_200': f"${sma_200:.2f}" if pd.notna(sma_200) else "N/A",
        }
        
        # Use pre-computed fund_ranks instead of re-calculating
        if ticker in fund_ranks:
            rank_summary = fund_ranks[ticker]
            # Handle both dict and Series formats
            if isinstance(rank_summary, dict):
                for metric in ['P/E', 'P/S', 'PEG']:
                    # Extract from nested structure if needed
                    metric_data = rank_summary.get(metric, {})
                    if isinstance(metric_data, dict):
                        short_terms = [metric_data.get(k) for k in ['30 days', '90 days', '120 days'] if metric_data.get(k) is not None]
                        long_terms = [metric_data.get(k) for k in ['1 year', '2 years', '3 years', '5 years', '7 years', '10 years', 'Full History'] if metric_data.get(k) is not None]
                        row[f'{metric}_Short_Term_Rank'] = np.mean(short_terms) if short_terms else None
                        row[f'{metric}_Long_Term_Rank'] = np.mean(long_terms) if long_terms else None
                    else:
                        # Direct summary format
                        row[f'{metric}_Short_Term_Rank'] = rank_summary.get(f'{metric}_Short_Term_Rank')
                        row[f'{metric}_Long_Term_Rank'] = rank_summary.get(f'{metric}_Long_Term_Rank')
            else:
                # Series format
                for metric in ['P/E', 'P/S', 'PEG']:
                    row[f'{metric}_Short_Term_Rank'] = rank_summary.get(f'{metric}_Short_Term_Rank') if hasattr(rank_summary, 'get') else None
                    row[f'{metric}_Long_Term_Rank'] = rank_summary.get(f'{metric}_Long_Term_Rank') if hasattr(rank_summary, 'get') else None
        tear_sheet_data.append(row)
    
    if tear_sheet_data:
        tear_sheet_df = pd.DataFrame(tear_sheet_data).set_index('Ticker')
        
        # Display cache status in Technical Dashboard tab
        daily_cache_key = get_daily_update_key()
        model_hash = get_model_hash()
        date_part = daily_cache_key.split('_MODEL_')[0]
        hash_parts = model_hash.split('_') if model_hash else []

        # Parse hash parts for display
        model_info = {}
        for part in hash_parts:
            if ':' in part:
                filename, hash_val = part.rsplit(':', 1)
                if filename.endswith('.py'):
                    model_info[filename] = hash_val

        # Inject CSS for frozen column and horizontal scrolling in metric details tables
        st.markdown("""
            <style>
            /* Enable horizontal scrolling - target all possible container structures */
            div[data-testid="stExpander"] .stDataFrame,
            div[data-testid="stExpander"] div[data-testid="stDataFrameContainer"],
            div[data-testid="stExpander"] .element-container .stDataFrame,
            div[data-testid="stExpander"] .element-container > div {
                overflow-x: auto !important;
                overflow-y: visible !important;
            }
            
            /* Ensure scrollable wrapper */
            div[data-testid="stExpander"] .stDataFrame > div,
            div[data-testid="stExpander"] div[data-testid="stDataFrameContainer"] > div {
                overflow-x: auto !important;
                position: relative !important;
            }
            
            /* Fix table layout for sticky positioning */
            div[data-testid="stExpander"] table,
            div[data-testid="stExpander"] .stDataFrame table {
                border-collapse: separate !important;
                border-spacing: 0 !important;
                width: 100% !important;
            }
            
            /* Freeze first column header - comprehensive selectors */
            div[data-testid="stExpander"] table thead tr th:first-child,
            div[data-testid="stExpander"] .stDataFrame table thead tr th:first-child,
            div[data-testid="stExpander"] div[data-testid="stDataFrameContainer"] table thead tr th:first-child,
            div[data-testid="stExpander"] .element-container table thead tr th:first-child {
                position: -webkit-sticky !important;
                position: sticky !important;
                left: 0 !important;
                z-index: 999 !important;
                background-color: rgb(17, 24, 39) !important;
                border-right: 3px solid rgba(16, 185, 129, 0.7) !important;
                box-shadow: 4px 0 8px rgba(0, 0, 0, 0.4) !important;
                min-width: 80px !important;
                white-space: nowrap !important;
            }
            
            /* Freeze first column body cells - comprehensive selectors */
            div[data-testid="stExpander"] table tbody tr td:first-child,
            div[data-testid="stExpander"] .stDataFrame table tbody tr td:first-child,
            div[data-testid="stExpander"] div[data-testid="stDataFrameContainer"] table tbody tr td:first-child,
            div[data-testid="stExpander"] .element-container table tbody tr td:first-child {
                position: -webkit-sticky !important;
                position: sticky !important;
                left: 0 !important;
                z-index: 998 !important;
                background-color: rgb(17, 24, 39) !important;
                border-right: 3px solid rgba(16, 185, 129, 0.7) !important;
                box-shadow: 4px 0 8px rgba(0, 0, 0, 0.4) !important;
                min-width: 80px !important;
                white-space: nowrap !important;
            }
            
            /* Light theme support */
            [data-theme="light"] div[data-testid="stExpander"] table thead tr th:first-child,
            [data-theme="light"] div[data-testid="stExpander"] table tbody tr td:first-child,
            [data-theme="light"] div[data-testid="stExpander"] .stDataFrame table thead tr th:first-child,
            [data-theme="light"] div[data-testid="stExpander"] .stDataFrame table tbody tr td:first-child {
                background-color: rgb(255, 255, 255) !important;
            }
            
            /* Alternating row backgrounds in dark theme */
            div[data-testid="stExpander"] table tbody tr:nth-child(even) td:first-child,
            div[data-testid="stExpander"] .stDataFrame table tbody tr:nth-child(even) td:first-child {
                background-color: rgb(22, 29, 43) !important;
            }
            
            /* Alternating row backgrounds in light theme */
            [data-theme="light"] div[data-testid="stExpander"] table tbody tr:nth-child(even) td:first-child,
            [data-theme="light"] div[data-testid="stExpander"] .stDataFrame table tbody tr:nth-child(even) td:first-child {
                background-color: rgb(250, 250, 250) !important;
            }
            
            /* Header row background */
            div[data-testid="stExpander"] table thead tr {
                background-color: rgb(17, 24, 39) !important;
            }
            [data-theme="light"] div[data-testid="stExpander"] table thead tr {
                background-color: rgb(255, 255, 255) !important;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Helper function to color-code scores
        def color_scores(val):
            if pd.isna(val) or val is None:
                return ''
            try:
                num_val = float(val)
                if num_val > 0:
                    return 'background-color: rgba(16, 185, 129, 0.2)'  # Green for positive
                elif num_val < 0:
                    return 'background-color: rgba(239, 68, 68, 0.2)'  # Red for negative
            except (ValueError, TypeError):
                pass
            return ''
        
        # Reset index to make Ticker a column for easier access
        tear_sheet_df = tear_sheet_df.reset_index()
        
        # Define column groups for expandable sections
        base_cols = ['Ticker', 'Close', 'Trend_Score', 'Reversion_Score']
        score_cols = [c for c in tear_sheet_df.columns if 'Score' in c and c not in ['Trend_Score', 'Reversion_Score']]
        metric_value_cols = ['RSI', 'Stoch_%K', 'MACD', 'MACD_Signal'] + [c for c in tear_sheet_df.columns if c.startswith('SMA_')]
        fund_cols = [c for c in tear_sheet_df.columns if 'Rank' in c]
        
        st.header("📈 Trend-Following Signals (Max Score: +4 / -4)")
        col1, col2 = st.columns(2)
        
        with col1: 
            st.subheader("Strong Long Trends (Top 25)")
            trend_df_long = tear_sheet_df.sort_values(by="Trend_Score", ascending=False, na_position='last').head(25)
            
            # Create column configuration for expandable sections
            column_config = {
                'Ticker': st.column_config.TextColumn('Ticker', width='small'),
                'Close': st.column_config.TextColumn('Price', width='small'),
                'Trend_Score': st.column_config.NumberColumn('Trend', width='small', format='%d'),
                'Reversion_Score': st.column_config.NumberColumn('Reversion', width='small', format='%d'),
            }
            
            # Hide detail columns by default, show in expandable
            display_cols = base_cols.copy()
            hidden_cols = score_cols + metric_value_cols + fund_cols
            
            with st.expander("📊 Metric Details (Scores & Values)", expanded=False):
                st.caption("Individual indicator scores paired with their actual values")
                detail_cols = ['Ticker', 'Close', 'Trend_Score', 'Reversion_Score']
                detail_cols.extend(['MA_Ribbon_Score', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_20'])
                detail_cols.extend(['50_SMA_Score', 'SMA_50', '200_SMA_Score', 'SMA_200'])
                detail_cols.extend(['MACD_Score', 'MACD', 'MACD_Signal', 'Cross_Score'])
                detail_cols.extend(['RSI_Score', 'RSI', 'Stoch_Score', 'Stoch_%K'])
                detail_cols.extend(['BBands_Score', 'Patterns_Score'])
                detail_cols = [c for c in detail_cols if c in trend_df_long.columns]
                detail_df = trend_df_long[detail_cols]
                # Ensure Ticker is first column and set as index temporarily for proper display
                if 'Ticker' in detail_df.columns:
                    detail_df = detail_df.set_index('Ticker').reset_index()
                st.dataframe(
                    detail_df.style.applymap(color_scores, subset=[c for c in score_cols if c in detail_df.columns]),
                    use_container_width=True,
                    height=400
                )
            
            with st.expander("💎 Fundamental Rankings", expanded=False):
                st.caption("Valuation percentile ranks (lower = cheaper)")
                fund_df = trend_df_long[[c for c in base_cols + fund_cols if c in trend_df_long.columns]]
                if not fund_df.empty:
                    st.dataframe(fund_df, use_container_width=True)
            
            # Main table - just essential columns
            main_df = trend_df_long[display_cols]
            st.dataframe(
                main_df.style.applymap(color_scores, subset=['Trend_Score', 'Reversion_Score']),
                use_container_width=True,
                column_config=column_config
            )
            
        with col2: 
            st.subheader("Strong Short Trends (Bottom 25)")
            trend_df_short = tear_sheet_df.sort_values(by="Trend_Score", ascending=True, na_position='last').head(25)
            
            with st.expander("📊 Metric Details (Scores & Values)", expanded=False):
                st.caption("Individual indicator scores paired with their actual values")
                detail_cols = ['Ticker', 'Close', 'Trend_Score', 'Reversion_Score']
                detail_cols.extend(['MA_Ribbon_Score', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_20'])
                detail_cols.extend(['50_SMA_Score', 'SMA_50', '200_SMA_Score', 'SMA_200'])
                detail_cols.extend(['MACD_Score', 'MACD', 'MACD_Signal', 'Cross_Score'])
                detail_cols.extend(['RSI_Score', 'RSI', 'Stoch_Score', 'Stoch_%K'])
                detail_cols.extend(['BBands_Score', 'Patterns_Score'])
                detail_cols = [c for c in detail_cols if c in trend_df_short.columns]
                detail_df = trend_df_short[detail_cols]
                # Ensure Ticker is first column and set as index temporarily for proper display
                if 'Ticker' in detail_df.columns:
                    detail_df = detail_df.set_index('Ticker').reset_index()
                st.dataframe(
                    detail_df.style.applymap(color_scores, subset=[c for c in score_cols if c in detail_df.columns]),
                    use_container_width=True,
                    height=400
                )
            
            with st.expander("💎 Fundamental Rankings", expanded=False):
                st.caption("Valuation percentile ranks (lower = cheaper)")
                fund_df = trend_df_short[[c for c in base_cols + fund_cols if c in trend_df_short.columns]]
                if not fund_df.empty:
                    st.dataframe(fund_df, use_container_width=True)
            
            main_df = trend_df_short[display_cols]
            st.dataframe(
                main_df.style.applymap(color_scores, subset=['Trend_Score', 'Reversion_Score']),
                use_container_width=True,
                column_config=column_config
            )
        
        st.header("📉 Mean Reversion Signals (Max Score: +4 / -4)") 
        col1, col2 = st.columns(2)
        
        with col1: 
            st.subheader("Overbought (Bearish)")
            rev_df_bear = tear_sheet_df[tear_sheet_df['Reversion_Score'] < 0].sort_values(by="Reversion_Score", na_position='last').head(25)
            
            with st.expander("📊 Metric Details (Scores & Values)", expanded=False):
                st.caption("Individual indicator scores paired with their actual values")
                detail_cols = ['Ticker', 'Close', 'Trend_Score', 'Reversion_Score']
                detail_cols.extend(['MA_Ribbon_Score', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_20'])
                detail_cols.extend(['50_SMA_Score', 'SMA_50', '200_SMA_Score', 'SMA_200'])
                detail_cols.extend(['MACD_Score', 'MACD', 'MACD_Signal', 'Cross_Score'])
                detail_cols.extend(['RSI_Score', 'RSI', 'Stoch_Score', 'Stoch_%K'])
                detail_cols.extend(['BBands_Score', 'Patterns_Score'])
                detail_cols = [c for c in detail_cols if c in rev_df_bear.columns]
                detail_df = rev_df_bear[detail_cols]
                # Ensure Ticker is first column and set as index temporarily for proper display
                if 'Ticker' in detail_df.columns:
                    detail_df = detail_df.set_index('Ticker').reset_index()
                st.dataframe(
                    detail_df.style.applymap(color_scores, subset=[c for c in score_cols if c in detail_df.columns]),
                    use_container_width=True,
                    height=400
                )
            
            with st.expander("💎 Fundamental Rankings", expanded=False):
                st.caption("Valuation percentile ranks (lower = cheaper)")
                fund_df = rev_df_bear[[c for c in base_cols + fund_cols if c in rev_df_bear.columns]]
                if not fund_df.empty:
                    st.dataframe(fund_df, use_container_width=True)
            
            main_df = rev_df_bear[display_cols]
            st.dataframe(
                main_df.style.applymap(color_scores, subset=['Trend_Score', 'Reversion_Score']),
                use_container_width=True,
                column_config=column_config
            )
            
        with col2: 
            st.subheader("Oversold (Bullish)")
            rev_df_bull = tear_sheet_df[tear_sheet_df['Reversion_Score'] > 0].sort_values(by="Reversion_Score", ascending=False, na_position='last').head(25)
            
            with st.expander("📊 Metric Details (Scores & Values)", expanded=False):
                st.caption("Individual indicator scores paired with their actual values")
                detail_cols = ['Ticker', 'Close', 'Trend_Score', 'Reversion_Score']
                detail_cols.extend(['MA_Ribbon_Score', 'SMA_5', 'SMA_10', 'SMA_15', 'SMA_20'])
                detail_cols.extend(['50_SMA_Score', 'SMA_50', '200_SMA_Score', 'SMA_200'])
                detail_cols.extend(['MACD_Score', 'MACD', 'MACD_Signal', 'Cross_Score'])
                detail_cols.extend(['RSI_Score', 'RSI', 'Stoch_Score', 'Stoch_%K'])
                detail_cols.extend(['BBands_Score', 'Patterns_Score'])
                detail_cols = [c for c in detail_cols if c in rev_df_bull.columns]
                detail_df = rev_df_bull[detail_cols]
                # Ensure Ticker is first column and set as index temporarily for proper display
                if 'Ticker' in detail_df.columns:
                    detail_df = detail_df.set_index('Ticker').reset_index()
                st.dataframe(
                    detail_df.style.applymap(color_scores, subset=[c for c in score_cols if c in detail_df.columns]),
                    use_container_width=True,
                    height=400
                )
            
            with st.expander("💎 Fundamental Rankings", expanded=False):
                st.caption("Valuation percentile ranks (lower = cheaper)")
                fund_df = rev_df_bull[[c for c in base_cols + fund_cols if c in rev_df_bull.columns]]
                if not fund_df.empty:
                    st.dataframe(fund_df, use_container_width=True)
            
            main_df = rev_df_bull[display_cols]
            st.dataframe(
                main_df.style.applymap(color_scores, subset=['Trend_Score', 'Reversion_Score']),
                use_container_width=True,
                column_config=column_config
            )
        st.header("💎 Fundamental Valuation Rankings (Long-Term)") 
        fund_rank_cols = [c for c in tear_sheet_df.columns if 'Long_Term_Rank' in c]
        if fund_rank_cols:
            # Include Ticker in the summary dataframe
            fund_summary_df = tear_sheet_df[['Ticker'] + fund_rank_cols].copy().dropna(subset=fund_rank_cols, how='all')
            if not fund_summary_df.empty:
                fund_summary_df['Overall_Rank'] = fund_summary_df[fund_rank_cols].mean(axis=1, skipna=True)
                
                # Color coding for ranks (lower is better/cheaper)
                def color_ranks(val):
                    if pd.isna(val) or val is None:
                        return ''
                    try:
                        num_val = float(val)
                        if num_val <= 25:
                            return 'background-color: rgba(16, 185, 129, 0.3)'  # Green - very cheap
                        elif num_val <= 50:
                            return 'background-color: rgba(16, 185, 129, 0.1)'  # Light green - cheap
                        elif num_val >= 75:
                            return 'background-color: rgba(239, 68, 68, 0.3)'  # Red - expensive
                        elif num_val >= 50:
                            return 'background-color: rgba(239, 68, 68, 0.1)'  # Light red - somewhat expensive
                    except (ValueError, TypeError):
                        pass
                    return ''
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Cheapest (Lowest Ranks)")
                    cheap_df = fund_summary_df.sort_values(by='Overall_Rank', na_position='last').dropna(subset=['Overall_Rank']).head(25)
                    # Ensure Ticker is first column
                    if 'Ticker' in cheap_df.columns:
                        column_order = ['Ticker'] + [c for c in cheap_df.columns if c != 'Ticker']
                        cheap_df = cheap_df[column_order]
                    # Format only numeric columns (not Ticker)
                    format_dict = {col: '{:.2f}' for col in cheap_df.columns if col != 'Ticker' and col in fund_rank_cols + ['Overall_Rank']}
                    st.dataframe(
                        cheap_df.style.applymap(color_ranks, subset=fund_rank_cols + ['Overall_Rank']).format(format_dict, na_rep='N/A'),
                        use_container_width=True
                    )
                with col2:
                    st.subheader("Most Expensive (Highest Ranks)")
                    exp_df = fund_summary_df.sort_values(by='Overall_Rank', ascending=False, na_position='last').dropna(subset=['Overall_Rank']).head(25)
                    # Ensure Ticker is first column
                    if 'Ticker' in exp_df.columns:
                        column_order = ['Ticker'] + [c for c in exp_df.columns if c != 'Ticker']
                        exp_df = exp_df[column_order]
                    # Format only numeric columns (not Ticker)
                    format_dict = {col: '{:.2f}' for col in exp_df.columns if col != 'Ticker' and col in fund_rank_cols + ['Overall_Rank']}
                    st.dataframe(
                        exp_df.style.applymap(color_ranks, subset=fund_rank_cols + ['Overall_Rank']).format(format_dict, na_rep='N/A'),
                        use_container_width=True
                    )
        st.header("🔍 On-Demand Ticker Lookup")
        all_tickers = list(tech_data.keys())
        selected_ticker = st.selectbox("Select or type any analyzed stock ticker:", options=all_tickers, index=all_tickers.index("SPY") if "SPY" in all_tickers else 0)
        if selected_ticker and selected_ticker in tech_data:
            st.plotly_chart(create_analysis_chart(selected_ticker, tech_data[selected_ticker]), use_container_width=True)
        else:
            st.warning("Please select a ticker to view its detailed technical analysis chart.")
elif page == "Fundamental Explorer":
    st.title("💎 Fundamental Valuation Explorer")
    ticker_list = list(fund_data.keys())
    if not ticker_list: 
        st.warning("No stocks with fundamental data were found."); st.stop()
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

elif page == "Model Explanation":
    st.title("Model Scoring Explanation")
    st.header("Trading Score Methodology")

    # --- FIX: Changed single quotes to triple quotes for multi-line string ---
    st.markdown("""
    This model uses two core scores: **Trend Score** and **Reversion Score**, both ranging from **+4 to -4**.
    
    A **Positive Score** indicates a bullish signal (Buy/Long).
    A **Negative Score** indicates a bearish signal (Sell/Short).
    """)

    st.subheader("📈 Trend Score Components (Max: +4 / -4)")
    # --- FIX: Changed single quotes to triple quotes for multi-line string ---
    st.markdown("""
    | Indicator | Score | Condition |
    | :--- | :--- | :--- |
    | **MA Ribbon (5/10/15/20)** | **+1** / **-1** | Perfect Bullish (`5>10>15>20`) or Bearish (`5<10<15<20`) stacking |
    | **Price vs 200-Day SMA** | **+1** / **-1** | Close is Above / Below 200-Day SMA |
    | **MACD** | **+1** / **-1** | MACD line Above / Below signal line |
    | **Golden/Death Cross (50/200)** | **+1** / **-1** | 50D crosses Above / Below 200D |
    """)

    st.subheader("📉 Reversion Score Components (Max: +4 / -4)")
    # --- FIX: Changed single quotes to triple quotes for multi-line string ---
    st.markdown("""
    | Indicator | Score | Condition |
    | :--- | :--- | :--- |
    | **RSI (14)** | **+2** / **+1** | RSI ≤ 20 (Extreme Oversold) or ≤ 30 (Oversold) |
    |  | **-1** / **-2** | RSI ≥ 70 (Overbought) or ≥ 80 (Extreme Overbought) |
    | **Candlestick Patterns** | **+1** / **-1** | Bullish (Hammer, Bull Engulfing, Morning Star) / Bearish (Bear Engulfing) |
    | **Stochastics (%K)** | **+1** / **-1** | %K < 20 (Oversold) / > 80 (Overbought) |
    | **Bollinger Bands (1σ, 20-day)**| **+1** / **-1** | Close vs Previous Day's Lower/Upper Band (length=20, std=1.0) |
    | **Signal Cluster (info)** | — | Conf = count of active signals this bar (0..4) |
    """)
    
    st.subheader("📊 Scoring System Details")
    st.markdown("""
    **Trend Score Calculation (Max: +4 / -4):**
    - **MA Ribbon Score**: +1 if 5 SMA > 10 SMA > 15 SMA > 20 SMA (bullish alignment), -1 if reversed (bearish)
    - **200 SMA Score**: +1 if Close > 200 SMA, -1 if Close < 200 SMA
    - **MACD Score**: +1 if MACD > Signal, -1 if MACD < Signal
    - **Cross Score**: +1 on Golden Cross (50 crosses above 200), -1 on Death Cross (50 crosses below 200)
    - **Note**: 50 SMA Score is calculated but not included in Trend Score (shown for reference)
    - **Trend Score** = MA_Ribbon + 200_SMA + MACD + Cross (capped at +4/-4)
    
    **Reversion Score Calculation (Max: +4 / -4):**
    - **RSI Score**: +2 if RSI ≤ 20 (extreme oversold), +1 if RSI ≤ 30 (oversold), -1 if RSI ≥ 70 (overbought), -2 if RSI ≥ 80 (extreme overbought)
    - **Stochastic Score**: +1 if %K < 20 (oversold), -1 if %K > 80 (overbought)
    - **Bollinger Bands Score**: +1 if Close < Previous Day's Lower Band, -1 if Close > Previous Day's Upper Band (20-day, 1σ)
    - **Patterns Score**: +1 if bullish pattern detected (Hammer, Bull Engulfing, Morning Star), -1 if bearish pattern (Bear Engulfing)
    - **Reversion Score** = Sum of all reversion component scores (capped at +4/-4)
    
    **Moving Averages:**
    - **SMA_5**: 5-day Simple Moving Average
    - **SMA_10**: 10-day Simple Moving Average  
    - **SMA_15**: 15-day Simple Moving Average
    - **SMA_20**: 20-day Simple Moving Average
    - **SMA_50**: 50-day Simple Moving Average (used for Golden/Death Cross)
    - **SMA_200**: 200-day Simple Moving Average (long-term trend indicator)
    
    **MACD Calculation:**
    - **MACD Line**: 12-period EMA - 26-period EMA
    - **Signal Line**: 9-period EMA of MACD Line
    - Signal is bullish when MACD > Signal, bearish when MACD < Signal
    """)
    
    st.subheader("🕯️ Candlestick Pattern Details")
    st.markdown("""
    **Bullish Patterns (+1 score when detected):**
    
    - **Hammer** (Single Candle):
      - **Visual**: Small body at the top, long lower shadow/wick, tiny or no upper shadow
      - **Calculation**: 
        - Body size (|close - open|) < 30% of the total candle range (high - low)
        - Lower wick > 50% of the total range
        - Upper wick < 20% of the total range
      - **Interpretation**: After a downtrend, shows sellers pushed price down but buyers fought back. 
        Indicates potential bullish reversal.
    
    - **Bullish Engulfing** (Two Candles):
      - **Visual**: First candle is bearish (red), second candle is larger bullish (green) that completely 
        covers/swallows the first candle's body
      - **Calculation**:
        - Previous candle: close < open (bearish/red)
        - Current candle: close > open (bullish/green)
        - Current open ≤ previous close AND current close ≥ previous open (engulfs the body)
      - **Interpretation**: Strong buying pressure overwhelms selling pressure. Signals momentum shift to bullish.
    
    - **Morning Star** (Three Candles):
      - **Visual**: First candle is bearish, second is a small-bodied "star" with gaps, third is bullish
      - **Calculation**:
        - Candle 1 (previous day): close < open (bearish)
        - Candle 2 (current day): Small body (< 25% of range) with close below previous close
        - Candle 3 (next day): close > open AND close > midpoint of candle 1's body
      - **Interpretation**: Shows exhaustion of selling, indecision (star), then strong buying. 
        Classic reversal pattern from bearish to bullish.
    
    **Bearish Patterns (-1 score when detected):**
    
    - **Bearish Engulfing** (Two Candles):
      - **Visual**: First candle is bullish (green), second candle is larger bearish (red) that completely 
        covers/swallows the first candle's body
      - **Calculation**:
        - Previous candle: close > open (bullish/green)
        - Current candle: close < open (bearish/red)
        - Current open ≥ previous close AND current close ≤ previous open (engulfs the body)
      - **Interpretation**: Strong selling pressure overwhelms buying pressure. Signals momentum shift to bearish.
    
    **How They Work:**
    - Patterns are calculated using OHLC (Open, High, Low, Close) price data
    - Each pattern is detected independently and assigned a binary value (1 = detected, 0 = not detected)
    - Multiple patterns can occur on the same day (though rare)
    - The final **Patterns_Score** is +1 if any bullish pattern is detected, -1 if any bearish pattern is detected, 
      or 0 if no patterns are detected
    - This score contributes to the **Reversion_Score** which ranges from -4 to +4
    - Patterns are most reliable when they occur after strong trends (oversold for bullish, overbought for bearish)
    """)

elif page == "Backtest":
    st.title("🧪 Strategy Backtest")
    st.markdown("Run backtests on historical data to evaluate strategy performance.")
    
    # Fast check - use cached ticker list from session state to avoid slow dict access
    if 'available_tickers_cache' in st.session_state and st.session_state.available_tickers_cache:
        available_tickers = st.session_state.available_tickers_cache
    else:
        # Cache ticker list for fast access on tab switches
        try:
            if tech_data and isinstance(tech_data, dict) and len(tech_data) > 0:
                available_tickers = list(tech_data.keys())
                st.session_state.available_tickers_cache = available_tickers
            else:
                available_tickers = []
        except Exception:
            available_tickers = []
    
    if not available_tickers:
        st.warning("⚠️ No analysis data available. Please go to the Technical Dashboard tab and run an analysis first.")
        st.info("💡 **Tip:** Select 'Analyze only watchlist' for faster analysis, or uncheck it for full S&P 500 analysis.")
        st.stop()
    
    # Show data availability - only calculate if needed (lazy)
    if 'data_availability_shown' not in st.session_state or not st.session_state.data_availability_shown:
        try:
            if tech_data and isinstance(tech_data, dict) and len(tech_data) > 0:
                sample_ticker = available_tickers[0]
                sample_data = tech_data.get(sample_ticker)
                if sample_data is not None and not sample_data.empty and hasattr(sample_data, 'index'):
                    available_start = sample_data.index.min()
                    st.info(f"📅 **Data Availability:** Historical data available from {available_start.strftime('%Y-%m-%d')} onwards. Backtests before this date will use available data only.")
                    st.session_state.data_availability_shown = True
        except Exception:
            pass  # Skip if we can't get the date
    
    # Collapsible Backtest Parameters Section - Improved Layout
    with st.expander("📋 Backtest Parameters", expanded=True):
        # Top row: Stock Selection and Portfolio Setup
        top_col1, top_col2 = st.columns([2, 1])
        
        with top_col1:
            st.markdown("**1. Stock Selection**")
            ticker_input_method = st.radio("Selection Method", ["Single Ticker", "Multiple Tickers"], index=0, horizontal=True, label_visibility="collapsed")
            
            if ticker_input_method == "Single Ticker":
                selected_tickers = [st.selectbox("Select Ticker", available_tickers, index=0 if "SPY" in available_tickers else 0, label_visibility="collapsed")]
            else:
                ticker_input = st.text_input("Enter Tickers (comma-separated)", value="SPY, QQQ, AAPL", help="Example: SPY, QQQ, AAPL, MSFT", label_visibility="collapsed")
                selected_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
                invalid_tickers = [t for t in selected_tickers if t not in available_tickers]
                if invalid_tickers:
                    st.warning(f"⚠️ Tickers not available: {', '.join(invalid_tickers)}")
                selected_tickers = [t for t in selected_tickers if t in available_tickers]
                if not selected_tickers:
                    st.error("No valid tickers selected.")
                    st.stop()
            
            backtest_start_date = st.date_input("**Backtest Start Date**", value=date(2010, 1, 1), min_value=date(1990, 1, 1), max_value=date.today())
        
        with top_col2:
            st.markdown("**2. Portfolio Setup**")
            initial_portfolio = st.number_input("Initial Value ($)", min_value=1000, value=10000, step=1000, format="%d", help="Starting portfolio value")
            trade_direction = st.radio("Trade Direction", ["calls", "puts", "both"], index=0, horizontal=True)
        
        st.markdown("---")
        
        # Middle section: Strategy Configuration in 3 columns
        mid_col1, mid_col2, mid_col3 = st.columns(3)
        
        with mid_col1:
            st.markdown("**3. Signal Type**")
            signal_type = st.radio(
                "Which signals to trade:",
                ["Reversion Only", "Trend Only", "Both (no confirmation)", "Both (with confirmation)"],
                index=0,
                help="Reversion = Oversold/Overbought. Trend = Follow trend. Both = Run independently or with confirmation.",
                label_visibility="collapsed"
            )
            signal_type_param = {
                'Reversion Only': 'reversion', 
                'Trend Only': 'trend', 
                'Both (no confirmation)': 'both_no_confirmation',
                'Both (with confirmation)': 'both'
            }[signal_type]
            
            st.markdown("**4. Entry Filters**")
            if signal_type in ['Reversion Only', 'Both (no confirmation)', 'Both (with confirmation)']:
                entry_reversion_min = st.slider("Min Reversion Score", 1, 4, 1)
            else:
                entry_reversion_min = 1
            
            if signal_type == 'Trend Only':
                trend_entry_min = st.slider("Min Trend Score", 1, 4, 1)
            elif signal_type in ['Both (no confirmation)', 'Both (with confirmation)']:
                trend_entry_min = st.slider("Min Trend Score", 1, 4, 1)
            else:
                trend_entry_min = 1
            
            if signal_type == "Reversion Only":
                use_trend_filter = st.checkbox("Use Trend as Confirmation", value=False)
                if use_trend_filter:
                    trend_min = st.slider("Trend Min", -4, 4, 1)
                    trend_max = st.slider("Trend Max", -4, 4, 4)
                else:
                    trend_min = -4
                    trend_max = 4
            elif signal_type in ['Both (no confirmation)', 'Both (with confirmation)']:
                # For "Both (with confirmation)" mode, add confirmation threshold slider
                if signal_type == 'Both (with confirmation)':
                    confirmation_threshold = st.slider("Confirmation Threshold", 0, 4, 0, 
                        help="How much trend can deviate from reversion. 0 = strict (must match direction), 2 = trend can be 2 away from reversion")
                else:
                    confirmation_threshold = 0  # Not used in no-confirmation mode
                # For "Both" modes, set trend range to allow both directions (-4 to 4)
                trend_min = -4
                trend_max = 4
                use_trend_filter = False  # Not used in "Both" modes, confirmation handles it differently
            else:
                use_trend_filter = False
                trend_min = -4
                trend_max = 4
                confirmation_threshold = 0
        
        with mid_col2:
            st.markdown("**5. Exit Parameters**")
            trailing_stop_val = st.slider("Trailing Stop %", 0.05, 0.30, 0.15, 0.05)
            trailing_stop = trailing_stop_val
            profit_target_val = st.slider("Profit Target %", 0.10, 2.00, 0.50, 0.05)
            profit_target = profit_target_val
            stop_loss_val = st.slider("Stop Loss %", -0.90, -0.10, -0.60, 0.05)
            stop_loss = stop_loss_val
            expiration_days = st.selectbox("Expiration (days)", [30, 60, 90, 120, 180, 240, 365], index=1)
        
        with mid_col3:
            st.markdown("**6. Position Management**")
            initial_position_pct_val = st.slider("Initial Entry %", 0.01, 0.25, 0.10, 0.01, help="Max % of portfolio per initial entry")
            initial_position_pct = initial_position_pct_val
            max_position_pct_val = st.slider("Max Position %", 0.05, 0.50, 0.20, 0.01, help="Max % including double downs")
            max_position_pct = max_position_pct_val
            
            st.markdown("**Position Limits**")
            enable_max_contracts = st.checkbox("Limit Max Contracts", value=True, help="Enable maximum contract quantity limit")
            if enable_max_contracts:
                max_contracts = st.number_input("Max Contracts", min_value=1, max_value=10000, value=1000, step=100, help="Maximum number of option contracts per position (including double downs)")
            else:
                max_contracts = None  # No limit
            
            enable_min_price = st.checkbox("Limit Min Option Price", value=True, help="Enable minimum option price filter")
            if enable_min_price:
                min_option_price = st.number_input("Min Option Price ($)", min_value=0.01, max_value=10.00, value=1.00, step=0.10, help="Minimum option price per share (per lot = $100 minimum)")
            else:
                min_option_price = None  # No minimum
            
            strike_increment = st.select_slider("Strike Increment", options=[0.5, 1.0, 2.5, 5.0, 10.0], value=1.0)
            risk_free_rate_val = st.slider("Risk-Free Rate %", 0.0, 10.0, 3.0, 0.1)
            risk_free_rate = risk_free_rate_val / 100.0
            
            st.markdown("**7. Double Down**")
            enable_dd = st.checkbox("Enable Double Down", value=True)
            if enable_dd:
                dd_level_val = st.slider("DD Level %", -0.40, -0.10, -0.25, 0.05)
                dd_level = dd_level_val
                dd_multiplier = st.slider("DD Multiplier", 2.0, 4.0, 2.0, 1.0)
            else:
                dd_level = None
                dd_multiplier = None
    
    st.markdown("---")
    
    # ===== OPTIMIZER SECTION (Optional) =====
    show_optimizer = st.checkbox("🤖 Show Strategy Optimizer", value=False, 
                                 help="Enable this to test hundreds of parameter combinations to find the optimal strategy. Uncheck to hide optimizer and just run backtests.")

    if show_optimizer:
        st.header("🤖 Strategy Optimizer")
        
        # Optimization Target Selection
        st.subheader("1. Optimization Target")
        optimizer_target_metric = st.radio(
            "What metric should the optimizer optimize for?",
            ["Risk-Adjusted Return (CAGR / Max Drawdown)", "Annualized Return (CAGR)", "Max Drawdown (Minimize)", "Win Rate", "Total Return"],
            index=0,
            key='opt_target_metric',
            help="Select which metric to maximize (or minimize for Max Drawdown). The optimizer will find the parameter combination that performs best on this metric."
        )
        st.markdown("Automatically test hundreds of parameter combinations to find the optimal strategy.")
        
        opt_col1, opt_col2 = st.columns(2)
        
        with opt_col1:
            st.subheader("1. Core Setup")
            optimizer_portfolio_value = st.number_input("Initial Portfolio Value ($)", min_value=1000, value=10000, step=1000, key='opt_portfolio')
            optimizer_tickers_str = st.text_area("Tickers (comma separated)", value=",".join(selected_tickers) if 'selected_tickers' in locals() else "SPY", key='opt_tickers').upper()
            optimizer_start_date = st.date_input("Backtest Start Date", value=backtest_start_date if 'backtest_start_date' in locals() else date(2010, 1, 1), min_value=date(1990, 1, 1), max_value=date.today(), key='opt_start_date')
            
            st.subheader("2. Model Selection")
            optimizer_model_selections = {
                'score_ma_ribbon': st.checkbox("MA Ribbon", value=True, key='opt_chk_ma_ribbon'),
                'score_50sma': st.checkbox("50 SMA", value=True, key='opt_chk_50sma'),
                'score_200sma': st.checkbox("200 SMA", value=True, key='opt_chk_200sma'),
                'score_macd': st.checkbox("MACD", value=True, key='opt_chk_macd'),
                'score_rsi': st.checkbox("RSI", value=True, key='opt_chk_rsi'),
                'score_stoch': st.checkbox("Stochastics", value=True, key='opt_chk_stoch'),
                'score_bbands': st.checkbox("Bollinger Bands", value=True, key='opt_chk_bbands'),
            }
            
            st.subheader("3. Strategy Type & Filters")
            st.caption("💡 **Tip:** For ~5,000 combinations, select 1 Signal Type option")
            optimizer_trade_direction = st.radio("Optimize Trade Direction:", ['calls', 'puts', 'both'], index=0, key='opt_radio_dir')
            optimizer_signal_types = st.multiselect("Signal Type to Optimize", ['reversion', 'trend', 'both_no_confirmation', 'both'], default=['reversion', 'trend'], key='opt_signal_types')
            if optimizer_signal_types:
                st.caption(f"  → Currently: {len(optimizer_signal_types)} option{'s' if len(optimizer_signal_types) > 1 else ''} selected ({', '.join(optimizer_signal_types)})")
            else:
                st.caption(f"  → Currently: 1 option (default: reversion)")
            
            st.subheader("4. Entry Filters (Optimization Range)")
            st.caption("💡 **Tip:** For ~5,000 combinations, use 2 options for Reversion, 2-3 for Trend")
            entry_rev_min_opt, entry_rev_max_opt = st.slider("Reversion Score (Abs Value)", 1, 4, (1, 3), key='opt_slider_ers')
            entry_rev_scores = list(range(entry_rev_min_opt, entry_rev_max_opt + 1))
            st.caption(f"  → Currently: {len(entry_rev_scores)} options ({entry_rev_scores})")
            entry_trend_min_opt, entry_trend_max_opt = st.slider("Trend Score (Abs Value)", 1, 4, (1, 3), key='opt_slider_tes')
            entry_trend_scores = list(range(entry_trend_min_opt, entry_trend_max_opt + 1))
            st.caption(f"  → Currently: {len(entry_trend_scores)} options ({entry_trend_scores})")
            
            if st.checkbox("Include Trend Score as Confirmation Filter?", value=False, key='opt_chk_trend_filter'):
                min_trend_val, max_trend_val = st.slider("Trend Confirmation Range", -4, 4, (1, 4), key='opt_slider_ts')
                trend_scores_grid = list(range(min_trend_val, max_trend_val + 1))
            else:
                trend_scores_grid = ["NoFilter"]
        
        with opt_col2:
            st.subheader("5. Exit Parameters (Optimization Range)")
            st.caption("💡 **Tip:** For ~5,000 combinations, select 2 options for Trailing Stop, 2 for Expiration, 1-2 for Profit Target, 2 for Stop Loss")
            optimizer_trailing_stops = st.multiselect("Trailing Stop %", [0.05, 0.10, 0.15, 0.20, 0.25], default=[0.10, 0.15, 0.20], key='opt_trailing')
            if not optimizer_trailing_stops:
                st.warning("Select at least one Trailing Stop %")
            else:
                st.caption(f"  → Currently: {len(optimizer_trailing_stops)} options selected")
            
            optimizer_expiration_days = st.multiselect("Expiration Days", [30, 60, 90, 120, 180, 240, 365], default=[60, 90, 120], key='opt_expiration')
            if not optimizer_expiration_days:
                st.warning("Select at least one Expiration Day value")
            else:
                st.caption(f"  → Currently: {len(optimizer_expiration_days)} options selected")
            
            optimizer_profit_targets = st.multiselect("Profit Target %", [10, 25, 50, 75, 100, 150, 200], default=[50, 100], key='opt_profit_target')
            if not optimizer_profit_targets:
                optimizer_profit_targets = [50]
            st.caption(f"  → Currently: {len(optimizer_profit_targets)} options selected")
            
            optimizer_stop_losses = st.multiselect("Stop Loss %", [-90, -80, -70, -60, -50, -40, -30, -20, -10], default=[-60, -50, -40], key='opt_stop_loss')
            st.caption(f"  → Currently: {len(optimizer_stop_losses)} options selected")
            
            st.subheader("6. Position Management (Optimization Range)")
            st.caption("💡 **Tip:** For ~5,000 combinations, select 2 options for Initial/Max Position, 1 for Strike Increment")
            optimizer_init_position_pct = st.multiselect("Max % Portfolio for Initial Entry", [0.01, 0.05, 0.10, 0.15, 0.20, 0.25], default=[0.05, 0.10, 0.15], key='opt_init_pos')
            if not optimizer_init_position_pct:
                optimizer_init_position_pct = [0.10]
            st.caption(f"  → Currently: {len(optimizer_init_position_pct)} options selected")
            
            optimizer_max_position_pct = st.multiselect("Max % Portfolio (Incl. DD)", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50], default=[0.15, 0.20, 0.25], key='opt_max_pos')
            if not optimizer_max_position_pct:
                optimizer_max_position_pct = [0.20]
            st.caption(f"  → Currently: {len(optimizer_max_position_pct)} options selected")
            
            optimizer_strike_increments = st.multiselect("Strike Price Increment", [0.5, 1.0, 2.5, 5.0, 10.0], default=[0.5, 1.0, 2.5], key='opt_strike_inc')
            st.caption(f"  → Currently: {len(optimizer_strike_increments)} options selected")
            
            st.subheader("7. Double Down (Optimization)")
            st.caption("💡 **Tip:** For ~5,000 combinations, select 2 for Enabled, 1-2 options for Level/Multiplier")
            if st.checkbox("Test Both Enabled/Disabled?", value=True, key='opt_chk_dd_both'):
                optimizer_dd_enabled = [True, False]
                st.caption(f"  → Currently: 2 options (True, False)")
            else:
                optimizer_dd_enabled = [st.checkbox("Enable Doubling Down?", value=True, key='opt_chk_dd')]
                st.caption(f"  → Currently: 1 option")
            
            optimizer_dd_levels = st.multiselect("Double Down Level %", [-40, -35, -30, -25, -20, -15, -10], default=[-30, -25, -20], key='opt_dd_level')
            if not optimizer_dd_levels:
                optimizer_dd_levels = [-25]
            st.caption(f"  → Currently: {len(optimizer_dd_levels)} options selected")
            
            optimizer_dd_multipliers = st.multiselect("Double Down Multiplier", [2.0, 2.5, 3.0, 3.5, 4.0], default=[2.0, 2.5, 3.0], key='opt_dd_mult')
            if not optimizer_dd_multipliers:
                optimizer_dd_multipliers = [2.0]
            st.caption(f"  → Currently: {len(optimizer_dd_multipliers)} options selected")
            
            optimizer_risk_free_rate = st.slider("Risk-Free Rate %", 0.0, 10.0, 3.0, 0.1, format='%.1f%%', key='opt_rf_rate') / 100.0
        
        # Calculate estimated combinations
        import math
        estimated_combinations = (
            len(optimizer_signal_types) if optimizer_signal_types else 1
        ) * 1 * (  # trade_direction is single choice
            len(entry_rev_scores)
        ) * (
            len(entry_trend_scores)
        ) * (
            len(optimizer_trailing_stops) if optimizer_trailing_stops else 1
        ) * (
            len(optimizer_expiration_days) if optimizer_expiration_days else 1
        ) * (
            len(optimizer_profit_targets) if optimizer_profit_targets else 1
        ) * (
            len(optimizer_stop_losses) if optimizer_stop_losses else 1
        ) * (
            len(optimizer_init_position_pct) if optimizer_init_position_pct else 1
        ) * (
            len(optimizer_max_position_pct) if optimizer_max_position_pct else 1
        ) * (
            len(optimizer_strike_increments) if optimizer_strike_increments else 1
        ) * (
            len(optimizer_dd_enabled) if isinstance(optimizer_dd_enabled, list) else 1
        ) * (
            len(optimizer_dd_levels) if optimizer_dd_levels else 1
        ) * (
            len(optimizer_dd_multipliers) if optimizer_dd_multipliers else 1
        )
        
        # Add trend filter if enabled
        if trend_scores_grid != ["NoFilter"]:
            estimated_combinations *= len(trend_scores_grid)
        
        # Display combination estimate
        st.markdown("---")
        if estimated_combinations > 5000:
            st.warning(f"⚠️ **Estimated Combinations: {estimated_combinations:,}** (per stock)\n\n"
                      f"This is above the recommended ~5,000 limit. Consider reducing some parameter options:\n\n"
                      f"- **For ~5,000 combinations**, try limiting selections to:\n"
                      f"  • Signal Type: 1 option\n"
                      f"  • Entry Reversion: 2 options\n"
                      f"  • Entry Trend: 2-3 options\n"
                      f"  • Trailing Stop: 2 options\n"
                      f"  • Expiration: 2 options\n"
                      f"  • Profit Target: 1-2 options\n"
                      f"  • Stop Loss: 2 options\n"
                      f"  • Initial Position: 2 options\n"
                      f"  • Max Position: 2 options\n"
                      f"  • Strike Increment: 1 option\n"
                      f"  • Double Down Enabled: 2 options\n"
                      f"  • DD Level: 1-2 options\n"
                      f"  • DD Multiplier: 1-2 options")
        elif estimated_combinations > 10000:
            st.error(f"❌ **Estimated Combinations: {estimated_combinations:,}** (per stock)\n\n"
                    f"This is very high and may take a very long time. Please reduce the number of options selected.")
        else:
            st.info(f"ℹ️ **Estimated Combinations: {estimated_combinations:,}** (per stock)\n\n"
                   f"Total backtests will be: **{estimated_combinations * len([t.strip() for t in optimizer_tickers_str.replace(',', '\n').split('\n') if t.strip()]):,}** "
                   f"(combinations × number of tickers)")
        
        run_optimizer = st.button("🚀 Run Optimizer", type="primary", use_container_width=True, key='run_optimizer_btn')
        
        # Run optimizer if button clicked
        if run_optimizer:
            import itertools
            import os
            DATA_DIR = "local_historical_data"
            
            optimizer_tickers = [t.strip() for t in optimizer_tickers_str.replace(',', '\n').split('\n') if t.strip()]
            if not optimizer_tickers:
                st.error("Please enter at least one ticker.")
            elif not optimizer_trailing_stops or not optimizer_expiration_days:
                st.warning("Please select values for Trailing Stop and Expiration Days.")
            else:
                # Build parameter grid
                optimizer_param_grid = {
                    "signal_type": optimizer_signal_types if optimizer_signal_types else ['reversion'],
                    "trade_direction": [optimizer_trade_direction] if optimizer_trade_direction else ['calls'],
                    "entry_reversion_score": entry_rev_scores,
                    "entry_trend_score": entry_trend_scores,
                    "trailing_stop_pct": optimizer_trailing_stops,
                    "time_to_expiration_days": optimizer_expiration_days,
                    "profit_target_pct": [v/100.0 for v in optimizer_profit_targets],
                    "stop_loss_pct": [v/100.0 for v in optimizer_stop_losses],
                    "initial_position_size_pct": optimizer_init_position_pct,
                    "max_position_size_pct_all_dd": optimizer_max_position_pct,
                    "strike_increment": optimizer_strike_increments,
                    "enable_dd": optimizer_dd_enabled,
                    "dd_level": [v/100.0 for v in optimizer_dd_levels],
                    "dd_multiplier": optimizer_dd_multipliers,
                }
                
                if trend_scores_grid != ["NoFilter"]:
                    optimizer_param_grid["min_trend_score"] = trend_scores_grid
                    optimizer_param_grid["max_trend_score"] = trend_scores_grid
                
                optimizer_common_params = {
                    "strike_otm_pct": 1.05,
                    "risk_free_rate": optimizer_risk_free_rate,
                    "volatility_lookback": 21,
                    "model_selections": optimizer_model_selections,
                }
                
                param_names = list(optimizer_param_grid.keys())
                param_values = list(optimizer_param_grid.values())
                all_combinations = list(itertools.product(*param_values))
                total_tests = len(optimizer_tickers) * len(all_combinations)
                
                st.info("Preparing to run %d combinations for each of %d tickers. Total backtests: %d. This may take a long time." % (len(all_combinations), len(optimizer_tickers), total_tests))
                progress_bar = st.progress(0)
                
                # Load SPY data
                spy_file = os.path.join(DATA_DIR, "SPY_daily.csv")
                spy_data = pd.DataFrame()
                if os.path.exists(spy_file):
                    try:
                        spy_data = pd.read_csv(spy_file, index_col='date', parse_dates=True)
                        if spy_data is not None and isinstance(spy_data, pd.DataFrame) and not spy_data.empty:
                            optimizer_start_timestamp = pd.to_datetime(optimizer_start_date)
                            spy_data = spy_data[spy_data.index >= optimizer_start_timestamp].copy()
                            if spy_data is None or spy_data.empty:
                                spy_data = pd.DataFrame()  # Reset to empty if filter removed all data
                    except Exception as e:
                        safe_print(f"Error loading SPY for optimizer: {e}")
                        spy_data = pd.DataFrame()
                
                best_results_per_stock = []
                overall_min_date = pd.Timestamp.max
                overall_max_date = pd.Timestamp.min
                
                for i, ticker in enumerate(optimizer_tickers):
                    ticker_file = os.path.join(DATA_DIR, "%s_daily.csv" % ticker)
                    if not os.path.exists(ticker_file):
                        st.warning(f"Data for {ticker} not found. Skipping.")
                        continue
                    
                    raw_data = pd.read_csv(ticker_file, index_col='date', parse_dates=True)
                    optimizer_start_timestamp = pd.to_datetime(optimizer_start_date)
                    raw_data = raw_data[raw_data.index >= optimizer_start_timestamp]
                    
                    analyzer = FinancialAnalyzer()
                    analyzed_data = analyzer.run_full_analysis(raw_data, optimizer_common_params['volatility_lookback'], optimizer_common_params['model_selections'], spy_data=spy_data if not spy_data.empty else None)
                    
                    if analyzed_data is None or analyzed_data.empty:
                        st.warning(f"Not enough data for {ticker}. Skipping.")
                        continue
                    
                    overall_min_date = min(overall_min_date, analyzed_data.index.min())
                    overall_max_date = max(overall_max_date, analyzed_data.index.max())
                    
                    results_for_this_stock = []
                    for j, combo in enumerate(all_combinations):
                        progress_bar.progress((i * len(all_combinations) + j + 1) / total_tests, f"Testing {ticker}: Combo {j+1}/{len(all_combinations)}")
                        
                        context_params = optimizer_common_params.copy()
                        context_params.update(dict(zip(param_names, combo)))
                        
                        # Handle double down
                        if context_params.get('enable_dd', False):
                            dd_level = context_params.get('dd_level', -0.25)
                            dd_multiplier = context_params.get('dd_multiplier', 2.0)
                            context_params['double_down_levels'] = {dd_level: dd_multiplier}
                        else:
                            context_params['double_down_levels'] = {}
                        
                        context_params.pop('enable_dd', None)
                        context_params.pop('dd_level', None)
                        context_params.pop('dd_multiplier', None)
                        
                        # Set entry conditions (same logic as optimizer)
                        signal_type_param_opt = context_params.get('signal_type', 'reversion')
                        trade_dir_opt = context_params.get('trade_direction', 'calls')
                        
                        if signal_type_param_opt in ['reversion', 'both', 'both_no_confirmation']:
                            entry_rev_min = context_params.get('entry_reversion_score', 1)
                            if trade_dir_opt == 'calls':
                                context_params['min_reversion_score'] = entry_rev_min
                                context_params['max_reversion_score'] = 4
                            elif trade_dir_opt == 'puts':
                                context_params['min_reversion_score'] = -4
                                context_params['max_reversion_score'] = -entry_rev_min
                            else:
                                context_params['min_reversion_score'] = -4
                                context_params['max_reversion_score'] = 4
                                context_params['entry_reversion_threshold'] = entry_rev_min
                        
                        if signal_type_param_opt in ['trend', 'both', 'both_no_confirmation']:
                            entry_trend_min = context_params.get('entry_trend_score', 1)
                            if trade_dir_opt == 'calls':
                                context_params['min_trend_score'] = entry_trend_min
                                context_params['max_trend_score'] = 4
                            elif trade_dir_opt == 'puts':
                                context_params['min_trend_score'] = -4
                                context_params['max_trend_score'] = -entry_trend_min
                            else:
                                context_params['min_trend_score'] = -4
                                context_params['max_trend_score'] = 4
                                context_params['entry_trend_threshold'] = entry_trend_min
                        
                        context_params.pop('entry_reversion_score', None)
                        context_params.pop('entry_trend_score', None)
                        
                        if "min_trend_score" in context_params and context_params["min_trend_score"] != "NoFilter" and signal_type_param_opt == 'reversion':
                            pass
                        elif "min_trend_score" not in context_params or context_params.get("min_trend_score") == "NoFilter":
                            if signal_type_param_opt in ['reversion', 'both']:
                                context_params['min_trend_score'] = -4
                                context_params['max_trend_score'] = 4
                        
                        result = run_portfolio_backtest(analyzed_data, context_params, optimizer_portfolio_value)
                        metrics = result['metrics']
                        
                        cagr = metrics['Annualized Return (CAGR)']
                        drawdown = metrics['Max Drawdown']
                        win_rate = metrics['Win Rate']
                        total_return = metrics['Total Return']
                        risk_adjusted_return = (cagr / drawdown) if drawdown > 0.01 else (cagr * 1000) if cagr > 0 else 0
                        
                        # Calculate optimization score based on selected target
                        if optimizer_target_metric == "Risk-Adjusted Return (CAGR / Max Drawdown)":
                            opt_score = risk_adjusted_return
                        elif optimizer_target_metric == "Annualized Return (CAGR)":
                            opt_score = cagr
                        elif optimizer_target_metric == "Max Drawdown (Minimize)":
                            opt_score = -drawdown  # Negative because we want to MINIMIZE drawdown (maximize -drawdown)
                        elif optimizer_target_metric == "Win Rate":
                            opt_score = win_rate
                        elif optimizer_target_metric == "Total Return":
                            opt_score = total_return
                        else:
                            opt_score = risk_adjusted_return  # Default
                        
                        results_for_this_stock.append({
                            "params": context_params,
                            "metrics": metrics,
                            "risk_adjusted_return": risk_adjusted_return,
                            "optimization_score": opt_score,
                            "trades": result.get('Trades', result.get('trades', []))
                        })
                    
                    if results_for_this_stock:
                        # Find best result based on optimization target metric
                        best_result = max(results_for_this_stock, key=lambda x: x['optimization_score'])
                        best_result['ticker'] = ticker
                        best_results_per_stock.append(best_result)
                        if 'all_optimizer_results' not in st.session_state:
                            st.session_state.all_optimizer_results = {}
                        st.session_state.all_optimizer_results[ticker] = results_for_this_stock
                
                if best_results_per_stock:
                    st.session_state.optimizer_results = best_results_per_stock
                    st.session_state.analyzed_data_range_display = f"{overall_min_date.strftime('%Y-%m-%d')} to {overall_max_date.strftime('%Y-%m-%d')}" if overall_min_date != pd.Timestamp.max else "N/A"
                    st.session_state.last_optimization_target = optimizer_target_metric  # Store the target metric
                    progress_bar.empty()
                    st.success(f"✅ Optimization complete! Optimized for: **{optimizer_target_metric}**")
                else:
                    st.warning("⚠️ No optimization results were generated.")
                    progress_bar.empty()
    
    st.markdown("---")
    
    # Explanation of entry signals
    with st.expander("ℹ️ How Entry Signals Work", expanded=False):
        st.markdown("""
        **Signal Type Options:**
        
        1. **Reversion Only**: Trade on oversold/overbought signals
           - Positive Reversion Score → CALLS (Buy/Long) - Stock is oversold, expecting bounce
           - Negative Reversion Score → PUTS (Sell/Short) - Stock is overbought, expecting decline
           - Optional: Use Trend Score as confirmation filter
        
        2. **Trend Only**: Trade by following the trend
           - Positive Trend Score → CALLS (Buy/Long) - Follow bullish trend
           - Negative Trend Score → PUTS (Sell/Short) - Follow bearish trend
           - Requires: Trend Score meets entry threshold
        
        3. **Both (no confirmation)**: Run both strategies independently
           - Runs separate backtests for trend and reversion signals
           - Each strategy gets half the portfolio allocation
           - Results are combined for comparison
           - No confirmation required between signal types
        
        4. **Both (with confirmation)**: Trade on either signal type, but require confirmation
           - Reversion signals need trend confirmation (trend filter must pass)
           - Trend signals need reversion confirmation (reversion must be in valid range)
           - More conservative, only trades when both signals align
        
        **Entry Requirements:**
        
        **For Reversion Signals:**
        - Reversion Score must meet threshold:
          - **For Calls**: `Reversion_Score >= min_reversion_score` (oversold)
          - **For Puts**: `Reversion_Score <= -min_reversion_score` (overbought)
        - If trend filter enabled: `min_trend_score <= Trend_Score <= max_trend_score`
        
        **For Trend Signals:**
        - Trend Score must meet threshold:
          - **For Calls**: `Trend_Score >= min_trend_score` (bullish trend)
          - **For Puts**: `Trend_Score <= -min_trend_score` (bearish trend)
        - If using "Both" mode: Reversion Score must be in valid range
        
        **Example: Meta (Trend=-2, Reversion=+1)**
        - **Reversion Only mode**: Would trigger CALL if `Min Reversion Score ≤ 1` (oversold signal)
        - **Trend Only mode**: Would NOT trigger (Trend=-2 is negative, but if puts allowed, might trigger PUT)
        - **Both mode**: Reversion signal needs trend confirmation; if trend filter blocks -2, no trade
        
        **Summary:** 
        - **Reversion Score** = Mean reversion signals (oversold/overbought)
        - **Trend Score** = Momentum/trend following signals (bullish/bearish trend)
        - You can trade on either, both, or use one to confirm the other
        """)
    
    # Model selections (enable all by default for dashboard backtest)
    model_selections = {
        'score_ma_ribbon': True,
        'score_50sma': True,
        'score_200sma': True,
        'score_macd': True,
        'score_rsi': True,
        'score_stoch': True,
        'score_bbands': True,
    }
    
    # Metrics selection dropdown
    st.subheader("📊 Select Metrics to Display")
    available_metrics = [
        'Final Portfolio Value',
        'Total Return', 
        'Annualized Return (CAGR)',
        'Max Drawdown',
        'Win Rate',
        'Total Trades',
        'Portfolio History Chart',
        'Trade Details Table',
        'Performance by Market Regime'
    ]
    selected_metrics = st.multiselect(
        "Choose which metrics to show in results:",
        available_metrics,
        default=available_metrics,
        help="Select the metrics you want to see. Uncheck to hide them."
    )
    
    st.markdown("---")
    
    run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
    
    if run_backtest:
        if not selected_tickers:
            st.error("Please select at least one ticker.")
            st.stop()
        
        try:
            # Prepare backtest parameters (same for all tickers)
            backtest_params = {
                'profit_target_pct': profit_target,
                'stop_loss_pct': stop_loss,
                'initial_position_size_pct': initial_position_pct,
                'max_position_size_pct_all_dd': max_position_pct,
                'max_contracts': max_contracts if max_contracts is not None else 999999,  # Maximum number of contracts (None = no limit, use very high number)
                'min_option_price_per_lot': (min_option_price * 100) if min_option_price is not None else 0.0,  # Convert per-share to per-lot, or 0 if disabled
                'strike_otm_pct': 1.05,
                'risk_free_rate': risk_free_rate,
                'volatility_lookback': 21,
                'double_down_levels': {},
                'trade_direction': trade_direction,
                'signal_type': signal_type_param,  # Add signal type parameter
                'model_selections': model_selections,
                'strike_increment': strike_increment,
                'min_trend_score': trend_min,
                'max_trend_score': trend_max,
                'trailing_stop_pct': trailing_stop,
                'time_to_expiration_days': expiration_days,
                'confirmation_threshold': confirmation_threshold if 'confirmation_threshold' in locals() else 0,  # Add confirmation threshold
            }
            
            # Set entry conditions based on trade direction and signal type
            if signal_type_param in ['reversion', 'both', 'both_no_confirmation']:
                # Reversion entry thresholds
                if trade_direction == 'calls':
                    backtest_params['min_reversion_score'] = entry_reversion_min
                    backtest_params['max_reversion_score'] = 4
                elif trade_direction == 'puts':
                    backtest_params['min_reversion_score'] = -4
                    backtest_params['max_reversion_score'] = -entry_reversion_min
                else:  # both directions
                    # For 'both' trade direction, we need to allow both positive AND negative scores
                    # Positive scores (oversold) -> calls, negative scores (overbought) -> puts
                    # Set range to allow all scores that could trigger either direction
                    backtest_params['min_reversion_score'] = -4  # Allow negative scores for puts (filter range)
                    backtest_params['max_reversion_score'] = 4   # Allow positive scores for calls (filter range)
                    # CRITICAL: Also pass the entry threshold separately so it doesn't get confused with the filter range
                    backtest_params['entry_reversion_threshold'] = entry_reversion_min  # Entry threshold for both directions
            
            if signal_type_param in ['trend', 'both', 'both_no_confirmation']:
                # Trend entry thresholds
                if trade_direction == 'calls':
                    backtest_params['min_trend_score'] = trend_entry_min if signal_type_param in ['trend', 'both_no_confirmation'] else trend_min
                    backtest_params['max_trend_score'] = 4
                elif trade_direction == 'puts':
                    backtest_params['min_trend_score'] = -4
                    backtest_params['max_trend_score'] = -trend_entry_min if signal_type_param in ['trend', 'both_no_confirmation'] else trend_max
                else:  # both directions
                    if signal_type_param in ['trend', 'both_no_confirmation']:
                        # For 'both' trade direction, we need to allow both positive AND negative scores
                        # Positive scores -> calls, negative scores -> puts
                        # Set range to allow all scores that could trigger either direction
                        backtest_params['min_trend_score'] = -4  # Allow negative scores for puts (filter range)
                        backtest_params['max_trend_score'] = 4   # Allow positive scores for calls (filter range)
                        # CRITICAL: Also pass the entry threshold separately so it doesn't get confused with the filter range
                        backtest_params['entry_trend_threshold'] = trend_entry_min  # Entry threshold for both directions
                    else:  # 'both' (with confirmation) mode
                        # CRITICAL FIX: When trading both directions, need to allow BOTH positive AND negative trend scores
                        # Otherwise puts can never confirm (need negative trend to match negative reversion)
                        backtest_params['min_trend_score'] = -4  # Allow negative scores for puts
                        backtest_params['max_trend_score'] = 4   # Allow positive scores for calls
                        # Also set entry threshold if provided
                        if trend_entry_min:
                            backtest_params['entry_trend_threshold'] = trend_entry_min
            
            if enable_dd and dd_level is not None:
                backtest_params['double_down_levels'] = {dd_level: dd_multiplier}
            
            # Run backtests for all selected tickers
            all_results = []
            failed_tickers = []
            start_ts = pd.to_datetime(backtest_start_date)
            
            # CRITICAL FIX: Clean tech_data of any None values BEFORE starting backtests
            # This prevents NoneType errors for all stocks
            if tech_data and isinstance(tech_data, dict):
                # Remove any None entries from tech_data (corrupted data)
                none_tickers = [t for t, data in tech_data.items() if data is None or not isinstance(data, pd.DataFrame)]
                if none_tickers:
                    safe_print(f"⚠️ Removing {len(none_tickers)} corrupted None entries from tech_data: {none_tickers[:5]}")
                    for ticker in none_tickers:
                        tech_data.pop(ticker, None)
                    st.session_state.tech_data = tech_data
                
                # Ensure SPY exists and is valid
                if 'SPY' not in tech_data or tech_data.get('SPY') is None:
                    # SPY is missing or None - load it now
                    try:
                        from data_providers.fmp_provider import FMPProvider
                        spy_provider = FMPProvider(api_key=FMP_API_KEY)
                        spy_df = spy_provider.get_daily_stock_data('SPY', '1990-01-01', pd.to_datetime('today').strftime('%Y-%m-%d'))
                        if (spy_df is not None and 
                            isinstance(spy_df, pd.DataFrame) and 
                            not spy_df.empty and 
                            len(spy_df) >= 200 and
                            'close' in spy_df.columns):
                            tech_data['SPY'] = spy_df.copy()
                            st.session_state.tech_data = tech_data
                            safe_print("✅ Loaded SPY data for backtesting")
                        else:
                            safe_print("⚠️ Could not load valid SPY data - backtests will use 'Unknown' market regime")
                    except Exception as e:
                        safe_print(f"⚠️ Error loading SPY for backtest: {e}")
                else:
                    # SPY exists - validate it's not corrupted
                    spy_check = tech_data.get('SPY')
                    if spy_check is None or not isinstance(spy_check, pd.DataFrame) or spy_check.empty:
                        # Corrupted SPY entry - remove and reload
                        safe_print("⚠️ SPY entry is corrupted, removing and reloading")
                        tech_data.pop('SPY', None)
                        try:
                            from data_providers.fmp_provider import FMPProvider
                            spy_provider = FMPProvider(api_key=FMP_API_KEY)
                            spy_df = spy_provider.get_daily_stock_data('SPY', '1990-01-01', pd.to_datetime('today').strftime('%Y-%m-%d'))
                            if (spy_df is not None and isinstance(spy_df, pd.DataFrame) and not spy_df.empty):
                                tech_data['SPY'] = spy_df.copy()
                                st.session_state.tech_data = tech_data
                        except:
                            pass
            
            progress_text = f"Running backtest for {len(selected_tickers)} ticker(s)..."
            progress_bar = st.progress(0, text=progress_text)
            
            for idx, ticker in enumerate(selected_tickers):
                progress_bar.progress((idx + 1) / len(selected_tickers), text=f"Processing {ticker} ({idx + 1}/{len(selected_tickers)})...")
                
                try:
                    # Get the stock data for the ticker - with validation
                    if not tech_data or not isinstance(tech_data, dict):
                        failed_tickers.append((ticker, "tech_data is invalid or empty"))
                        continue
                    
                    ticker_data = tech_data.get(ticker)
                    
                    # CRITICAL: Validate ticker_data is actually a DataFrame (not None)
                    if (ticker_data is None or 
                        not isinstance(ticker_data, pd.DataFrame) or 
                        ticker_data.empty or
                        not hasattr(ticker_data, 'index') or
                        ticker_data.index is None):
                        failed_tickers.append((ticker, "No valid data available (ticker data is None or invalid)"))
                        continue
                    
                    # Filter data from start date (use pre-computed data from dashboard)
                    # CRITICAL: Validate again before filtering
                    try:
                        filtered_data = ticker_data[ticker_data.index >= start_ts].copy()
                        # Validate the copy succeeded
                        if filtered_data is None or not isinstance(filtered_data, pd.DataFrame):
                            failed_tickers.append((ticker, "Failed to filter data"))
                            continue
                    except Exception as filter_error:
                        failed_tickers.append((ticker, f"Error filtering data: {filter_error}"))
                        continue
                    
                    if filtered_data.empty or len(filtered_data) < 200:
                        failed_tickers.append((ticker, f"Not enough data (need at least 200 days from {backtest_start_date})"))
                        continue
                    
                    # Check if required columns exist (from pre-computed analysis)
                    required_cols = ['Trend_Score', 'Reversion_Score', 'close']
                    missing_cols = [col for col in required_cols if col not in filtered_data.columns]
                    if missing_cols:
                        failed_tickers.append((ticker, f"Missing columns: {', '.join(missing_cols)}"))
                        continue
                    
                    # Only calculate volatility if missing (backtest needs it for Black-Scholes)
                    # CRITICAL: Validate filtered_data before any assignments
                    if filtered_data is None or not isinstance(filtered_data, pd.DataFrame) or filtered_data.empty:
                        failed_tickers.append((ticker, "Filtered data is invalid"))
                        continue
                    
                    if 'historical_volatility' not in filtered_data.columns:
                        # Calculate only volatility, don't re-run full analysis
                        try:
                            filtered_data['log_return'] = np.log(filtered_data['close'] / filtered_data['close'].shift(1))
                            filtered_data['historical_volatility'] = filtered_data['log_return'].rolling(window=21).std() * np.sqrt(252)
                            filtered_data = filtered_data.drop(columns=['log_return'], errors='ignore')
                            # Validate after assignment
                            if filtered_data is None:
                                failed_tickers.append((ticker, "Error calculating volatility"))
                                continue
                        except Exception as vol_error:
                            failed_tickers.append((ticker, f"Error calculating volatility: {vol_error}"))
                            continue
                    
                    # Use pre-computed data directly (no re-analysis needed)
                    # CRITICAL: Validate before copying
                    if filtered_data is None or not isinstance(filtered_data, pd.DataFrame):
                        failed_tickers.append((ticker, "Cannot copy filtered data"))
                        continue
                    analyzed_data = filtered_data.copy()
                    
                    # CRITICAL: Validate analyzed_data before using
                    if analyzed_data is None or not isinstance(analyzed_data, pd.DataFrame) or analyzed_data.empty:
                        failed_tickers.append((ticker, "Analyzed data is invalid after copy"))
                        continue
                    
                    # Ensure we have volatility (required for backtest)
                    if 'historical_volatility' not in analyzed_data.columns or analyzed_data['historical_volatility'].isna().all():
                        failed_tickers.append((ticker, "Unable to calculate historical volatility"))
                        continue
                    
                    # Fill any NaN values in volatility with forward fill
                    # CRITICAL: Validate analyzed_data before assignment
                    try:
                        analyzed_data['historical_volatility'] = analyzed_data['historical_volatility'].ffill().fillna(0.2)
                    except Exception as fill_error:
                        failed_tickers.append((ticker, f"Error filling volatility: {fill_error}"))
                        continue
                    
                    # Ensure MarketRegime is calculated (needed for backtest trade records)
                    # SIMPLIFIED: Just use SPY from tech_data if available, otherwise skip (no loading during backtest)
                    if 'MarketRegime' not in analyzed_data.columns:
                        # Get SPY data ONLY from tech_data (already loaded) - don't try to load during backtest
                        spy_data = None
                        try:
                            # Get from tech_data - SPY should have been loaded before backtest loop started
                            if tech_data and isinstance(tech_data, dict) and 'SPY' in tech_data:
                                potential_spy = tech_data.get('SPY')
                                # Validate it's a real DataFrame
                                if (potential_spy is not None and 
                                    isinstance(potential_spy, pd.DataFrame) and 
                                    not potential_spy.empty and 
                                    hasattr(potential_spy, 'index') and 
                                    potential_spy.index is not None and
                                    'close' in potential_spy.columns and
                                    len(potential_spy) >= 200):
                                    spy_data = potential_spy.copy()
                        except Exception as e:
                            safe_print(f"Warning: Could not get SPY data for {ticker}: {e}")
                            spy_data = None
                        
                        # Additional validation before using spy_data
                        if spy_data is not None and isinstance(spy_data, pd.DataFrame) and not spy_data.empty:
                            try:
                                # Filter SPY data to match backtest date range
                                # CRITICAL: Re-validate spy_data before EVERY operation
                                if (spy_data is not None and 
                                    isinstance(spy_data, pd.DataFrame) and 
                                    not spy_data.empty and 
                                    hasattr(spy_data, 'index') and 
                                    spy_data.index is not None and 
                                    'close' in spy_data.columns):
                                    try:
                                        # CRITICAL: Validate spy_data one more time before filtering
                                        # Double-check to prevent NoneType errors
                                        if (spy_data is None or 
                                            not isinstance(spy_data, pd.DataFrame) or 
                                            spy_data.empty or 
                                            not hasattr(spy_data, 'index') or
                                            spy_data.index is None or
                                            'close' not in spy_data.columns):
                                            if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                                analyzed_data['MarketRegime'] = 'Unknown'
                                        else:
                                            # Safe to filter now
                                            filtered = spy_data[spy_data.index >= start_ts].copy()
                                            if (filtered is not None and 
                                                isinstance(filtered, pd.DataFrame) and 
                                                not filtered.empty and 
                                                len(filtered) >= 200 and 
                                                'close' in filtered.columns):
                                                spy_filtered = filtered.copy()
                                                # CRITICAL: Validate copy before ANY assignment
                                                if (spy_filtered is not None and 
                                                    isinstance(spy_filtered, pd.DataFrame) and 
                                                    not spy_filtered.empty and
                                                    hasattr(spy_filtered, 'index') and
                                                    spy_filtered.index is not None and
                                                    'close' in spy_filtered.columns):
                                                    try:
                                                        # CRITICAL: Validate analyzed_data before join operation
                                                        if (analyzed_data is None or 
                                                            not isinstance(analyzed_data, pd.DataFrame) or 
                                                            analyzed_data.empty):
                                                            # Cannot assign to None - skip this ticker
                                                            failed_tickers.append((ticker, "analyzed_data became None during market regime calculation"))
                                                            continue
                                                        else:
                                                            spy_filtered['SMA_200'] = spy_filtered['close'].rolling(window=200).mean()
                                                            spy_filtered['IsBullMarket'] = (spy_filtered['close'] > spy_filtered['SMA_200']).astype(int)
                                                            # Join SPY market regime to analyzed data
                                                            # CRITICAL: Validate join result
                                                            join_result = analyzed_data.join(spy_filtered[['IsBullMarket']], how='left')
                                                            if (join_result is not None and 
                                                                isinstance(join_result, pd.DataFrame) and 
                                                                not join_result.empty):
                                                                analyzed_data = join_result
                                                                analyzed_data['IsBullMarket'] = analyzed_data['IsBullMarket'].ffill().bfill().infer_objects(copy=False)
                                                                analyzed_data['MarketRegime'] = analyzed_data['IsBullMarket'].apply(lambda x: 'Bull' if x == 1 else 'Bear')
                                                            else:
                                                                # Join failed - set to Unknown without modifying analyzed_data
                                                                if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                                                    analyzed_data['MarketRegime'] = 'Unknown'
                                                    except Exception as assign_error:
                                                        safe_print(f"Error assigning to SPY columns for {ticker}: {assign_error}")
                                                        # CRITICAL: Only assign if analyzed_data is valid
                                                        if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                                            analyzed_data['MarketRegime'] = 'Unknown'
                                                else:
                                                    if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                                        analyzed_data['MarketRegime'] = 'Unknown'
                                            else:
                                                if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                                    analyzed_data['MarketRegime'] = 'Unknown'
                                    except Exception as filter_error:
                                        safe_print(f"Error filtering SPY data for {ticker}: {filter_error}")
                                        if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                            analyzed_data['MarketRegime'] = 'Unknown'
                                else:
                                    if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                        analyzed_data['MarketRegime'] = 'Unknown'
                            except Exception as e:
                                safe_print(f"Error calculating market regime for {ticker}: {e}")
                                if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                    analyzed_data['MarketRegime'] = 'Unknown'
                        else:
                            if analyzed_data is not None and isinstance(analyzed_data, pd.DataFrame):
                                analyzed_data['MarketRegime'] = 'Unknown'
                    
                    # CRITICAL: Final validation of analyzed_data before running backtest
                    # This prevents NoneType errors inside the backtest engine
                    if (analyzed_data is None or 
                        not isinstance(analyzed_data, pd.DataFrame) or 
                        analyzed_data.empty):
                        failed_tickers.append((ticker, "analyzed_data is None or invalid before backtest"))
                        continue
                    
                    # Ensure all required columns exist before backtest
                    required_backtest_cols = ['close', 'Trend_Score', 'Reversion_Score', 'historical_volatility']
                    missing_backtest_cols = [col for col in required_backtest_cols if col not in analyzed_data.columns]
                    if missing_backtest_cols:
                        failed_tickers.append((ticker, f"Missing required columns for backtest: {', '.join(missing_backtest_cols)}"))
                        continue
                    
                    # Calculate portfolio allocation per ticker when multiple tickers
                    ticker_portfolio_value = initial_portfolio / len(selected_tickers) if len(selected_tickers) > 1 else initial_portfolio
                    
                    # If "both_no_confirmation" mode, run separate backtests for trend and reversion independently
                    if signal_type_param == 'both_no_confirmation':
                        # Run trend-only backtest
                        trend_params = backtest_params.copy()
                        trend_params['signal_type'] = 'trend'
                        # Ensure trend entry thresholds are set correctly for trend-only mode
                        # When trade_direction is 'both', entry_trend_threshold should already be set
                        # For calls/puts, ensure min_trend_score/max_trend_score are correct
                        if trade_direction == 'calls':
                            trend_params['min_trend_score'] = trend_entry_min
                            trend_params['max_trend_score'] = 4
                            trend_params.pop('entry_trend_threshold', None)  # Not needed for single direction
                        elif trade_direction == 'puts':
                            trend_params['min_trend_score'] = -4
                            trend_params['max_trend_score'] = -trend_entry_min
                            trend_params.pop('entry_trend_threshold', None)  # Not needed for single direction
                        # else: trade_direction == 'both', entry_trend_threshold should already be set
                        trend_result = run_portfolio_backtest(analyzed_data, trend_params, ticker_portfolio_value / 2)
                        # CRITICAL: Validate backtest result before modifying
                        if trend_result is None or not isinstance(trend_result, dict):
                            failed_tickers.append((ticker, "trend backtest returned None or invalid result"))
                            continue
                        trend_result['Ticker'] = ticker
                        trend_result['Initial Portfolio Value'] = ticker_portfolio_value / 2
                        trend_result['Signal_Type_Test'] = 'trend'  # Tag for later aggregation
                        # Tag all trades with 'trend'
                        trend_trades = trend_result.get('Trades', trend_result.get('trades', []))
                        if trend_trades is not None and isinstance(trend_trades, pd.DataFrame):
                            trend_trades['signal_type'] = signal_type_param
                            trend_result['Trades'] = trend_trades
                        elif trend_trades is not None and isinstance(trend_trades, list):
                            for trade in trend_trades:
                                if trade is not None and isinstance(trade, dict):
                                    trade['signal_type'] = 'trend'
                            trend_result['Trades'] = trend_trades
                        
                        # Run reversion-only backtest
                        reversion_params = backtest_params.copy()
                        reversion_params['signal_type'] = 'reversion'
                        # Ensure reversion entry thresholds are set correctly for reversion-only mode
                        if trade_direction == 'calls':
                            reversion_params['min_reversion_score'] = entry_reversion_min
                            reversion_params['max_reversion_score'] = 4
                            reversion_params.pop('entry_reversion_threshold', None)  # Not needed for single direction
                        elif trade_direction == 'puts':
                            reversion_params['min_reversion_score'] = -4
                            reversion_params['max_reversion_score'] = -entry_reversion_min
                            reversion_params.pop('entry_reversion_threshold', None)  # Not needed for single direction
                        # else: trade_direction == 'both', entry_reversion_threshold should already be set
                        reversion_result = run_portfolio_backtest(analyzed_data, reversion_params, ticker_portfolio_value / 2)
                        # CRITICAL: Validate backtest result before modifying
                        if reversion_result is None or not isinstance(reversion_result, dict):
                            failed_tickers.append((ticker, "reversion backtest returned None or invalid result"))
                            continue
                        reversion_result['Ticker'] = ticker
                        reversion_result['Initial Portfolio Value'] = ticker_portfolio_value / 2
                        reversion_result['Signal_Type_Test'] = 'reversion'  # Tag for later aggregation
                        # Tag all trades with 'reversion'
                        reversion_trades = reversion_result.get('Trades', reversion_result.get('trades', []))
                        if reversion_trades is not None and isinstance(reversion_trades, pd.DataFrame):
                            reversion_trades['signal_type'] = 'reversion'
                            reversion_result['Trades'] = reversion_trades
                        elif reversion_trades is not None and isinstance(reversion_trades, list):
                            for trade in reversion_trades:
                                if trade is not None and isinstance(trade, dict):
                                    trade['signal_type'] = 'reversion'
                            reversion_result['Trades'] = reversion_trades
                        
                        # Combine the two backtests into a single result
                        combined_trades = []
                        combined_history = []
                        
                        # Combine trades
                        for trade_list in [trend_trades, reversion_trades]:
                            if isinstance(trade_list, pd.DataFrame):
                                combined_trades.extend(trade_list.to_dict('records'))
                            elif isinstance(trade_list, list):
                                combined_trades.extend(trade_list)
                        
                        # Combine portfolio histories
                        # backtest_engine returns 'portfolio_history' (lowercase) as DataFrame with date index
                        trend_history = trend_result.get('portfolio_history', trend_result.get('Portfolio History', []))
                        reversion_history = reversion_result.get('portfolio_history', reversion_result.get('Portfolio History', []))
                        
                        # Convert histories to dataframes if needed and combine by date
                        all_history = []
                        for hist_list in [trend_history, reversion_history]:
                            if isinstance(hist_list, pd.DataFrame):
                                # Handle DataFrame - could have date as index or column
                                if isinstance(hist_list.index, pd.DatetimeIndex):
                                    # Date is the index, convert to column
                                    hist_df_copy = hist_list.reset_index()
                                    if 'index' in hist_df_copy.columns:
                                        hist_df_copy = hist_df_copy.rename(columns={'index': 'date'})
                                    elif 'date' not in hist_df_copy.columns and len(hist_df_copy.columns) > 0:
                                        # Try first column
                                        hist_df_copy.columns = ['date', 'value'] if len(hist_df_copy.columns) == 2 else ['date'] + list(hist_df_copy.columns[1:])
                                    all_history.extend(hist_df_copy.to_dict('records'))
                                elif 'date' in hist_list.columns:
                                    # Date is already a column
                                    all_history.extend(hist_list.to_dict('records'))
                                else:
                                    # Try to infer - might be list of dicts already
                                    all_history.extend(hist_list.to_dict('records') if hasattr(hist_list, 'to_dict') else [])
                            elif isinstance(hist_list, list):
                                all_history.extend(hist_list)
                        
                        # Aggregate portfolio history by date (sum values from both backtests)
                        if all_history:
                            try:
                                hist_df = pd.DataFrame(all_history)
                                # Handle different portfolio history formats
                                # Could be: {'date': ..., 'value': ...} or {'index': Timestamp, 'value': ...} or DataFrame with date index
                                if 'date' not in hist_df.columns and len(hist_df) > 0:
                                    # Check if index is datetime
                                    if isinstance(hist_df.index, pd.DatetimeIndex):
                                        hist_df = hist_df.reset_index()
                                        if 'index' in hist_df.columns:
                                            hist_df = hist_df.rename(columns={'index': 'date'})
                                    # Or check first row for date-like key
                                    elif len(hist_df.columns) > 0:
                                        first_col = hist_df.columns[0]
                                        if 'date' in first_col.lower() or pd.api.types.is_datetime64_any_dtype(hist_df[first_col]):
                                            hist_df = hist_df.rename(columns={first_col: 'date'})
                                
                                # Ensure we have both 'date' and 'value' columns
                                if 'date' in hist_df.columns and 'value' in hist_df.columns:
                                    hist_df['date'] = pd.to_datetime(hist_df['date'])
                                    combined_hist_df = hist_df.groupby('date')['value'].sum().reset_index()
                                    combined_hist_df = combined_hist_df.sort_values('date')
                                    combined_history = combined_hist_df.to_dict('records')
                                else:
                                    # Fallback: try to reconstruct from trades if history is missing
                                    safe_print(f"⚠️ Warning: Portfolio history format unexpected, attempting to reconstruct from trades")
                                    combined_history = []
                            except Exception as e:
                                safe_print(f"⚠️ Error processing portfolio history: {e}")
                                combined_history = []
                        
                        # Calculate combined metrics
                        final_trend = trend_result.get('metrics', {}).get('Final Portfolio Value', trend_result.get('Final Portfolio Value', ticker_portfolio_value / 2))
                        final_reversion = reversion_result.get('metrics', {}).get('Final Portfolio Value', reversion_result.get('Final Portfolio Value', ticker_portfolio_value / 2))
                        combined_final = final_trend + final_reversion
                        
                        # Recalculate metrics from combined history
                        if combined_history:
                            hist_values = [h['value'] for h in combined_history if 'value' in h]
                            if hist_values:
                                peak_value = max(hist_values)
                                max_dd = ((peak_value - min(hist_values)) / peak_value) if peak_value > 0 else 0
                                num_years = (pd.to_datetime(combined_history[-1]['date']) - pd.to_datetime(combined_history[0]['date'])).days / 365.25 if len(combined_history) > 1 else 1
                                total_return = (combined_final - ticker_portfolio_value) / ticker_portfolio_value if ticker_portfolio_value > 0 else 0
                                cagr = (combined_final / ticker_portfolio_value) ** (1 / num_years) - 1 if num_years > 0 and ticker_portfolio_value > 0 else 0
                                win_rate = len([t for t in combined_trades if t.get('pnl_dollars', 0) > 0]) / len(combined_trades) if combined_trades else 0
                            else:
                                max_dd = max(
                                    trend_result.get('metrics', {}).get('Max Drawdown', 0),
                                    reversion_result.get('metrics', {}).get('Max Drawdown', 0)
                                )
                                total_return = (combined_final - ticker_portfolio_value) / ticker_portfolio_value if ticker_portfolio_value > 0 else 0
                                cagr = total_return
                                win_rate = len([t for t in combined_trades if t.get('pnl_dollars', 0) > 0]) / len(combined_trades) if combined_trades else 0
                        else:
                            max_dd = max(
                                trend_result.get('metrics', {}).get('Max Drawdown', 0),
                                reversion_result.get('metrics', {}).get('Max Drawdown', 0)
                            )
                            total_return = (combined_final - ticker_portfolio_value) / ticker_portfolio_value if ticker_portfolio_value > 0 else 0
                            cagr = total_return
                            win_rate = len([t for t in combined_trades if t.get('pnl_dollars', 0) > 0]) / len(combined_trades) if combined_trades else 0
                        
                        # Store separate histories for charting (ensure proper format)
                        # backtest_engine returns 'portfolio_history' (lowercase) as DataFrame with date index
                        trend_history_for_chart = trend_result.get('portfolio_history', trend_result.get('Portfolio History', []))
                        reversion_history_for_chart = reversion_result.get('portfolio_history', reversion_result.get('Portfolio History', []))
                        
                        # Convert to consistent format (list of dicts with 'date' and 'value')
                        def normalize_history(hist):
                            if isinstance(hist, pd.DataFrame):
                                if isinstance(hist.index, pd.DatetimeIndex):
                                    hist = hist.reset_index()
                                    if 'index' in hist.columns:
                                        hist = hist.rename(columns={'index': 'date'})
                                if 'date' in hist.columns and 'value' in hist.columns:
                                    return hist[['date', 'value']].to_dict('records')
                                elif len(hist.columns) >= 2:
                                    hist = hist.rename(columns={hist.columns[0]: 'date', hist.columns[1]: 'value'})
                                    return hist[['date', 'value']].to_dict('records')
                            elif isinstance(hist, list):
                                return hist
                            return []
                        
                        trend_history_for_chart = normalize_history(trend_history_for_chart)
                        reversion_history_for_chart = normalize_history(reversion_history_for_chart)
                        
                        # Create combined result
                        result = {
                            'Ticker': ticker,
                            'Initial Portfolio Value': ticker_portfolio_value,
                            'Portfolio History': combined_history,
                            'Trades': combined_trades,
                            'metrics': {
                                'Final Portfolio Value': combined_final,
                                'Total Return': total_return,
                                'Annualized Return (CAGR)': cagr,
                                'Max Drawdown': max_dd,
                                'Win Rate': win_rate,
                                'Total Trades': len(combined_trades),
                            },
                            'Final Portfolio Value': combined_final,
                            'Total Return': total_return,
                            'Annualized Return (CAGR)': cagr,
                            'Max Drawdown': max_dd,
                            'Win Rate': win_rate,
                            'Signal_Type_Test': 'both_no_confirmation',  # Tag as combined (no confirmation)
                            '_trend_history': trend_history_for_chart,  # Store separately for chart
                            '_reversion_history': reversion_history_for_chart  # Store separately for chart
                        }
                        all_results.append(result)
                    else:
                        # Run normal backtest (trend-only or reversion-only)
                        result = run_portfolio_backtest(analyzed_data, backtest_params, ticker_portfolio_value)
                        # CRITICAL: Validate backtest result before modifying
                        if result is None or not isinstance(result, dict):
                            failed_tickers.append((ticker, "backtest returned None or invalid result"))
                            continue
                        result['Ticker'] = ticker
                        result['Initial Portfolio Value'] = ticker_portfolio_value  # Store initial value for aggregation
                        # Tag trades with signal type
                        trades = result.get('Trades', result.get('trades', []))
                        if trades is not None and isinstance(trades, pd.DataFrame):
                            trades['signal_type'] = signal_type_param
                            result['Trades'] = trades
                        elif trades is not None and isinstance(trades, list):
                            for trade in trades:
                                if trade is not None and isinstance(trade, dict):
                                    trade['signal_type'] = signal_type_param
                            result['Trades'] = trades
                        all_results.append(result)
                    
                except Exception as e:
                    # Get full error details to debug NoneType assignment issues
                    import traceback
                    error_details = f"{str(e)}"
                    error_trace = traceback.format_exc()
                    # Extract the specific line causing the error
                    if "'NoneType' object does not support item assignment" in str(e):
                        safe_print(f"⚠️ NoneType assignment error for {ticker}: {error_trace}")
                    failed_tickers.append((ticker, error_details))
                    continue
            
            progress_bar.empty()
            
            # Show warnings for failed tickers
            if failed_tickers:
                for ticker, reason in failed_tickers:
                    st.warning(f"⚠️ {ticker}: {reason}")
            
            if not all_results:
                st.error("No backtests completed successfully. Please check the errors above.")
                st.stop()
            
            # Aggregate results if multiple tickers
            if len(all_results) > 1:
                # Combine portfolio histories
                combined_history = []
                combined_trades = []
                # Access initial portfolio value (stored when result created) or calculate from metrics
                total_initial = sum(r.get('Initial Portfolio Value', r.get('metrics', {}).get('Initial Portfolio Value', initial_portfolio / len(selected_tickers))) for r in all_results)
                # Access final portfolio value from metrics dictionary
                total_final = sum(r.get('metrics', {}).get('Final Portfolio Value', r.get('Final Portfolio Value', 0)) for r in all_results)
                
                for result in all_results:
                    trades = result.get('Trades', result.get('trades', []))
                    portfolio_history = result.get('Portfolio History', result.get('portfolio_history', []))
                    
                    if trades is not None:
                        if isinstance(trades, pd.DataFrame):
                            if not trades.empty:
                                # Ensure signal_type column exists and has valid values
                                if 'signal_type' in trades.columns:
                                    # Fill NaN/None values with 'unknown'
                                    trades['signal_type'] = trades['signal_type'].fillna('unknown')
                                trades = trades.to_dict('records')
                        if isinstance(trades, list) and len(trades) > 0:
                            for trade in trades:
                                if isinstance(trade, dict):
                                    # Ensure signal_type exists in trade dict
                                    if 'signal_type' not in trade or trade.get('signal_type') is None:
                                        trade['signal_type'] = 'unknown'
                                    trade['Ticker'] = result['Ticker']
                                    combined_trades.append(trade)
                    
                    if portfolio_history is not None:
                        if isinstance(portfolio_history, pd.DataFrame):
                            if not portfolio_history.empty:
                                portfolio_history = portfolio_history.reset_index().to_dict('records')
                        if isinstance(portfolio_history, list) and len(portfolio_history) > 0:
                            for hist in portfolio_history:
                                if isinstance(hist, dict):
                                    combined_history.append(hist)
                
                # Aggregate portfolio history by date
                if combined_history:
                    hist_df = pd.DataFrame(combined_history)
                    hist_df['date'] = pd.to_datetime(hist_df['date'])
                    combined_hist_df = hist_df.groupby('date')['value'].sum().reset_index()
                    combined_hist_df = combined_hist_df.sort_values('date')
                    
                    # Calculate combined metrics
                    peak_value = combined_hist_df['value'].max()
                    max_dd = ((peak_value - combined_hist_df['value']) / peak_value).max()
                    num_years = (combined_hist_df['date'].max() - combined_hist_df['date'].min()).days / 365.25
                    total_return = (total_final - total_initial) / total_initial if total_initial > 0 else 0
                    cagr = (total_final / total_initial) ** (1 / num_years) - 1 if num_years > 0 and total_initial > 0 else 0
                    
                    win_rate = len([t for t in combined_trades if t.get('pnl_dollars', 0) > 0]) / len(combined_trades) if combined_trades else 0
                    
                    aggregated_result = {
                        'Ticker': 'PORTFOLIO (Combined)',
                        'Initial Portfolio Value': total_initial,
                        'Final Portfolio Value': total_final,
                        'Total Return': total_return,
                        'Annualized Return (CAGR)': cagr,
                        'Max Drawdown': max_dd,
                        'Win Rate': win_rate,
                        'Portfolio History': combined_hist_df.to_dict('records'),
                        'Trades': combined_trades,
                        'Metrics': {
                            'Final Portfolio Value': total_final,
                            'Total Return': total_return,  # Store as numeric, not formatted string
                            'Annualized Return (CAGR)': cagr,  # Store as numeric
                            'Max Drawdown': max_dd,  # Store as numeric
                            'Win Rate': win_rate,  # Store as numeric
                            'Total Trades': len(combined_trades),
                        }
                    }
                    all_results.insert(0, aggregated_result)
            
            # Store backtest results in session state
            st.session_state.backtest_results = all_results
        except Exception as e:
            st.error(f"Error running backtest: {str(e)}")
            st.exception(e)

    # Results toggle and display section (outside of if run_backtest block)
    st.markdown("---")
    st.header("📊 Results")
    
    # Determine which results to show
    has_backtest_results = 'backtest_results' in st.session_state and st.session_state.backtest_results
    has_optimizer_results = 'optimizer_results' in st.session_state and st.session_state.optimizer_results
    
    if has_backtest_results or has_optimizer_results:
        # Toggle button to switch between results
        if has_backtest_results and has_optimizer_results:
            view_mode = st.radio("View Results:", ["Backtest Results", "Optimized Results"], horizontal=True, key='results_view_toggle')
        elif has_backtest_results:
            view_mode = "Backtest Results"
            st.info("💡 Run the optimizer to compare with optimized results.")
        else:
            view_mode = "Optimized Results"
        
        # Display selected results
        if view_mode == "Optimized Results" and has_optimizer_results:
            # Get optimization target from session state if available
            opt_target_display = st.session_state.get('last_optimization_target', 'Risk-Adjusted Return (CAGR / Max Drawdown)')
            st.info(f"📊 **Optimized for:** {opt_target_display}")
            st.subheader("🏆 Optimal Strategy Results")
            if 'analyzed_data_range_display' in st.session_state:
                st.caption(f"Optimization Date Range: {st.session_state.analyzed_data_range_display}")
            
            # Display optimizer results table
            optimizer_data = []
            for res in st.session_state.optimizer_results:
                signal_type = res['params'].get('signal_type', 'reversion')
                trade_dir = res['params'].get('trade_direction', 'calls')
                
                optimizer_data.append({
                    "Ticker": res['ticker'],
                    "Signal Type": signal_type.replace('_', ' ').title(),
                    "Trade Dir": trade_dir.title(),
                    "Risk-Adj Return": res['risk_adjusted_return'],
                    "CAGR": res['metrics']['Annualized Return (CAGR)'],
                    "Max DD": res['metrics']['Max Drawdown'],
                    "Win Rate": res['metrics']['Win Rate'],
                    "Total Trades": res['metrics']['Total Trades'],
                })
            
            if optimizer_data:
                opt_df = pd.DataFrame(optimizer_data).sort_values("Risk-Adj Return", ascending=False).set_index("Ticker")
                st.dataframe(opt_df.style.format({
                    "Risk-Adj Return": "{:.2f}",
                    "CAGR": "{:.2%}",
                    "Max DD": "{:.2%}",
                    "Win Rate": "{:.2%}",
                    "Total Trades": "{:,.0f}",
                }).background_gradient(cmap='Greens', subset=['Risk-Adj Return', 'CAGR']), use_container_width=True, hide_index=False)
            
            # Comparison section
            if 'all_optimizer_results' in st.session_state and st.session_state.all_optimizer_results:
                st.markdown("---")
                st.subheader("📊 Compare Your Selection vs Optimal")
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    available_tickers_opt = [r['ticker'] for r in st.session_state.optimizer_results if r['ticker'] in st.session_state.all_optimizer_results]
                    if available_tickers_opt:
                        compare_ticker = st.selectbox("Select Ticker to Compare", available_tickers_opt, key='compare_ticker_dash')
                    else:
                        compare_ticker = None
                with comp_col2:
                    compare_metric = st.selectbox("Optimization Metric", ["Risk-Adjusted Return", "Annualized Return (CAGR)", "Win Rate", "Max Drawdown"], key='compare_metric_dash')
                
                if compare_ticker and compare_ticker in st.session_state.all_optimizer_results:
                    all_results_opt = st.session_state.all_optimizer_results[compare_ticker]
                    optimal_result = next((r for r in st.session_state.optimizer_results if r['ticker'] == compare_ticker), None)
                    
                    if optimal_result:
                        metric_key_map = {
                            "Risk-Adjusted Return": "risk_adjusted_return",
                            "Annualized Return (CAGR)": lambda x: x['metrics']['Annualized Return (CAGR)'],
                            "Win Rate": lambda x: x['metrics']['Win Rate'],
                            "Max Drawdown": lambda x: -x['metrics']['Max Drawdown']
                        }
                        metric_key = metric_key_map[compare_metric]
                        if callable(metric_key):
                            sorted_results = sorted(all_results_opt, key=metric_key, reverse=True)[:10]
                        else:
                            sorted_results = sorted(all_results_opt, key=lambda x: x[metric_key], reverse=True)[:10]
                        
                        comparison_data = []
                        for idx, r in enumerate(sorted_results):
                            is_optimal = r['params'] == optimal_result['params']
                            comparison_data.append({
                                "Rank": idx + 1,
                                "Is Optimal": "✅ Yes" if is_optimal else "",
                                "Signal Type": r['params'].get('signal_type', 'reversion').replace('_', ' ').title(),
                                "Trade Dir": r['params'].get('trade_direction', 'calls').title(),
                                "Risk-Adj Return": r['risk_adjusted_return'],
                                "CAGR": r['metrics']['Annualized Return (CAGR)'],
                                "Max DD": r['metrics']['Max Drawdown'],
                                "Win Rate": r['metrics']['Win Rate'],
                                "Trades": r['metrics']['Total Trades'],
                            })
                        
                        if comparison_data:
                            comp_df = pd.DataFrame(comparison_data)
                            st.dataframe(comp_df.style.format({
                                "Risk-Adj Return": "{:.2f}",
                                "CAGR": "{:.2%}",
                                "Max DD": "{:.2%}",
                                "Win Rate": "{:.2%}",
                                "Trades": "{:,.0f}",
                            }).background_gradient(cmap='Greens', subset=['Risk-Adj Return', 'CAGR']), use_container_width=True, hide_index=True)
        
        elif view_mode == "Backtest Results" and has_backtest_results:
            # Display backtest results (existing code)
            all_results = st.session_state.backtest_results
            # Display results for all tickers in requested order
            for result in all_results:
                ticker_name = result.get('Ticker', 'Unknown')
                st.header(f"📊 Results for {ticker_name}")
                
                # The order is now: Optimized Signals -> Regime -> Current Signal -> Performance -> Charts/Tables
                
                # 0. Optimized Entry Signals - MOVED TO TOP
                st.subheader("🎯 Optimized Entry Signals")
                signal_info = []
                signal_info.append(f"**Signal Type:** {signal_type}")
                
                if signal_type_param in ['reversion', 'both']:
                    if trade_direction == 'calls':
                        signal_info.append(f"**Reversion Calls Entry:** Reversion Score ≥ {entry_reversion_min} (oversold/buy signal)")
                    elif trade_direction == 'puts':
                        signal_info.append(f"**Reversion Puts Entry:** Reversion Score ≤ -{entry_reversion_min} (overbought/sell signal)")
                    else:
                        signal_info.append(f"**Reversion Calls/Puts Entry:** |Reversion Score| ≥ {entry_reversion_min}")
                
                if signal_type_param in ['trend', 'both']:
                    if trade_direction == 'calls':
                        signal_info.append(f"**Trend Calls Entry:** Trend Score ≥ {trend_entry_min} (bullish trend/follow)")
                    elif trade_direction == 'puts':
                        signal_info.append(f"**Trend Puts Entry:** Trend Score ≤ -{trend_entry_min} (bearish trend/follow)")
                    else:
                        signal_info.append(f"**Trend Calls/Puts Entry:** |Trend Score| ≥ {trend_entry_min}")
                
                if signal_type_param == 'both':
                    signal_info.append(f"**Testing Mode:** When 'Both' is selected, the backtester runs **separate independent backtests** for trend and reversion signals, then combines the results. Each signal type is tested with half the portfolio allocation.")
                
                if use_trend_filter and signal_type_param == 'reversion':
                    signal_info.append(f"**Trend Confirmation Filter:** {trend_min} ≤ Trend Score ≤ {trend_max}")
                elif not use_trend_filter:
                    signal_info.append(f"**Trend Filter:** Disabled (allows any Trend Score)")
                
                signal_info.append(f"**Trade Direction:** {trade_direction.title()}")
                st.markdown("\n".join(signal_info))
                
                # 1. Performance by Market Regime (if available and selected)
                if 'Performance by Market Regime' in selected_metrics:
                    trades_for_regime = result.get('Trades', result.get('trades', []))
                    if trades_for_regime is not None:
                        if isinstance(trades_for_regime, list):
                            regime_trades_df = pd.DataFrame(trades_for_regime)
                        else:
                            regime_trades_df = trades_for_regime
                        
                        if not regime_trades_df.empty and 'market_regime' in regime_trades_df.columns:
                            # Ensure signal_type column exists and fill missing values
                            if 'signal_type' in regime_trades_df.columns:
                                regime_trades_df['signal_type'] = regime_trades_df['signal_type'].fillna('unknown')
                            else:
                                # If signal_type doesn't exist, add it with default value
                                regime_trades_df['signal_type'] = 'unknown'
                            
                            st.subheader("📊 Performance by Market Regime")
                            
                            # Calculate comprehensive regime statistics
                            regime_stats_list = []
                            for regime in regime_trades_df['market_regime'].unique():
                                regime_trades = regime_trades_df[regime_trades_df['market_regime'] == regime]
                                wins = regime_trades[regime_trades['pnl_dollars'] > 0]
                                losses = regime_trades[regime_trades['pnl_dollars'] <= 0]
                                
                                regime_stats_list.append({
                                    'Market Regime': regime,
                                    'Total PnL ($)': regime_trades['pnl_dollars'].sum(),
                                    'Avg PnL ($)': regime_trades['pnl_dollars'].mean(),
                                    'Total Trades': len(regime_trades),
                                    'Winning Trades': len(wins),
                                    'Losing Trades': len(losses),
                                    'Win Rate (%)': (len(wins) / len(regime_trades) * 100) if len(regime_trades) > 0 else 0,
                                    'Avg Win ($)': wins['pnl_dollars'].mean() if len(wins) > 0 else 0,
                                    'Avg Loss ($)': losses['pnl_dollars'].mean() if len(losses) > 0 else 0,
                                })
                            
                            regime_stats_df = pd.DataFrame(regime_stats_list)
                            st.dataframe(regime_stats_df.style.format({
                                'Total PnL ($)': "${:,.2f}",
                                'Avg PnL ($)': "${:,.2f}",
                                'Total Trades': "{:,.0f}",
                                'Winning Trades': "{:,.0f}",
                                'Losing Trades': "{:,.0f}",
                                'Win Rate (%)': "{:.1f}%",
                                'Avg Win ($)': "${:,.2f}",
                                'Avg Loss ($)': "${:,.2f}"
                            }), use_container_width=True, hide_index=True)
                            
                            # NEW: Performance by Signal Type × Market Regime (Trend Bull/Bear, Reversion Bull/Bear)
                            if 'signal_type' in regime_trades_df.columns:
                                st.subheader("📈 Performance by Signal Type × Market Regime")
                                
                                # Define color mapping for trend (blue) and reversion
                                # Use colors that work in both dark and light modes
                                def get_row_color(signal_type_val, regime_val):
                                    """Return background color based on signal type"""
                                    if 'trend' in str(signal_type_val).lower():
                                        return 'rgba(59, 130, 246, 0.15)'  # Blue tint for trend
                                    elif 'reversion' in str(signal_type_val).lower():
                                        return 'rgba(16, 185, 129, 0.15)'  # Green tint for reversion (keeping existing)
                                    return None
                                
                                performance_list = []
                                # Create all combinations: Trend Bull, Trend Bear, Reversion Bull, Reversion Bear
                                # Handle different signal_type_param values and ensure we check what's actually in the data
                                valid_signal_types_in_data = regime_trades_df['signal_type'].dropna().unique()
                                
                                # Determine which signal types to check
                                if signal_type_param in ['both', 'both_no_confirmation']:
                                    signal_types = ['trend', 'reversion']
                                elif signal_type_param == 'trend':
                                    signal_types = ['trend']
                                elif signal_type_param == 'reversion':
                                    signal_types = ['reversion']
                                else:
                                    # Fallback: use what's actually in the data
                                    signal_types = [str(s).lower() for s in valid_signal_types_in_data if pd.notna(s)]
                                
                                regimes = ['Bull', 'Bear']
                                
                                for signal_type_val in signal_types:
                                    for regime in regimes:
                                        # Filter trades by signal type and regime
                                        # Handle case-insensitive matching and handle NaN/None values
                                        signal_match = regime_trades_df['signal_type'].astype(str).str.lower() == signal_type_val.lower()
                                        regime_match = regime_trades_df['market_regime'].astype(str).str.lower() == regime.lower()
                                        combined_trades_filtered = regime_trades_df[signal_match & regime_match]
                                        
                                        if len(combined_trades_filtered) > 0:
                                            wins = combined_trades_filtered[combined_trades_filtered['pnl_dollars'] > 0]
                                            losses = combined_trades_filtered[combined_trades_filtered['pnl_dollars'] <= 0]
                                    
                                            performance_list.append({
                                        'Signal Type': signal_type_val.title(),
                                                'Market Regime': regime,
                                                'Total PnL ($)': combined_trades_filtered['pnl_dollars'].sum(),
                                                'Avg PnL ($)': combined_trades_filtered['pnl_dollars'].mean(),
                                                'Total Trades': len(combined_trades_filtered),
                                        'Winning Trades': len(wins),
                                        'Losing Trades': len(losses),
                                                'Win Rate (%)': (len(wins) / len(combined_trades_filtered) * 100) if len(combined_trades_filtered) > 0 else 0,
                                        'Avg Win ($)': wins['pnl_dollars'].mean() if len(wins) > 0 else 0,
                                        'Avg Loss ($)': losses['pnl_dollars'].mean() if len(losses) > 0 else 0,
                                    })
                                
                                if performance_list:
                                    performance_df = pd.DataFrame(performance_list)
                                    # Sort: Trend Bull, Trend Bear, Reversion Bull, Reversion Bear
                                    performance_df['sort_key'] = performance_df['Signal Type'].apply(lambda x: 1 if x == 'Trend' else 2) + \
                                                                 performance_df['Market Regime'].apply(lambda x: 0 if x == 'Bull' else 1)
                                    performance_df = performance_df.sort_values('sort_key').drop(columns=['sort_key'])
                                    
                                    # Create styled dataframe with color coding
                                    def style_row(row):
                                        signal = row['Signal Type']
                                        if 'Trend' in signal:
                                            return ['background-color: rgba(59, 130, 246, 0.15)'] * len(row)
                                        elif 'Reversion' in signal:
                                            return ['background-color: rgba(16, 185, 129, 0.15)'] * len(row)
                                        return [''] * len(row)
                                    
                                    styled_df = performance_df.style.apply(style_row, axis=1).format({
                                    'Total PnL ($)': "${:,.2f}",
                                    'Avg PnL ($)': "${:,.2f}",
                                    'Total Trades': "{:,.0f}",
                                    'Winning Trades': "{:,.0f}",
                                    'Losing Trades': "{:,.0f}",
                                    'Win Rate (%)': "{:.1f}%",
                                    'Avg Win ($)': "${:,.2f}",
                                    'Avg Loss ($)': "${:,.2f}"
                                    })
                                    
                                    st.dataframe(styled_df, use_container_width=True, hide_index=True)
                                else:
                                    st.info("No trades found for the selected signal type and market regime combinations.")
                
                # 2. Current Signal Check (only for individual tickers, not combined portfolio) - MOVED AFTER REGIME
                if ticker_name != 'PORTFOLIO (Combined)' and ticker_name in selected_tickers:
                    st.subheader("🔍 Current Signal Analysis")
                    ticker_df = tech_data.get(ticker_name)
                    if ticker_df is not None and not ticker_df.empty:
                        latest = ticker_df.iloc[-1]
                        current_trend = latest.get('Trend_Score')
                        current_reversion = latest.get('Reversion_Score')
                        current_price = latest.get('close')
                        
                        # Check if current signals match entry conditions based on signal_type_param
                        detected_signals = []
                        
                        # Check reversion signals
                        if signal_type_param in ['reversion', 'both', 'both_no_confirmation']:
                            reversion_ok = False
                            trend_filter_ok = True
                            
                            # Check trend filter (if enabled for reversion signals)
                            if use_trend_filter and signal_type_param == 'reversion':
                                trend_filter_ok = (pd.notna(current_trend) and trend_min <= current_trend <= trend_max)
                            elif signal_type_param == 'both':
                                # In "both (with confirmation)" mode, reversion signals need trend confirmation
                                trend_filter_ok = (pd.notna(current_trend) and trend_min <= current_trend <= trend_max)
                            elif signal_type_param == 'both_no_confirmation':
                                # In "both (no confirmation)" mode, no filter needed
                                trend_filter_ok = True
                            else:
                                trend_filter_ok = True
                            
                            if pd.notna(current_reversion) and trend_filter_ok:
                                if trade_direction == 'calls' and current_reversion >= entry_reversion_min:
                                    detected_signals.append("CALL (Reversion Signal - Oversold)")
                                    reversion_ok = True
                                elif trade_direction == 'puts' and current_reversion <= -entry_reversion_min:
                                    detected_signals.append("PUT (Reversion Signal - Overbought)")
                                    reversion_ok = True
                                elif trade_direction == 'both':
                                    if current_reversion >= entry_reversion_min:
                                        detected_signals.append("CALL (Reversion Signal - Oversold)")
                                        reversion_ok = True
                                    elif current_reversion <= -entry_reversion_min:
                                        detected_signals.append("PUT (Reversion Signal - Overbought)")
                                        reversion_ok = True
                        
                        # Check trend signals
                        if signal_type_param in ['trend', 'both', 'both_no_confirmation']:
                            trend_ok = False
                            reversion_filter_ok = True
                            
                            # Check reversion filter (if needed for trend signals)
                            if signal_type_param == 'both':
                                # For "both (with confirmation)" mode, reversion must be in valid range
                                if trade_direction == 'calls':
                                    reversion_filter_ok = pd.notna(current_reversion) and current_reversion >= -4
                                elif trade_direction == 'puts':
                                    reversion_filter_ok = pd.notna(current_reversion) and current_reversion <= 4
                                else:
                                    reversion_filter_ok = pd.notna(current_reversion) and abs(current_reversion) >= 0
                            elif signal_type_param == 'both_no_confirmation':
                                # For "both (no confirmation)" mode, no filter needed
                                reversion_filter_ok = True
                            else:
                                reversion_filter_ok = True
                            
                            if pd.notna(current_trend) and reversion_filter_ok:
                                # For "both (with confirmation)" mode, check if trend is in filter range
                                if signal_type_param == 'both':
                                    if use_trend_filter:
                                        trend_ok_range = trend_min <= current_trend <= trend_max
                                    else:
                                        # No trend filter - any trend score is acceptable
                                        trend_ok_range = True
                                else:
                                    trend_ok_range = trend_min <= current_trend <= trend_max
                                
                                if signal_type_param == 'trend':
                                    # For trend-only, check entry threshold
                                    if trade_direction == 'calls' and current_trend >= trend_entry_min:
                                        detected_signals.append("CALL (Trend Signal - Bullish)")
                                        trend_ok = True
                                    elif trade_direction == 'puts' and current_trend <= -trend_entry_min:
                                        detected_signals.append("PUT (Trend Signal - Bearish)")
                                        trend_ok = True
                                    elif trade_direction == 'both':
                                        if current_trend >= trend_entry_min:
                                            detected_signals.append("CALL (Trend Signal - Bullish)")
                                            trend_ok = True
                                        elif current_trend <= -trend_entry_min:
                                            detected_signals.append("PUT (Trend Signal - Bearish)")
                                            trend_ok = True
                                elif signal_type_param == 'both':
                                    # For both mode, trend signals need to meet entry threshold AND be in range
                                    # The range check ensures trend is not filtered out, but entry threshold determines signal
                                    if trend_ok_range:
                                        if trade_direction == 'calls' and current_trend >= trend_entry_min:
                                            detected_signals.append("CALL (Trend Signal - Bullish)")
                                            trend_ok = True
                                        elif trade_direction == 'puts' and current_trend <= -trend_entry_min:
                                            detected_signals.append("PUT (Trend Signal - Bearish)")
                                            trend_ok = True
                                    elif trade_direction == 'both':
                                        if current_trend >= trend_entry_min:
                                            detected_signals.append("CALL (Trend Signal - Bullish)")
                                            trend_ok = True
                                        elif current_trend <= -trend_entry_min:
                                            detected_signals.append("PUT (Trend Signal - Bearish)")
                                            trend_ok = True
                        
                        # Display results
                        if detected_signals:
                            st.success(f"✅ **Current Signal(s) Detected:** {', '.join(detected_signals)}")
                            st.write(f"**Current Scores:** Trend = {int(current_trend) if pd.notna(current_trend) else 'N/A'}, Reversion = {int(current_reversion) if pd.notna(current_reversion) else 'N/A'}")
                            st.write(f"**Current Price:** ${current_price:.2f}" if pd.notna(current_price) else "**Current Price:** N/A")
                            
                            # Explain "both" mode behavior
                            if signal_type_param == 'both' and len(detected_signals) > 1:
                                # Check if there are both signal types
                                has_reversion = any('Reversion Signal' in s for s in detected_signals)
                                has_trend = any('Trend Signal' in s for s in detected_signals)
                                if has_reversion and has_trend:
                                    st.info(f"🎯 **Backtester Behavior:** In 'Both' mode, the backtester tests **both signal types independently**. Reversion and trend signals are run as separate backtests (each with half the portfolio), then results are combined. Both signal types will be included in the performance tables and charts.")
                            
                            st.caption("ℹ️ **Note:** These signals show what would trigger a trade *right now* based on current scores. Actual trades only appear in 'Trade Details Table' if they were executed during the backtest period. If the backtest end date doesn't include today, or if a trade hasn't been executed yet, you may see signals but no corresponding trades.")
                        else:
                            st.info("⚠️ **No Current Signal** - Current scores do not meet entry criteria based on selected signal type.")
                            reasons = []
                            if signal_type_param in ['reversion', 'both']:
                                if not pd.notna(current_reversion):
                                    reasons.append(f"Reversion Score is N/A")
                                elif use_trend_filter and not trend_filter_ok:
                                    reasons.append(f"Trend Score ({int(current_trend) if pd.notna(current_trend) else 'N/A'}) outside filter range [{trend_min}, {trend_max}]")
                                elif not reversion_ok:
                                    if trade_direction == 'calls':
                                        reasons.append(f"Reversion Score ({int(current_reversion) if pd.notna(current_reversion) else 'N/A'}) < {entry_reversion_min} (need ≥ {entry_reversion_min})")
                                    elif trade_direction == 'puts':
                                        reasons.append(f"Reversion Score ({int(current_reversion) if pd.notna(current_reversion) else 'N/A'}) > -{entry_reversion_min} (need ≤ -{entry_reversion_min})")
                                    else:
                                        reasons.append(f"|Reversion Score| ({abs(int(current_reversion)) if pd.notna(current_reversion) else 'N/A'}) < {entry_reversion_min} (need ≥ {entry_reversion_min})")
                            
                            if signal_type_param in ['trend', 'both', 'both_no_confirmation']:
                                if not pd.notna(current_trend):
                                    reasons.append(f"Trend Score is N/A")
                                elif signal_type_param == 'both' and not reversion_filter_ok:
                                    reasons.append(f"Reversion Score ({int(current_reversion) if pd.notna(current_reversion) else 'N/A'}) outside confirmation range")
                                elif not trend_ok:
                                    if trade_direction == 'calls':
                                        reasons.append(f"Trend Score ({int(current_trend) if pd.notna(current_trend) else 'N/A'}) < {trend_entry_min} (need ≥ {trend_entry_min})")
                                    elif trade_direction == 'puts':
                                        reasons.append(f"Trend Score ({int(current_trend) if pd.notna(current_trend) else 'N/A'}) > -{trend_entry_min} (need ≤ -{trend_entry_min})")
                                    else:
                                        reasons.append(f"|Trend Score| ({abs(int(current_trend)) if pd.notna(current_trend) else 'N/A'}) < {trend_entry_min} (need ≥ {trend_entry_min})")
                            
                            if reasons:
                                st.write("  - " + "\n  - ".join(reasons))
                            else:
                                st.write("  - Check signal type and entry threshold settings.")
                
                # 3. Performance Metrics
                if any(m in selected_metrics for m in ['Final Portfolio Value', 'Total Return', 'Annualized Return (CAGR)', 'Max Drawdown', 'Win Rate', 'Total Trades']):
                    st.subheader("📈 Performance Metrics")
                    metrics = result.get('Metrics', result.get('metrics', {}))
                    if not metrics:
                        # Extract metrics from result if not in Metrics dict
                        metrics = {
                            'Final Portfolio Value': result.get('Final Portfolio Value', 0),
                            'Annualized Return (CAGR)': result.get('Annualized Return (CAGR)', 0),
                            'Max Drawdown': result.get('Max Drawdown', 0),
                            'Win Rate': result.get('Win Rate', 0),
                            'Total Trades': len(result.get('Trades', []))
                        }
                    
                    # Helper function to safely convert to float
                    def safe_float(val, default=0.0):
                        """Convert value to float, handling strings and None"""
                        if val is None:
                            return default
                        if isinstance(val, str):
                            # Try to parse string (remove % sign if present and convert percentage)
                            try:
                                cleaned = val.replace('%', '').strip()
                                num_val = float(cleaned)
                                # If original had %, it was a percentage (e.g., "10%" = 0.10), so divide by 100
                                # But if it's already a decimal like "0.10", don't divide
                                if '%' in val and num_val > 1:
                                    num_val = num_val / 100
                                return num_val
                            except (ValueError, AttributeError):
                                return default
                        try:
                            return float(val)
                        except (ValueError, TypeError):
                            return default
                    
                    # Display metrics in columns
                    metric_cols = []
                    if 'Final Portfolio Value' in selected_metrics:
                        final_val = safe_float(metrics.get('Final Portfolio Value', 0))
                        metric_cols.append(("Final Portfolio Value", f"${final_val:,.2f}"))
                    if 'Total Return' in selected_metrics:
                        final_val = safe_float(metrics.get('Final Portfolio Value', initial_portfolio))
                        total_return_pct = ((final_val - initial_portfolio) / initial_portfolio * 100) if initial_portfolio > 0 else 0
                        metric_cols.append(("Total Return", f"{total_return_pct:+.2f}%"))
                    if 'Annualized Return (CAGR)' in selected_metrics:
                        cagr_val = safe_float(metrics.get('Annualized Return (CAGR)', 0))
                        metric_cols.append(("Annualized Return (CAGR)", f"{cagr_val:.2%}"))
                    if 'Max Drawdown' in selected_metrics:
                        dd_val = safe_float(metrics.get('Max Drawdown', 0))
                        metric_cols.append(("Max Drawdown", f"{dd_val:.2%}"))
                    if 'Win Rate' in selected_metrics:
                        wr_val = safe_float(metrics.get('Win Rate', 0))
                        metric_cols.append(("Win Rate", f"{wr_val:.2%}"))
                    if 'Total Trades' in selected_metrics:
                        trades_val = safe_float(metrics.get('Total Trades', 0))
                        metric_cols.append(("Total Trades", f"{int(trades_val):,}"))
                    
                    if metric_cols:
                        num_cols = min(len(metric_cols), 4)
                        cols = st.columns(num_cols)
                        for idx, (label, value) in enumerate(metric_cols):
                            with cols[idx % num_cols]:
                                st.metric(label, value)
                    
                    # Calculate and display Strategy vs Market correlation (portfolio returns vs SPY returns)
                    try:
                        portfolio_history = result.get('Portfolio History', result.get('portfolio_history', []))
                        trades_data = result.get('Trades', result.get('trades', []))
                        
                        # Get SPY data for market returns
                        spy_data = None
                        try:
                            if tech_data and isinstance(tech_data, dict) and 'SPY' in tech_data:
                                potential_spy = tech_data.get('SPY')
                                # CRITICAL: Validate SPY data is actually a DataFrame (not None)
                                if (potential_spy is not None and 
                                    isinstance(potential_spy, pd.DataFrame) and 
                                    not potential_spy.empty):
                                    spy_data = potential_spy.copy()
                        except Exception as e:
                            safe_print(f"Warning: Could not retrieve SPY for correlation calculation: {e}")
                            spy_data = None
                        
                        if (spy_data is not None and 
                            isinstance(spy_data, pd.DataFrame) and 
                            not spy_data.empty and 
                            portfolio_history is not None):
                            # Convert portfolio history using helper function (reduces code duplication)
                            port_history_list = normalize_portfolio_history(portfolio_history)
                            if port_history_list:
                                port_df = pd.DataFrame(port_history_list)
                            else:
                                port_df = None
                            
                            if port_df is not None and 'date' in port_df.columns and 'value' in port_df.columns and len(port_df) > 50:
                                port_df['date'] = pd.to_datetime(port_df['date'])
                                port_df = port_df.sort_values('date').set_index('date')
                                
                                # Get initial portfolio value
                                initial_val = result.get('Initial Portfolio Value', initial_portfolio)
                                
                                # Calculate portfolio returns (daily % change)
                                port_df['portfolio_return'] = port_df['value'].pct_change()
                                
                                # Filter SPY data to match portfolio date range - with proper checks
                                spy_filtered = None
                                try:
                                    if (spy_data is not None and 
                                        isinstance(spy_data, pd.DataFrame) and 
                                        not spy_data.empty and
                                        hasattr(spy_data, 'index') and
                                        spy_data.index is not None and
                                        'close' in spy_data.columns):
                                        spy_filtered = spy_data[spy_data.index >= port_df.index.min()].copy()
                                        if spy_filtered is not None and isinstance(spy_filtered, pd.DataFrame) and not spy_filtered.empty:
                                            spy_filtered = spy_filtered[spy_filtered.index <= port_df.index.max()]
                                            if spy_filtered is not None and not spy_filtered.empty and 'close' in spy_filtered.columns:
                                                spy_filtered['market_return'] = spy_filtered['close'].pct_change()
                                            else:
                                                spy_filtered = None
                                        else:
                                            spy_filtered = None
                                except Exception as e:
                                    safe_print(f"Error processing SPY data: {e}")
                                    spy_filtered = None
                                
                                if spy_filtered is not None and not spy_filtered.empty and 'market_return' in spy_filtered.columns:
                                    # Merge on date
                                    merged = port_df[['portfolio_return']].join(spy_filtered[['market_return']], how='inner')
                                    merged = merged.dropna()
                                    
                                    if len(merged) > 50:
                                        # Calculate overall correlation (combined strategy)
                                        overall_corr = merged['portfolio_return'].corr(merged['market_return'])
                                        
                                        if pd.notna(overall_corr):
                                            st.subheader("📊 Strategy vs Market Correlation")
                                            st.caption("Correlation of portfolio returns vs SPY (market) returns")
                                            
                                            correlations = {}
                                            correlations['Combined Strategy'] = overall_corr
                                            
                                            # If "both_no_confirmation" mode, calculate separate correlations
                                            if signal_type_param == 'both_no_confirmation':
                                                # Check if we have separate trend/reversion histories
                                                trend_history = result.get('_trend_history', [])
                                                reversion_history = result.get('_reversion_history', [])
                                                
                                                # Calculate trend correlation
                                                if trend_history:
                                                    trend_history_list = normalize_portfolio_history(trend_history)
                                                    if trend_history_list:
                                                        trend_df = pd.DataFrame(trend_history_list)
                                                    else:
                                                        trend_df = None
                                                    
                                                    if trend_df is not None and 'date' in trend_df.columns and 'value' in trend_df.columns:
                                                        trend_df['date'] = pd.to_datetime(trend_df['date'])
                                                        trend_df = trend_df.sort_values('date').set_index('date')
                                                        trend_df['trend_return'] = trend_df['value'].pct_change()
                                                        trend_merged = trend_df[['trend_return']].join(spy_filtered[['market_return']], how='inner').dropna()
                                                        if len(trend_merged) > 50:
                                                            trend_corr = trend_merged['trend_return'].corr(trend_merged['market_return'])
                                                            if pd.notna(trend_corr):
                                                                correlations['Trend Strategy'] = trend_corr
                                                
                                                # Calculate reversion correlation
                                                if reversion_history:
                                                    rev_history_list = normalize_portfolio_history(reversion_history)
                                                    if rev_history_list:
                                                        rev_df = pd.DataFrame(rev_history_list)
                                                    else:
                                                        rev_df = None
                                                    
                                                    if rev_df is not None and 'date' in rev_df.columns and 'value' in rev_df.columns:
                                                        rev_df['date'] = pd.to_datetime(rev_df['date'])
                                                        rev_df = rev_df.sort_values('date').set_index('date')
                                                        rev_df['reversion_return'] = rev_df['value'].pct_change()
                                                        rev_merged = rev_df[['reversion_return']].join(spy_filtered[['market_return']], how='inner').dropna()
                                                        if len(rev_merged) > 50:
                                                            rev_corr = rev_merged['reversion_return'].corr(rev_merged['market_return'])
                                                            if pd.notna(rev_corr):
                                                                correlations['Reversion Strategy'] = rev_corr
                                            elif signal_type_param == 'trend':
                                                correlations['Trend Strategy'] = overall_corr
                                            elif signal_type_param == 'reversion':
                                                correlations['Reversion Strategy'] = overall_corr
                                            
                                            # Calculate monthly correlations
                                            monthly_correlations = {}
                                            
                                            # Resample to monthly returns
                                            port_monthly = port_df['value'].resample('M').last().pct_change()
                                            spy_monthly = spy_filtered['close'].resample('M').last().pct_change()
                                            monthly_merged = pd.DataFrame({
                                                'portfolio_return': port_monthly,
                                                'market_return': spy_monthly
                                            }).dropna()
                                            
                                            if len(monthly_merged) > 12:  # Need at least 12 months
                                                monthly_overall_corr = monthly_merged['portfolio_return'].corr(monthly_merged['market_return'])
                                                if pd.notna(monthly_overall_corr):
                                                    monthly_correlations['Combined Strategy'] = monthly_overall_corr
                                                    
                                                    # If "both_no_confirmation" mode, calculate separate monthly correlations
                                                    if signal_type_param == 'both_no_confirmation':
                                                        trend_history = result.get('_trend_history', [])
                                                        reversion_history = result.get('_reversion_history', [])
                                                        
                                                        # Trend monthly correlation
                                                        if trend_history:
                                                            trend_history_list = normalize_portfolio_history(trend_history)
                                                            if trend_history_list:
                                                                trend_df = pd.DataFrame(trend_history_list)
                                                            else:
                                                                trend_df = None
                                                            
                                                            if trend_df is not None and 'date' in trend_df.columns and 'value' in trend_df.columns:
                                                                trend_df['date'] = pd.to_datetime(trend_df['date'])
                                                                trend_df = trend_df.sort_values('date').set_index('date')
                                                                trend_monthly = trend_df['value'].resample('M').last().pct_change()
                                                                trend_monthly_merged = pd.DataFrame({
                                                                    'trend_return': trend_monthly,
                                                                    'market_return': spy_monthly
                                                                }).dropna()
                                                                if len(trend_monthly_merged) > 12:
                                                                    trend_monthly_corr = trend_monthly_merged['trend_return'].corr(trend_monthly_merged['market_return'])
                                                                    if pd.notna(trend_monthly_corr):
                                                                        monthly_correlations['Trend Strategy'] = trend_monthly_corr
                                                        
                                                        # Reversion monthly correlation
                                                        if reversion_history:
                                                            rev_history_list = normalize_portfolio_history(reversion_history)
                                                            if rev_history_list:
                                                                rev_df = pd.DataFrame(rev_history_list)
                                                            else:
                                                                rev_df = None
                                                            
                                                            if rev_df is not None and 'date' in rev_df.columns and 'value' in rev_df.columns:
                                                                rev_df['date'] = pd.to_datetime(rev_df['date'])
                                                                rev_df = rev_df.sort_values('date').set_index('date')
                                                                rev_monthly = rev_df['value'].resample('M').last().pct_change()
                                                                rev_monthly_merged = pd.DataFrame({
                                                                    'reversion_return': rev_monthly,
                                                                    'market_return': spy_monthly
                                                                }).dropna()
                                                                if len(rev_monthly_merged) > 12:
                                                                    rev_monthly_corr = rev_monthly_merged['reversion_return'].corr(rev_monthly_merged['market_return'])
                                                                    if pd.notna(rev_monthly_corr):
                                                                        monthly_correlations['Reversion Strategy'] = rev_monthly_corr
                                                    elif signal_type_param == 'trend':
                                                        monthly_correlations['Trend Strategy'] = monthly_overall_corr
                                                    elif signal_type_param == 'reversion':
                                                        monthly_correlations['Reversion Strategy'] = monthly_overall_corr
                                            
                                            # Display daily correlations
                                            st.markdown("**Daily Return Correlations:**")
                                            corr_cols = st.columns(len(correlations))
                                            for idx, (strategy_name, corr_val) in enumerate(correlations.items()):
                                                with corr_cols[idx % len(correlations)]:
                                                    abs_corr = abs(corr_val)
                                                    if abs_corr < 0.2:
                                                        corr_label = "Very Low"
                                                    elif abs_corr < 0.4:
                                                        corr_label = "Low"
                                                    elif abs_corr < 0.6:
                                                        corr_label = "Moderate"
                                                    else:
                                                        corr_label = "High"
                                                    
                                                    st.metric(
                                                        strategy_name,
                                                        f"{corr_val:.3f}",
                                                        help=f"{corr_label} daily correlation with market (SPY). {'Positive' if corr_val > 0 else 'Negative'} correlation means strategy moves {'with' if corr_val > 0 else 'against'} the market."
                                                    )
                                            
                                            # Display monthly correlations if available
                                            if monthly_correlations:
                                                st.markdown("**Monthly Return Correlations:**")
                                                monthly_corr_cols = st.columns(len(monthly_correlations))
                                                for idx, (strategy_name, corr_val) in enumerate(monthly_correlations.items()):
                                                    with monthly_corr_cols[idx % len(monthly_correlations)]:
                                                        abs_corr = abs(corr_val)
                                                        if abs_corr < 0.2:
                                                            corr_label = "Very Low"
                                                        elif abs_corr < 0.4:
                                                            corr_label = "Low"
                                                        elif abs_corr < 0.6:
                                                            corr_label = "Moderate"
                                                        else:
                                                            corr_label = "High"
                                                        
                                                        st.metric(
                                                            strategy_name,
                                                            f"{corr_val:.3f}",
                                                            help=f"{corr_label} monthly correlation with market (SPY). Monthly correlations smooth out daily volatility and may better reflect the overall strategy-market relationship."
                                                        )
                                            
                                            st.caption(f"Daily correlation analysis based on {len(merged):,} daily returns. "
                                                      f"{f'Monthly correlation based on {len(monthly_merged)} monthly returns. ' if monthly_correlations else ''}"
                                                      f"Values range from -1 (perfect inverse) to +1 (perfect alignment) with 0 being uncorrelated.")
                    except Exception as e:
                        safe_print(f"⚠️ Error calculating market correlation: {e}")
                        pass
                
                # Drawdown Optimization Suggestions
                max_dd = result.get('metrics', {}).get('Max Drawdown', 0)
                if max_dd > 0.50:  # If drawdown > 50%
                    st.subheader("💡 Drawdown Optimization Suggestions")
                    with st.expander("🔍 Tips to Reduce Drawdown", expanded=True):
                        suggestions = []
                        if max_dd > 0.80:
                            suggestions.append("**⚠️ CRITICAL:** Drawdown >80% indicates significant risk. Consider:")
                        else:
                            suggestions.append("**High drawdown detected.** Consider optimizing:")
                        
                        suggestions.extend([
                            f"1. **Reduce Position Size:** Lower `Initial Entry %` and `Max Position %` (currently {initial_position_pct:.1%} / {max_position_pct:.1%}) to limit exposure per trade",
                            f"2. **Tighter Stop Loss:** Reduce `Stop Loss %` (currently {abs(stop_loss):.1%}) to exit losing positions sooner",
                            f"3. **Trailing Stop:** Optimize `Trailing Stop %` (currently {trailing_stop:.1%}) to protect profits",
                            "4. **Signal Confirmation:** Use 'Both (with confirmation)' mode to require both trend and reversion alignment",
                            "5. **Limit Double Downs:** Disable or reduce `Double Down` multiplier to avoid over-leveraging",
                            "6. **Market Regime Filter:** Review 'Performance by Market Regime' to see if drawdowns occur in specific conditions",
                            f"7. **Shorter Expiration:** Reduce `Expiration Days` (currently {expiration_days} days) to limit time at risk per trade"
                        ])
                        
                        st.markdown("\n".join(suggestions))
                
                # Daily Performance Download
                st.subheader("📥 Download Daily Performance")
                portfolio_history = result.get('Portfolio History', result.get('portfolio_history', []))
                
                # Check if portfolio_history is valid (handle DataFrame, list, or None)
                has_history = False
                if portfolio_history is not None:
                    if isinstance(portfolio_history, pd.DataFrame):
                        has_history = not portfolio_history.empty
                    elif isinstance(portfolio_history, list):
                        has_history = len(portfolio_history) > 0
                
                if has_history:
                    # Convert to DataFrame
                    if isinstance(portfolio_history, pd.DataFrame):
                        daily_df = portfolio_history.copy()
                        if isinstance(daily_df.index, pd.DatetimeIndex):
                            daily_df = daily_df.reset_index()
                            if 'index' in daily_df.columns:
                                daily_df = daily_df.rename(columns={'index': 'date'})
                    elif isinstance(portfolio_history, list):
                        daily_df = pd.DataFrame(portfolio_history)
                    else:
                        daily_df = None
                    
                    if daily_df is not None and 'date' in daily_df.columns and 'value' in daily_df.columns:
                        daily_df['date'] = pd.to_datetime(daily_df['date'])
                        daily_df = daily_df.sort_values('date').reset_index(drop=True)
                        
                        # Get initial portfolio value
                        initial_val = result.get('Initial Portfolio Value', initial_portfolio / len(selected_tickers) if len(selected_tickers) > 1 else initial_portfolio)
                        
                        # Calculate running peak (Portfolio Max)
                        daily_df['Portfolio_Max'] = daily_df['value'].cummax()
                        
                        # Calculate daily return
                        daily_df['Daily_Return'] = daily_df['value'].pct_change().fillna(0)
                        
                        # Calculate drawdown (exact formula from CSV)
                        daily_df['Drawdown'] = (daily_df['value'] - daily_df['Portfolio_Max']) / daily_df['Portfolio_Max']
                        
                        # Format for export
                        export_df = daily_df[['date', 'value', 'Portfolio_Max', 'Daily_Return', 'Drawdown']].copy()
                        export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
                        export_df['value'] = export_df['value'].apply(lambda x: f"${x:,.2f}")
                        export_df['Portfolio_Max'] = export_df['Portfolio_Max'].apply(lambda x: f"${x:,.2f}")
                        export_df['Daily_Return'] = export_df['Daily_Return'].apply(lambda x: f"{x:.4%}")
                        export_df['Drawdown'] = export_df['Drawdown'].apply(lambda x: f"{x:.2%}")
                        
                        # Rename columns for clarity
                        export_df.columns = ['Date', 'Portfolio_Value', 'Portfolio_Max', 'Daily_Return', 'Drawdown']
                        
                        csv_daily = export_df.to_csv(index=False)
                        
                        st.download_button(
                            label=f"📥 Download Daily Performance ({ticker_name})",
                            data=csv_daily,
                            file_name=f"daily_performance_{ticker_name}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            key=f"download_daily_{ticker_name}"
                        )
                        st.caption(f"Daily portfolio performance from {export_df['Date'].iloc[0]} to {export_df['Date'].iloc[-1]}")
                
                st.subheader("📥 Download Monthly Returns")
                portfolio_history = result.get('Portfolio History', result.get('portfolio_history', []))
                trades_data = result.get('Trades', result.get('trades', []))
                
                # Check if portfolio_history is valid (handle DataFrame, list, or None)
                has_history = False
                if portfolio_history is not None:
                    if isinstance(portfolio_history, pd.DataFrame):
                        has_history = not portfolio_history.empty
                    elif isinstance(portfolio_history, list):
                        has_history = len(portfolio_history) > 0
                    else:
                        has_history = False
                
                if has_history:
                    # Convert portfolio history using helper function
                    port_history_list = normalize_portfolio_history(portfolio_history)
                    if port_history_list:
                        hist_df = pd.DataFrame(port_history_list)
                    else:
                        hist_df = pd.DataFrame()
                    
                    if 'date' in hist_df.columns and 'value' in hist_df.columns:
                        hist_df['date'] = pd.to_datetime(hist_df['date'])
                        hist_df = hist_df.sort_values('date')
                        
                        # Get initial portfolio value (first value or from result)
                        initial_value = result.get('Initial Portfolio Value', initial_portfolio / len(selected_tickers) if len(selected_tickers) > 1 else initial_portfolio)
                        
                        # Calculate monthly returns with Portfolio Value, Max, and Drawdown
                        hist_df['year_month'] = hist_df['date'].dt.to_period('M')
                        
                        # Calculate Portfolio Max (running maximum) for the entire history
                        hist_df['Portfolio_Max'] = hist_df['value'].cummax()
                        
                        # Calculate Drawdown for each point
                        hist_df['Drawdown'] = (hist_df['value'] - hist_df['Portfolio_Max']) / hist_df['Portfolio_Max']
                        hist_df.loc[hist_df['value'] == hist_df['Portfolio_Max'], 'Drawdown'] = 0.0
                        
                        # Group by month and get first and last values
                        monthly_data = []
                        for month, month_data in hist_df.groupby('year_month'):
                            first_value = month_data.iloc[0]['value']
                            last_value = month_data.iloc[-1]['value']
                            monthly_return = ((last_value - first_value) / first_value * 100) if first_value > 0 else 0.0
                            
                            # Get Portfolio Max and Drawdown for the last day of the month
                            last_max = month_data.iloc[-1]['Portfolio_Max']
                            last_drawdown = month_data.iloc[-1]['Drawdown']
                            
                            monthly_data.append({
                                'Date': month.to_timestamp().strftime('%Y-%m'),
                                ticker_name: f"{monthly_return:.3f}%",
                                'Portfolio_Value': f"${last_value:,.2f}",
                                'Portfolio_Max': f"${last_max:,.2f}",
                                'Drawdown': f"{last_drawdown:.2%}"
                            })
                        
                        monthly_df = pd.DataFrame(monthly_data)
                        
                        if len(selected_tickers) == 1:
                            # Single stock - simple format
                            download_df = monthly_df.copy()
                        else:
                            # Multiple stocks - combine all
                            # This is already being handled per ticker, so we'll aggregate later
                            download_df = monthly_df.copy()
                        
                        # Create CSV string
                        csv = download_df.to_csv(index=False)
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.download_button(
                                label=f"📥 Download Monthly Returns ({ticker_name})",
                                data=csv,
                                file_name=f"monthly_returns_{ticker_name}.csv",
                                mime="text/csv",
                                key=f"download_monthly_{ticker_name}"
                            )
                        with col2:
                            st.caption(f"Monthly returns from {monthly_df['Date'].iloc[0]} to {monthly_df['Date'].iloc[-1]}")
                        
                        # Show trades in tabs if available
                        has_trades = False
                        if trades_data is not None:
                            if isinstance(trades_data, pd.DataFrame):
                                has_trades = not trades_data.empty
                            elif isinstance(trades_data, list):
                                has_trades = len(trades_data) > 0
                            else:
                                has_trades = False
                        
                        if has_trades:
                            with st.expander(f"📋 View All Trades for {ticker_name}", expanded=False):
                                if isinstance(trades_data, pd.DataFrame):
                                    trades_display = trades_data.copy()
                                elif isinstance(trades_data, list):
                                    trades_display = pd.DataFrame(trades_data)
                                else:
                                    trades_display = pd.DataFrame()
                                
                                if not trades_display.empty:
                                    st.dataframe(trades_display, use_container_width=True)
                
                # 4. Portfolio Value Chart (conditional)
                # (Keep existing chart code here)
                if 'Portfolio History Chart' in selected_metrics:
                    portfolio_history = result.get('Portfolio History', result.get('portfolio_history', []))
                    if portfolio_history is not None and (isinstance(portfolio_history, list) and len(portfolio_history) > 0 or (isinstance(portfolio_history, pd.DataFrame) and not portfolio_history.empty)):
                        if isinstance(portfolio_history, list):
                            hist_df = pd.DataFrame(portfolio_history)
                            if 'date' in hist_df.columns:
                                hist_df['date'] = pd.to_datetime(hist_df['date'])
                                hist_df = hist_df.set_index('date').sort_index()
                            else:
                                hist_df = hist_df.set_index(hist_df.columns[0]).sort_index()
                        else:
                            hist_df = portfolio_history
                        
                        if not hist_df.empty:
                            st.subheader("📈 Portfolio Value Over Time")
                            fig = go.Figure()
                            ticker_portfolio = initial_portfolio / len(selected_tickers) if len(selected_tickers) > 1 else initial_portfolio
                            
                            # If "both" mode, show separate trend and reversion lines
                            if signal_type_param == 'both_no_confirmation' and result.get('Signal_Type_Test') == 'both_no_confirmation':
                                # Use stored separate histories if available
                                trend_history = result.get('_trend_history', [])
                                reversion_history = result.get('_reversion_history', [])
                                initial_per_signal = ticker_portfolio / 2
                                
                                if trend_history and reversion_history:
                                    # Convert to dataframes
                                    if isinstance(trend_history, list):
                                        trend_df = pd.DataFrame(trend_history)
                                        if 'date' in trend_df.columns:
                                            trend_df['date'] = pd.to_datetime(trend_df['date'])
                                            trend_df = trend_df.set_index('date').sort_index()
                                    else:
                                        trend_df = trend_history
                                    
                                    if isinstance(reversion_history, list):
                                        reversion_df = pd.DataFrame(reversion_history)
                                        if 'date' in reversion_df.columns:
                                            reversion_df['date'] = pd.to_datetime(reversion_df['date'])
                                            reversion_df = reversion_df.set_index('date').sort_index()
                                    else:
                                        reversion_df = reversion_history
                                    
                                    # Get all dates
                                    all_dates = sorted(set(list(trend_df.index) + list(reversion_df.index) + list(hist_df.index)))
                                    
                                    # Forward fill values
                                    trend_values = []
                                    reversion_values = []
                                    last_trend = initial_per_signal
                                    last_reversion = initial_per_signal
                                    
                                    for date in all_dates:
                                        if date in trend_df.index:
                                            if 'value' in trend_df.columns:
                                                last_trend = trend_df.loc[date, 'value']
                                            else:
                                                last_trend = trend_df.loc[date].iloc[0]
                                        if date in reversion_df.index:
                                            if 'value' in reversion_df.columns:
                                                last_reversion = reversion_df.loc[date, 'value']
                                            else:
                                                last_reversion = reversion_df.loc[date].iloc[0]
                                        trend_values.append(last_trend)
                                        reversion_values.append(last_reversion)
                                    
                                    # Add separate lines
                                    fig.add_trace(go.Scatter(
                                        x=all_dates,
                                        y=trend_values,
                                        mode='lines',
                                        name='Trend Signals',
                                        line=dict(color='#3b82f6', width=2),  # Blue for trend
                                        hovertemplate='Trend: $%{y:,.2f}<extra></extra>'
                                    ))
                                    fig.add_trace(go.Scatter(
                                        x=all_dates,
                                        y=reversion_values,
                                        mode='lines',
                                        name='Reversion Signals',
                                        line=dict(color='#10b981', width=2),  # Green for reversion
                                        hovertemplate='Reversion: $%{y:,.2f}<extra></extra>'
                                    ))
                                    # Combined line
                                    combined_values = [t + r for t, r in zip(trend_values, reversion_values)]
                                    fig.add_trace(go.Scatter(
                                        x=all_dates,
                                        y=combined_values,
                                        mode='lines',
                                        name='Combined Portfolio',
                                        line=dict(color='#ffffff', width=1.5, dash='dash'),  # White dashed for combined
                                        hovertemplate='Combined: $%{y:,.2f}<extra></extra>'
                                    ))
                                else:
                                    # Fallback: show combined line
                                    fig.add_trace(go.Scatter(
                                        x=hist_df.index if hasattr(hist_df, 'index') else hist_df.iloc[:, 0],
                                        y=hist_df['value'] if 'value' in hist_df.columns else hist_df.iloc[:, 0],
                                        mode='lines',
                                        name='Combined Portfolio',
                                        line=dict(color='#ffffff', width=2)
                                    ))
                            elif signal_type_param == 'both':
                                # "Both (with confirmation)" mode - single backtest, show single line
                                # Color based on which signal type has more trades if available
                                trades_for_chart = result.get('Trades', result.get('trades', []))
                                trend_count = 0
                                reversion_count = 0
                                
                                # Check if trades_for_chart is valid (handle DataFrame, list, or None)
                                has_trades_for_chart = False
                                if trades_for_chart is not None:
                                    if isinstance(trades_for_chart, pd.DataFrame):
                                        has_trades_for_chart = not trades_for_chart.empty
                                    elif isinstance(trades_for_chart, list):
                                        has_trades_for_chart = len(trades_for_chart) > 0
                                
                                if has_trades_for_chart:
                                    if isinstance(trades_for_chart, pd.DataFrame):
                                        if 'signal_type' in trades_for_chart.columns:
                                            trend_count = len(trades_for_chart[trades_for_chart['signal_type'].str.lower().str.contains('trend', na=False)])
                                            reversion_count = len(trades_for_chart[trades_for_chart['signal_type'].str.lower().str.contains('reversion', na=False)])
                                    elif isinstance(trades_for_chart, list):
                                        for trade in trades_for_chart:
                                            if isinstance(trade, dict):
                                                signal = str(trade.get('signal_type', '')).lower()
                                                if 'trend' in signal:
                                                    trend_count += 1
                                                elif 'reversion' in signal:
                                                    reversion_count += 1
                                
                                # Default to green if both equal or can't determine, but show mixed if both exist
                                if trend_count > 0 and reversion_count > 0:
                                    line_color = '#10b981'  # Green for mixed (both signal types)
                                    line_name = 'Portfolio Value (Mixed Signals)'
                                elif trend_count > reversion_count:
                                    line_color = '#3b82f6'  # Blue for mostly trend
                                    line_name = 'Portfolio Value (Trend)'
                                else:
                                    line_color = '#10b981'  # Green for mostly reversion
                                    line_name = 'Portfolio Value (Reversion)'
                                
                                fig.add_trace(go.Scatter(
                                    x=hist_df.index if hasattr(hist_df, 'index') else hist_df.iloc[:, 0],
                                    y=hist_df['value'] if 'value' in hist_df.columns else hist_df.iloc[:, 0],
                                    mode='lines',
                                    name=line_name,
                                    line=dict(color=line_color, width=2)
                                ))
                            else:
                                # Single signal type - show single line
                                fig.add_trace(go.Scatter(
                                    x=hist_df.index if hasattr(hist_df, 'index') else hist_df.iloc[:, 0],
                                    y=hist_df['value'] if 'value' in hist_df.columns else hist_df.iloc[:, 0],
                                    mode='lines',
                                    name='Portfolio Value',
                                    line=dict(color='#3b82f6' if signal_type_param == 'trend' else '#10b981', width=2)
                                ))
                            
                            fig.add_hline(y=ticker_portfolio, line_dash="dash", line_color="gray", 
                                         annotation_text=f"Initial: ${ticker_portfolio:,.0f}")
                            fig.update_layout(
                                title=f"Portfolio Value Over Time - {ticker_name}",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                hovermode='x unified',
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                # 5. Trade Details Table (conditional)
                if 'Trade Details Table' in selected_metrics:
                    trades = result.get('Trades', result.get('trades', []))
                    if trades is not None and (isinstance(trades, list) and len(trades) > 0 or (isinstance(trades, pd.DataFrame) and not trades.empty)):
                        if isinstance(trades, list):
                            trades_df = pd.DataFrame(trades)
                        else:
                            trades_df = trades
                        
                        if not trades_df.empty:
                            st.subheader("📋 Trade Details")
                            
                            # Ensure signal_type column is visible and move it to front if it exists
                            if 'signal_type' in trades_df.columns:
                                # Reorder columns to put signal_type first (after entry_date if it exists)
                                cols = list(trades_df.columns)
                                if 'entry_date' in cols:
                                    # Put signal_type right after entry_date
                                    cols.remove('signal_type')
                                    entry_date_idx = cols.index('entry_date')
                                    cols.insert(entry_date_idx + 1, 'signal_type')
                                else:
                                    # Put signal_type first
                                    cols.remove('signal_type')
                                    cols.insert(0, 'signal_type')
                                trades_df = trades_df[cols]
                            
                            # Format columns if they exist
                            format_dict = {}
                            for col in ["pnl_dollars", "entry_stock_price", "exit_stock_price", "strike_price", 
                                       "entry_option_price_per_share", "exit_option_price_per_share", "total_cost"]:
                                if col in trades_df.columns:
                                    format_dict[col] = "${:,.2f}"
                            
                            # Color code rows by signal type if signal_type column exists
                            def style_trade_row(row):
                                """Color code rows based on signal_type"""
                                if 'signal_type' in row.index:
                                    signal = str(row['signal_type']).lower()
                                    if 'trend' in signal:
                                        return ['background-color: rgba(59, 130, 246, 0.15)'] * len(row)
                                    elif 'reversion' in signal:
                                        return ['background-color: rgba(16, 185, 129, 0.15)'] * len(row)
                                return [''] * len(row)
                            
                            styled_trades = trades_df.style.apply(style_trade_row, axis=1)
                            if format_dict:
                                styled_trades = styled_trades.format(format_dict)
                            
                            # Add legend explaining colors
                            if 'signal_type' in trades_df.columns:
                                col1, col2 = st.columns([1, 1])
                                with col1:
                                    st.caption("🔵 **Blue rows** = Trend signals")
                                with col2:
                                    st.caption("🟢 **Green rows** = Reversion signals")
                            
                            st.dataframe(
                                styled_trades,
                                use_container_width=True,
                                height=400
                            )
                
                st.markdown("---")
            
            # Add combined monthly returns download for multiple tickers (after all results displayed)
            if len(selected_tickers) > 1 and all_results:
                st.header("📥 Combined Monthly Returns Download")
                st.markdown("Download monthly returns for all selected stocks in a single CSV file.")
                
                # Aggregate monthly returns from all results
                all_monthly_data = {}
                
                for result in all_results:
                    ticker = result.get('Ticker', 'Unknown')
                    if ticker == 'PORTFOLIO (Combined)':
                        continue  # Skip combined portfolio, use individual tickers
                    
                    portfolio_history = result.get('Portfolio History', result.get('portfolio_history', []))
                    
                    # Check if portfolio_history is valid (handle DataFrame, list, or None)
                    has_history = False
                    if portfolio_history is not None:
                        if isinstance(portfolio_history, pd.DataFrame):
                            has_history = not portfolio_history.empty
                        elif isinstance(portfolio_history, list):
                            has_history = len(portfolio_history) > 0
                    
                    if not has_history:
                        continue
                    
                    # Convert to DataFrame
                    if isinstance(portfolio_history, pd.DataFrame):
                        hist_df = portfolio_history.copy()
                        if isinstance(hist_df.index, pd.DatetimeIndex):
                            hist_df = hist_df.reset_index()
                            if 'index' in hist_df.columns:
                                hist_df = hist_df.rename(columns={'index': 'date'})
                    elif isinstance(portfolio_history, list):
                        hist_df = pd.DataFrame(portfolio_history)
                    else:
                        continue
                    
                    if 'date' not in hist_df.columns or 'value' not in hist_df.columns:
                        continue
                    
                    hist_df['date'] = pd.to_datetime(hist_df['date'])
                    hist_df = hist_df.sort_values('date')
                    hist_df['year_month'] = hist_df['date'].dt.to_period('M')
                    
                    # Calculate Portfolio Max (running maximum) for the entire history
                    hist_df['Portfolio_Max'] = hist_df['value'].cummax()
                    
                    # Calculate Drawdown for each point
                    hist_df['Drawdown'] = (hist_df['value'] - hist_df['Portfolio_Max']) / hist_df['Portfolio_Max']
                    hist_df.loc[hist_df['value'] == hist_df['Portfolio_Max'], 'Drawdown'] = 0.0
                    
                    # Calculate monthly returns for this ticker
                    for month, month_data in hist_df.groupby('year_month'):
                        first_value = month_data.iloc[0]['value']
                        last_value = month_data.iloc[-1]['value']
                        monthly_return = ((last_value - first_value) / first_value * 100) if first_value > 0 else 0.0
                        
                        # Get Portfolio Max and Drawdown for the last day of the month
                        last_max = month_data.iloc[-1]['Portfolio_Max']
                        last_drawdown = month_data.iloc[-1]['Drawdown']
                        
                        month_str = month.to_timestamp().strftime('%Y-%m')
                        if month_str not in all_monthly_data:
                            all_monthly_data[month_str] = {}
                        
                        # Store return, value, max, and drawdown for this ticker
                        all_monthly_data[month_str][f'{ticker}_Return'] = f"{monthly_return:.3f}%"
                        all_monthly_data[month_str][f'{ticker}_Portfolio_Value'] = f"${last_value:,.2f}"
                        all_monthly_data[month_str][f'{ticker}_Portfolio_Max'] = f"${last_max:,.2f}"
                        all_monthly_data[month_str][f'{ticker}_Drawdown'] = f"{last_drawdown:.2%}"
                
                if all_monthly_data:
                    # Convert to DataFrame
                    combined_monthly_df = pd.DataFrame(all_monthly_data).T
                    combined_monthly_df = combined_monthly_df.reset_index()
                    combined_monthly_df.columns = ['Date'] + list(combined_monthly_df.columns[1:])
                    
                    # Fill missing months with 0.000% (months with no trades)
                    # Get date range from all tickers
                    all_dates = sorted(all_monthly_data.keys())
                    if all_dates:
                        start_date = pd.to_datetime(all_dates[0] + '-01')
                        end_date = pd.to_datetime(all_dates[-1] + '-01')
                        
                        # Create complete month range
                        all_months = pd.period_range(start_date, end_date, freq='M')
                        complete_months = [m.strftime('%Y-%m') for m in all_months]
                        
                        # Reindex to include all months
                        combined_monthly_df['Date'] = pd.to_datetime(combined_monthly_df['Date'] + '-01')
                        combined_monthly_df = combined_monthly_df.set_index('Date')
                        complete_index = pd.period_range(start_date, end_date, freq='M').to_timestamp()
                        
                        # Determine default fill values based on column names
                        fill_dict = {}
                        for col in combined_monthly_df.columns:
                            if 'Return' in col:
                                fill_dict[col] = '0.000%'
                            elif 'Drawdown' in col:
                                fill_dict[col] = '0.00%'
                            elif 'Portfolio_Value' in col or 'Portfolio_Max' in col:
                                fill_dict[col] = ''
                            else:
                                fill_dict[col] = ''
                        
                        combined_monthly_df = combined_monthly_df.reindex(complete_index, fill_value='')
                        combined_monthly_df = combined_monthly_df.reset_index()
                        combined_monthly_df['Date'] = combined_monthly_df['Date'].dt.strftime('%Y-%m')
                        
                        # Fill NaN values with appropriate defaults
                        for col in combined_monthly_df.columns:
                            if col != 'Date':
                                if 'Return' in col:
                                    combined_monthly_df[col] = combined_monthly_df[col].fillna('0.000%')
                                elif 'Drawdown' in col:
                                    combined_monthly_df[col] = combined_monthly_df[col].fillna('0.00%')
                                else:
                                    combined_monthly_df[col] = combined_monthly_df[col].fillna('')
                    
                    # Create CSV
                    combined_csv = combined_monthly_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Combined Monthly Returns (All Stocks)",
                        data=combined_csv,
                        file_name="monthly_returns_combined.csv",
                        mime="text/csv",
                        key="download_monthly_combined"
                    )
                    
                    st.caption(f"Monthly returns from {combined_monthly_df['Date'].iloc[0]} to {combined_monthly_df['Date'].iloc[-1]} for {len(selected_tickers)} stocks")