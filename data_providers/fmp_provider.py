import pandas as pd
import requests
import numpy as np
import os
from datetime import datetime, timedelta

# Define a local directory for storing the cached historical stock data
DATA_DIR = "local_historical_data"
# NEW: Define a file for storing manual daily price overrides
MANUAL_PRICE_FILE = os.path.join(DATA_DIR, "manual_daily_prices.csv")

class BaseDataProvider:
    """An empty base class for data provider inheritance."""
    pass

class FMPProvider(BaseDataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

    # NEW FUNCTION: Handles reading manual data overrides
    def get_manual_prices(self) -> pd.DataFrame:
        """Loads the manual price override file."""
        if os.path.exists(MANUAL_PRICE_FILE):
            try:
                df = pd.read_csv(MANUAL_PRICE_FILE, index_col='date', parse_dates=True)
                return df
            except Exception as e:
                print(f"Error loading manual prices: {e}")
                return pd.DataFrame()
        return pd.DataFrame()
    
    # NEW FUNCTION: Handles saving manual data overrides
    def save_manual_prices(self, df: pd.DataFrame):
        """Saves the manual price override data."""
        df.to_csv(MANUAL_PRICE_FILE)

    def _get_ticker_file_path(self, ticker: str) -> str:
        """Returns the local path for a ticker's data file."""
        return os.path.join(DATA_DIR, f"{ticker.upper()}_daily.csv")
    
    def _get_fundamentals_file_path(self, ticker: str) -> str:
        """Returns the local path for a ticker's fundamentals file."""
        return os.path.join(DATA_DIR, f"{ticker.upper()}_fundamentals.csv")

    def get_daily_stock_data(self, ticker: str, start_date_hint: str, end_date: str) -> pd.DataFrame:
        """
        Loads local historical data and only fetches new data (delta) from FMP API.
        Includes a fallback to manual prices on API failure.
        Skips weekend fetches (Saturday/Sunday) since markets are closed.
        """
        file_path = self._get_ticker_file_path(ticker)
        local_df = pd.DataFrame()
        start_date = start_date_hint # Default to full history hint

        # 1. Check for and Load Local Data
        if os.path.exists(file_path):
            try:
                local_df = pd.read_csv(file_path, index_col='date', parse_dates=True)
                local_df.sort_index(inplace=True)
                
                last_local_date = local_df.index.max().strftime('%Y-%m-%d')
                delta_start_date = (datetime.strptime(last_local_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                
                # If delta_start_date > end_date, we're already up to date (or beyond)
                if delta_start_date > end_date:
                    return local_df[['open', 'high', 'low', 'close', 'volume']]
                
                # Skip weekend fetches (Saturday=6, Sunday=7 in isoweekday)
                delta_datetime = datetime.strptime(delta_start_date, '%Y-%m-%d')
                if delta_datetime.isoweekday() in [6, 7]:  # Saturday or Sunday
                    return local_df[['open', 'high', 'low', 'close', 'volume']]
                
                start_date = delta_start_date
                
            except Exception as e:
                print(f"Error loading local data for {ticker}: {e}. Performing full fetch.")

        # 2. Fetch Delta/Full Data from API
        try:
            # Use historical-price-full endpoint for full history (not limited date range)
            # Only use date parameters if we're fetching a delta (partial date range)
            if local_df.empty:
                # Full fetch - use historical-price-full endpoint for complete history
                # Include date range to ensure we get the full history based on subscription plan
                url = f"{self.base_url}/historical-price-full/{ticker}"
                params = {"from": start_date, "to": end_date, "apikey": self.api_key}
            else:
                # Delta fetch - use historical-chart endpoint with date range
                url = f"{self.base_url}/historical-chart/1day/{ticker}"
                params = {"from": start_date, "to": end_date, "apikey": self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status() # This is where the error will be raised
            
            # If successful, process, merge, and save data
            data = response.json()
            if not data:
                return local_df[['open', 'high', 'low', 'close', 'volume']] if not local_df.empty else pd.DataFrame()
            
            # Handle different response formats
            if isinstance(data, dict) and 'historical' in data:
                # historical-price-full returns {symbol: "...", historical: [...]}
                api_df = pd.DataFrame(data['historical'])
            else:
                # historical-chart returns a list directly
                api_df = pd.DataFrame(data)
            
            api_df['date'] = pd.to_datetime(api_df['date'])
            api_df.set_index('date', inplace=True)
            api_df.sort_index(inplace=True)
            api_df = api_df[['open', 'high', 'low', 'close', 'volume']]
            
            if not local_df.empty:
                final_df = pd.concat([local_df, api_df]).drop_duplicates(keep='last')
            else:
                final_df = api_df
            
            final_df.sort_index(inplace=True)
            final_df.to_csv(file_path)
            
            return final_df

        except requests.exceptions.HTTPError as he:
            # Fallback for API errors (like 429 Rate Limit)
            print(f"HTTP Error fetching data for {ticker} from {start_date}: {he}")
            if local_df.empty: return pd.DataFrame()
            
            # --- MANUAL FALLBACK LOGIC ---
            manual_prices = self.get_manual_prices()
            if not manual_prices.empty and ticker in manual_prices.columns:
                # Get the latest manual price and date
                latest_manual_date = manual_prices.index.max()
                latest_close_value = manual_prices[ticker].iloc[-1]

                # Create a single row DataFrame for the manual data point
                manual_row = pd.DataFrame([{
                    'open': latest_close_value, 
                    'high': latest_close_value, 
                    'low': latest_close_value, 
                    'close': latest_close_value, 
                    'volume': 0 
                }], index=[latest_manual_date], columns=['open', 'high', 'low', 'close', 'volume'])
                
                # Merge manual data with local data
                final_df = pd.concat([local_df, manual_row]).drop_duplicates(keep='last')
                final_df.sort_index(inplace=True)
                return final_df
            
            # Fallback to only returning old local data if no manual data is present
            return local_df[['open', 'high', 'low', 'close', 'volume']] 

        except Exception as e:
            print(f"General Error fetching API data for {ticker} from {start_date}: {e}")
            return local_df[['open', 'high', 'low', 'close', 'volume']] if not local_df.empty else pd.DataFrame()
            
    def get_latest_price(self, ticker: str) -> dict:
        try:
            url = f"{self.base_url}/quote/{ticker}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params); response.raise_for_status()
            data = response.json()
            if data and data.get('price') is not None:
                latest_data = data[0]
                return {
                    'price': latest_data.get('price'),
                    'date': pd.to_datetime(latest_data.get('timestamp'), unit='s').strftime('%Y-%m-%d %H:%M:%S')
                }
            return {}
        except Exception as e:
            print(f"Error fetching latest price for {ticker}: {e}")
            return {}
            
    def get_historical_fundamentals(self, ticker: str, limit: int = 200, force_refresh: bool = False) -> pd.DataFrame:
        """
        Loads local fundamentals cache and only fetches new data from FMP API.
        Fundamentals refresh every Sunday to catch earnings releases throughout the week.
        Otherwise, cache is used if recent (<90 days).
        """
        fundamentals_file = self._get_fundamentals_file_path(ticker)
        local_fundamentals = pd.DataFrame()
        
        # 1. Check for and Load Local Fundamentals
        if not force_refresh and os.path.exists(fundamentals_file):
            try:
                local_fundamentals = pd.read_csv(fundamentals_file, index_col='date', parse_dates=True)
                local_fundamentals.sort_index(inplace=True)
                
                # Refresh on Sundays (isoweekday() = 7 = Sunday)
                today = datetime.now()
                if today.isoweekday() == 7:  # Sunday
                    print(f"Sunday detected - refreshing fundamentals for {ticker}")
                    # Continue to API fetch below
                else:
                    # Check if cache is recent (within last 90 days for quarterly data)
                    last_fundamental_date = local_fundamentals.index.max() if not local_fundamentals.empty else None
                    if last_fundamental_date:
                        days_since_last = (today - last_fundamental_date).days
                        if days_since_last < 90:
                            # Cache is recent, return it
                            return local_fundamentals
                
            except Exception as e:
                print(f"Error loading local fundamentals for {ticker}: {e}. Performing full fetch.")
        
        # 2. Fetch Fundamentals from API
        try:
            url_income = f"{self.base_url}/income-statement/{ticker}?period=quarter&limit={limit}"
            url_growth = f"{self.base_url}/financial-growth/{ticker}?period=quarter&limit={limit}"
            params = {"apikey": self.api_key}
            
            resp_income = requests.get(url_income, params=params); resp_income.raise_for_status()
            data_income = resp_income.json()
            resp_growth = requests.get(url_growth, params=params); resp_growth.raise_for_status()
            data_growth = resp_growth.json()

            if not data_income or not data_growth: return pd.DataFrame()

            df_income = pd.DataFrame(data_income); df_growth = pd.DataFrame(data_growth)
            df_income['date'] = pd.to_datetime(df_income['date']); df_growth['date'] = pd.to_datetime(df_growth['date'])
            df = pd.merge(df_income, df_growth, on='date', how='inner')
            
            if df.empty or 'date' not in df.columns: return pd.DataFrame() 

            df.set_index('date', inplace=True); df.sort_index(inplace=True)
            
            df['eps'] = pd.to_numeric(df.get('epsdiluted'), errors='coerce')
            df['revenue'] = pd.to_numeric(df.get('revenue'), errors='coerce')
            df['shares_out'] = pd.to_numeric(df.get('weightedAverageShsOutDil'), errors='coerce')
            df['eps_growth'] = pd.to_numeric(df.get('epsgrowth'), errors='coerce')

            # Calculate TTM: Use min_periods=1 to allow calculation with available data
            # This enables longer historical analysis while still preferring full 4-quarter data
            df['EPS_TTM'] = df['eps'].rolling(window=4, min_periods=1).sum()
            df['Sales_TTM'] = df['revenue'].rolling(window=4, min_periods=1).sum()
            # Fallback EPS growth if API field missing
            if 'epsgrowth' not in df.columns or df['epsgrowth'].isna().all():
                df['epsgrowth'] = df['EPS_TTM'].pct_change(periods=4)
            
            # Save to cache
            df[['EPS_TTM', 'Sales_TTM', 'shares_out', 'eps_growth']].to_csv(fundamentals_file)
            
            # Return all data, don't drop NaN rows yet - we need the full history for forward fill
            return df[['EPS_TTM', 'Sales_TTM', 'shares_out', 'eps_growth']]
        except Exception as e:
            print(f"Error fetching historical fundamentals for {ticker}: {e}")
            return pd.DataFrame()
            
    def get_daily_fundamental_ratios(self, ticker: str, daily_prices: pd.DataFrame) -> pd.DataFrame:
        quarterly_fundamentals = self.get_historical_fundamentals(ticker)

        if daily_prices.empty or quarterly_fundamentals.empty: 
            return pd.DataFrame()
        
        daily_prices_reset = daily_prices.reset_index()
        quarterly_fundamentals_reset = quarterly_fundamentals.reset_index()

        merged_df = pd.merge_asof(
            daily_prices_reset.sort_values('date'), 
            quarterly_fundamentals_reset.sort_values('date'), 
            on='date', 
            direction='backward'
        )
        merged_df.set_index('date', inplace=True)
        
        merged_df['P/E'] = merged_df['close'] / merged_df['EPS_TTM']
        merged_df['P/S'] = (merged_df['close'] * merged_df['shares_out']) / merged_df['Sales_TTM']
        # Use eps_growth if available else fallback from quarterly epsgrowth (ffill)
        merged_df['eps_growth'] = merged_df['eps_growth'].ffill()
        # Avoid division by near-zero growth
        eps_g = merged_df['eps_growth'].where(merged_df['eps_growth'].abs() > 1e-6, np.nan)
        merged_df['PEG'] = merged_df['P/E'] / (eps_g * 100)
        
        final_cols = ['P/E', 'P/S', 'PEG']
        final_df = merged_df[final_cols].replace([np.inf, -np.inf], np.nan)
        # Forward-fill and backward-fill sparse fundamentals to enable longer timeframes
        # Forward-fill bridges gaps between quarterly reports (typically 60-90 days apart)
        # Backward-fill extends data back in time for 5-10 year historical analysis
        # This ensures we have enough data points for percentile calculations
        final_df = final_df.ffill().bfill()  # Fill forward first, then backward to cover full date range
        
        return final_df