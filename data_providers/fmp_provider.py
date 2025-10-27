import pandas as pd
import requests
import numpy as np
import os
from datetime import datetime, timedelta

# Define a local directory for storing the cached historical stock data
# This directory will be created in the root of your project
DATA_DIR = "local_historical_data"

class BaseDataProvider:
    """An empty base class for data provider inheritance."""
    pass

class FMPProvider(BaseDataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        # Ensure the local data directory exists
        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)
    
    def _get_ticker_file_path(self, ticker: str) -> str:
        """Returns the local path for a ticker's data file."""
        return os.path.join(DATA_DIR, f"{ticker.upper()}_daily.csv")

    def get_daily_stock_data(self, ticker: str, start_date_hint: str, end_date: str) -> pd.DataFrame:
        """
        Loads local historical data and only fetches new data (delta) from FMP API.
        This optimizes API calls significantly by using local storage first.
        """
        file_path = self._get_ticker_file_path(ticker)
        local_df = pd.DataFrame()
        start_date = start_date_hint # Default to full history hint

        # 1. Check for and Load Local Data
        if os.path.exists(file_path):
            try:
                # Read local file, ensuring index is datetime and sorted
                local_df = pd.read_csv(file_path, index_col='date', parse_dates=True)
                local_df.sort_index(inplace=True)
                
                # Determine the start date for the API call (Day after the last local record)
                last_local_date = local_df.index.max().strftime('%Y-%m-%d')
                
                # Add one day to the last local date for the new fetch
                delta_start_date = (datetime.strptime(last_local_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                
                # If the delta start date is today or later, we have the latest data
                if delta_start_date >= end_date:
                    return local_df[['open', 'high', 'low', 'close', 'volume']]
                
                start_date = delta_start_date
                
            except Exception as e:
                print(f"Error loading local data for {ticker}: {e}. Performing full fetch.")
                # If local load fails, start date remains the full history hint (1990-01-01)
        # Else: file does not exist, start date remains the full history hint (1990-01-01)

        # 2. Fetch Delta/Full Data from API
        try:
            url = f"{self.base_url}/historical-chart/1day/{ticker}"
            params = {"from": start_date, "to": end_date, "apikey": self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if not data:
                # No new data since last run, return local data if it exists
                return local_df[['open', 'high', 'low', 'close', 'volume']] if not local_df.empty else pd.DataFrame()
            
            # Process new API data
            api_df = pd.DataFrame(data)
            api_df['date'] = pd.to_datetime(api_df['date'])
            api_df.set_index('date', inplace=True)
            api_df.sort_index(inplace=True)
            api_df = api_df[['open', 'high', 'low', 'close', 'volume']]
            
            # 3. Merge and Save
            if not local_df.empty:
                # Concatenate local and new data, dropping any duplicates that might occur 
                # (e.g. if the last local date and the first API date overlap slightly)
                final_df = pd.concat([local_df, api_df]).drop_duplicates(keep='last')
            else:
                final_df = api_df
            
            final_df.sort_index(inplace=True)
            
            # Save the updated data locally for the next run (CRITICAL step for delta-fetch)
            final_df.to_csv(file_path)
            
            return final_df

        except Exception as e:
            print(f"Error fetching API data for {ticker} from {start_date}: {e}")
            # Fallback: return the local data even if the API fetch failed
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
            
    def get_historical_fundamentals(self, ticker: str, limit: int = 120) -> pd.DataFrame:
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

            df['EPS_TTM'] = df['eps'].rolling(window=4, min_periods=4).sum()
            df['Sales_TTM'] = df['revenue'].rolling(window=4, min_periods=4).sum()
            
            return df[['EPS_TTM', 'Sales_TTM', 'shares_out', 'eps_growth']].dropna(how='any')
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
        merged_df['PEG'] = merged_df['P/E'] / (merged_df['eps_growth'] * 100)
        
        final_cols = ['P/E', 'P/S', 'PEG']
        final_df = merged_df[final_cols].replace([np.inf, -np.inf], np.nan).dropna(how='all')
        
        return final_df