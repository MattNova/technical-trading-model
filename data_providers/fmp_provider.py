import pandas as pd
import requests
import numpy as np

class BaseDataProvider:
    pass

class FMPProvider(BaseDataProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"

    def get_daily_stock_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            url = f"{self.base_url}/historical-chart/1day/{ticker}"
            params = {"from": start_date, "to": end_date, "apikey": self.api_key}
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if not data: 
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            print(f"Error fetching daily stock data for {ticker}: {e}")
            return pd.DataFrame()

    def get_latest_price(self, ticker: str) -> dict:
        try:
            url = f"{self.base_url}/quote/{ticker}"
            params = {"apikey": self.api_key}
            response = requests.get(url, params=params); response.raise_for_status()
            data = response.json()
            if data and data[0].get('price') is not None:
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
        """
        Calculates daily fundamental ratios.
        OPTIMIZED: This function now ACCEPTS the daily_prices DataFrame instead of fetching it again.
        """
        quarterly_fundamentals = self.get_historical_fundamentals(ticker)

        # Check if either of the required DataFrames is empty
        if daily_prices.empty or quarterly_fundamentals.empty: 
            return pd.DataFrame()
        
        # Prepare DataFrames for merging
        daily_prices_reset = daily_prices.reset_index()
        quarterly_fundamentals_reset = quarterly_fundamentals.reset_index()

        # Merge the daily price data with the quarterly fundamental data
        merged_df = pd.merge_asof(
            daily_prices_reset.sort_values('date'), 
            quarterly_fundamentals_reset.sort_values('date'), 
            on='date', 
            direction='backward'
        )
        merged_df.set_index('date', inplace=True)
        
        # Calculate the ratios
        merged_df['P/E'] = merged_df['close'] / merged_df['EPS_TTM']
        merged_df['P/S'] = (merged_df['close'] * merged_df['shares_out']) / merged_df['Sales_TTM']
        merged_df['PEG'] = merged_df['P/E'] / (merged_df['eps_growth'] * 100)
        
        final_cols = ['P/E', 'P/S', 'PEG']
        final_df = merged_df[final_cols].replace([np.inf, -np.inf], np.nan).dropna(how='all')
        
        return final_df