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
            url = f"{self.base_url}/historical-price-full/{ticker}"
            params = {"from": start_date, "to": end_date, "apikey": self.api_key}
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json().get('historical', [])
            if not data: return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error fetching daily stock data for {ticker}: {e}")
            return pd.DataFrame()

    def get_historical_fundamentals(self, ticker: str, limit: int = 120) -> pd.DataFrame:
        """Fetches and calculates historical TTM fundamentals from quarterly reports."""
        try:
            url_income = f"{self.base_url}/income-statement/{ticker}?period=quarter&limit={limit}"
            url_growth = f"{self.base_url}/financial-growth/{ticker}?period=quarter&limit={limit}"
            params = {"apikey": self.api_key}
            
            resp_income = requests.get(url_income, params=params); resp_income.raise_for_status()
            data_income = resp_income.json()
            
            resp_growth = requests.get(url_growth, params=params); resp_growth.raise_for_status()
            data_growth = resp_growth.json()

            if not data_income or not data_growth: return pd.DataFrame()

            df_income = pd.DataFrame(data_income)
            df_growth = pd.DataFrame(data_growth)

            df_income['date'] = pd.to_datetime(df_income['date'])
            df_growth['date'] = pd.to_datetime(df_growth['date'])
            df = pd.merge(df_income, df_growth, on='date', how='inner')
            
            if df.empty or 'date' not in df.columns: return pd.DataFrame() 

            df.set_index('date', inplace=True)
            df.sort_index(inplace=True)
            
            df['eps'] = pd.to_numeric(df.get('epsdiluted'), errors='coerce')
            df['revenue'] = pd.to_numeric(df.get('revenue'), errors='coerce')
            df['shares_out'] = pd.to_numeric(df.get('weightedAverageShsOutDil'), errors='coerce')
            df['eps_growth'] = pd.to_numeric(df.get('epsgrowth'), errors='coerce')

            df['EPS_TTM'] = df['eps'].rolling(window=4, min_periods=4).sum()
            df['Sales_TTM'] = df['revenue'].rolling(window=4, min_periods=4).sum()
            
            return df[['EPS_TTM', 'Sales_TTM', 'shares_out', 'eps_growth']].dropna()

        except Exception as e:
            print(f"Error fetching historical fundamentals for {ticker}: {e}")
            return pd.DataFrame()
            
    def get_daily_fundamental_ratios(self, ticker: str) -> pd.DataFrame:
        """Orchestrates creating a daily time series of P/E, P/S, and PEG ratios."""
        
        end_date_str = pd.to_datetime('today').strftime('%Y-%m-%d')
        daily_prices = self.get_daily_stock_data(ticker, '1990-01-01', end_date_str)
        quarterly_fundamentals = self.get_historical_fundamentals(ticker)

        if daily_prices.empty or quarterly_fundamentals.empty:
            return pd.DataFrame()
        
        daily_prices.reset_index(inplace=True)
        quarterly_fundamentals.reset_index(inplace=True)

        merged_df = pd.merge_asof(
            daily_prices.sort_values('date'),
            quarterly_fundamentals.sort_values('date'),
            on='date',
            direction='backward'
        )
        merged_df.set_index('date', inplace=True)
        
        merged_df['P/E'] = merged_df['close'] / merged_df['EPS_TTM']
        market_cap = merged_df['close'] * merged_df['shares_out']
        merged_df['P/S'] = market_cap / merged_df['Sales_TTM']
        merged_df['PEG'] = merged_df['P/E'] / (merged_df['eps_growth'] * 100)
        
        final_cols = ['P/E', 'P/S', 'PEG']
        final_df = merged_df[final_cols].replace([np.inf, -np.inf], np.nan).dropna(how='all')
        
        return final_df