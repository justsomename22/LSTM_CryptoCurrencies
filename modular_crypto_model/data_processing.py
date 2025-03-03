#data_processing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from arch import arch_model
import warnings

def add_technical_indicators(df, price_col='price'):
    """
    Add comprehensive technical indicators to the dataframe
    
    Parameters:
    - df (pd.DataFrame): DataFrame with price data
    - price_col (str): Column name for price data
    
    Returns:
    - pd.DataFrame: DataFrame with added technical indicators
    """
    df_result = df.copy()
    
    # Trend indicators
    for window in [7, 14, 30]:
        df_result[f'sma_{window}'] = SMAIndicator(df_result[price_col], window=window).sma_indicator()
        # Price relative to moving average
        df_result[f'price_sma_ratio_{window}'] = df_result[price_col] / df_result[f'sma_{window}']
    
    # MACD
    macd = MACD(df_result[price_col])
    df_result['macd'] = macd.macd()
    df_result['macd_signal'] = macd.macd_signal()
    df_result['macd_diff'] = macd.macd_diff()
    
    # Momentum indicators
    df_result['rsi'] = RSIIndicator(df_result[price_col], window=14).rsi()
    
    stoch = StochasticOscillator(df_result['price'], df_result['price'], df_result['price'])
    df_result['stoch_k'] = stoch.stoch()
    df_result['stoch_d'] = stoch.stoch_signal()
    
    # Volatility indicators
    bollinger = BollingerBands(df_result[price_col])
    df_result['bollinger_hband'] = bollinger.bollinger_hband()
    df_result['bollinger_lband'] = bollinger.bollinger_lband()
    df_result['bollinger_width'] = (df_result['bollinger_hband'] - df_result['bollinger_lband']) / df_result[price_col]

    # Price changes at different timeframes
    for window in [1, 3, 7, 14]:
        df_result[f'pct_change_{window}d'] = df_result[price_col].pct_change(window)
    
    # Volatility measures
    for window in [7, 14, 30]:
        df_result[f'volatility_{window}d'] = df_result[price_col].pct_change().rolling(window).std()
    
    # Volume indicators
    if 'volume' in df_result.columns:
        # Volume relative to moving average
        df_result['volume_sma_7'] = df_result['volume'].rolling(7).mean()
        df_result['volume_ratio'] = df_result['volume'] / df_result['volume_sma_7']
        
        # Price-volume relationship
        df_result['price_volume_ratio'] = df_result[price_col] / (df_result['volume'] + 1)  # Add 1 to avoid division by zero
    
    # Fill NAs created by indicators
    numeric_cols = df_result.select_dtypes(include=[np.number]).columns
    df_result[numeric_cols] = df_result[numeric_cols].fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df_result

def normalize_and_fill_data(df):
    """
    Normalize price and volume data, fill missing values, convert date, and add price change columns
    
    Parameters:
    - df (pd.DataFrame): DataFrame with cryptocurrency data
    
    Returns:
    - pd.DataFrame: Normalized DataFrame with added columns
    """
    df = df.copy()
    
    # Convert date if needed
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # First, sort the data by date for each cryptocurrency
    df = df.sort_values(['crypto_id', 'date'])
    
    # Separate processing for each cryptocurrency
    crypto_dfs = []
    
    for crypto_id, group in df.groupby('crypto_id'):
        group = group.copy()
        
        # Check for missing values before interpolation
        missing_values = group.isnull().sum().sum()
        if missing_values > 0:
            print(f"Filling {missing_values} missing values in {crypto_id} data")
        
        # Handle negative or zero values in price/volume (these shouldn't exist in valid data)
        if (group['price'] <= 0).any():
            print(f"Warning: Found {(group['price'] <= 0).sum()} non-positive price values in {crypto_id}")
            # Replace with a small positive value instead of dropping
            group.loc[group['price'] <= 0, 'price'] = group.loc[group['price'] > 0, 'price'].min() * 0.1
            
        if 'volume' in group.columns and (group['volume'] < 0).any():
            print(f"Warning: Found {(group['volume'] < 0).sum()} negative volume values in {crypto_id}")
            # Set negative volumes to 0
            group.loc[group['volume'] < 0, 'volume'] = 0
        
        # Fill missing values using linear interpolation (better for time series)
        # Use forward fill and backward fill as fallbacks to ensure no NaN values remain
        for col in ['price', 'volume'] if 'volume' in group.columns else ['price']:
            group[col] = group[col].interpolate(method='linear').ffill().bfill()
        
        # Apply log transformation to price and volume to stabilize variance
        # This is especially important for cryptocurrencies which can have extreme volatility
        group['log_price'] = np.log1p(group['price'])  # log1p = log(1+x) to handle small values better
        
        if 'volume' in group.columns:
            # Handle zero volume with small epsilon
            group['log_volume'] = np.log1p(group['volume'] + 1e-8)  # Add small epsilon to avoid log(0)
        
        # Add columns for price difference and direction
        group['price_diff'] = group['price'].diff().fillna(0)
        group['price_direction'] = (group['price_diff'] > 0).astype(int)
        
        # Add log returns (better than simple returns for financial data)
        group['log_return'] = group['log_price'].diff().fillna(0)
        
        # Add rolling standard deviation (volatility)
        group['volatility_7d'] = group['log_return'].rolling(7).std().fillna(0)
        
        crypto_dfs.append(group)
    
    # Combine all processed dataframes
    df_processed = pd.concat(crypto_dfs)
    
    # Final check to make sure there are no NaN values
    if df_processed.isnull().any().any():
        print("Warning: There are still NaN values after processing.")
        print(df_processed.isnull().sum())
        # Fill any remaining NaNs with sensible defaults
        df_processed = df_processed.fillna(0)
        
    return df_processed

def add_advanced_features(df, price_col='price'):
    """Add more advanced predictive features including GARCH"""
    df = df.copy()
    
    # Process each cryptocurrency separately
    crypto_dfs = []
    
    for crypto_id, group in df.groupby('crypto_id'):
        group = group.copy()
        
        # Add GARCH Volatility Forecast
        returns = group[price_col].pct_change().dropna() * 100
        if len(returns) > 50:  # Ensure enough data
            try:
                # Use more robust GARCH specifications
                garch = arch_model(returns, vol='Garch', p=1, q=1, dist='skewt')  # Use skewed t distribution for better fit
                garch_fit = garch.fit(disp='off', update_freq=0)  # Turn off convergence display
                
                # Add conditional volatility
                group['garch_vol'] = garch_fit.conditional_volatility.reindex(group.index, method='ffill').fillna(0)
                
                # Add forecasted volatility for next period
                try:
                    forecast = garch_fit.forecast(horizon=1)
                    volatility_forecast = np.sqrt(forecast.variance.iloc[-1, 0])
                    group['garch_vol_forecast'] = group['garch_vol'].shift(1)
                    group.loc[group.index[-1], 'garch_vol_forecast'] = volatility_forecast
                except Exception as e:
                    print(f"Volatility forecast failed for {crypto_id}: {str(e)}")
                    group['garch_vol_forecast'] = group['garch_vol'].shift(1).fillna(0)
                
                # Add volatility regime based on GARCH
                try:
                    group['garch_vol_normalized'] = (group['garch_vol'] - group['garch_vol'].mean()) / group['garch_vol'].std()
                    group['garch_regime'] = pd.qcut(
                        group['garch_vol'].fillna(group['garch_vol'].median()), 
                        5, 
                        labels=[0, 1, 2, 3, 4]
                    ).astype(int)
                except Exception as e:
                    print(f"Error creating volatility regime for {crypto_id}: {str(e)}")
                    group['garch_vol_normalized'] = 0
                    group['garch_regime'] = 0
            except Exception as e:
                print(f"GARCH modeling failed for {crypto_id}: {str(e)}. Skipping GARCH features.")
        
        # Add momentum features (trend following)
        # Compute the rate of change over different periods
        for window in [3, 7, 14, 30]:
            # Rate of change (momentum indicator)
            group[f'price_roc_{window}d'] = group[price_col].pct_change(window).fillna(0)
        
        crypto_dfs.append(group)
    
    # Combine processed dataframes
    result_df = pd.concat(crypto_dfs)
    
    # Add day of week and other time features
    if 'date' in result_df.columns:
        result_df['day_of_week'] = result_df['date'].dt.dayofweek
        result_df['day_of_month'] = result_df['date'].dt.day
        result_df['month'] = result_df['date'].dt.month
        result_df['quarter'] = result_df['date'].dt.quarter
        result_df['year'] = result_df['date'].dt.year
        result_df['is_month_start'] = result_df['date'].dt.is_month_start.astype(int)
        result_df['is_month_end'] = result_df['date'].dt.is_month_end.astype(int)
        
    return result_df

def find_balanced_threshold(price_diff_scaled):
    """
    Calculate a balanced threshold to split price differences into three classes.
    
    Parameters:
    - price_diff_scaled (np.ndarray): Scaled price differences
    
    Returns:
    - float: Threshold value for stable class
    """
    sorted_abs_diff = np.sort(np.abs(price_diff_scaled))
    n = len(sorted_abs_diff)
    class_size = n // 3  # Aim for ~33% in each class
    threshold_stable = sorted_abs_diff[class_size]
    return threshold_stable 