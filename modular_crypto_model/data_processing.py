#data_processing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from arch import arch_model
import warnings
from joblib import Parallel, delayed

def add_target_columns(df):
    """
    Add target columns for prediction: price_change, log_return, and direction.
    
    Parameters:
        df (pd.DataFrame): DataFrame with 'price' and 'crypto_id' columns.
    
    Returns:
        pd.DataFrame: DataFrame with added target columns.
    """
    df = df.copy()
    # Calculate price change: next price - current price
    df['price_change'] = df.groupby('crypto_id')['price'].shift(-1) - df['price']
    # Calculate log return: log(next price / current price)
    df['log_return'] = np.log(df.groupby('crypto_id')['price'].shift(-1) / df['price'])
    # Calculate direction: 1 if next price > current price, 0 otherwise
    df['direction'] = (df.groupby('crypto_id')['price'].shift(-1) > df['price']).astype(int)
    return df

def add_technical_indicators(df, price_col='price', use_bollinger=True, use_macd=True, use_moving_avg=True):
    """
    Add comprehensive technical indicators to the dataframe
    
    Parameters:
    - df (pd.DataFrame): DataFrame with price data
    - price_col (str): Column name for price data
    - use_bollinger (bool): Whether to include Bollinger Bands indicators
    - use_macd (bool): Whether to include MACD indicators
    - use_moving_avg (bool): Whether to include Moving Average indicators
    
    Returns:
    - pd.DataFrame: DataFrame with added technical indicators
    """
    df = df.copy()
    
    # Relative Strength Index (RSI)
    df['rsi'] = RSIIndicator(df[price_col], window=14).rsi()
    
    # Rolling volatility
    df['volatility_7d'] = df[price_col].pct_change().rolling(7).std()
    
    # Moving Averages (if enabled)
    if use_moving_avg:
        # Simple Moving Averages
        df['sma_5'] = SMAIndicator(df[price_col], window=5).sma_indicator()
        df['sma_10'] = SMAIndicator(df[price_col], window=10).sma_indicator()
        df['sma_20'] = SMAIndicator(df[price_col], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(df[price_col], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(df[price_col], window=200).sma_indicator()
        
        # Exponential Moving Averages
        # Using pandas' ewm (exponential weighted moving average) function
        df['ema_5'] = df[price_col].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df[price_col].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df[price_col].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df[price_col].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df[price_col].ewm(span=200, adjust=False).mean()
        
        # Moving Average Crossover Signals
        # Golden Cross/Death Cross (50-day MA crosses 200-day MA)
        df['ma_cross_50_200'] = df['sma_50'] - df['sma_200']
        
        # Fast/Slow MA Cross (10-day MA crosses 50-day MA)
        df['ma_cross_10_50'] = df['sma_10'] - df['sma_50']
        
        # Short-term Cross (5-day MA crosses 20-day MA)
        df['ma_cross_5_20'] = df['sma_5'] - df['sma_20']
        
        # EMA vs Price - Shows if price is above/below trend
        df['price_vs_ema_20'] = df[price_col] - df['ema_20']
        
        # Percent distance from moving averages (normalized)
        df['pct_from_sma_50'] = (df[price_col] - df['sma_50']) / df['sma_50']
        df['pct_from_sma_200'] = (df[price_col] - df['sma_200']) / df['sma_200']
    
    # Bollinger Bands (if enabled)
    if use_bollinger:
        # Initialize Bollinger Bands with 20-day SMA and 2 standard deviations
        bollinger = BollingerBands(df[price_col], window=20, window_dev=2)
        
        # Add Bollinger Band metrics
        df['bollinger_mavg'] = bollinger.bollinger_mavg()  # Middle band (20-day SMA)
        df['bollinger_hband'] = bollinger.bollinger_hband()  # Upper band
        df['bollinger_lband'] = bollinger.bollinger_lband()  # Lower band
        
        # Calculate distance from price to bands (as percentage)
        df['bollinger_width'] = (df['bollinger_hband'] - df['bollinger_lband']) / df['bollinger_mavg']  # Band width (volatility indicator)
        df['bollinger_pct_b'] = bollinger.bollinger_pband()  # Relative position within bands (0 to 1)
        
        # Calculate distance from current price to bands
        df['dist_to_upper'] = (df['bollinger_hband'] - df[price_col]) / df[price_col]
        df['dist_to_lower'] = (df[price_col] - df['bollinger_lband']) / df[price_col]
    
    # MACD (Moving Average Convergence Divergence) indicators (if enabled)
    if use_macd:
        # Initialize MACD with standard settings (12, 26, 9)
        macd_indicator = MACD(df[price_col], window_fast=12, window_slow=26, window_sign=9)
        
        # MACD Line: Difference between 12-period EMA and 26-period EMA
        df['macd_line'] = macd_indicator.macd()
        
        # Signal Line: 9-period EMA of the MACD Line
        df['macd_signal'] = macd_indicator.macd_signal()
        
        # MACD Histogram: Difference between MACD Line and Signal Line
        df['macd_hist'] = macd_indicator.macd_diff()
        
        # MACD Divergence: Normalized difference between price and MACD
        # Helps detect divergence between price movement and momentum
        try:
            price_norm = (df[price_col] - df[price_col].rolling(26).mean()) / df[price_col].rolling(26).std()
            macd_norm = (df['macd_line'] - df['macd_line'].rolling(26).mean()) / df['macd_line'].rolling(26).std()
            df['macd_divergence'] = price_norm - macd_norm
            df['macd_divergence'] = df['macd_divergence'].fillna(0)
        except Exception as e:
            warnings.warn(f"Could not calculate MACD divergence: {str(e)}")
            df['macd_divergence'] = 0
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].bfill().ffill().fillna(0)
    
    return df

def normalize_and_fill_data(df):
    """
    Normalize price and volume data, fill missing values, convert date, and add price change columns
    
    Parameters:
    - df (pd.DataFrame): DataFrame with cryptocurrency data
    
    Returns:
    - pd.DataFrame: Normalized DataFrame with added columns
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['crypto_id', 'date'])
    
    def process_group(group):
        """
        Process each group of cryptocurrency data to interpolate missing price values.
        
        Parameters:
        - group (pd.DataFrame): DataFrame for a single cryptocurrency.
        
        Returns:
        - pd.DataFrame: Processed group with interpolated prices.
        """
        group['price'] = group['price'].interpolate(method='linear').ffill().bfill()
        return group
    
    df = Parallel(n_jobs=-1)(delayed(process_group)(group) for _, group in df.groupby('crypto_id'))
    return pd.concat(df)

def add_advanced_features(df, price_col='price'):
    """
    Add more advanced predictive features including GARCH volatility modeling.
    
    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models are
    particularly valuable for cryptocurrency analysis as they capture volatility
    clustering (periods of high/low volatility that tend to group together).
    
    Parameters:
    - df (pd.DataFrame): DataFrame with cryptocurrency data
    - price_col (str): Column name for price data
    
    Returns:
    - pd.DataFrame: DataFrame with added advanced features including:
        - garch_vol: Historical volatility estimated by GARCH model
        - garch_vol_forecast: Forecasted volatility for next period
        - garch_regime: Volatility regime classification (0-4, low to high)
        - price_roc_*: Price rate of change over different time periods
        - Seasonal time features (day of week, month, etc.)
    """
    df = df.copy()
    
    # Process each cryptocurrency separately
    crypto_dfs = []
    
    for crypto_id, group in df.groupby('crypto_id'):
        try:
            group = group.copy()
            
            # Add GARCH Volatility Forecast
            returns = group[price_col].pct_change().dropna() * 100
            if len(returns) > 50:  # Ensure enough data for GARCH modeling
                try:
                    # Use more robust GARCH specifications
                    # p=1, q=1 is standard for crypto but could be optimized per currency
                    garch = arch_model(returns, vol='Garch', p=1, q=1, dist='skewt')
                    res = garch.fit(disp='off', update_freq=0, last_obs=len(returns))
                    
                    # Add conditional volatility (historical volatility estimated by GARCH)
                    group['garch_vol'] = pd.Series(
                        index=group.index,
                        data=np.nan
                    )
                    vol_index = returns.index.intersection(group.index)
                    group.loc[vol_index, 'garch_vol'] = res.conditional_volatility
                    
                    # Forward-fill missing values then backfill any remaining NaNs
                    group['garch_vol'] = group['garch_vol'].ffill().bfill()
                    
                    # Add forecasted volatility for next period (1-step ahead forecast)
                    try:
                        forecast = res.forecast(horizon=1)
                        forecast_vol = np.sqrt(forecast.variance.values[-1, 0])
                        
                        # Shift garch_vol by 1 to get yesterday's forecast for today
                        group['garch_vol_forecast'] = group['garch_vol'].shift(1)
                        
                        # For the most recent day, use the actual forecast
                        last_date = group.index[-1]
                        if last_date in group.index:
                            group.loc[last_date, 'garch_vol_forecast'] = forecast_vol
                    except Exception as e:
                        warnings.warn(f"Volatility forecast failed for {crypto_id}: {str(e)}")
                        group['garch_vol_forecast'] = group['garch_vol'].shift(1)
                    
                    # Fill any NaNs with median to avoid data loss
                    group['garch_vol_forecast'] = group['garch_vol_forecast'].fillna(
                        group['garch_vol_forecast'].median() if not group['garch_vol_forecast'].isna().all() else 0
                    )
                    
                    # Add volatility regime based on GARCH (0=very low, 4=very high)
                    try:
                        # Normalized volatility (z-score)
                        group['garch_vol_normalized'] = (group['garch_vol'] - group['garch_vol'].mean()) / group['garch_vol'].std()
                        
                        # Create 5 volatility regimes using quantile-based binning
                        group['garch_regime'] = pd.qcut(
                            group['garch_vol'].rank(method='first'),  # Rank method handles duplicates
                            5, 
                            labels=False  # Use integers 0-4
                        ).astype(int)
                    except Exception as e:
                        warnings.warn(f"Error creating volatility regime for {crypto_id}: {str(e)}")
                        group['garch_vol_normalized'] = 0
                        group['garch_regime'] = 2  # Default to middle regime
                except Exception as e:
                    warnings.warn(f"GARCH modeling failed for {crypto_id}: {str(e)}. Using simple volatility instead.")
                    # Fallback to simple volatility calculation
                    group['garch_vol'] = returns.rolling(window=20).std().reindex(group.index).ffill().bfill()
                    group['garch_vol_forecast'] = group['garch_vol'].shift(1)
                    group['garch_regime'] = pd.qcut(
                        group['garch_vol'].rank(method='first'),
                        5,
                        labels=False
                    ).fillna(2).astype(int)
            else:
                # Not enough data - use standard deviation as a simpler alternative
                warnings.warn(f"Not enough data for GARCH modeling for {crypto_id}. Using simple volatility.")
                vol = group[price_col].pct_change().rolling(window=7).std() * 100
                group['garch_vol'] = vol.fillna(vol.median() if not vol.isna().all() else 0)
                group['garch_vol_forecast'] = group['garch_vol'].shift(1).fillna(group['garch_vol'].median() if not group['garch_vol'].isna().all() else 0)
                group['garch_regime'] = 2  # Default to middle regime
            
            # Add momentum features (trend following) - rate of change over different periods
            for window in [3, 7, 14, 30]:
                group[f'price_roc_{window}d'] = group[price_col].pct_change(window).fillna(0)
            
            crypto_dfs.append(group)
        except Exception as e:
            warnings.warn(f"Error processing advanced features for {crypto_id}: {str(e)}")
            # Add the original group without advanced features to avoid data loss
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