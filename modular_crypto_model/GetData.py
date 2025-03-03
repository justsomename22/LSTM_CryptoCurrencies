#GetData.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import numpy as np
from ta.trend import SMAIndicator
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
# Import from data_processing
from .data_processing import add_advanced_features
from arch import arch_model

# --- Configuration ---
base_url = "https://api.coingecko.com/api/v3"
crypto_tickers = ["bitcoin", "ethereum", "monero"]
end_date = datetime.now()
start_date = end_date - timedelta(days=364)
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")
csv_filename = "cryptocurrency_data.csv"  # Adjusted path for simplicity
api_key = None  # Replace with your API key if available

# --- Functions ---

def fetch_crypto_historical_data(crypto_id, start_date, end_date, max_retries=3):
    """
    Fetch historical cryptocurrency data from the CoinGecko API.
    
    Parameters:
    -----------
    crypto_id : str
        The ID of the cryptocurrency to fetch data for.
    start_date : str
        The start date for the historical data (format: YYYY-MM-DD).
    end_date : str
        The end date for the historical data (format: YYYY-MM-DD).
    max_retries : int, default=3
        Maximum number of retries for failed requests.
    
    Returns:
    --------
    dict or None
        The historical data for the cryptocurrency or None if the request fails.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")  # Convert start date string to datetime object
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")  # Convert end date string to datetime object
    start_timestamp = int(start_dt.timestamp())  # Convert start date to timestamp
    end_timestamp = int(end_dt.timestamp())  # Convert end date to timestamp
    
    url = f"{base_url}/coins/{crypto_id}/market_chart/range"  # Construct the API endpoint URL
    params = {"vs_currency": "usd", "from": start_timestamp, "to": end_timestamp}  # Set parameters for the API request
    headers = {"x-cg-api-key": api_key} if api_key else {}  # Set API key in headers if available
    
    print(f"Requesting data from {start_date} to {end_date} for {crypto_id}")  # Log the request
    for attempt in range(max_retries):  # Retry logic for failed requests
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)  # Make the API request
            status_code = response.status_code  # Get the response status code
            print(f"Attempt {attempt+1}/{max_retries} - Status code: {status_code}")  # Log the attempt and status code
            if status_code == 429:  # Check for rate limit error
                retry_after = int(response.headers.get('Retry-After', 2 ** attempt))  # Get retry time from headers
                print(f"Rate limit hit. Waiting {retry_after} seconds before retry.")  # Log rate limit hit
                time.sleep(retry_after)  # Wait before retrying
                continue  # Retry the request
            response.raise_for_status()  # Raise an error for bad responses
            data = response.json()  # Parse the response JSON
            print(f"Successfully retrieved data for {crypto_id}")  # Log successful data retrieval
            return data  # Return the retrieved data
        except requests.exceptions.Timeout:
            wait_time = 2 ** attempt  # Exponential backoff for timeout errors
            print(f"Timeout error. Retrying in {wait_time} seconds...")  # Log timeout error
            time.sleep(wait_time)  # Wait before retrying
        except requests.exceptions.HTTPError as e:
            if response.status_code == 400:  # Handle bad request error
                print(f"Bad request for {crypto_id}: {response.text}")  # Log bad request error
                return None  # Return None for bad requests
            wait_time = 2 ** attempt  # Exponential backoff for other HTTP errors
            print(f"HTTP error {response.status_code}. Retrying in {wait_time} seconds...")  # Log HTTP error
            time.sleep(wait_time)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            wait_time = 2 ** attempt  # Exponential backoff for request exceptions
            print(f"Request error: {str(e)}. Retrying in {wait_time} seconds...")  # Log request error
            time.sleep(wait_time)  # Wait before retrying
    print(f"Failed to fetch data for {crypto_id} after {max_retries} attempts.")  # Log failure after retries
    return None  # Return None if all attempts fail

def process_crypto_data(crypto_id, historical_data):
    """
    Process the historical cryptocurrency data into a structured format.
    
    Parameters:
    -----------
    crypto_id : str
        The ID of the cryptocurrency being processed.
    historical_data : dict
        The historical data retrieved from the API.
    
    Returns:
    --------
    list
        A list of processed data points for the cryptocurrency.
    """
    processed_data = []  # Initialize a list to hold processed data
    if historical_data and 'prices' in historical_data and 'total_volumes' in historical_data:  # Check for valid data
        prices = historical_data['prices']  # Extract prices from historical data
        volumes = historical_data['total_volumes']  # Extract volumes from historical data
        data_points = min(len(prices), len(volumes))  # Determine the number of data points to process
        for i in range(data_points):  # Loop through the data points
            timestamp = prices[i][0]  # Get the timestamp
            date = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d')  # Convert timestamp to date
            price = prices[i][1]  # Get the price
            volume = volumes[i][1]  # Get the volume
            processed_data.append({  # Append processed data to the list
                "crypto_id": crypto_id,
                "date": date,
                "price": price,
                "volume": volume
            })
    else:
        print(f"No valid historical data found for {crypto_id}")  # Log if no valid data is found
    return processed_data  # Return the processed data

def add_basic_features(df, price_col='price'):
    """Add basic features to the dataframe including volatility measures"""
    df = df.copy()
    
    # Convert date if needed
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Daily returns
    df['daily_return'] = df[price_col].pct_change()
    
    # Volatility measures for different windows
    for window in [7, 14, 30, 60]:
        df[f'volatility_{window}d'] = df['daily_return'].rolling(window).std() * np.sqrt(window)
    
    # Simple moving averages
    for window in [7, 14, 30, 60]:
        df[f'sma_{window}'] = df[price_col].rolling(window).mean()
    
    # Price momentum
    for window in [1, 3, 7, 14]:
        df[f'momentum_{window}d'] = df[price_col].pct_change(window)
    
    # Volume changes
    if 'volume' in df.columns:
        df['volume_change'] = df['volume'].pct_change()
        df['avg_volume_7d'] = df['volume'].rolling(7).mean()
        df['volume_ratio'] = df['volume'] / df['avg_volume_7d'].replace(0, 0.001)
    
    return df

def fetch_historical_data(crypto_id, vs_currency='usd', days=365, interval='daily'):
    """Fetch historical data for a cryptocurrency using the CoinGecko API"""
    url = f"{base_url}/coins/{crypto_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": interval
    }
    headers = {"x-cg-api-key": api_key} if api_key else {}
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching historical data for {crypto_id}: {str(e)}")
        return None

def add_advanced_features(df, price_col='price'):
    """Add more advanced predictive features"""
    df = df.copy()
    
    # Make sure basic features are added first if they don't exist
    if 'volatility_7d' not in df.columns:
        df = add_basic_features(df, price_col)
    
    # Add longer-term technical indicators
    for window in [30, 60, 90]:
        df[f'sma_{window}'] = SMAIndicator(df[price_col], window=window).sma_indicator()
        df[f'price_sma_ratio_{window}'] = df[price_col] / df[f'sma_{window}']
    
    # Add market regime indicators
    df['volatility_regime'] = pd.qcut(df['volatility_30d'].fillna(0), 5, labels=[0, 1, 2, 3, 4])
    
    # Add cyclical features
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    
    # Momentum indicators with different lookback periods
    for window in [3, 7, 14, 30]:
        df[f'momentum_{window}d'] = df[price_col].pct_change(window)
    
    # Add volatility clustering features
    df['volatility_ratio'] = df['volatility_7d'] / df['volatility_30d'].replace(0, 0.001)
    
    # Add mean-reversion indicators
    for window in [7, 14, 30]:
        rolling_mean = df[price_col].rolling(window).mean()
        rolling_std = df[price_col].rolling(window).std()
        df[f'z_score_{window}'] = (df[price_col] - rolling_mean) / rolling_std.replace(0, 0.001)
    
    # GARCH implementation removed - now using the more robust implementation from data_processing.py
    
    # Bitcoin Correlation (for non-BTC cryptocurrencies)
    if 'crypto_id' in df.columns and df['crypto_id'].iloc[0].lower() != 'bitcoin':
        # Find Bitcoin data in the same dataframe (assuming multiple cryptos in the same df)
        btc_data = df[df['crypto_id'].str.lower() == 'bitcoin'] if 'crypto_id' in df.columns else None
        
        # If we have Bitcoin data, calculate correlation
        if btc_data is not None and not btc_data.empty:
            df_with_btc = pd.merge(
                df, 
                btc_data[['date', 'price']].rename(columns={'price': 'bitcoin_price'}),
                on='date', 
                how='left'
            )
            
            for window in [7, 14, 30]:
                df[f'btc_corr_{window}d'] = df_with_btc[price_col].rolling(window).corr(df_with_btc['bitcoin_price'])
    
    # Fill missing values with appropriate methods
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    return df

def train_ensemble(self, crypto_id, n_models=5):
    """Train multiple models and use ensemble averaging for predictions"""
    if crypto_id not in self.data_by_crypto:
        raise ValueError(f"No data prepared for {crypto_id}")
    
    data = self.data_by_crypto[crypto_id]
    X_train, y_train = data['X_train'], data['y_train']
    
    # Train multiple models with different seeds/configurations
    ensemble_models = []
    for i in range(n_models):
        print(f"Training ensemble model {i+1}/{n_models} for {crypto_id}...")
        
        # Vary model parameters slightly for diversity
        hidden_size = self.model_params['hidden_size'] + np.random.randint(-16, 17) 
        dropout = self.model_params['dropout'] + np.random.uniform(-0.05, 0.05)
        
        # Create and train model
        model = ImprovedCryptoLSTM(
            input_size=X_train.shape[2],
            hidden_size=hidden_size,
            num_layers=self.model_params['num_layers'],
            dropout=max(0.1, min(0.5, dropout)),
            use_batch_norm=self.model_params['use_batch_norm'],
            prediction_type=self.prediction_type
        ).to(self.device)
        
        # Simple training loop
        train_loader = self.create_dataloader(X_train, y_train)
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        for epoch in range(50):  # Shorter training for ensemble members
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        ensemble_models.append(model)
    
    self.ensemble_models[crypto_id] = ensemble_models
    return ensemble_models

def predict_with_ensemble(self, crypto_id, current_data):
    """Make predictions using the ensemble"""
    if crypto_id not in self.ensemble_models:
        raise ValueError(f"No ensemble available for {crypto_id}")
    
    # Prepare input data
    if len(current_data) < self.sequence_length:
        raise ValueError(f"Need at least {self.sequence_length} data points")
    
    current_data = current_data[self.feature_columns].tail(self.sequence_length)
    scaler = self.scalers[crypto_id]
    scaled_data = scaler.transform(current_data)
    X = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
    
    # Get predictions from all models
    ensemble_preds = []
    for model in self.ensemble_models[crypto_id]:
        model.eval()
        with torch.no_grad():
            pred = model(X).item()
            ensemble_preds.append(pred)
    
    # Ensemble prediction (mean with outlier removal)
    ensemble_preds = np.array(ensemble_preds)
    q1, q3 = np.percentile(ensemble_preds, [25, 75])
    iqr = q3 - q1
    mask = (ensemble_preds >= q1 - 1.5*iqr) & (ensemble_preds <= q3 + 1.5*iqr)
    ensemble_pred = ensemble_preds[mask].mean() if mask.any() else ensemble_preds.mean()
    
    # Calculate confidence interval
    std_dev = ensemble_preds.std()
    confidence_interval = (ensemble_pred - 1.96*std_dev, ensemble_pred + 1.96*std_dev)
    
    last_price = current_data['price'].iloc[-1]
    return ensemble_pred, {
        'last_price': last_price,
        'predicted_price': last_price + ensemble_pred,
        'confidence_interval': confidence_interval,
        'ensemble_variance': std_dev**2
    }

def time_series_cross_validate(self, crypto_id, n_splits=5, test_window=20):
    """Perform time-series cross-validation to better evaluate future performance"""
    if crypto_id not in self.data_by_crypto:
        raise ValueError(f"No data for {crypto_id}")
    
    data = self.data_by_crypto[crypto_id]
    X, y = np.concatenate([data['X_train'], data['X_test']]), np.concatenate([data['y_train'], data['y_test']])
    
    # Define splits for time series data
    n_samples = len(X)
    indices = np.arange(n_samples)
    test_starts = [n_samples - (i+1)*test_window for i in range(n_splits)]
    test_starts = [max(0, start) for start in test_starts]
    
    cv_results = []
    for i, test_start in enumerate(test_starts):
        train_idx, test_idx = indices[:test_start], indices[test_start:test_start+test_window]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
            
        # Train model on this split
        model = self.train_on_split(crypto_id, X[train_idx], y[train_idx])
        
        # Evaluate
        X_test_cv, y_test_cv = X[test_idx], y[test_idx]
        test_loader = self.create_dataloader(X_test_cv, y_test_cv, shuffle=False)
        metrics = self.evaluate_model(model, test_loader, crypto_id, f'cv_split_{i}')
        
        cv_results.append({
            'split': i,
            'metrics': metrics,
            'test_size': len(test_idx)
        })
    
    # Calculate weighted average of metrics
    avg_metrics = {}
    total_weight = sum(r['test_size'] for r in cv_results)
    for metric in cv_results[0]['metrics']:
        avg_metrics[metric] = sum(r['metrics'][metric] * r['test_size'] for r in cv_results) / total_weight
    
    return avg_metrics, cv_results

def train_multi_horizon_model(self, crypto_id, horizons=[1, 3, 7, 14]):
    """Train models to predict at multiple horizons"""
    if crypto_id not in self.data_by_crypto:
        raise ValueError(f"No data for {crypto_id}")
    
    data = self.data_by_crypto[crypto_id]
    feature_df = data['feature_df']
    
    horizon_models = {}
    for horizon in horizons:
        print(f"Training {horizon}-day horizon model for {crypto_id}...")
        
        # Create targets for this horizon
        targets = feature_df['price'].diff(horizon).shift(-horizon)
        valid_idx = ~targets.isna()
        
        # Use only valid data for this horizon
        X = feature_df.values[:-horizon]  # Features
        y = targets.values[:-horizon]     # Targets
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i+self.sequence_length])
            y_seq.append(y[i+self.sequence_length-1])
            
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        
        # Split into train/test
        train_size = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Train model for this horizon
        model = self.train_horizon_model(crypto_id, X_train, y_train, horizon)
        
        # Evaluate
        test_loader = self.create_dataloader(X_test, y_test, shuffle=False)
        metrics = self.evaluate_model(model, test_loader, crypto_id, f'horizon_{horizon}')
        
        horizon_models[horizon] = {
            'model': model,
            'metrics': metrics
        }
    
    self.horizon_models[crypto_id] = horizon_models
    return horizon_models

def add_market_correlation_features(crypto_df, crypto_id):
    """Add features related to correlations with broader markets"""
    try:
        # Load market data (stocks, commodities, etc.)
        market_data = pd.read_csv('market_data.csv')
        market_data['date'] = pd.to_datetime(market_data['date'])
        
        # Merge with crypto data
        crypto_df['date'] = pd.to_datetime(crypto_df['date'])
        merged_df = pd.merge(crypto_df, market_data, on='date', how='left')
        
        # Calculate rolling correlations
        for col in ['sp500', 'gold', 'vix', 'dollar_index']:
            if col in merged_df.columns:
                for window in [14, 30]:
                    merged_df[f'corr_{col}_{window}d'] = merged_df['price'].rolling(window).corr(merged_df[col])
        
        return merged_df
    except:
        # If market data is not available, return original dataframe
        return crypto_df

# Define helper functions at the module level instead of nested
def recalibration_check(model_timestamp, current_time, frequency_days=30):
    """Check if model needs recalibration"""
    time_diff = current_time - model_timestamp
    return time_diff.days >= frequency_days

def recalibrate_model(self, crypto_id, new_data):
    """Recalibrate model with new data"""
    print(f"Recalibrating model for {crypto_id}...")
    
    # Update dataset with new data
    self.update_dataset(crypto_id, new_data)
    
    # Retrain model (can use faster training for recalibration)
    self.train_models(n_splits=3, optimize=False)
    
    # Update model timestamp
    self.model_timestamps[crypto_id] = datetime.now()

def implement_recalibration_strategy(self, crypto_id, recalibration_frequency=30):
    """Implement a strategy to recalibrate models periodically"""
    # This would be used in production environments
    
    # Use module-level functions instead of nested definitions
    if crypto_id not in self.model_timestamps:
        self.model_timestamps[crypto_id] = datetime.now()
        return
        
    # Check if recalibration is needed
    current_time = datetime.now()
    if recalibration_check(self.model_timestamps[crypto_id], current_time, recalibration_frequency):
        # Get new data since last training
        new_data = self.fetch_new_data(crypto_id, self.model_timestamps[crypto_id])
        # Recalibrate the model
        recalibrate_model(self, crypto_id, new_data)

def analyze_feature_importance(self, crypto_id):
    """Analyze which features are most important for predictions"""
    if crypto_id not in self.data_by_crypto:
        raise ValueError(f"No data for {crypto_id}")
    
    data = self.data_by_crypto[crypto_id]
    X_test, y_test = data['X_test'], data['y_test']
    
    # Create baseline prediction
    test_loader = self.create_dataloader(X_test, y_test, shuffle=False)
    
    # Check if best_models contains the model directly or in a dictionary
    if crypto_id not in self.best_models:
        raise ValueError(f"No trained model found for {crypto_id}")
        
    model = self.best_models[crypto_id]
    # Handle different structures of best_models (direct model or dict with 'model' key)
    if isinstance(model, dict) and 'model' in model:
        model = model['model']
    
    if model is None:
        raise ValueError(f"Model for {crypto_id} exists but is None. Training likely failed.")
        
    baseline_metrics = self.evaluate_model(model, test_loader, crypto_id, 'baseline')
    
    # Test feature importance by permutation
    feature_importance = {}
    for i, feature_name in enumerate(self.feature_columns):
        # Create permuted version of test data
        X_permuted = X_test.copy()
        X_permuted[:, :, i] = np.random.permutation(X_permuted[:, :, i])
        
        # Evaluate with permuted feature
        permuted_loader = self.create_dataloader(X_permuted, y_test, shuffle=False)
        permuted_metrics = self.evaluate_model(model, permuted_loader, crypto_id, f'permute_{feature_name}')
        
        # Calculate importance as increase in error
        if self.prediction_type == 'price':
            importance = permuted_metrics['orig_rmse'] - baseline_metrics['orig_rmse']
        else:
            importance = permuted_metrics['val_loss'] - baseline_metrics['val_loss']
            
        feature_importance[feature_name] = importance
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    features, values = zip(*sorted_features[:20])  # Top 20 features
    plt.barh(features, values)
    plt.title(f'Feature Importance for {crypto_id}')
    plt.xlabel('Increase in Error When Feature is Permuted')
    plt.tight_layout()
    plt.savefig(f"{crypto_id}_feature_importance.png")
    plt.close()
    
    return sorted_features

# --- Main Script ---

all_crypto_data = []  # Initialize a list to hold all cryptocurrency data
print(f"Fetching cryptocurrency data for {len(crypto_tickers)} cryptos...")  # Log the number of cryptocurrencies to fetch
print(f"Time range: {start_date_str} to {end_date_str} (within 365-day limit)")  # Log the time range for data fetching

for crypto_id in crypto_tickers:  # Loop through each cryptocurrency ticker
    print(f"Fetching data for {crypto_id}...")  # Log the current cryptocurrency being fetched
    historical_data = fetch_crypto_historical_data(crypto_id, start_date_str, end_date_str)  # Fetch historical data
    if historical_data:  # Check if data was successfully retrieved
        processed_data = process_crypto_data(crypto_id, historical_data)  # Process the retrieved data
        all_crypto_data.extend(processed_data)  # Add processed data to the main list
        print(f"Successfully processed data for {crypto_id} - {len(processed_data)} data points")  # Log success
    wait_time = 6 if not api_key else 2  # Set wait time based on API key availability
    print(f"Waiting {wait_time} seconds before next request...")  # Log wait time
    time.sleep(wait_time)  # Wait before making the next request

# Create a DataFrame from the collected data
df = pd.DataFrame(all_crypto_data)  # Convert the list of data points to a DataFrame
if not df.empty:  # Check if the DataFrame is not empty
    try:
        # Apply feature engineering to the data
        print("Applying feature engineering...")
        
        # Group by crypto_id to process each cryptocurrency separately
        enhanced_dfs = []
        for crypto_id, group_df in df.groupby('crypto_id'):
            print(f"Adding features for {crypto_id}...")
            
            # Sort by date
            group_df = group_df.sort_values('date')
            
            # Add basic features first
            group_df = add_basic_features(group_df)
            
            # Add advanced features
            try:
                group_df = add_advanced_features(group_df)
                print(f"Successfully added advanced features for {crypto_id}")
            except Exception as e:
                print(f"Error adding advanced features for {crypto_id}: {str(e)}")
            
            enhanced_dfs.append(group_df)
        
        # Combine enhanced dataframes
        enhanced_df = pd.concat(enhanced_dfs)
        
        # Save the enhanced DataFrame to CSV
        enhanced_csv_filename = "cryptocurrency_data_enhanced.csv"
        enhanced_df.to_csv(enhanced_csv_filename, index=False)
        print(f"Enhanced cryptocurrency data saved to '{enhanced_csv_filename}'. Rows: {len(enhanced_df)}")
        
        # Also save the original data
        df.to_csv(csv_filename, index=False)  # Save the DataFrame to a CSV file
        print(f"Original cryptocurrency data saved to '{csv_filename}'. Rows: {len(df)}")
        
        # Display feature columns
        print("\nFeature columns:")
        print(", ".join(enhanced_df.columns))
        
        print("\nData sample:")  # Log a sample of the data
        print(enhanced_df.head())  # Print the first few rows of the DataFrame
    except PermissionError:
        print(f"Error: Permission denied when saving to {csv_filename}")  # Log permission error
    except Exception as e:
        print(f"Error saving CSV: {str(e)}")  # Log any other errors during saving
else:
    print("No cryptocurrency data fetched and processed.")  # Log if no data was fetched

# --- Add the following method to your class (e.g., CryptoModel) ---

def calculate_metrics(self, y_true, y_pred, crypto_id, X=None):
    """
    Calculate performance metrics for model evaluation on the original price scale.
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        The true target values (scaled).
    y_pred : numpy.ndarray
        The predicted values from the model (scaled).
    crypto_id : str
        The cryptocurrency ID being evaluated.
    X : numpy.ndarray, optional
        The input sequences (batch_size, sequence_length, input_size) for context.
    
    Returns:
    --------
    dict
        A dictionary containing various evaluation metrics on the original scale.
    """
    # Inverse transform to original scale
    y_true_orig = self._inverse_transform_prices(crypto_id, y_true.flatten())
    y_pred_orig = self._inverse_transform_prices(crypto_id, y_pred.flatten())

    # Compute metrics on original scale
    mse = np.mean((y_true_orig - y_pred_orig) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_orig - y_pred_orig))
    mask = y_true_orig != 0
    mape = np.mean(np.abs((y_true_orig[mask] - y_pred_orig[mask]) / y_true_orig[mask])) * 100 if mask.any() else float('inf')
    r2 = 1 - (np.sum((y_true_orig - y_pred_orig) ** 2) / np.sum((y_true_orig - np.mean(y_true_orig)) ** 2))
    directional_accuracy = np.mean(np.sign(y_true_orig[1:] - y_true_orig[:-1]) == np.sign(y_pred_orig[1:] - y_pred_orig[:-1]))

    # SMAPE on original scale
    smape = 100 / len(y_true_orig) * np.sum(2 * np.abs(y_pred_orig - y_true_orig) / 
                                            (np.abs(y_true_orig) + np.abs(y_pred_orig) + 1e-8))

    # MASE on original scale
    scaling_factor = self.mase_scaling_factors[crypto_id]  # Already in original scale from prepare_data
    mase = mae / scaling_factor if scaling_factor > 0 else float('inf')

    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'DirectionalAccuracy': directional_accuracy,
        'SMAPE': smape,
        'MASE': mase
    }

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: [batch, seq_len, hidden]
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attn_weights * lstm_output, dim=1)
        return context, attn_weights

class ImprovedAttentionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, 
                 use_batch_norm=True, prediction_type='price'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_type = prediction_type
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, 
                           bidirectional=True)
        
        lstm_output_size = hidden_size * 2  # bidirectional doubles output
        
        # Attention mechanism
        self.attention = AttentionLayer(lstm_output_size)
        
        # Normalization and dense layers
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(lstm_output_size)
        
        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.relu2 = nn.ReLU()
        
        # Output layer
        if prediction_type == 'direction':
            self.fc_out = nn.Linear(hidden_size // 2, 3)
        else:
            self.fc_out = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        context, _ = self.attention(lstm_out)
        
        # Apply batch normalization if specified
        if self.use_batch_norm:
            context = self.bn(context)
        
        # Dense layers
        out = self.fc1(context)
        out = self.dropout1(out)
        out = self.relu1(out)
        
        out = self.fc2(out)
        out = self.dropout2(out)
        out = self.relu2(out)
        
        # Output layer
        out = self.fc_out(out)
        
        return out