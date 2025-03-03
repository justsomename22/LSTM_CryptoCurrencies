#trainer.py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tqdm import tqdm
import os
import pickle
import torch.nn as nn
import math

from data_processing import add_technical_indicators, normalize_and_fill_data, add_advanced_features
from models import ImprovedCryptoLSTM
from utils import EarlyStopping

# Define SimpleLSTM at module level
class SimpleLSTM(nn.Module):
    """
    Simple LSTM model for cryptocurrency prediction.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1, output_dim=1, is_direction=False):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.is_direction = is_direction
        self.sigmoid = nn.Sigmoid() if is_direction else nn.Identity()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

# Define Positional Encoding for Transformer
class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models to maintain temporal order.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # For each dimension, calculate the appropriate frequency
        for i in range(0, d_model, 2):
            # Calculate frequency for this dimension
            div_term = math.exp(-(i) * math.log(10000.0) / d_model)
            
            # Apply sin to even indices
            pe[:, i] = torch.sin(position.squeeze(-1) * div_term)
            
            # Apply cos to odd indices (if still within bounds)
            if i + 1 < d_model:  # Check if we've reached the dimension limit
                pe[:, i + 1] = torch.cos(position.squeeze(-1) * div_term)
        
        # Store without batch dimension for flexibility
        self.register_buffer('pe', pe)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Apply positional encoding to the sequence length dimension, not batch dimension
        seq_len = x.size(1)
        pos_encoding = self.pe[:seq_len, :]
        
        # Add positional encoding to each position in the sequence across all batches
        x = x + pos_encoding.unsqueeze(0)
        return self.dropout(x)

# Define Transformer Model
class CryptoTransformer(nn.Module):
    """
    Transformer model for cryptocurrency price prediction.
    """
    def __init__(self, input_dim, d_model=64, n_heads=8, num_layers=2, 
                 dropout=0.1, sequence_length=20, prediction_type='price'):
        super().__init__()
        
        # Store original d_model for reference
        self.original_d_model = d_model
        
        # Ensure d_model is divisible by n_heads
        if d_model % n_heads != 0:
            # Adjust d_model to be divisible by n_heads
            d_model = ((d_model + n_heads - 1) // n_heads) * n_heads
            print(f"Adjusted d_model to {d_model} to be divisible by {n_heads} heads")
        
        # Store the adjusted d_model
        self.adjusted_d_model = d_model
            
        self.prediction_type = prediction_type
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length, dropout=dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, 
                                                  dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, features]
        x = self.input_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.pos_encoder(x)  # Add positional encoding
        x = self.transformer_encoder(x)  # Apply transformer
        
        # Get the last timestep for prediction
        x = x[:, -1, :]  # [batch_size, d_model]
        
        # Output layer
        x = self.output_layer(x)  # [batch_size, 1]
        
        if self.prediction_type == 'direction':
            x = torch.sigmoid(x)
            
        return x.squeeze(-1)

class ImprovedCryptoTrainer:
    """
    Improved trainer class for managing the training and evaluation of cryptocurrency prediction models.
    """
    def __init__(self, data_path, crypto_ids=None, sequence_length=20, test_size=0.2, 
                 target_column='price', device=None, batch_size=32, epochs=100, model_type='lstm',
                 verbosity=1):
        """
        Initialize the trainer with data and model type
        
        Parameters:
        - data_path (str): Path to the cryptocurrency data CSV file
        - crypto_ids (list): List of cryptocurrency IDs to train models for
        - sequence_length (int): Length of the input sequences
        - test_size (float): Proportion of data to use for testing
        - target_column (str): Target column to predict ('price' or 'direction')
        - device (str): Device to use for training ('cuda' or 'cpu')
        - batch_size (int): Batch size for training
        - epochs (int): Maximum number of training epochs
        - model_type (str): Type of model to use ('lstm' or 'transformer')
        - verbosity (int): Level of output verbosity (0=minimal, 1=normal, 2=detailed)
        """
        # Set device
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Store parameters
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.target_column = target_column
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_type = model_type.lower()  # 'lstm' or 'transformer'
        self.verbosity = verbosity
        
        # Validate model_type
        if self.model_type not in ['lstm', 'transformer']:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'lstm' or 'transformer'.")
        
        # Initialize containers
        self.data = None
        self.crypto_ids = crypto_ids
        self.X_train = {}
        self.y_train = {}
        self.X_val = {}
        self.y_val = {}
        self.X_test = {}
        self.y_test = {}
        self.best_params = {}
        self.test_metrics = {}
        
        # Storage for models and data
        self.scalers = {}
        self.best_models = {}
        self.data_by_crypto = {}
        self.feature_columns = None
        self.performance_metrics = {}
        self.target_transforms = {}
        
        # Load and prepare data
        self.prepare_data(data_path)
        
        # Default model parameters
        self.model_params = {
            'hidden_size': 32 if self.model_type == 'lstm' else 32,  # d_model for transformer
            'num_layers': 2,
            'dropout': 0.2,
            'n_heads': 2 if self.model_type == 'transformer' else None  # Only for transformer
        }
            
    def prepare_data(self, data_path):
        """
        Load and prepare data for training
        
        Parameters:
        - data_path (str): Path to the cryptocurrency data
        """
        # Load data
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            self.data['date'] = pd.to_datetime(self.data['date'])
            if self.crypto_ids is None:
                self.crypto_ids = self.data['crypto_id'].unique().tolist()
            
            self.data = normalize_and_fill_data(self.data)
            self.data = add_technical_indicators(self.data)
            self.data = add_advanced_features(self.data)
            
            for crypto_id in self.crypto_ids:
                crypto_data = self.data[self.data['crypto_id'] == crypto_id].copy()
                
                if len(crypto_data) < 100:
                    print(f"Warning: Not enough data for {crypto_id}. Skipping.")
                    continue
                
                crypto_data = crypto_data.sort_values('date').reset_index(drop=True)
                crypto_data = self._apply_r2_enhancing_features(crypto_data)
                
                if self.feature_columns is None:
                    numeric_cols = crypto_data.select_dtypes(include=[np.number]).columns.tolist()
                    self.feature_columns = [col for col in numeric_cols if col != self.target_column and col not in ['date', 'crypto_id', 'index']]
                
                train_size = int(len(crypto_data) * (1 - self.test_size))
                train_data = crypto_data.iloc[:train_size]
                test_data = crypto_data.iloc[train_size:]
                
                X_train_raw = train_data[self.feature_columns].values
                X_test_raw = test_data[self.feature_columns].values
                
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train_raw)
                X_test_scaled = scaler.transform(X_test_raw)
                self.scalers[crypto_id] = scaler
                
                y_train_raw = train_data[self.target_column].values
                y_test_raw = test_data[self.target_column].values
                
                target_scaler = RobustScaler()
                y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
                y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
                self.target_transforms[crypto_id] = target_scaler
                
                X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_scaled, self.target_column == 'direction')
                X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test_scaled, self.target_column == 'direction')
                
                self.data_by_crypto[crypto_id] = {
                    'train_data': train_data,
                    'test_data': test_data,
                    'X_train': X_train_seq,
                    'y_train': y_train_seq,
                    'X_test': X_test_seq,
                    'y_test': y_test_seq,
                    'y_test_raw': y_test_raw,
                    'feature_names': self.feature_columns
                }
                
                print(f"Prepared data for {crypto_id}. Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")
            
            print("Data preparation completed.")
        except Exception as e:
            print(f"Error in data preparation: {str(e)}")
            raise

    def _apply_r2_enhancing_features(self, data):
        """
        Apply additional features to enhance R2 score
        
        Parameters:
        - data (pd.DataFrame): DataFrame to enhance
        
        Returns:
        - pd.DataFrame: Enhanced DataFrame
        """
        df = data.copy()
        
        # Add more complex indicators for better R2
        try:
            # Price momentum features
            df['price_momentum'] = df['price'].pct_change(5) / df['price'].pct_change(1)
            df['price_acceleration'] = df['price'].pct_change(1).diff(1)
            
            # Volatility based features
            df['volatility_ratio'] = df['volatility_7d'] / df['volatility_30d']
            
            # Mean reversion features
            for window in [7, 14, 30]:
                # Z-score (how many std devs price is from moving average)
                ma = df['price'].rolling(window).mean()
                std = df['price'].rolling(window).std()
                df[f'z_score_{window}'] = (df['price'] - ma) / std
                
                # Distance from moving average
                df[f'ma_distance_{window}'] = (df['price'] - ma) / ma
            
            # Polynomial features of key indicators
            for indicator in ['rsi', 'macd', 'volatility_7d']:
                if indicator in df.columns:
                    df[f'{indicator}_squared'] = df[indicator] ** 2
            
            # Interaction terms
            if 'rsi' in df.columns and 'macd' in df.columns:
                df['rsi_macd_interaction'] = df['rsi'] * df['macd']
            
            # Support and resistance features
            df['high_low_ratio'] = df['price'].rolling(20).max() / df['price'].rolling(20).min()
            
            # Fill NAs created by these calculations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(method='bfill').fillna(method='ffill').fillna(0)
        except Exception as e:
            print(f"Warning: Error applying R2 enhancing features: {str(e)}")
        
        return df

    def _create_sequences(self, data, target_col_idx, is_classification=False):
        """
        Create sequences for time series prediction
        
        Parameters:
        - data (np.ndarray): Feature data
        - target_col_idx (np.ndarray): Target data
        - is_classification (bool): Whether this is a classification task
        
        Returns:
        - tuple: (X_sequences, y_sequences)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target_col_idx[i + self.sequence_length])
        
        return np.array(X), np.array(y)

    def create_dataloader(self, X, y, batch_size=None, shuffle=True):
        """
        Create PyTorch DataLoader
        
        Parameters:
        - X (np.ndarray): Feature data
        - y (np.ndarray): Target data
        - batch_size (int): Batch size
        - shuffle (bool): Whether to shuffle the data
        
        Returns:
        - DataLoader: PyTorch DataLoader
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.reshape(-1, 1) if len(y.shape) == 1 else y)
        
        # Create TensorDataset
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Create DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        return dataloader

    def optimize_hyperparameters(self, crypto_id, X_train, y_train, n_trials=20):
        """
        Optimize hyperparameters for a specific cryptocurrency model.
        """
        print(f"Optimizing hyperparameters for {crypto_id}...")
        
        # Create a validation set for optimization
        train_size = int(0.8 * len(X_train))
        X_train_opt, X_val_opt = X_train[:train_size], X_train[train_size:]
        y_train_opt, y_val_opt = y_train[:train_size], y_train[train_size:]
        
        # Create a study
        study = optuna.create_study(direction="minimize")
        
        def objective(trial):
            if self.model_type == 'transformer':
                n_heads = trial.suggest_int('n_heads', 2, 4)  # Reduced from 2-8
                d_model_min = max(32, n_heads * 4)
                d_model_max = 128  # Reduced from 256
                d_model = trial.suggest_int('d_model', d_model_min, d_model_max, step=n_heads)
                
                params = {
                    'd_model': d_model,
                    'n_heads': n_heads,
                    'num_layers': trial.suggest_int('num_layers', 1, 2),  # Reduced from 1-4
                    'dropout': trial.suggest_float('dropout', 0.1, 0.3),  # Reduced from 0.1-0.5
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),  # Adjusted lower bound up
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),  # Adjusted lower bound up
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),  # Removed 16, favoring larger batches
                    'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])  # Removed AdamW for simplicity
                }
            else:  # LSTM
                params = {
                    'hidden_size': trial.suggest_int('hidden_size', 16, 64),  # Reduced from 32-256
                    'num_layers': trial.suggest_int('num_layers', 1, 2),  # Reduced from 1-3
                    'dropout': trial.suggest_float('dropout', 0.1, 0.3),  # Reduced from 0.1-0.5
                    'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                    'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
                    'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
                    'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'RMSprop'])
                }
            
            # Create model based on type
            if self.model_type == 'transformer':
                original_d_model = params.get('d_model', 64)
                
                model = CryptoTransformer(
                    input_dim=X_train.shape[2],
                    d_model=original_d_model,
                    n_heads=params.get('n_heads', 4),
                    num_layers=params.get('num_layers', 2),
                    dropout=params.get('dropout', 0.1),
                    sequence_length=self.sequence_length,
                    prediction_type=self.target_column
                ).to(self.device)
                
                # Log if d_model was adjusted
                if hasattr(model, 'adjusted_d_model') and model.adjusted_d_model != original_d_model:
                    # Update the params dictionary with the actual value used
                    params['d_model'] = model.adjusted_d_model
                    
                    # Only try to access trial if we're in an optimization context
                    # Check if 'trial' variable exists in local scope (during optimization)
                    if 'trial' in locals() or 'trial' in globals():
                        # Store original d_model in trial's user_attrs for reference
                        trial.set_user_attr('original_d_model', original_d_model)
                        if self.verbosity >= 1:
                            print(f"Trial #{trial.number}: d_model adjusted from {original_d_model} to {model.adjusted_d_model} to be divisible by n_heads={params.get('n_heads', 4)}")
                    else:
                        # We're not in optimization context, just print the info
                        if self.verbosity >= 1:
                            print(f"d_model adjusted from {original_d_model} to {model.adjusted_d_model} to be divisible by n_heads={params.get('n_heads', 4)}")
            else:
                model = ImprovedCryptoLSTM(
                    input_dim=X_train.shape[2],
                    hidden_dim=params['hidden_size'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout'],
                    output_dim=1,
                    sequence_length=self.sequence_length,
                    prediction_type=self.target_column
                ).to(self.device)
            
            # Select optimizer
            if params['optimizer'] == 'Adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            else:
                optimizer = torch.optim.RMSprop(model.parameters(), lr=params['learning_rate'], weight_decay=params['weight_decay'])
            
            # Define loss function
            criterion = torch.nn.HuberLoss(delta=1.0) if self.target_column != 'direction' else torch.nn.BCELoss()
            
            # Create dataloaders with suggested batch size
            train_loader_trial = self.create_dataloader(X_train_opt, y_train_opt, batch_size=params['batch_size'])
            val_loader_trial = self.create_dataloader(X_val_opt, y_val_opt, batch_size=params['batch_size'], shuffle=False)
            
            # Train for a few epochs
            early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=False)
            best_val_loss = float('inf')
            
            for epoch in range(10):
                model.train()
                train_loss = 0
                for inputs, targets in train_loader_trial:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    inputs = inputs.float()
                    targets = targets.float().view(-1, 1)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader_trial:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        inputs = inputs.float()
                        targets = targets.float().view(-1, 1)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader_trial)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                if early_stopping.step(val_loss):
                    break
            
            return best_val_loss
        
        optuna_verbosity = 1 if self.verbosity >= 1 else 0
        study.optimize(objective, n_trials=n_trials, show_progress_bar=self.verbosity >= 2)
        
        best_params = study.best_params.copy()
        best_trial = study.best_trial
        
        # Check if there was an adjusted d_model value
        if 'd_model' in best_params and self.model_type == 'transformer' and best_trial.user_attrs.get('original_d_model') is not None:
            original_d_model = best_trial.user_attrs['original_d_model']
            if original_d_model != best_params['d_model']:
                if self.verbosity >= 1:
                    print(f"Note: Original d_model={original_d_model} was adjusted to {best_params['d_model']} to be divisible by n_heads={best_params['n_heads']}")
        
        if self.verbosity >= 1:
            print(f"\nBest hyperparameters for {crypto_id}: {best_params}")
            print(f"Best validation loss: {study.best_value:.6f}")
        
        return best_params

    def train_models(self, n_splits=5, optimize=True, n_trials=100):
        """
        Train multiple cryptocurrency models with optional cross-validation and hyperparameter optimization.
        
        Parameters:
        - n_splits (int): Number of cross-validation splits
        - optimize (bool): Whether to optimize hyperparameters
        - n_trials (int): Number of optimization trials
        """
        crypto_ids_to_train = [cid for cid in self.crypto_ids if cid in self.data_by_crypto]
        
        if self.verbosity >= 1:
            print(f"\n{'='*50}\nTraining models for {len(crypto_ids_to_train)} cryptocurrencies\n{'='*50}")
            crypto_iterator = tqdm(enumerate(crypto_ids_to_train), total=len(crypto_ids_to_train), desc="Cryptocurrencies")
        else:
            crypto_iterator = enumerate(crypto_ids_to_train)
            
        for i, crypto_id in crypto_iterator:
            if self.verbosity >= 1:
                if not isinstance(crypto_iterator, tqdm):
                    print(f"\n{'='*50}\nTraining model for {crypto_id} ({i+1}/{len(crypto_ids_to_train)})\n{'='*50}")
            
            X_train = self.data_by_crypto[crypto_id]['X_train']
            y_train = self.data_by_crypto[crypto_id]['y_train']
            X_test = self.data_by_crypto[crypto_id]['X_test']
            y_test = self.data_by_crypto[crypto_id]['y_test']
            
            if optimize:
                if self.verbosity >= 1:
                    print(f"\nStarting hyperparameter optimization for {crypto_id}...")
                best_params = self.optimize_hyperparameters(crypto_id, X_train, y_train, n_trials=n_trials)
                if self.verbosity >= 1:
                    print(f"Best parameters for {crypto_id}: {best_params}")
                    # If d_model was adjusted, make sure to mention this
                    if self.model_type == 'transformer' and 'd_model' in best_params:
                        d_model = best_params['d_model']
                        n_heads = best_params['n_heads']
                        if d_model % n_heads == 0:
                            print(f"Verified: d_model={d_model} is divisible by n_heads={n_heads}")
                self.best_params[crypto_id] = best_params
            else:
                if self.verbosity >= 1:
                    print(f"Using default hyperparameters for {crypto_id}")
                best_params = self.model_params
                
            if n_splits > 1:
                if self.verbosity >= 1:
                    print(f"Performing {n_splits}-fold cross-validation for {crypto_id}...")
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = []
                
                if self.verbosity >= 1:
                    fold_iterator = tqdm(enumerate(kf.split(X_train)), total=n_splits, desc=f"CV Folds", leave=False)
                else:
                    fold_iterator = enumerate(kf.split(X_train))
                
                for fold, (train_idx, val_idx) in fold_iterator:
                    X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
                    y_train_cv, y_val_cv = y_train[train_idx], y_train[val_idx]
                    
                    if self.verbosity >= 1 and not isinstance(fold_iterator, tqdm):
                        print(f"Training fold {fold+1}/{n_splits}")
                    
                    model, metrics = self.train_single_model(crypto_id, {**best_params, 'fold': fold}, trial_num=fold+1)
                    
                    if metrics is not None:
                        cv_scores.append(metrics)
                    else:
                        cv_scores.append({'val_loss': float('inf')})
                
                # Extract valid val_loss values
                val_losses = []
                for score in cv_scores:
                    if isinstance(score, dict) and 'val_loss' in score:
                        if isinstance(score['val_loss'], (float, int)):
                            val_losses.append(score['val_loss'])
                        elif isinstance(score['val_loss'], list) and score['val_loss']:
                            # Take the last value if it's a list
                            val_losses.append(score['val_loss'][-1])
                
                if val_losses:
                    mean_val_loss = np.mean(val_losses)
                    if self.verbosity >= 1:
                        print(f"\nCross-validation results for {crypto_id}:")
                        print(f"Mean validation loss: {mean_val_loss:.6f}")
            
            if self.verbosity >= 1:
                print(f"\nTraining final model for {crypto_id} on all training data...")
                
            final_model, final_metrics = self.train_single_model(crypto_id, best_params, trial_num='final')
            
            if final_model is not None:
                self.best_models[crypto_id] = final_model
                
                if self.verbosity >= 1:
                    print(f"Evaluating {crypto_id} model on test set...")
                    
                self.X_test[crypto_id] = X_test
                self.y_test[crypto_id] = y_test
                
                test_predictions = self.predict(crypto_id, X_test)
                
                if self.target_column == 'direction':
                    test_predictions = (test_predictions > 0.5).astype(int)
                
                if self.target_column == 'price' and crypto_id in self.target_transforms:
                    test_predictions_2d = test_predictions.reshape(-1, 1)
                    y_test_2d = y_test.reshape(-1, 1)
                    test_predictions = self.target_transforms[crypto_id].inverse_transform(test_predictions_2d).flatten()
                    y_test_actual = self.target_transforms[crypto_id].inverse_transform(y_test_2d).flatten()
                else:
                    y_test_actual = y_test
                
                mse = np.mean((test_predictions - y_test_actual) ** 2)
                mae = np.mean(np.abs(test_predictions - y_test_actual))
                
                if self.target_column == 'price':
                    direction_pred = np.diff(test_predictions, prepend=test_predictions[0])
                    direction_actual = np.diff(y_test_actual, prepend=y_test_actual[0])
                    direction_accuracy = np.mean((direction_pred > 0) == (direction_actual > 0))
                    y_mean = np.mean(y_test_actual)
                    ss_total = np.sum((y_test_actual - y_mean) ** 2)
                    ss_residual = np.sum((y_test_actual - test_predictions) ** 2)
                    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else float('-inf')
                    self.test_metrics[crypto_id] = {'mse': mse, 'mae': mae, 'direction_accuracy': direction_accuracy, 'r2': r2}
                else:
                    accuracy = np.mean(test_predictions == y_test_actual)
                    self.test_metrics[crypto_id] = {'mse': mse, 'mae': mae, 'accuracy': accuracy}
                
                if self.verbosity >= 1:
                    print(f"Saving model for {crypto_id}...")
                    
                torch.save(final_model.state_dict(), f"{crypto_id}_best_model.pth")
                with open(f"{crypto_id}_best_model_{crypto_id}_{self.target_column}.pkl", 'wb') as f:
                    pickle.dump(self, f)
            else:
                print(f"Warning: Training failed for {crypto_id}. No model to evaluate.")
        
        if self.verbosity >= 1:
            print("\nTraining completed for all cryptocurrencies.")
            
    def train_single_model(self, crypto_id, params, trial_num=1):
        """
        Train a single cryptocurrency model (LSTM or Transformer).
        """
        try:
            X_train = self.data_by_crypto[crypto_id]['X_train']
            y_train = self.data_by_crypto[crypto_id]['y_train']
            
            train_size = int(0.8 * len(X_train))
            X_train_split, X_val_split = X_train[:train_size], X_train[train_size:]
            y_train_split, y_val_split = y_train[:train_size], y_train[train_size:]
            
            input_dim = X_train.shape[2]
            if self.verbosity >= 1:
                print(f"Creating model with input dimension: {input_dim}")
            
            # Create model based on type
            if self.model_type == 'transformer':
                original_d_model = params.get('d_model', 64)
                
                model = CryptoTransformer(
                    input_dim=input_dim,
                    d_model=original_d_model,
                    n_heads=params.get('n_heads', 4),
                    num_layers=params.get('num_layers', 2),
                    dropout=params.get('dropout', 0.1),
                    sequence_length=self.sequence_length,
                    prediction_type=self.target_column
                ).to(self.device)
                
                # Log if d_model was adjusted
                if hasattr(model, 'adjusted_d_model') and model.adjusted_d_model != original_d_model:
                    # Update the params dictionary with the actual value used
                    params['d_model'] = model.adjusted_d_model
                    
                    # Only try to access trial if we're in an optimization context
                    # Check if 'trial' variable exists in local scope (during optimization)
                    if 'trial' in locals() or 'trial' in globals():
                        # Store original d_model in trial's user_attrs for reference
                        trial.set_user_attr('original_d_model', original_d_model)
                        if self.verbosity >= 1:
                            print(f"Trial #{trial.number}: d_model adjusted from {original_d_model} to {model.adjusted_d_model} to be divisible by n_heads={params.get('n_heads', 4)}")
                    else:
                        # We're not in optimization context, just print the info
                        if self.verbosity >= 1:
                            print(f"d_model adjusted from {original_d_model} to {model.adjusted_d_model} to be divisible by n_heads={params.get('n_heads', 4)}")
            else:
                try:
                    model = ImprovedCryptoLSTM(
                        input_dim=input_dim,
                        hidden_dim=params.get('hidden_size', 128),
                        num_layers=params.get('num_layers', 2),
                        dropout=params.get('dropout', 0.3),
                        sequence_length=self.sequence_length,
                        prediction_type=self.target_column
                    ).to(self.device)
                except Exception as e:
                    print(f"Error creating ImprovedCryptoLSTM: {str(e)}")
                    print("Falling back to SimpleLSTM")
                    is_direction = self.target_column == 'direction'
                    model = SimpleLSTM(
                        input_dim=input_dim,
                        hidden_dim=params.get('hidden_size', 64),
                        num_layers=params.get('num_layers', 1),
                        dropout=params.get('dropout', 0.1),
                        is_direction=is_direction
                    ).to(self.device)
            
            if self.verbosity >= 1:
                print(f"Model created successfully: {model.__class__.__name__}")
            
            criterion = torch.nn.HuberLoss(delta=1.0) if self.target_column != 'direction' else torch.nn.BCELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=params.get('learning_rate', 0.001), 
                                       weight_decay=params.get('weight_decay', 1e-5))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=self.verbosity >= 2)
            
            train_loader = self.create_dataloader(X_train_split, y_train_split, batch_size=params.get('batch_size', 32))
            val_loader = self.create_dataloader(X_val_split, y_val_split, batch_size=params.get('batch_size', 32), shuffle=False)
            
            # Cap epochs at 20
            epochs = min(self.epochs, 20)
            patience = params.get('patience', 5)
            
            early_stopping = EarlyStopping(patience=patience, min_delta=0.001, verbose=self.verbosity >= 2)
            epochs = min(self.epochs, 20)
            
            best_val_loss = float('inf')
            best_model_state = None
            training_history = {'train_loss': [], 'val_loss': []}
            
            if self.verbosity >= 1:
                print(f"Starting training for {crypto_id} (trial {'final' if trial_num == 'final' else trial_num})...")
                print(f"Training with {len(train_loader)} batches, validation with {len(val_loader)} batches")
            
            # Use tqdm for progress bar if verbosity >= 1
            epoch_iterator = tqdm(range(epochs), desc=f"Training {crypto_id}", disable=self.verbosity < 1)
            
            for epoch in epoch_iterator:
                model.train()
                train_loss = 0
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    inputs = inputs.float()
                    targets = targets.float().view(-1, 1)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader) if len(train_loader) > 0 else float('inf')
                
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        inputs = inputs.float()
                        targets = targets.float().view(-1, 1)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader) if len(val_loader) > 0 else float('inf')
                
                scheduler.step(val_loss)
                
                # Update progress bar with loss values
                if self.verbosity >= 1:
                    epoch_iterator.set_postfix({"Train Loss": f"{train_loss:.6f}", "Val Loss": f"{val_loss:.6f}"})
                
                # Only print detailed epoch logs if verbosity is high
                if self.verbosity >= 2:
                    print(f"Epoch {epoch+1}/{epochs}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
                
                training_history['train_loss'].append(train_loss)
                training_history['val_loss'].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    if self.verbosity >= 2:
                        print(f"New best model saved (val_loss: {best_val_loss:.6f})")
                
                if early_stopping.step(val_loss):
                    if self.verbosity >= 1:
                        print(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            self.performance_metrics[crypto_id] = {
                'training_history': training_history,
                'best_val_loss': best_val_loss,
                'params': params
            }
            
            return model, training_history
        
        except Exception as e:
            print(f"Error in train_single_model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def predict(self, crypto_id, X=None):
        """
        Make predictions with the trained model for a given cryptocurrency.
        
        Parameters:
        - crypto_id (str): Cryptocurrency ID
        - X (np.ndarray, optional): Input data for prediction. If None, uses test data.
        
        Returns:
        - np.ndarray: Predictions
        """
        if crypto_id not in self.best_models:
            raise ValueError(f"No trained model found for {crypto_id}")
        
        if X is None:
            if crypto_id not in self.X_test:
                raise ValueError(f"No test data found for {crypto_id}")
            X = self.X_test[crypto_id]
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        model = self.best_models[crypto_id]
        if model is None:
            raise ValueError(f"Model for {crypto_id} exists but is None. Training likely failed.")
            
        model.eval()
        with torch.no_grad():
            predictions = model(X_tensor).cpu().numpy()
        predictions = predictions.reshape(-1)
        return predictions 