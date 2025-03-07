#trainer.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import os
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm
import pickle
import traceback
from utils import EarlyStopping
from data_processing import add_technical_indicators, normalize_and_fill_data, add_advanced_features, add_target_columns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_trainer.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CryptoTrainer")

class SimpleLSTM(nn.Module):    
    """
    Simple LSTM model for cryptocurrency price prediction.

    Attributes:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in the LSTM layer.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate for regularization.
        output_dim (int): Number of output features.
        is_direction (bool): If True, applies sigmoid activation for binary classification.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, dropout=0.1, output_dim=1, is_direction=False):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid() if is_direction else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output predictions.
        """
        try:
            lstm_out, _ = self.lstm(x)
            return self.sigmoid(self.fc(lstm_out[:, -1, :]))
        except Exception as e:
            logger.error(f"Error in SimpleLSTM forward pass: {str(e)}")
            raise

class CryptoTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=1, dropout=0.1, sequence_length=10, prediction_type='price'):
        super().__init__()
        if d_model % n_heads != 0:
            logger.warning(f"d_model ({d_model}) is not divisible by n_heads ({n_heads}). Adjusting d_model.")
            
        self.d_model = d_model if d_model % n_heads == 0 else ((d_model + n_heads - 1) // n_heads) * n_heads
        self.prediction_type = prediction_type
        self.input_embedding = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = nn.Dropout(dropout)  # Simplified: no full PositionalEncoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(self.d_model, 1)

    def forward(self, x):
        try:
            x = self.input_embedding(x)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = self.output_layer(x[:, -1, :])
            return torch.sigmoid(x.squeeze(-1)) if self.prediction_type == 'direction' else x.squeeze(-1)
        except Exception as e:
            logger.error(f"Error in CryptoTransformer forward pass: {str(e)}")
            raise

class ModelNotFoundError(Exception):
    """Exception raised when a model for a specific crypto ID is not found."""
    pass

class DataPreparationError(Exception):
    """Exception raised when there is an error in data preparation."""
    pass

class ModelTrainingError(Exception):
    """Exception raised when there is an error in model training."""
    pass

class ImprovedCryptoTrainer:
    """
    Trainer class for managing the training process of cryptocurrency prediction models.

    Attributes:
        data_path (str): Path to the dataset.
        crypto_ids (list): List of cryptocurrency IDs to train on.
        sequence_length (int): Length of input sequences for the model.
        test_size (float): Proportion of data to use for testing.
        target_column (str): The target column to predict (e.g., 'price').
        device (str): Device to run the model on ('cpu' or 'cuda').
        batch_size (int): Number of samples per batch.
        epochs (int): Number of training epochs.
        model_type (str): Type of model to use ('lstm' or 'transformer').
        verbosity (int): Level of logging verbosity.
        cache (bool): Whether to use cached data.
        use_garch (bool): Whether to include GARCH volatility features in the model.
        use_bollinger (bool): Whether to include Bollinger Bands indicators in the model.
        use_macd (bool): Whether to include MACD indicators in the model.
        use_moving_avg (bool): Whether to include Moving Average indicators in the model.
    """
    def __init__(self, data_path, crypto_ids=None, sequence_length=10, test_size=0.2, target_column='price',
                 device=None, batch_size=128, epochs=20, model_type='transformer', verbosity=1, cache=False, 
                 use_garch=True, use_bollinger=True, use_macd=True, use_moving_avg=True):
        # Initialize device
        self.device = None
        try:
            if device:
                self.device = device
            else:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
        except Exception as e:
            logger.error(f"Error setting device: {str(e)}. Defaulting to CPU.")
            self.device = 'cpu'
        
        # Initialize other parameters
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.target_column = target_column
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_type = model_type.lower()
        self.verbosity = verbosity
        self.cache = cache
        self.use_garch = use_garch
        self.use_bollinger = use_bollinger
        self.use_macd = use_macd
        self.use_moving_avg = use_moving_avg
        
        if self.use_garch:
            logger.info("GARCH volatility modeling enabled")
        else:
            logger.info("GARCH volatility modeling disabled")
            
        if self.use_bollinger:
            logger.info("Bollinger Bands indicators enabled")
        else:
            logger.info("Bollinger Bands indicators disabled")
            
        if self.use_macd:
            logger.info("MACD indicators enabled")
        else:
            logger.info("MACD indicators disabled")
            
        if self.use_moving_avg:
            logger.info("Moving Average indicators enabled")
        else:
            logger.info("Moving Average indicators disabled")
        
        # Validate model type
        if not (self.model_type in ['lstm', 'transformer'] or 
                self.model_type.startswith('ensemble_') or 
                self.model_type.startswith('feature_')):
            error_msg = f"Invalid model type: {self.model_type}. Must be 'lstm', 'transformer', or an ensemble/feature variant."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Initialize data-related attributes
        self.data = None
        self.crypto_ids = crypto_ids
        self.best_models = {}
        self.scalers = {}
        self.target_transforms = {}
        self.data_by_crypto = {}
        self.test_metrics = {}
        self.model_timestamps = {}
        
        # Prepare data
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            self.prepare_data(data_path)
        except FileNotFoundError as e:
            logger.error(f"File not found: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            logger.debug(traceback.format_exc())
            raise DataPreparationError(f"Failed to initialize trainer: {str(e)}")

    def prepare_data(self, data_path):
        """
        Prepare and preprocess the data for training.

        Parameters:
            data_path (str): Path to the CSV data file.

        Raises:
            DataPreparationError: If there is an error in data preparation.
        """
        logger.info(f"Preparing data from {data_path}")
        
        # Try loading cached data if available
        if self.cache and self.crypto_ids:
            try:
                all_cached = True
                for cid in self.crypto_ids:
                    if not all(os.path.exists(f"{cid}_{file}") for file in 
                              ["X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy", "scaler.pkl", "target_scaler.pkl"]):
                        all_cached = False
                        break
                
                if all_cached:
                    logger.info("Loading cached preprocessed data...")
                    for crypto_id in self.crypto_ids:
                        try:
                            self.data_by_crypto[crypto_id] = {
                                'X_train': np.load(f"{crypto_id}_X_train.npy"),
                                'y_train': np.load(f"{crypto_id}_y_train.npy"),
                                'X_test': np.load(f"{crypto_id}_X_test.npy"),
                                'y_test': np.load(f"{crypto_id}_y_test.npy"),
                            }
                            
                            with open(f"{crypto_id}_scaler.pkl", 'rb') as f:
                                self.scalers[crypto_id] = pickle.load(f)
                                
                            with open(f"{crypto_id}_target_scaler.pkl", 'rb') as f:
                                self.target_transforms[crypto_id] = pickle.load(f)
                                
                        except Exception as e:
                            logger.error(f"Error loading cached data for {crypto_id}: {str(e)}")
                            all_cached = False
                            break
                            
                    if all_cached:
                        logger.info("Successfully loaded all cached data.")
                        return
                    else:
                        logger.warning("Failed to load some cached data. Processing data from scratch.")
            except Exception as e:
                logger.error(f"Error checking cached data: {str(e)}")
                logger.info("Proceeding with data processing from scratch.")

        # Process data from scratch
        try:
            self.data = pd.read_csv(data_path)
            logger.info(f"Loaded data with shape: {self.data.shape}")
            
            if 'date' not in self.data.columns:
                raise DataPreparationError("'date' column not found in dataset")
                
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
            
            # Check for NaT values after conversion
            if self.data['date'].isna().any():
                invalid_dates = self.data.loc[self.data['date'].isna(), :].index.tolist()
                logger.warning(f"Found {len(invalid_dates)} invalid date entries. First few: {invalid_dates[:5]}")
                self.data = self.data.dropna(subset=['date'])
                logger.info(f"Dropped rows with invalid dates. New shape: {self.data.shape}")
                
            if 'crypto_id' not in self.data.columns:
                raise DataPreparationError("'crypto_id' column not found in dataset")
                
            if self.crypto_ids is None:
                self.crypto_ids = self.data['crypto_id'].unique().tolist()
                logger.info(f"Found {len(self.crypto_ids)} unique crypto IDs")
            
            try:
                from data_processing import normalize_and_fill_data, add_technical_indicators, add_advanced_features
                self.data = normalize_and_fill_data(self.data)
                self.data = add_technical_indicators(self.data, use_bollinger=self.use_bollinger, use_macd=self.use_macd, use_moving_avg=self.use_moving_avg)
                self.data = add_target_columns(self.data)
                # Add GARCH and other advanced features if enabled
                if self.use_garch:
                    logger.info("Adding GARCH volatility features to data")
                    self.data = add_advanced_features(self.data)
                else:
                    logger.info("Skipping GARCH volatility features")
            except ImportError as e:
                logger.error(f"Failed to import data_processing module: {str(e)}")
                raise
            except Exception as e:
                logger.error(f"Error in data preprocessing: {str(e)}")
                raise DataPreparationError(f"Data preprocessing failed: {str(e)}")
            
            # Process each crypto ID
            for crypto_id in self.crypto_ids:
                try:
                    crypto_data = self.data[self.data['crypto_id'] == crypto_id].sort_values('date')
                    logger.info(f"{crypto_id} - Raw rows: {len(crypto_data)}")
                    
                    train_size = int(len(crypto_data) * (1 - self.test_size))
                    logger.info(f"{crypto_id} - Train rows: {train_size}, Test rows: {len(crypto_data) - train_size}")
                    
                    train_data, test_data = crypto_data.iloc[:train_size], crypto_data.iloc[train_size:]
                    
                    # Expanded feature set including GARCH volatility features if enabled
                    features = ['price', 'rsi', 'volatility_7d']
                    
                    # Add GARCH features if they exist in the data and GARCH is enabled
                    if self.use_garch:
                        garch_features = ['garch_vol', 'garch_vol_forecast', 'garch_regime']
                        for feature in garch_features:
                            if feature in crypto_data.columns:
                                features.append(feature)
                                logger.info(f"Added GARCH feature: {feature}")
                    
                    # Add Bollinger Bands features if they exist in the data and Bollinger Bands are enabled
                    if self.use_bollinger:
                        bollinger_features = ['bollinger_mavg', 'bollinger_hband', 'bollinger_lband', 
                                            'bollinger_width', 'bollinger_pct_b', 
                                            'dist_to_upper', 'dist_to_lower']
                        for feature in bollinger_features:
                            if feature in crypto_data.columns:
                                features.append(feature)
                                logger.info(f"Added Bollinger Bands feature: {feature}")
                    
                    # Add MACD features if they exist in the data and MACD is enabled
                    if self.use_macd:
                        macd_features = ['macd_line', 'macd_signal', 'macd_hist', 'macd_divergence']
                        for feature in macd_features:
                            if feature in crypto_data.columns:
                                features.append(feature)
                                logger.info(f"Added MACD feature: {feature}")
                    
                    # Add Moving Average features if they exist in the data and Moving Averages are enabled
                    if self.use_moving_avg:
                        ma_features = [
                            # Simple Moving Averages
                            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_200',
                            # Exponential Moving Averages
                            'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
                            # Moving Average Crossover Signals
                            'ma_cross_50_200', 'ma_cross_10_50', 'ma_cross_5_20',
                            # Price vs EMA
                            'price_vs_ema_20',
                            # Percent distance from moving averages
                            'pct_from_sma_50', 'pct_from_sma_200'
                        ]
                        for feature in ma_features:
                            if feature in crypto_data.columns:
                                features.append(feature)
                                logger.info(f"Added Moving Average feature: {feature}")
                    
                    # Check if all required features exist
                    missing_features = [f for f in features if f not in crypto_data.columns]
                    if missing_features:
                        logger.error(f"Missing features for {crypto_id}: {missing_features}")
                        continue
                    
                    X_train_raw, X_test_raw = train_data[features].values, test_data[features].values
                    
                    # Check for NaN values
                    if np.isnan(X_train_raw).any() or np.isnan(X_test_raw).any():
                        logger.warning(f"NaN values found in {crypto_id} data. Filling with zeros.")
                        X_train_raw = np.nan_to_num(X_train_raw, nan=0.0)
                        X_test_raw = np.nan_to_num(X_test_raw, nan=0.0)
                    
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train_raw)
                    X_test_scaled = scaler.transform(X_test_raw)
                    self.scalers[crypto_id] = scaler
                    
                    if self.target_column not in train_data.columns:
                        logger.error(f"Target column '{self.target_column}' not found in {crypto_id} data")
                        continue
                        
                    y_train_raw, y_test_raw = train_data[self.target_column].values, test_data[self.target_column].values
                    target_scaler = RobustScaler()
                    y_train_scaled = target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
                    y_test_scaled = target_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()
                    self.target_transforms[crypto_id] = target_scaler
                    
                    # Create sequences
                    X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_scaled)
                    X_test_seq, y_test_seq = self._create_sequences(X_test_scaled, y_test_scaled)
                    
                    # Log the number of sequences created
                    logger.info(f"{crypto_id} - Train sequences: {len(X_train_seq)}, Test sequences: {len(X_test_seq)}")
                    
                    if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                        logger.warning(f"No sequences created for {crypto_id}. Skipping.")
                        continue
                    elif len(X_train_seq) < self.batch_size or len(X_test_seq) < self.batch_size:
                        logger.info(f"{crypto_id} has fewer sequences than batch_size ({len(X_train_seq)} train, {len(X_test_seq)} test), but proceeding.")
                    
                    self.data_by_crypto[crypto_id] = {
                        'X_train': X_train_seq, 'y_train': y_train_seq,
                        'X_test': X_test_seq, 'y_test': y_test_seq
                    }
                    
                    logger.info(f"Successfully processed {crypto_id} data. Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")
                    
                    # Save cached data if requested
                    if self.cache:
                        try:
                            np.save(f"{crypto_id}_X_train.npy", X_train_seq)
                            np.save(f"{crypto_id}_y_train.npy", y_train_seq)
                            np.save(f"{crypto_id}_X_test.npy", X_test_seq)
                            np.save(f"{crypto_id}_y_test.npy", y_test_seq)
                            with open(f"{crypto_id}_scaler.pkl", 'wb') as f:
                                pickle.dump(scaler, f)
                            with open(f"{crypto_id}_target_scaler.pkl", 'wb') as f:
                                pickle.dump(target_scaler, f)
                            logger.info(f"Cached data for {crypto_id}")
                        except Exception as e:
                            logger.error(f"Failed to cache data for {crypto_id}: {str(e)}")
                            
                except Exception as e:
                    logger.error(f"Error processing data for {crypto_id}: {str(e)}")
                    logger.debug(traceback.format_exc())
            
            if not self.data_by_crypto:
                raise DataPreparationError("No valid crypto data could be processed")
                
            logger.info(f"Data preparation complete. Processed {len(self.data_by_crypto)} crypto IDs.")
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            logger.debug(traceback.format_exc())
            raise DataPreparationError(f"Failed to prepare data: {str(e)}")

    def _create_sequences(self, data, target, is_classification=False):
        """
        Create sequences from the data for time series prediction.

        Parameters:
            data (np.ndarray): Input data.
            target (np.ndarray): Target values.
            is_classification (bool): If True, indicates classification task.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of sequences and corresponding targets.
        """
        try:
            if len(data) <= self.sequence_length:
                raise ValueError(f"Data length ({len(data)}) must be greater than sequence length ({self.sequence_length})")
                
            X, y = [], []
            for i in range(len(data) - self.sequence_length):
                X.append(data[i:i + self.sequence_length])
                y.append(target[i + self.sequence_length])
            
            if not X or not y:
                raise ValueError("Failed to create sequences, empty result")
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            raise DataPreparationError(f"Sequence creation failed: {str(e)}")

    def create_dataloader(self, X, y, batch_size=None, shuffle=True):
        """
        Create a DataLoader for the training or validation data.

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            batch_size (int, optional): Batch size for the DataLoader.
            shuffle (bool): Whether to shuffle the data.

        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        try:
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Mismatch between X and y lengths: {X.shape[0]} vs {y.shape[0]}")
                
            if batch_size is None:
                batch_size = min(self.batch_size, X.shape[0])
                
            dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y.reshape(-1, 1)))
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
            
        except Exception as e:
            logger.error(f"Error creating dataloader: {str(e)}")
            raise DataPreparationError(f"DataLoader creation failed: {str(e)}")

    def train_single_model(self, crypto_id, params):
        """
        Train a single model for a specific cryptocurrency.

        Parameters:
            crypto_id (str): The ID of the cryptocurrency to train.
            params (dict): Additional parameters for training.

        Returns:
            nn.Module: The trained model.
        
        Raises:
            ModelTrainingError: If there is an error during training.
        """
        try:
            if crypto_id not in self.data_by_crypto:
                raise ModelTrainingError(f"No data available for {crypto_id}")
                
            logger.info(f"Training model for {crypto_id}")
            X_train = self.data_by_crypto[crypto_id]['X_train']
            y_train = self.data_by_crypto[crypto_id]['y_train']
            
            train_size = int(0.8 * len(X_train))
            if train_size <= 0 or train_size >= len(X_train):
                raise ModelTrainingError(f"Invalid train/val split for {crypto_id}: {train_size}/{len(X_train) - train_size}")
                
            X_train_split, X_val_split = X_train[:train_size], X_train[train_size:]
            y_train_split, y_val_split = y_train[:train_size], y_train[train_size:]
            
            input_dim = X_train.shape[2]
            
            # Check if we're using an ensemble model
            is_ensemble = self.model_type.startswith('ensemble')
            ensemble_method = 'average'  # Default method
            
            # Parse ensemble method if specified (e.g., 'ensemble_weighted')
            if is_ensemble and '_' in self.model_type:
                ensemble_method = self.model_type.split('_')[1]
                logger.info(f"Using ensemble method: {ensemble_method}")
            
            # Check if we're using a feature ensemble
            is_feature_ensemble = self.model_type.startswith('feature')
            feature_ensemble_method = 'average'  # Default method
            
            # Parse feature ensemble method if specified (e.g., 'feature_average')
            if is_feature_ensemble and '_' in self.model_type:
                parts = self.model_type.split('_')
                if len(parts) > 1:
                    feature_ensemble_method = parts[1]
                logger.info(f"Using feature ensemble with method: {feature_ensemble_method}")
            
            # Create model
            try:
                if is_feature_ensemble:
                    # Import the feature ensemble model
                    from models import FeatureEnsemble
                    
                    logger.info(f"Creating feature ensemble model with method: {feature_ensemble_method}")
                    model = FeatureEnsemble(
                        input_dim=input_dim,
                        hidden_dim=64,
                        model_type='transformer',  # Use transformer as base model for all feature sets
                        ensemble_method=feature_ensemble_method,
                        prediction_type=self.target_column,
                        use_full_feature_model=True
                    ).to(self.device)
                    
                    # Create feature masks for each feature set
                    # We need to identify the indices of different feature groups in the input
                    # This depends on how features are added in prepare_data
                    
                    # Initialize all masks with zeros
                    feature_masks = []
                    
                    # Basic price features (always included)
                    basic_feature_count = 3  # price, rsi, volatility
                    
                    # Count feature groups
                    garch_features = 3 if self.use_garch else 0  # garch_vol, garch_vol_forecast, garch_regime
                    bollinger_features = 7 if self.use_bollinger else 0  # bollinger_mavg, bollinger_hband, bollinger_lband, etc.
                    macd_features = 4 if self.use_macd else 0  # macd_line, macd_signal, macd_hist, macd_divergence
                    ma_features = 3 if self.use_moving_avg else 0  # sma_10, sma_30, sma_diff
                    
                    # Calculate starting indices for each feature group
                    garch_start = basic_feature_count
                    bollinger_start = garch_start + garch_features
                    macd_start = bollinger_start + bollinger_features
                    ma_start = macd_start + macd_features
                    
                    # 1. Create mask for price + GARCH features
                    mask1 = torch.zeros(input_dim)
                    # Basic features always included
                    mask1[:basic_feature_count] = 1.0
                    # Include GARCH features if used
                    if self.use_garch:
                        mask1[garch_start:bollinger_start] = 1.0
                    feature_masks.append(mask1)
                    
                    # 2. Create mask for price + Bollinger features
                    mask2 = torch.zeros(input_dim)
                    # Basic features always included
                    mask2[:basic_feature_count] = 1.0
                    # Include Bollinger features if used
                    if self.use_bollinger:
                        mask2[bollinger_start:macd_start] = 1.0
                    feature_masks.append(mask2)
                    
                    # 3. Create mask for price + MACD features
                    mask3 = torch.zeros(input_dim)
                    # Basic features always included
                    mask3[:basic_feature_count] = 1.0
                    # Include MACD features if used
                    if self.use_macd:
                        mask3[macd_start:ma_start] = 1.0
                    feature_masks.append(mask3)
                    
                    # 4. Create mask for price + Moving Average features
                    mask4 = torch.zeros(input_dim)
                    # Basic features always included
                    mask4[:basic_feature_count] = 1.0
                    # Include Moving Average features if used
                    if self.use_moving_avg:
                        mask4[ma_start:] = 1.0
                    feature_masks.append(mask4)
                    
                    # 5. Create mask for all features
                    mask5 = torch.ones(input_dim)
                    feature_masks.append(mask5)
                    
                    # Set the feature masks in the model
                    model.set_feature_masks(feature_masks)
                    
                    logger.info(f"Created feature masks for {len(feature_masks)} feature sets")
                
                elif is_ensemble:
                    # Import the ensemble model
                    from models import CryptoEnsemble
                    
                    logger.info(f"Creating ensemble model with method: {ensemble_method}")
                    model = CryptoEnsemble(
                        input_dim=input_dim,
                        hidden_dim=64,
                        models_config=[
                            {'type': 'lstm', 'layers': 1, 'dropout': 0.1},  # Simple LSTM
                            {'type': 'gru', 'layers': 2, 'dropout': 0.2},   # GRU
                            {'type': 'transformer', 'layers': 1, 'heads': 4, 'dropout': 0.1}  # Transformer
                        ],
                        ensemble_method=ensemble_method,
                        prediction_type=self.target_column
                    ).to(self.device)
                elif self.model_type == 'transformer':
                    from models import CryptoTransformer
                    model = CryptoTransformer(
                        input_dim=input_dim,
                        d_model=64,
                        n_heads=4,
                        num_layers=3,
                        dropout=0.1,
                        sequence_length=self.sequence_length,
                        prediction_type=self.target_column
                    ).to(self.device)
                else:  # LSTM
                    from trainer import SimpleLSTM
                    model = SimpleLSTM(
                        input_dim=input_dim,
                        hidden_dim=64,
                        num_layers=1,
                        dropout=0.1,
                        output_dim=1,
                        is_direction=self.target_column == 'direction'
                    ).to(self.device)
            except Exception as e:
                logger.error(f"Error creating model for {crypto_id}: {str(e)}")
                raise ModelTrainingError(f"Model creation failed: {str(e)}")
            
            # Setup optimizer and criterion
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.HuberLoss() if self.target_column == 'price' else nn.BCELoss()
            logger.info(f"Criterion type: {type(criterion)}")
            
            # Create dataloaders
            try:
                train_loader = self.create_dataloader(X_train_split, y_train_split, batch_size=16)
                val_loader = self.create_dataloader(X_val_split, y_val_split, batch_size=32, shuffle=False)
            except Exception as e:
                logger.error(f"Error creating data loaders for {crypto_id}: {str(e)}")
                raise ModelTrainingError(f"DataLoader creation failed: {str(e)}")
            
            # Initialize best validation loss
            best_val_loss = float('inf')
            best_model_state = None
            early_stop_count = 0
            max_early_stop = 5
            
            # For stacking ensemble, train base models first
            if is_ensemble and ensemble_method == 'stacking':
                logger.info("Training base models for ensemble...")
                self._train_base_models(model, X_train_split, y_train_split, X_val_split, y_val_split, criterion, crypto_id)
                
                # Freeze base models for stacking
                model.freeze_base_models()
                logger.info("Training meta-model for stacking ensemble...")
            
            # For feature ensemble, train base models separately
            if is_feature_ensemble:
                val_losses = self._train_feature_ensemble_models(model, X_train_split, y_train_split, X_val_split, y_val_split, criterion, crypto_id)
                
                # Update weights if using weighted ensemble
                if feature_ensemble_method == 'weighted':
                    model.update_weights(val_losses)
                    logger.info(f"Updated feature ensemble weights: {model.weights.data.cpu().numpy()}")
                
                # For stacking, freeze base models to train meta-model
                if feature_ensemble_method == 'stacking':
                    model.freeze_base_models()
                    logger.info("Training meta-model for feature stacking ensemble...")
            
            # Use automatic mixed precision for faster training if available
            use_amp = torch.cuda.is_available()
            scaler = torch.cuda.amp.GradScaler() if use_amp else None
            
            # Training loop
            for epoch in tqdm(range(self.epochs), desc=f"Training {crypto_id}", disable=self.verbosity < 1):
                try:
                    # Training phase
                    model.train()
                    train_loss = 0
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        try:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                            targets = targets.squeeze(-1)
                            
                            if use_amp:
                                with torch.cuda.amp.autocast():
                                    outputs = model(inputs)
                                    loss = criterion(outputs, targets)
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                outputs = model(inputs)
                                loss = criterion(outputs, targets)
                                loss.backward()
                                optimizer.step()
                                
                            optimizer.zero_grad()
                            train_loss += loss.item()
                            
                        except Exception as e:
                            logger.error(f"Error in batch {batch_idx} for {crypto_id}: {str(e)}")
                            continue
                    
                    train_loss /= len(train_loader)
                    
                    # Validation phase
                    val_loss = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, targets in val_loader:
                            try:
                                inputs, targets = inputs.to(self.device), targets.to(self.device)
                                targets = targets.squeeze(-1)
                                outputs = model(inputs)
                                val_loss += criterion(outputs, targets).item()
                            except Exception as e:
                                logger.error(f"Error in validation for {crypto_id}: {str(e)}")
                                continue
                    
                    val_loss /= len(val_loader)
                    logger.info(f"{crypto_id} - Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = model.state_dict()
                        early_stop_count = 0
                    else:
                        early_stop_count += 1
                        
                    # For weighted ensemble, update weights based on validation performance of base models
                    if is_ensemble and ensemble_method == 'weighted' and epoch % 5 == 0:
                        self._update_ensemble_weights(model, X_val_split, y_val_split, criterion)
                    
                    if early_stop_count >= max_early_stop:
                        logger.info(f"Early stopping for {crypto_id} at epoch {epoch}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in epoch {epoch} for {crypto_id}: {str(e)}")
                    continue
            
            # Load best model
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                logger.info(f"Training completed for {crypto_id}. Best val loss: {best_val_loss:.6f}")
                return model
            else:
                raise ModelTrainingError(f"No valid model state found for {crypto_id}")
                
        except Exception as e:
            logger.error(f"Error training model for {crypto_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise ModelTrainingError(f"Training failed for {crypto_id}: {str(e)}")
            
    def _train_base_models(self, ensemble_model, X_train, y_train, X_val, y_val, criterion, crypto_id):
        """
        Train the base models of an ensemble separately.
        
        Parameters:
            ensemble_model (CryptoEnsemble): The ensemble model containing base models.
            X_train (Tensor): Training features.
            y_train (Tensor): Training targets.
            X_val (Tensor): Validation features.
            y_val (Tensor): Validation targets.
            criterion (nn.Module): Loss criterion.
            crypto_id (str): Cryptocurrency ID.
            
        Returns:
            None - modifies the ensemble_model in-place
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor.reshape(-1, 1))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_performances = []
        
        # Train each base model separately
        for i, model in enumerate(ensemble_model.base_models):
            logger.info(f"Training base model {i+1}/{len(ensemble_model.base_models)} for {crypto_id}")
            
            # Create optimizer for this base model
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
            
            best_val_loss = float('inf')
            early_stopping = EarlyStopping(patience=3, verbose=self.verbosity >= 2)
            
            # Train for fewer epochs for base models
            for epoch in range(min(10, self.epochs)):
                # Training
                model.train()
                train_loss = 0
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets.squeeze(-1))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                model.eval()
                with torch.no_grad():
                    outputs = model(X_val_tensor)
                    val_loss = criterion(outputs, y_val_tensor).item()
                
                logger.info(f"{crypto_id} - Base Model {i+1} - Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                if early_stopping.step(val_loss):
                    logger.info(f"Early stopping for base model {i+1} at epoch {epoch}")
                    break
            
            val_performances.append(best_val_loss)
        
        # Update ensemble weights based on validation performance
        if ensemble_model.ensemble_method == 'weighted':
            ensemble_model.update_weights(val_performances)
            logger.info(f"Updated ensemble weights: {ensemble_model.weights.data.cpu().numpy()}")
    
    def _train_feature_ensemble_models(self, ensemble_model, X_train, y_train, X_val, y_val, criterion, crypto_id):
        """
        Train the base models of a feature ensemble separately.
        
        Parameters:
            ensemble_model (FeatureEnsemble): The feature ensemble model containing base models.
            X_train (Tensor): Training features.
            y_train (Tensor): Training targets.
            X_val (Tensor): Validation features.
            y_val (Tensor): Validation targets.
            criterion (nn.Module): Loss criterion.
            crypto_id (str): Cryptocurrency ID.
            
        Returns:
            list: Validation losses for each base model.
        """
        try:
            logger.info("Training feature ensemble base models...")
            logger.info(f"Criterion type in _train_feature_ensemble_models: {type(criterion)}")
            logger.info(f"X_train type: {type(X_train)}, y_train type: {type(y_train)}")
            if isinstance(X_train, torch.Tensor):
                logger.info(f"X_train shape: {X_train.shape}")
            if isinstance(y_train, torch.Tensor):
                logger.info(f"y_train shape: {y_train.shape}")
            
            # Convert to PyTorch tensors if they aren't already
            if not isinstance(X_train, torch.Tensor):
                X_train = torch.FloatTensor(X_train).to(self.device)
            if not isinstance(y_train, torch.Tensor):
                y_train = torch.FloatTensor(y_train).to(self.device)
            if not isinstance(X_val, torch.Tensor):
                X_val = torch.FloatTensor(X_val).to(self.device)
            if not isinstance(y_val, torch.Tensor):
                y_val = torch.FloatTensor(y_val).to(self.device)
            
            logger.info(f"After conversion - X_train type: {type(X_train)}, y_train type: {type(y_train)}")
            logger.info(f"After conversion - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            
            # Get base models and feature masks
            base_models = ensemble_model.base_models
            feature_masks = ensemble_model.feature_masks
            feature_sets = ensemble_model.feature_sets
            
            # Create dataloaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Train each base model separately
            val_losses = []
            for i, model in enumerate(base_models):
                logger.info(f"Training feature model {i+1}/{len(base_models)} for {crypto_id}: {feature_sets[i]}")
                
                # Set up optimizer for this model
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                mask = feature_masks[i]
                
                # Initialize best validation loss
                best_val_loss = float('inf')
                early_stop_count = 0
                max_early_stop = 3  # Stop after 3 epochs without improvement
                
                # Train for a few epochs
                for epoch in range(10):  # Fewer epochs per base model
                    # Training phase
                    model.train()
                    train_loss = 0.0
                    
                    for batch_X, batch_y in train_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        
                        # Apply feature mask if available
                        if mask is not None:
                            # Create a version of batch_X with only the relevant features
                            mask_expanded = mask.expand(batch_X.shape[0], batch_X.shape[1], -1).to(self.device)
                            masked_X = batch_X * mask_expanded
                            
                            # Add small values to maintain tensor structure
                            small_value = 1e-8
                            inverted_mask = 1.0 - mask_expanded
                            masked_X = masked_X + (small_value * inverted_mask)
                            
                            # Forward pass and compute loss
                            optimizer.zero_grad()
                            output = model(masked_X)
                        else:
                            # Use all features
                            optimizer.zero_grad()
                            output = model(batch_X)
                        
                        loss = criterion(output, batch_y)
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                    
                    train_loss /= len(train_loader)
                    
                    # Validation phase
                    model.eval()
                    val_loss = 0.0
                    
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                            
                            # Apply feature mask if available
                            if mask is not None:
                                # Create a version of batch_X with only the relevant features
                                mask_expanded = mask.expand(batch_X.shape[0], batch_X.shape[1], -1).to(self.device)
                                masked_X = batch_X * mask_expanded
                                
                                # Add small values to maintain tensor structure
                                small_value = 1e-8
                                inverted_mask = 1.0 - mask_expanded
                                masked_X = masked_X + (small_value * inverted_mask)
                                
                                # Forward pass
                                output = model(masked_X)
                            else:
                                # Use all features
                                output = model(batch_X)
                            
                            loss = criterion(output, batch_y)
                            val_loss += loss.item()
                    
                    val_loss /= len(val_loader)
                    
                    # Log progress
                    logger.info(f"Epoch {epoch+1}/10, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                    
                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stop_count = 0
                    else:
                        early_stop_count += 1
                        if early_stop_count >= max_early_stop:
                            logger.info(f"Early stopping after {epoch+1} epochs")
                            break
                
                # Store validation loss for this model
                val_losses.append(best_val_loss)
                logger.info(f"Model {i+1} best validation loss: {best_val_loss:.4f}")
            
            return val_losses
        except Exception as e:
            logger.error(f"Error in _train_feature_ensemble_models: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _update_ensemble_weights(self, ensemble_model, X_val, y_val, criterion):
        """
        Update weights for a weighted ensemble model based on validation performance.
        
        Parameters:
            ensemble_model (CryptoEnsemble): The ensemble model to update weights for
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            criterion (torch.nn.Module): Loss function
            
        Returns:
            None - modifies the ensemble_model in-place
        """
        # Convert to PyTorch tensors
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        val_performances = []
        
        # Evaluate each base model on validation data
        for i, model in enumerate(ensemble_model.base_models):
            model.eval()
            with torch.no_grad():
                outputs = model(X_val_tensor)
                val_loss = criterion(outputs, y_val_tensor).item()
            val_performances.append(val_loss)
        
        # Update weights based on validation performance
        ensemble_model.update_weights(val_performances)
        logger.info(f"Updated ensemble weights: {ensemble_model.weights.data.cpu().numpy()}")

    def train_models(self, n_splits=3, optimize=False, n_trials=10):
        """
        Train models for all specified cryptocurrencies.

        Parameters:
            n_splits (int): Number of cross-validation splits.
            optimize (bool): Whether to optimize hyperparameters.
            n_trials (int): Number of trials for hyperparameter optimization.

        Returns:
            dict: Dictionary of best models for each cryptocurrency.
        """
        logger.info(f"Training models for {len(self.crypto_ids)} crypto IDs")
        failed_models = []
        
        for crypto_id in self.crypto_ids:
            try:
                if crypto_id not in self.data_by_crypto:
                    logger.warning(f"No data available for {crypto_id}. Skipping.")
                    continue
                
                model = self.train_single_model(crypto_id, {})
                self.best_models[crypto_id] = model
                
                try:
                    model_path = f"{crypto_id}_best_model.pth"
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved model for {crypto_id} to {model_path}")
                except Exception as e:
                    logger.error(f"Error saving model for {crypto_id}: {str(e)}")
                
                self.model_timestamps[crypto_id] = datetime.now()
                
            except Exception as e:
                logger.error(f"Failed to train model for {crypto_id}: {str(e)}")
                failed_models.append(crypto_id)
                
        if failed_models:
            logger.warning(f"Training failed for {len(failed_models)} models: {failed_models}")
            
        logger.info(f"Model training complete. Successfully trained {len(self.best_models)} models.")
        return self.best_models

    def predict(self, crypto_id, X=None):
        """
        Make predictions using the trained model for a specific cryptocurrency.

        Parameters:
            crypto_id (str): The ID of the cryptocurrency to predict.
            X (np.ndarray, optional): Input features for prediction.

        Returns:
            np.ndarray: Predicted values.

        Raises:
            ModelNotFoundError: If no model is available for the specified cryptocurrency.
        """
        try:
            if crypto_id not in self.best_models:
                raise ModelNotFoundError(f"No model available for {crypto_id}")
                
            if X is None:
                if crypto_id not in self.data_by_crypto:
                    raise DataPreparationError(f"No test data available for {crypto_id}")
                X = self.data_by_crypto[crypto_id]['X_test']
                
            if len(X) == 0:
                raise ValueError(f"Empty input data for prediction for {crypto_id}")
                
            model = self.best_models[crypto_id]
            model.eval()
            
            with torch.no_grad():
                # Process in batches to avoid memory issues
                batch_size = 128
                all_preds = []
                
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i+batch_size]
                    batch_tensor = torch.FloatTensor(batch_X).to(self.device)
                    
                    try:
                        preds = model(batch_tensor).cpu().numpy()
                        
                        # Handle different output shapes
                        # If predictions come out as shape (batch_size, 1), flatten to (batch_size,)
                        if len(preds.shape) > 1 and preds.shape[1] == 1:
                            preds = preds.flatten()
                            
                        all_preds.append(preds)
                    except Exception as e:
                        logger.error(f"Error in prediction batch for {crypto_id}: {str(e)}")
                        logger.error(f"Input tensor shape: {batch_tensor.shape}, Model type: {type(model).__name__}")
                        # Continue with other batches
                
                if not all_preds:
                    raise ValueError(f"No valid predictions generated for {crypto_id}")
                
                try:
                    # Try to concatenate the predictions
                    predictions = np.concatenate(all_preds)
                    
                    # Check if predictions are empty or malformed
                    if predictions.size == 0:
                        raise ValueError(f"Empty predictions array for {crypto_id}")
                    
                    # Log prediction shape for debugging
                    logger.info(f"Predictions shape for {crypto_id}: {predictions.shape}")
                    
                    # Make sure predictions match the expected length
                    expected_length = len(self.data_by_crypto[crypto_id]['y_test'])
                    if len(predictions) != expected_length:
                        logger.warning(f"Prediction length mismatch: got {len(predictions)}, expected {expected_length}")
                        # Adjust prediction length if needed - this handles some edge cases
                        if len(predictions) > expected_length:
                            predictions = predictions[:expected_length]
                        elif len(predictions) < expected_length:
                            # Pad with last value or zeros
                            padding = np.zeros(expected_length - len(predictions))
                            if len(predictions) > 0:
                                padding[:] = predictions[-1]
                            predictions = np.concatenate([predictions, padding])
                    
                    return predictions
                except Exception as e:
                    logger.error(f"Error processing predictions for {crypto_id}: {str(e)}")
                    # Return an empty array of the correct shape as fallback
                    return np.array([])
                
        except ModelNotFoundError as e:
            logger.error(str(e))
            raise
        except Exception as e:
            logger.error(f"Error in prediction for {crypto_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise

    def retrain_if_needed(self, crypto_id, days_threshold=30):
        """
        Check if a model needs to be retrained based on its age.

        Parameters:
            crypto_id (str): The ID of the cryptocurrency to check.
            days_threshold (int): Number of days after which retraining is needed.

        Returns:
            bool: True if retraining was performed, False otherwise.
        """
        try:
            needs_retrain = False
            
            if crypto_id not in self.model_timestamps:
                logger.info(f"No timestamp for {crypto_id} model. Retraining required.")
                needs_retrain = True
            elif (datetime.now() - self.model_timestamps[crypto_id]).days >= days_threshold:
                logger.info(f"Model for {crypto_id} is {(datetime.now() - self.model_timestamps[crypto_id]).days} days old. Retraining required.")
                needs_retrain = True
                
            if needs_retrain:
                logger.info(f"Retraining model for {crypto_id}...")
                try:
                    model = self.train_single_model(crypto_id, {})
                    self.best_models[crypto_id] = model
                    torch.save(model.state_dict(), f"{crypto_id}_best_model.pth")
                    self.model_timestamps[crypto_id] = datetime.now()
                    logger.info(f"Retraining complete for {crypto_id}")
                    return True
                except Exception as e:
                    logger.error(f"Retraining failed for {crypto_id}: {str(e)}")
                    return False
            else:
                logger.info(f"No retraining needed for {crypto_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error checking if retraining needed for {crypto_id}: {str(e)}")
            return False 
