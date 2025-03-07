#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Processing Module for Cryptocurrency Price Prediction

This module handles all data processing tasks for cryptocurrency price prediction,
including data loading, preprocessing, normalization, and preparation for model training.
It transforms raw cryptocurrency data into structured sequences suitable for
time series forecasting models.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split

from GetData import CryptoDataFetcher

class DataProcessor:
    """
    A class for processing cryptocurrency data for model training and evaluation.
    
    This class handles data loading, preprocessing, normalization, and sequence creation
    for time series forecasting with cryptocurrencies. It prepares data in the format
    required by PyTorch models.
    
    Attributes:
        start_date (str): Start date for historical data retrieval
        end_date (str): End date for historical data retrieval
        sequence_length (int): Number of time steps in each input sequence
        prediction_horizon (int): Number of future time steps to predict
        train_test_split (float): Ratio of training data to total data
        validation_split (float): Ratio of validation data to training data
        normalization (str): Method for data normalization
        features (list): List of features to include in the model
        logger (logging.Logger): Logger for recording data processing information
        scalers (dict): Dictionary to store data scalers for each cryptocurrency
    """
    
    def __init__(self, start_date='2015-01-01', end_date=None, sequence_length=60,
                 prediction_horizon=1, train_test_split=0.8, validation_split=0.1,
                 normalization='minmax', features=None):
        """
        Initialize the DataProcessor with the provided parameters.
        
        Args:
            start_date (str): Start date for historical data in YYYY-MM-DD format
            end_date (str): End date for historical data in YYYY-MM-DD format (default: current date)
            sequence_length (int): Number of time steps in each input sequence
            prediction_horizon (int): Number of future time steps to predict
            train_test_split (float): Ratio of training data to total data
            validation_split (float): Ratio of validation data to training data
            normalization (str): Method for data normalization ('minmax', 'zscore', or 'robust')
            features (list): List of features to include in the model (default: ['Close', 'Volume', 'Market_Cap'])
        """
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.now().strftime('%Y-%m-%d')
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.train_test_split = train_test_split
        self.validation_split = validation_split
        self.normalization = normalization
        self.features = features if features else ['Close', 'Volume', 'Market_Cap']
        
        # Setup logger
        self.logger = logging.getLogger('crypto_trainer')
        
        # Dictionary to store scalers for each cryptocurrency
        self.scalers = {}
        
    def prepare_data(self, cryptocurrency):
        """
        Prepare data for a specific cryptocurrency for model training and testing.
        
        This method handles the complete data preparation pipeline:
        1. Fetches historical data
        2. Preprocesses and cleans the data
        3. Normalizes the features
        4. Creates sequences for time series forecasting
        5. Splits data into training, validation, and test sets
        
        Args:
            cryptocurrency (str): Name of the cryptocurrency
            
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test, scaler
        """
        self.logger.info(f"Preparing data for {cryptocurrency}")
        
        # Fetch historical data
        df = self._fetch_data(cryptocurrency)
        
        if df is None or df.empty:
            self.logger.error(f"Failed to fetch data for {cryptocurrency}")
            return None, None, None, None, None, None, None
            
        # Preprocess data
        df = self._preprocess_data(df)
        
        # Normalize data
        df_normalized, scaler = self._normalize_data(df, cryptocurrency)
        
        # Create sequences
        X, y = self._create_sequences(df_normalized)
        
        # Split data into train, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=self.train_test_split, shuffle=False
        )
        
        # Further split training data to create validation set
        val_size = self.validation_split / self.train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, train_size=1-val_size, shuffle=False
        )
        
        self.logger.info(f"Data preparation completed for {cryptocurrency}")
        self.logger.info(f"Train set shape: {X_train.shape}, {y_train.shape}")
        self.logger.info(f"Validation set shape: {X_val.shape}, {y_val.shape}")
        self.logger.info(f"Test set shape: {X_test.shape}, {y_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler
        
    def _fetch_data(self, cryptocurrency):
        """
        Fetch historical data for a cryptocurrency.
        
        Args:
            cryptocurrency (str): Name of the cryptocurrency
            
        Returns:
            pandas.DataFrame: DataFrame containing historical price data
        """
        try:
            data_fetcher = CryptoDataFetcher()
            df = data_fetcher.get_crypto_data(
                cryptocurrency, self.start_date, self.end_date
            )
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for {cryptocurrency}: {e}")
            return None
            
    def _preprocess_data(self, df):
        """
        Preprocess the cryptocurrency data.
        
        This method performs data cleaning, handles missing values,
        and prepares the data for normalization.
        
        Args:
            df (pandas.DataFrame): DataFrame containing raw cryptocurrency data
            
        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            self.logger.warning(f"Found {df.isnull().sum().sum()} missing values in data")
            # Forward fill missing values
            df = df.fillna(method='ffill')
            # If there are still missing values (at the beginning), use backward fill
            df = df.fillna(method='bfill')
            
        # Select only the required features
        available_features = [f for f in self.features if f in df.columns]
        if len(available_features) < len(self.features):
            missing_features = set(self.features) - set(available_features)
            self.logger.warning(f"Some requested features are not available: {missing_features}")
            
        df = df[available_features]
        
        # Remove outliers (optional)
        # You could add outlier detection and removal logic here
        
        return df
        
    def _normalize_data(self, df, cryptocurrency):
        """
        Normalize the data using the specified normalization method.
        
        Args:
            df (pandas.DataFrame): DataFrame to normalize
            cryptocurrency (str): Name of the cryptocurrency
            
        Returns:
            tuple: (pandas.DataFrame, sklearn.preprocessing.Scaler) - Normalized data and the scaler
        """
        # Initialize the appropriate scaler
        if self.normalization == 'minmax':
            scaler = MinMaxScaler()
        elif self.normalization == 'zscore':
            scaler = StandardScaler()
        elif self.normalization == 'robust':
            scaler = RobustScaler()
        else:
            self.logger.warning(f"Unknown normalization method: {self.normalization}, using MinMax scaling")
            scaler = MinMaxScaler()
            
        # Fit and transform the data
        normalized_data = scaler.fit_transform(df.values)
        df_normalized = pd.DataFrame(normalized_data, columns=df.columns, index=df.index)
        
        # Store the scaler for later use
        self.scalers[cryptocurrency] = scaler
        
        return df_normalized, scaler
        
    def _create_sequences(self, df):
        """
        Create sequences for time series forecasting.
        
        This method transforms the normalized data into sequences of consecutive time steps
        for training time series models. Each sequence consists of 'sequence_length' time steps
        as input (X) and 'prediction_horizon' future time steps as target (y).
        
        Args:
            df (pandas.DataFrame): Normalized DataFrame
            
        Returns:
            tuple: (numpy.ndarray, numpy.ndarray) - Input sequences (X) and target values (y)
        """
        data = df.values
        X, y = [], []
        
        # Loop through the data to create sequences
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            sequence = data[i:(i + self.sequence_length)]
            X.append(sequence)
            
            # Target (future values)
            if self.prediction_horizon == 1:
                target = data[i + self.sequence_length, 0]  # Typically forecasting the Close price
            else:
                # Multiple time step forecasting
                target = [data[i + self.sequence_length + j, 0] for j in range(self.prediction_horizon)]
                
            y.append(target)
            
        return np.array(X), np.array(y)
        
    def prepare_prediction_data(self, cryptocurrency):
        """
        Prepare the most recent data for making predictions.
        
        This method fetches the most recent data for a cryptocurrency and prepares it
        for making future predictions. It returns the last sequence of data points
        that can be used as input to a trained model.
        
        Args:
            cryptocurrency (str): Name of the cryptocurrency
            
        Returns:
            numpy.ndarray: The most recent sequence of data for prediction
        """
        self.logger.info(f"Preparing prediction data for {cryptocurrency}")
        
        # Fetch historical data
        df = self._fetch_data(cryptocurrency)
        
        if df is None or df.empty:
            self.logger.error(f"Failed to fetch data for {cryptocurrency}")
            return None
            
        # Preprocess data
        df = self._preprocess_data(df)
        
        # Get or create scaler
        if cryptocurrency in self.scalers:
            scaler = self.scalers[cryptocurrency]
            df_normalized = pd.DataFrame(
                scaler.transform(df.values),
                columns=df.columns,
                index=df.index
            )
        else:
            # If no scaler exists for this cryptocurrency, create one
            df_normalized, scaler = self._normalize_data(df, cryptocurrency)
            
        # Get the most recent sequence
        latest_sequence = df_normalized.values[-self.sequence_length:]
        
        # Ensure the sequence has the right shape
        if len(latest_sequence) < self.sequence_length:
            self.logger.warning(
                f"Not enough data points for {cryptocurrency}. "
                f"Need {self.sequence_length}, but only have {len(latest_sequence)}"
            )
            # Pad with zeros if necessary (not ideal, but prevents errors)
            padding = np.zeros((self.sequence_length - len(latest_sequence), len(df.columns)))
            latest_sequence = np.vstack([padding, latest_sequence])
            
        return latest_sequence
        
    def inverse_transform_predictions(self, predictions, scaler):
        """
        Transform normalized predictions back to their original scale.
        
        Args:
            predictions (numpy.ndarray): Normalized predictions
            scaler: The scaler used for normalization
            
        Returns:
            numpy.ndarray: Predictions in the original scale
        """
        # Handle single point prediction
        if predictions.ndim == 1:
            # Create a dummy array with zeros for other features
            dummy = np.zeros((len(predictions), scaler.scale_.shape[0]))
            dummy[:, 0] = predictions  # Assuming prediction is for the first feature (Close price)
            # Inverse transform and extract the first feature
            original_scale = scaler.inverse_transform(dummy)[:, 0]
            return original_scale
            
        # Handle multi-step predictions
        elif predictions.ndim == 2:
            # Create a dummy array with zeros for other features
            dummy = np.zeros((predictions.shape[0], predictions.shape[1], scaler.scale_.shape[0]))
            dummy[:, :, 0] = predictions  # Assuming prediction is for the first feature (Close price)
            # Reshape for inverse transform
            reshaped = dummy.reshape(-1, scaler.scale_.shape[0])
            # Inverse transform
            original_scale = scaler.inverse_transform(reshaped)[:, 0]
            # Reshape back to original shape
            original_scale = original_scale.reshape(predictions.shape[0], predictions.shape[1])
            return original_scale
            
        # Handle batch of multi-step predictions
        elif predictions.ndim == 3:
            batch_size, sequence_len, n_features = predictions.shape
            # Assuming the model predicts only the first feature
            dummy = np.zeros((batch_size, sequence_len, scaler.scale_.shape[0]))
            dummy[:, :, 0] = predictions[:, :, 0]
            # Reshape for inverse transform
            reshaped = dummy.reshape(-1, scaler.scale_.shape[0])
            # Inverse transform
            original_scale = scaler.inverse_transform(reshaped)[:, 0]
            # Reshape back to original batch shape
            original_scale = original_scale.reshape(batch_size, sequence_len)
            return original_scale
            
        else:
            raise ValueError(f"Unexpected prediction shape: {predictions.shape}")
            
    def get_scaler(self, cryptocurrency):
        """
        Get the scaler for a specific cryptocurrency.
        
        Args:
            cryptocurrency (str): Name of the cryptocurrency
            
        Returns:
            sklearn.preprocessing.Scaler: The scaler for the specified cryptocurrency
        """
        return self.scalers.get(cryptocurrency, None)

# Example usage when run directly
if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example data processor configuration
    processor = DataProcessor(
        start_date='2020-01-01',
        end_date='2023-01-01',
        sequence_length=60,
        prediction_horizon=7,
        features=['Close', 'Volume', 'Market_Cap']
    )
    
    # Prepare data for Bitcoin
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = processor.prepare_data('bitcoin')
    
    # Print shapes
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}") 