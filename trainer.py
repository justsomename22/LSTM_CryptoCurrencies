#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cryptocurrency Model Trainer Module

This module handles the training, validation, and prediction processes for cryptocurrency price models.
It contains the CryptoTrainer class which manages data preparation, model initialization,
training loops, hyperparameter tuning, and prediction generation.
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta

# Import local modules
from data_processing import DataProcessor
from models import (
    LSTMModel, GRUModel, RNNModel, TransformerModel, 
    CNN_LSTMModel, create_model
)
from utils import (
    plot_training_history, plot_predictions, save_model, 
    load_model, setup_logging
)

class CryptoTrainer:
    """
    A class for training, validating, and generating predictions with cryptocurrency price models.
    
    This class handles the complete workflow for cryptocurrency price prediction, including:
    - Data loading and preprocessing
    - Model initialization and configuration
    - Training and validation loops
    - Early stopping and model checkpointing
    - Performance evaluation
    - Prediction generation
    
    Attributes:
        args (argparse.Namespace): Command-line arguments or configuration parameters
        logger (logging.Logger): Logger for recording training information
        device (torch.device): Device for model training (CPU or GPU)
        data_processor (DataProcessor): Data processing utility for cryptocurrency data
    """
    
    def __init__(self, args):
        """
        Initialize the CryptoTrainer with the provided arguments.
        
        Args:
            args (argparse.Namespace): Command-line arguments or configuration parameters
        """
        self.args = args
        self.logger = logging.getLogger('crypto_trainer')
        
        # Set device for model training (CPU/GPU)
        self.device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            start_date=args.data_start_date,
            end_date=args.data_end_date,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            train_test_split=args.train_test_split,
            validation_split=args.validation_split,
            normalization=args.normalization,
            features=args.features
        )
        
        # Create necessary directories
        os.makedirs(args.output_dir, exist_ok=True)
        
    def train_models(self):
        """
        Train models for all specified cryptocurrencies.
        
        This method iterates through each cryptocurrency specified in the arguments,
        prepares data, initializes models, and executes the training process.
        """
        self.logger.info("Starting model training process")
        
        # Train models for each cryptocurrency
        for crypto in self.args.cryptocurrencies:
            self.logger.info(f"Training model for {crypto}")
            self._train_single_model(crypto)
            
        self.logger.info("Model training completed for all cryptocurrencies")
        
    def _train_single_model(self, cryptocurrency):
        """
        Train a model for a single cryptocurrency.
        
        Args:
            cryptocurrency (str): The name of the cryptocurrency to train for
            
        Returns:
            dict: Training history containing loss values
        """
        # Process data for the specific cryptocurrency
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = self.data_processor.prepare_data(cryptocurrency)
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.args.batch_size)
        
        # Initialize model
        input_dim = X_train.shape[2]  # Number of features
        output_dim = y_train.shape[1]  # Prediction horizon
        
        model = create_model(
            model_type=self.args.model_type,
            input_dim=input_dim,
            hidden_dim=64,  # Could be parameterized
            num_layers=2,   # Could be parameterized
            output_dim=output_dim,
            dropout=0.2     # Could be parameterized
        ).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        
        # Training loop
        best_val_loss = float('inf')
        early_stopping_counter = 0
        training_history = {'train_loss': [], 'val_loss': []}
        
        self.logger.info(f"Starting training loop for {cryptocurrency} using {self.args.model_type} model")
        
        for epoch in range(self.args.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            avg_train_loss = train_loss / len(train_loader)
            training_history['train_loss'].append(avg_train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
            avg_val_loss = val_loss / len(val_loader)
            training_history['val_loss'].append(avg_val_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.args.epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
                
            # Check for early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                
                # Save the best model
                model_filename = f"{cryptocurrency}_best_model.pth"
                save_model(model, os.path.join(self.args.output_dir, model_filename))
                self.logger.info(f"Saved best model checkpoint with validation loss: {best_val_loss:.4f}")
            else:
                early_stopping_counter += 1
                
            if self.args.early_stopping and early_stopping_counter >= self.args.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
                
        # Plot training history
        plot_training_history(
            training_history,
            title=f"{cryptocurrency.capitalize()} {self.args.model_type} Training History",
            output_path=os.path.join(self.args.output_dir, f"{cryptocurrency}_training_history.png")
        )
        
        # Evaluate on test set
        self._evaluate_model(cryptocurrency, model, X_test, y_test, scaler)
        
        return training_history
        
    def _evaluate_model(self, cryptocurrency, model, X_test, y_test, scaler):
        """
        Evaluate a trained model on test data.
        
        Args:
            cryptocurrency (str): The name of the cryptocurrency being evaluated
            model (nn.Module): The trained model
            X_test (numpy.ndarray): Test input data
            y_test (numpy.ndarray): Test target data
            scaler: The scaler used for data normalization
        """
        self.logger.info(f"Evaluating model for {cryptocurrency}")
        
        # Convert test data to PyTorch tensors
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()
            
        # Inverse transform predictions and actual values if needed
        if hasattr(scaler, 'inverse_transform'):
            # This logic depends on your preprocessing steps and may need adjustment
            predictions = self.data_processor.inverse_transform_predictions(predictions, scaler)
            y_test_inv = self.data_processor.inverse_transform_predictions(y_test, scaler)
        else:
            y_test_inv = y_test
            
        # Calculate metrics
        mse = np.mean((predictions - y_test_inv) ** 2)
        mae = np.mean(np.abs(predictions - y_test_inv))
        rmse = np.sqrt(mse)
        
        self.logger.info(f"{cryptocurrency.upper()} Test Metrics - MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        # Plot predictions
        plot_predictions(
            y_true=y_test_inv,
            y_pred=predictions,
            title=f"{cryptocurrency.capitalize()} Price Predictions",
            output_path=os.path.join(self.args.output_dir, f"{cryptocurrency}_predictions.png")
        )
        
    def make_predictions(self):
        """
        Generate predictions using trained models.
        
        This method loads trained models for each cryptocurrency and generates
        predictions for future time points based on the latest available data.
        """
        self.logger.info("Generating predictions for specified cryptocurrencies")
        
        # Check if model path is provided
        if not self.args.model_path and not os.path.exists(self.args.output_dir):
            self.logger.error("Model path or output directory must exist for predictions")
            return
            
        for crypto in self.args.cryptocurrencies:
            # Determine model path
            if self.args.model_path:
                model_path = self.args.model_path
            else:
                model_path = os.path.join(self.args.output_dir, f"{crypto}_best_model.pth")
                
            if not os.path.exists(model_path):
                self.logger.warning(f"Model for {crypto} not found at {model_path}, skipping")
                continue
                
            # Load data for prediction
            latest_data = self.data_processor.prepare_prediction_data(crypto)
            
            # Load model
            model = load_model(model_path, self.device)
            
            # Generate predictions
            model.eval()
            with torch.no_grad():
                latest_data_tensor = torch.FloatTensor(latest_data).unsqueeze(0).to(self.device)
                prediction = model(latest_data_tensor).cpu().numpy()
                
            # Inverse transform prediction if needed
            if hasattr(self.data_processor, 'get_scaler'):
                scaler = self.data_processor.get_scaler(crypto)
                if scaler:
                    prediction = self.data_processor.inverse_transform_predictions(prediction, scaler)
                    
            # Format and display predictions
            prediction_dates = [datetime.now() + timedelta(days=i+1) for i in range(prediction.shape[1])]
            prediction_df = pd.DataFrame({
                'Date': [d.strftime('%Y-%m-%d') for d in prediction_dates],
                'Predicted_Price': prediction[0]
            })
            
            self.logger.info(f"Predictions for {crypto.capitalize()}:")
            self.logger.info(prediction_df.to_string(index=False))
            
            # Save predictions to CSV
            output_file = os.path.join(self.args.output_dir, f"{crypto}_predictions.csv")
            prediction_df.to_csv(output_file, index=False)
            self.logger.info(f"Saved predictions to {output_file}")

def parse_args():
    """
    Parse command-line arguments for the trainer module.
    
    This function can be used when running the trainer module directly.
    
    Returns:
        argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction Trainer')
    
    # General configuration arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='Operation mode: train (train models) or predict (make predictions)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the configuration JSON file containing model parameters')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level for application messages')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model outputs, checkpoints, and results')

    # Model and training configuration arguments
    parser.add_argument('--cryptocurrencies', nargs='+', default=['bitcoin', 'ethereum', 'monero'],
                        help='List of cryptocurrencies to train/predict (space-separated)')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'SimpleRNN', 'Transformer', 'CNN_LSTM'],
                        help='Type of neural network model to use')
    parser.add_argument('--features', nargs='+', default=['Close', 'Volume', 'Market_Cap'],
                        help='List of features to include in the model training (space-separated)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training the model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='Length of input sequence (number of time steps) for time series predictions')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                        help='Number of future time steps to predict')
    
    # Data processing arguments
    parser.add_argument('--train_test_split', type=float, default=0.8,
                        help='Ratio of training data to total data (0.8 means 80% training, 20% testing)')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Ratio of validation data to training data')
    parser.add_argument('--data_start_date', type=str, default='2015-01-01',
                        help='Start date for historical data collection (YYYY-MM-DD format)')
    parser.add_argument('--data_end_date', type=str, default=None,
                        help='End date for historical data collection (YYYY-MM-DD format, defaults to current date)')
    parser.add_argument('--normalization', type=str, default='minmax', choices=['minmax', 'zscore', 'robust'],
                        help='Data normalization method to use')
    
    # Model loading arguments
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load a pre-trained model (required for predict mode)')
    
    # Advanced options
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training if available')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping to prevent overfitting')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """
    Main function for executing the trainer module directly.
    
    This function parses command-line arguments, sets up logging, initializes the
    appropriate components, and executes the training or prediction workflow.
    """
    # Parse arguments
    args = parse_args()
    
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as config_file:
            config = json.load(config_file)
            # Override args with config values
            for key, value in config.items():
                setattr(args, key, value)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"crypto_trainer_{timestamp}.log"
    setup_logging(args.log_level, log_filename)
    logger = logging.getLogger('crypto_trainer')
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create trainer instance
    trainer = CryptoTrainer(args)
    
    # Execute based on mode
    if args.mode == 'train':
        trainer.train_models()
    elif args.mode == 'predict':
        trainer.make_predictions()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)
        
    logger.info(f"Cryptocurrency {args.mode} process completed")

if __name__ == "__main__":
    main() 