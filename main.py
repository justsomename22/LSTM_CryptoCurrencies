#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Module for Cryptocurrency Price Prediction

This module serves as the entry point for the cryptocurrency price prediction system.
It handles command-line arguments, initializes the training process, and manages
the overall workflow of the application.
"""

import argparse
import logging
import os
import sys
import torch
from datetime import datetime

from trainer import CryptoTrainer
from evaluation import ModelEvaluator
from utils import setup_logging

def parse_arguments():
    """
    Parse command-line arguments for the application.
    
    Returns:
        argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Cryptocurrency Price Prediction')
    
    # General configuration arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate', 'predict'],
                        help='Operation mode: train (train models), evaluate (evaluate models), or predict (make predictions)')
    parser.add_argument('--config', type=str, default='configs/default_config.json',
                        help='Path to the configuration JSON file containing model parameters')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level for application messages')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save model outputs, checkpoints, and results')

    # Model and training configuration arguments
    parser.add_argument('--cryptocurrencies', nargs='+', default=['bitcoin', 'ethereum', 'monero'],
                        help='List of cryptocurrencies to train/evaluate/predict (space-separated)')
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
    
    # Model evaluation arguments
    parser.add_argument('--metrics', nargs='+', default=['mse', 'mae', 'rmse', 'mape', 'r2'],
                        help='Evaluation metrics to calculate (space-separated)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to load a pre-trained model (required for evaluate and predict modes)')
    
    # Advanced options
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training if available')
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping to prevent overfitting')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--resume_training', action='store_true',
                        help='Resume training from a saved checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """
    Main function for executing the cryptocurrency prediction system.
    
    This function parses command-line arguments, sets up logging, initializes the
    appropriate components based on the selected mode, and executes the workflow.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"crypto_trainer_{timestamp}.log"
    setup_logging(args.log_level, log_filename)
    logger = logging.getLogger('crypto_trainer')
    
    # Log basic information
    logger.info(f"Starting cryptocurrency prediction system in {args.mode} mode")
    logger.info(f"Processing cryptocurrencies: {', '.join(args.cryptocurrencies)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Choose device (CPU/GPU)
    device = torch.device("cuda:0" if args.use_gpu and torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.mode == 'train':
        # Initialize and run the training process
        trainer = CryptoTrainer(args)
        trainer.train_models()
        
    elif args.mode == 'evaluate':
        # Check if model path is provided
        if not args.model_path:
            logger.error("Model path must be provided for evaluation mode")
            sys.exit(1)
            
        # Initialize and run the evaluation process
        evaluator = ModelEvaluator(args)
        evaluator.evaluate_models()
        
    elif args.mode == 'predict':
        # Check if model path is provided
        if not args.model_path:
            logger.error("Model path must be provided for prediction mode")
            sys.exit(1)
            
        # Initialize and run the prediction process
        trainer = CryptoTrainer(args)
        trainer.make_predictions()
    
    logger.info("Cryptocurrency prediction system completed successfully")

if __name__ == '__main__':
    main() 