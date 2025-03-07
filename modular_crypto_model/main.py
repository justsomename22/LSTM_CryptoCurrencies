#main.py
"""
Main entry point for the cryptocurrency prediction pipeline.

This script handles the training and evaluation of models for cryptocurrency price prediction.
It includes command line argument parsing and logging setup.
"""
from evaluation import (
    evaluate_model,
    analyze_model_performance,
    plot_actual_vs_predicted,
    generate_residual_plot,
    generate_error_distribution_plot,
    predict_future,
    compare_models,
    generate_anomaly_detection_plot,
    visualize_ensemble_predictions,
    visualize_feature_ensemble
)
from trainer import ImprovedCryptoTrainer
import os
import argparse
import torch
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional, Union
import random
import time
from tqdm import tqdm
import logging  # Import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Main")

def test_transformer_model():
    """
    Test the CryptoTransformer model with various batch sizes to ensure it handles input correctly.

    Returns:
        bool: True if all tests pass, False otherwise.
    """
    from trainer import CryptoTransformer
    try:
        # Test with small batch size
        input_dim = 60  # This matches your data's feature dimension
        seq_len = 20
        
        # Test with different batch sizes to verify batch handling
        batch_sizes = [2, 16, 64, 128]
        
        for batch_size in batch_sizes:
            print(f"Testing with batch size: {batch_size}")
            test_input = torch.randn(batch_size, seq_len, input_dim)
            
            # Create the model
            model = CryptoTransformer(
                input_dim=input_dim,
                d_model=64,
                n_heads=8,
                num_layers=2,
                dropout=0.1,
                sequence_length=seq_len,
                prediction_type='price'
            )
            
            # Test a forward pass
            output = model(test_input)
            
            # Verify the output shape matches the batch size
            assert output.shape == torch.Size([batch_size]), f"Shape mismatch: {output.shape} vs {torch.Size([batch_size])}"
        
        print("All batch size tests passed!")
        return True
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main(args):
    """
    Main function to execute the training and evaluation of models.

    Parameters:
        args (Namespace): Command line arguments parsed by argparse.
    """
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory for plots
    os.makedirs("model_evaluation", exist_ok=True)
    
    # Log data information
    try:
        df = pd.read_csv(args.data_path)
        logger.info(f"Raw data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Rows per crypto: {df.groupby('crypto_id').size().to_dict()}")
    except Exception as e:
        logger.error(f"Error reading data file: {str(e)}")
        return
    
    # Check if using an ensemble model
    is_ensemble = args.model_type.startswith('ensemble')
    ensemble_type = args.model_type.split('_')[1] if is_ensemble else ''
    
    logger.info("Initializing enhanced trainer with GARCH volatility modeling")
    if is_ensemble:
        logger.info(f"Using ensemble model with {ensemble_type} method")
    
    # Initialize the trainer with specified parameters
    trainer = ImprovedCryptoTrainer(
        data_path=args.data_path,
        sequence_length=5,  # Reduced from 10
        batch_size=16,      # Reduced from 128
        epochs=20,
        model_type=args.model_type,
        use_garch=args.use_garch,
        use_bollinger=args.use_bollinger,
        use_macd=args.use_macd,
        use_moving_avg=args.use_moving_avg
    )
    
    logger.info(f"Training models with {' enhanced feature set including' if args.use_garch or args.use_bollinger else ' standard features'}{' GARCH volatility modeling' if args.use_garch else ''}{' and' if args.use_garch and args.use_bollinger else ''}{' Bollinger Bands indicators' if args.use_bollinger else ''}{' and MACD indicators' if args.use_macd else ''}{' and Moving Average indicators' if args.use_moving_avg else ''}")
    
    # Train models
    trainer.train_models()

    # Existing code for evaluation...
    print("\n" + "="*70)
    features_text = []
    if args.use_garch:
        features_text.append("GARCH VOLATILITY")
    if args.use_bollinger:
        features_text.append("BOLLINGER BANDS")
    if args.use_macd:
        features_text.append("MACD")
    if args.use_moving_avg:
        features_text.append("MOVING AVERAGES")
    
    features_str = " WITH " + " & ".join(features_text) + " FEATURES" if features_text else ""
    print(f"COMPREHENSIVE MODEL EVALUATION{features_str}")
    print("="*70)
    
    trained_cryptos = [crypto_id for crypto_id in trainer.crypto_ids if crypto_id in trainer.best_models]
    if not trained_cryptos:
        print("No trained models available. Train models first or check logs.")
        return
    
    print(f"Evaluating {len(trained_cryptos)} models: {', '.join(trained_cryptos)}")
    
    for crypto_id in trained_cryptos:
        print(f"\n{'#'*20} Evaluating {crypto_id.upper()} {'#'*20}")
        try:
            evaluate_model(trainer, crypto_id)
            analyze_model_performance(trainer, crypto_id)
            if args.plot:
                plot_actual_vs_predicted(trainer, crypto_id, f"model_evaluation/{crypto_id}_actual_vs_predicted.png")
                generate_residual_plot(trainer, crypto_id, f"model_evaluation/{crypto_id}_residuals.png")
                generate_error_distribution_plot(trainer, crypto_id, f"model_evaluation/{crypto_id}_error_distribution.png")
                # Generate ensemble visualization if it's an ensemble model
                if args.model_type.startswith('ensemble'):
                    visualize_ensemble_predictions(trainer, crypto_id, f"model_evaluation/{crypto_id}_ensemble_comparison.png")
                # Generate feature ensemble visualization if it's a feature ensemble model
                if args.model_type.startswith('feature'):
                    visualize_feature_ensemble(trainer, crypto_id, f"model_evaluation/{crypto_id}_feature_ensemble.png")
            if args.predict_days > 0:
                predict_future(trainer, crypto_id, days=args.predict_days, plot=args.plot,
                              save_path=f"model_evaluation/{crypto_id}_future_prediction.png")
            print(f"Completed evaluation for {crypto_id}")
        except Exception as e:
            print(f"Error evaluating {crypto_id}: {str(e)}")
    
    if len(trained_cryptos) > 1:
        # Only generate comparison plots if plotting is enabled
        compare_models(trainer, metric='mae', plot=args.plot)

if __name__ == "__main__":
    """
    Entry point for the script. Runs the transformer model test and starts the main process.
    """
    # First run our test to verify the fix
    print("Testing Transformer model...")
    if not test_transformer_model():
        print("Transformer model test failed. Exiting.")
        exit(1)
    print("Transformer model test passed. Continuing with execution.")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Cryptocurrency Prediction Pipeline')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="cryptocurrency_data.csv", 
                        help='Path to cryptocurrency data CSV file')
    parser.add_argument('--crypto_ids', type=str, default=None, 
                        help='Comma-separated list of cryptocurrency IDs to train (e.g., "bitcoin,ethereum")')
    
    # Model parameters
    # ensemble_average best performance
    parser.add_argument('--model_type', type=str, default='feature_average', 
                       choices=['lstm', 'transformer', 
                                'ensemble_average', 'ensemble_weighted', 'ensemble_stacking',
                                'feature_average', 'feature_weighted', 'feature_stacking'],
                       help='Type of model to use (lstm, transformer, ensemble variants, or feature ensemble variants)')
    parser.add_argument('--use_garch', action='store_true', default=True,
                       help='Whether to use GARCH volatility modeling features (helps capture volatility clustering in price time series)')
    parser.add_argument('--use_bollinger', action='store_true', default=True,
                       help='Whether to use Bollinger Bands indicators (helps identify overbought/oversold conditions)')
    parser.add_argument('--use_macd', action='store_true', default=True,
                       help='Whether to use MACD indicators (helps identify trend strength and potential reversals)')
    parser.add_argument('--use_moving_avg', action='store_true', default=False,
                       help='Whether to use Moving Average indicators (helps identify trends and support/resistance levels)')
    parser.add_argument('--sequence_length', type=int, default=10, 
                        help='Sequence length for time series prediction')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proportion of data to use for testing')
    parser.add_argument('--target_column', type=str, default='log_return', 
                     choices=['price', 'price_change', 'log_return', 'direction'],
                     help='Target column to predict (price, price_change, log_return, or direction)')
    
    # New plotting parameter
    parser.add_argument('--plot', action='store_true', default=True,
                        help='Whether to generate plots for evaluation results')
    
    # Training parameters
    parser.add_argument('--train', action='store_true', default=True,
                        help='Whether to train models or just evaluate existing ones')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30, 
                        help='Maximum number of training epochs')
    parser.add_argument('--cv_splits', type=int, default=3, 
                        help='Number of cross-validation splits')
    parser.add_argument('--optimize', action='store_true', default=False,
                        help='Whether to optimize hyperparameters')
    parser.add_argument('--n_trials', type=int, default=10, 
                        help='Number of hyperparameter optimization trials')
    
    # Evaluation parameters
    parser.add_argument('--predict_days', type=int, default=30, 
                        help='Number of days to predict into the future')
    
    # Verbosity control
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2],
                        help='Output verbosity level (0=minimal, 1=normal, 2=detailed)')
    
    # New parameters
    parser.add_argument('--cache', action='store_true', default=False, help='Use cached preprocessed data/models')
    
    args = parser.parse_args()
    
    # Check if running without explicit command line arguments
    # If so, use these specific default settings
    if len(sys.argv) == 1:  # Only the script name in sys.argv, no other arguments
        args.train = True
        # args.plot is now controlled by the --plot flag
        # args.plot = True  # This was overriding the default False setting
        args.predict_days = 30
        args.n_trials = 10  # Reduce the number of trials for testing
        print(f"Running with default settings: Training=ON, Plotting={'ON' if args.plot else 'OFF'}, Trials=10")
    
    main(args) 