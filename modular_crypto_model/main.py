#main.py
from evaluation import (
    evaluate_model,
    analyze_model_performance,
    plot_actual_vs_predicted,
    generate_residual_plot,
    generate_error_distribution_plot,
    predict_future,
    compare_models,
    generate_anomaly_detection_plot
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

# For testing our fix
def test_transformer_model():
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
            print(f"Output shape: {output.shape}, expected: torch.Size([{batch_size}])")
            
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
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory for plots
    os.makedirs("model_evaluation", exist_ok=True)
    
    # Initialize the trainer
    trainer = ImprovedCryptoTrainer(
        data_path=args.data_path,
        crypto_ids=args.crypto_ids.split(',') if args.crypto_ids else None,
        sequence_length=args.sequence_length,
        test_size=args.test_size,
        target_column=args.target_column,
        batch_size=args.batch_size,
        epochs=args.epochs,
        model_type=args.model_type,
        verbosity=args.verbosity
    )
    
    # Train models if requested
    if args.train:
        print("\n" + "="*70)
        print("TRAINING CRYPTOCURRENCY MODELS")
        print("="*70)
        trainer.train_models(
            n_splits=args.cv_splits, 
            optimize=args.optimize, 
            n_trials=args.n_trials
        )
    
    # Print comprehensive evaluation for each cryptocurrency
    print("\n" + "="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    
    # Get list of cryptocurrencies that have trained models
    trained_cryptos = [crypto_id for crypto_id in trainer.crypto_ids if crypto_id in trainer.best_models]
    
    if not trained_cryptos:
        print("No cryptocurrency models were successfully trained. Check the training logs for errors.")
        return
        
    print(f"Evaluating {len(trained_cryptos)} trained cryptocurrency models: {', '.join(trained_cryptos)}")
    
    for crypto_id in trained_cryptos:
        print(f"\n{'#'*20} Evaluating {crypto_id.upper()} {'#'*20}")
        try:
            # Calculate all standard metrics
            evaluate_model(trainer, crypto_id)
            
            # Perform detailed analysis and display metrics
            analysis_results = analyze_model_performance(trainer, crypto_id)
            
            # Generate plots for the model
            print(f"\nGenerating visualization plots for {crypto_id}...")
            
            # Basic plots
            plot_path = plot_actual_vs_predicted(
                trainer, 
                crypto_id, 
                save_path=f"model_evaluation/{crypto_id}_actual_vs_predicted.png"
            )
            
            residual_path = generate_residual_plot(
                trainer, 
                crypto_id, 
                save_path=f"model_evaluation/{crypto_id}_residuals.png"
            )
            
            error_dist_path = generate_error_distribution_plot(
                trainer, 
                crypto_id, 
                save_path=f"model_evaluation/{crypto_id}_error_distribution.png"
            )
            
            # Advanced anomaly detection plot
            anomaly_path = generate_anomaly_detection_plot(
                trainer,
                crypto_id,
                save_path=f"model_evaluation/{crypto_id}_anomalies.png"
            )
            
            # Predict future values
            if args.predict_days > 0:
                print(f"\nGenerating future predictions for {crypto_id} ({args.predict_days} days)...")
                future_predictions = predict_future(
                    trainer, 
                    crypto_id, 
                    days=args.predict_days, 
                    plot=True, 
                    save_path=f"model_evaluation/{crypto_id}_future_prediction.png"
                )
            
            print(f"Successfully completed evaluation for {crypto_id}")
        except Exception as e:
            print(f"Error evaluating {crypto_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Compare all models (only if we have multiple models)
    if len(trained_cryptos) > 1:
        try:
            print("\nComparing model performance across cryptocurrencies...")
            # Compare using different metrics
            for metric in ['mse', 'mae', 'direction_accuracy']:
                compare_models(trainer, metric=metric)
        except Exception as e:
            print(f"Error comparing models: {str(e)}")
    
    print("\nEvaluation completed. All plots saved to the 'model_evaluation' directory.")

if __name__ == "__main__":
    # First run our test to verify the fix
    print("Testing Transformer model...")
    if not test_transformer_model():
        print("Transformer model test failed. Exiting.")
        exit(1)
    print("Transformer model test passed. Continuing with execution.")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate cryptocurrency prediction models')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default="cryptocurrency_data.csv", 
                        help='Path to cryptocurrency data CSV file')
    parser.add_argument('--crypto_ids', type=str, default=None, 
                        help='Comma-separated list of cryptocurrency IDs to train (e.g., "bitcoin,ethereum")')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='transformer', choices=['lstm', 'transformer'],
                       help='Type of model to use (lstm or transformer)')
    parser.add_argument('--sequence_length', type=int, default=10, 
                        help='Sequence length for time series prediction')
    parser.add_argument('--test_size', type=float, default=0.2, 
                        help='Proportion of data to use for testing')
    parser.add_argument('--target_column', type=str, default='price', choices=['price', 'direction'],
                        help='Target column to predict (price or direction)')
    
    # Training parameters
    parser.add_argument('--train', action='store_true',
                        help='Whether to train models or just evaluate existing ones')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Maximum number of training epochs')
    parser.add_argument('--cv_splits', type=int, default=3, 
                        help='Number of cross-validation splits')
    parser.add_argument('--optimize', action='store_true',
                        help='Whether to optimize hyperparameters')
    parser.add_argument('--n_trials', type=int, default=10, 
                        help='Number of hyperparameter optimization trials')
    
    # Evaluation parameters
    parser.add_argument('--predict_days', type=int, default=30, 
                        help='Number of days to predict into the future')
    
    # Verbosity control
    parser.add_argument('--verbosity', type=int, default=1, choices=[0, 1, 2],
                        help='Output verbosity level (0=minimal, 1=normal, 2=detailed)')
    
    args = parser.parse_args()
    
    # Check if running without explicit command line arguments
    # If so, set training to True by default
    if len(sys.argv) == 1:  # Only the script name in sys.argv, no other arguments
        args.train = True
        args.optimize = True
        args.predict_days = 30
        args.n_trials = 10  # Reduce the number of trials for testing
        print("Running with default settings: Training=ON, Optimization=ON, Trials=10")
    
    main(args) 