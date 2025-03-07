#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model Evaluation Module

This module handles the evaluation of trained cryptocurrency price prediction models.
It provides functionality to assess model performance using various metrics,
compare different models, and visualize prediction results.
"""

import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
from datetime import datetime

from data_processing import DataProcessor
from models import create_model, load_model
from utils import setup_logging, plot_predictions

class ModelEvaluator:
    """
    A class for evaluating cryptocurrency price prediction models.
    
    This class provides methods for loading trained models, evaluating their
    performance on test data, calculating various metrics, and visualizing results.
    
    Attributes:
        args (argparse.Namespace): Command-line arguments or configuration parameters
        logger (logging.Logger): Logger for recording evaluation information
        device (torch.device): Device for model evaluation (CPU or GPU)
        data_processor (DataProcessor): Data processing utility for cryptocurrency data
        results_dict (dict): Dictionary to store evaluation results
    """
    
    def __init__(self, args):
        """
        Initialize the ModelEvaluator with the provided arguments.
        
        Args:
            args (argparse.Namespace): Command-line arguments or configuration parameters
        """
        self.args = args
        self.logger = logging.getLogger('crypto_trainer')
        
        # Set device for model evaluation (CPU/GPU)
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
        
        # Create a dictionary to store evaluation results
        self.results_dict = {}
        
        # Create necessary directories
        os.makedirs(args.output_dir, exist_ok=True)
        
    def evaluate_models(self):
        """
        Evaluate models for all specified cryptocurrencies.
        
        This method iterates through each cryptocurrency specified in the arguments,
        loads the corresponding model, and evaluates its performance on test data.
        """
        self.logger.info("Starting model evaluation process")
        
        # Evaluate models for each cryptocurrency
        for crypto in self.args.cryptocurrencies:
            self.logger.info(f"Evaluating model for {crypto}")
            self._evaluate_single_model(crypto)
            
        # Calculate aggregate results
        self.calculate_aggregate_results()
        
        # Save results to file
        self._save_results()
        
        self.logger.info("Model evaluation completed for all cryptocurrencies")
        
    def _evaluate_single_model(self, cryptocurrency):
        """
        Evaluate a model for a single cryptocurrency.
        
        Args:
            cryptocurrency (str): The name of the cryptocurrency to evaluate
            
        Returns:
            dict: Evaluation metrics
        """
        # Process data for the specific cryptocurrency
        _, _, X_test, _, _, y_test, scaler = self.data_processor.prepare_data(cryptocurrency)
        
        # Determine model path
        if self.args.model_path:
            # Check if the model_path is a directory or a specific file
            if os.path.isdir(self.args.model_path):
                model_path = os.path.join(self.args.model_path, f"{cryptocurrency}_best_model.pth")
            else:
                model_path = self.args.model_path
        else:
            model_path = os.path.join(self.args.output_dir, f"{cryptocurrency}_best_model.pth")
            
        if not os.path.exists(model_path):
            self.logger.warning(f"Model for {cryptocurrency} not found at {model_path}, skipping")
            return None
            
        # Load model
        model = load_model(model_path, self.device)
        
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
        metrics = self._calculate_metrics(y_test_inv, predictions)
        
        # Log metrics
        self.logger.info(f"Evaluation metrics for {cryptocurrency.capitalize()}:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
            
        # Store results in the results dictionary
        for metric_name, metric_value in metrics.items():
            self.results_dict[f"{cryptocurrency}_{self.args.model_type}_{metric_name}"] = f"{metric_value:.4f}"
            
        # Plot predictions
        plot_predictions(
            y_true=y_test_inv,
            y_pred=predictions,
            title=f"{cryptocurrency.capitalize()} Price Predictions",
            output_path=os.path.join(self.args.output_dir, f"{cryptocurrency}_predictions.png")
        )
        
        return metrics
        
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate various evaluation metrics between true and predicted values.
        
        Args:
            y_true (numpy.ndarray): Ground truth values
            y_pred (numpy.ndarray): Predicted values
            
        Returns:
            dict: Dictionary containing calculated metrics
        """
        metrics = {}
        
        # Calculate MSE
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # Calculate MAE
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Calculate RMSE
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # Calculate R²
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Calculate MAPE (handling zero values)
        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        metrics['mape'] = mean_absolute_percentage_error(y_true + epsilon, y_pred + epsilon)
        
        return metrics
        
    def calculate_aggregate_results(self):
        """
        Calculate aggregate results across multiple cryptocurrencies.
        
        This method computes average metrics across all evaluated cryptocurrencies
        and stores them in the results dictionary.
        """
        self.logger.info("Calculating aggregate results")
        
        # Define the metrics to aggregate
        metrics_to_aggregate = ['mse', 'mae', 'rmse', 'r2', 'mape']
        
        # Calculate aggregates for each metric
        for metric in metrics_to_aggregate:
            self.calculate_total_aggregate_results(self.args.cryptocurrencies, metric, self.args.model_type)
            
    def calculate_total_aggregate_results(self, cryptocurrencies, metric_label, model_type):
        """
        Calculate aggregate results for specified cryptocurrencies and add to results_dict.
        
        This method computes the average of a specific metric across all specified
        cryptocurrencies and prints the individual values for each cryptocurrency.
        
        Args:
            cryptocurrencies (list): List of cryptocurrency names
            metric_label (str): The metric to aggregate (e.g., 'mse', 'mae')
            model_type (str): Model type identifier
        """
        total_sum = 0.0
        for crypto in cryptocurrencies:
            # Get the metric value for this cryptocurrency from the results dictionary
            value = float(self.results_dict.get(f"{crypto}_{model_type}_{metric_label}", 0))
            total_sum += value
            # Print the individual cryptocurrency metric value
            # This is the line that generates the output seen in the logs: "BITCOIN: 2.8629", etc.
            print(f"{crypto.upper()}: {value:.4f}")
        
        # Calculate average of the metric across cryptocurrencies
        average = total_sum / len(cryptocurrencies) if cryptocurrencies else 0
        self.results_dict[f"AVERAGE_{model_type}_{metric_label}"] = f"{average:.4f}"
        
    def _save_results(self):
        """
        Save evaluation results to a file.
        
        This method saves the calculated metrics to CSV and JSON files for later reference.
        """
        # Save results as CSV
        results_df = pd.DataFrame(list(self.results_dict.items()), columns=['Metric', 'Value'])
        csv_path = os.path.join(self.args.output_dir, 'evaluation_results.csv')
        results_df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved evaluation results to {csv_path}")
        
        # Save results as JSON
        json_path = os.path.join(self.args.output_dir, 'evaluation_results.json')
        with open(json_path, 'w') as json_file:
            json.dump(self.results_dict, json_file, indent=4)
        self.logger.info(f"Saved evaluation results to {json_path}")
        
    def compare_models(self, model_types, cryptocurrencies=None):
        """
        Compare different model types across specified cryptocurrencies.
        
        Args:
            model_types (list): List of model types to compare
            cryptocurrencies (list, optional): List of cryptocurrencies to compare.
                                              If None, uses all cryptocurrencies from args.
        """
        if cryptocurrencies is None:
            cryptocurrencies = self.args.cryptocurrencies
            
        self.logger.info(f"Comparing models: {', '.join(model_types)} for cryptocurrencies: {', '.join(cryptocurrencies)}")
        
        # Create a DataFrame to store comparison results
        comparison_data = []
        
        # Iterate through cryptocurrencies and model types
        for crypto in cryptocurrencies:
            for model_type in model_types:
                model_metrics = {}
                model_metrics['Cryptocurrency'] = crypto.capitalize()
                model_metrics['Model Type'] = model_type
                
                # Get metrics for this model and cryptocurrency
                for metric in ['mse', 'mae', 'rmse', 'r2', 'mape']:
                    key = f"{crypto}_{model_type}_{metric}"
                    if key in self.results_dict:
                        model_metrics[metric.upper()] = float(self.results_dict[key])
                    else:
                        model_metrics[metric.upper()] = None
                        
                comparison_data.append(model_metrics)
                
        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save comparison to CSV
        csv_path = os.path.join(self.args.output_dir, 'model_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        self.logger.info(f"Saved model comparison to {csv_path}")
        
        # Visualize comparison
        self._plot_model_comparison(comparison_df)
        
    def _plot_model_comparison(self, comparison_df):
        """
        Plot a comparison of different models.
        
        Args:
            comparison_df (pandas.DataFrame): DataFrame containing comparison data
        """
        # Plot MSE comparison
        plt.figure(figsize=(12, 8))
        
        # Group by cryptocurrency and model type
        pivot_table = comparison_df.pivot_table(
            index='Cryptocurrency',
            columns='Model Type',
            values='MSE'
        )
        
        # Plot bar chart
        pivot_table.plot(kind='bar', ax=plt.gca())
        plt.title('MSE Comparison Across Models and Cryptocurrencies')
        plt.ylabel('Mean Squared Error (lower is better)')
        plt.xlabel('Cryptocurrency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.args.output_dir, 'model_mse_comparison.png')
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved MSE comparison plot to {plot_path}")
        
        # Additional plots can be created for other metrics

def parse_args():
    """
    Parse command-line arguments for the evaluator module.
    
    This function can be used when running the evaluator module directly.
    
    Returns:
        argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Cryptocurrency Model Evaluator')
    
    # General configuration arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to the configuration JSON file containing evaluation parameters')
    parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level for application messages')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation outputs and visualizations')

    # Model configuration arguments
    parser.add_argument('--cryptocurrencies', nargs='+', default=['bitcoin', 'ethereum', 'monero'],
                        help='List of cryptocurrencies to evaluate (space-separated)')
    parser.add_argument('--model_type', type=str, default='LSTM', choices=['LSTM', 'GRU', 'SimpleRNN', 'Transformer', 'CNN_LSTM'],
                        help='Type of neural network model to evaluate')
    parser.add_argument('--features', nargs='+', default=['Close', 'Volume', 'Market_Cap'],
                        help='List of features used in the model (space-separated)')
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='Length of input sequence used in the model')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                        help='Number of future time steps predicted by the model')
    
    # Data processing arguments
    parser.add_argument('--train_test_split', type=float, default=0.8,
                        help='Ratio of training data to total data used during model training')
    parser.add_argument('--validation_split', type=float, default=0.1,
                        help='Ratio of validation data to training data used during model training')
    parser.add_argument('--data_start_date', type=str, default='2015-01-01',
                        help='Start date for historical data (YYYY-MM-DD format)')
    parser.add_argument('--data_end_date', type=str, default=None,
                        help='End date for historical data (YYYY-MM-DD format, defaults to current date)')
    parser.add_argument('--normalization', type=str, default='minmax', choices=['minmax', 'zscore', 'robust'],
                        help='Data normalization method used')
    
    # Model loading arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to load trained models (directory containing model files or a specific model file)')
    
    # Evaluation options
    parser.add_argument('--metrics', nargs='+', default=['mse', 'mae', 'rmse', 'mape', 'r2'],
                        help='Evaluation metrics to calculate (space-separated)')
    parser.add_argument('--compare_models', action='store_true',
                        help='Compare multiple model types (requires models of different types to be available)')
    parser.add_argument('--model_types_to_compare', nargs='+', default=[],
                        help='List of model types to compare if compare_models is enabled (space-separated)')
    
    # Advanced options
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for evaluation if available')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def main():
    """
    Main function for executing the evaluator module directly.
    
    This function parses command-line arguments, sets up logging, initializes the
    evaluator, and executes the evaluation workflow.
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
    log_filename = f"crypto_evaluator_{timestamp}.log"
    setup_logging(args.log_level, log_filename)
    logger = logging.getLogger('crypto_trainer')
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create evaluator instance
    evaluator = ModelEvaluator(args)
    
    # Evaluate models
    evaluator.evaluate_models()
    
    # Compare models if requested
    if args.compare_models and args.model_types_to_compare:
        evaluator.compare_models(args.model_types_to_compare)
        
    logger.info("Cryptocurrency model evaluation completed")

if __name__ == "__main__":
    main() 