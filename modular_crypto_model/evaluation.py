#evaluation.py
"""
Evaluation module for cryptocurrency price prediction models.

This module provides functions for calculating metrics and generating
visualizations to evaluate model performance.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from datetime import timedelta, datetime

# Define standard metrics that should be calculated for all cryptocurrencies
STANDARD_METRICS = ['mse', 'mae', 'rmse', 'direction_accuracy', 'r2_score', 'mape', 'smape', 'mdape']

def calculate_standard_metrics(predictions, y_test):
    """
    Calculate a standard set of metrics for model evaluation.
    
    Parameters:
    - predictions: Model predictions
    - y_test: Actual target values
    
    Returns:
    - dict: Dictionary containing all standard metrics
    """
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = float(np.mean((predictions - y_test) ** 2))
    
    # Mean Absolute Error
    metrics['mae'] = float(np.mean(np.abs(predictions - y_test)))
    
    # Root Mean Squared Error
    metrics['rmse'] = float(np.sqrt(metrics['mse']))
    
    # Direction Accuracy (correct prediction of up/down movement)
    predict_diff = np.diff(predictions, prepend=predictions[0])
    actual_diff = np.diff(y_test, prepend=y_test[0])
    metrics['direction_accuracy'] = float(np.mean((predict_diff > 0) == (actual_diff > 0)))
    
    # R-squared (coefficient of determination)
    y_mean = np.mean(y_test)
    ss_total = np.sum((y_test - y_mean) ** 2)
    ss_residual = np.sum((y_test - predictions) ** 2)
    metrics['r2_score'] = float(1 - (ss_residual / ss_total) if ss_total != 0 else 0)
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    abs_percentage_errors = np.abs((y_test - predictions) / (np.abs(y_test) + epsilon)) * 100
    metrics['mape'] = float(np.mean(abs_percentage_errors))
    
    # Symmetric Mean Absolute Percentage Error (SMAPE)
    denominator = np.abs(y_test) + np.abs(predictions) + epsilon
    smape_errors = 2 * np.abs(y_test - predictions) / denominator * 100
    metrics['smape'] = float(np.mean(smape_errors))
    
    # Median Absolute Percentage Error (MdAPE)
    metrics['mdape'] = float(np.median(abs_percentage_errors))
    
    return metrics

def evaluate_model(trainer, crypto_id):
    """
    Evaluate the model for a specific cryptocurrency.
    
    Parameters:
    - trainer: The trainer object containing the model and data
    - crypto_id: The ID of the cryptocurrency to evaluate
    
    Returns:
    - dict: Dictionary of metrics
    """
    if crypto_id not in trainer.best_models:
        print(f"No trained model available for {crypto_id}")
        return None
    
    # Get the test data and predictions
    if hasattr(trainer, 'X_test') and hasattr(trainer, 'y_test'):
        X_test = trainer.X_test[crypto_id]
        y_test = trainer.y_test[crypto_id]
        
        # Get the predictions using the trainer's predict method
        predictions = trainer.predict(crypto_id, X_test)
        
        # Calculate all standard metrics
        metrics = calculate_standard_metrics(predictions, y_test)
        
        # Update the trainer's test metrics
        if crypto_id in trainer.test_metrics:
            trainer.test_metrics[crypto_id].update(metrics)
        else:
            trainer.test_metrics[crypto_id] = metrics
        
        return metrics
    else:
        print(f"Test data not available for {crypto_id}")
        return None

def plot_actual_vs_predicted(trainer, crypto_id, save_path=None):
    """
    Plot actual vs predicted values for the model.
    
    Parameters:
    - trainer: The trainer object containing the model and data
    - crypto_id: The ID of the cryptocurrency to plot
    - save_path: Optional path to save the plot
    
    Returns:
    - None
    """
    # Plotting logic here...
    pass

def generate_residual_plot(trainer, crypto_id, save_path=None):
    """
    Generate a residual plot for the model.
    
    Parameters:
    - trainer: The trainer object containing the model and data
    - crypto_id: The ID of the cryptocurrency to plot
    - save_path: Optional path to save the plot
    
    Returns:
    - None
    """
    # Residual plotting logic here...
    pass

def generate_error_distribution_plot(trainer, crypto_id, save_path=None):
    """
    Generate a plot of the error distribution for the model.
    
    Parameters:
    - trainer: The trainer object containing the model and data
    - crypto_id: The ID of the cryptocurrency to plot
    - save_path: Optional path to save the plot
    
    Returns:
    - None
    """
    # Error distribution plotting logic here...
    pass

def analyze_model_performance(trainer, crypto_id):
    """
    Analyze the performance of the model for a specific cryptocurrency.
    
    Parameters:
    - trainer: The trainer object containing the model and data
    - crypto_id: The ID of the cryptocurrency to analyze
    
    Returns:
    - None
    """
    if crypto_id not in trainer.test_metrics:
        print(f"No test metrics available for {crypto_id}")
        return None
    
    # Get the metrics from the trainer object
    metrics = trainer.test_metrics[crypto_id]
    
    # Print a nice formatted table of metrics
    print(f"\n{'='*50}")
    print(f"PERFORMANCE METRICS FOR {crypto_id.upper()}")
    print(f"{'='*50}")
    
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name.upper().ljust(20)}: {metric_value:.4f}")
    
    print(f"{'='*50}\n")
    
    # Calculate additional metrics if needed
    # For example, we might want to calculate RMSE from MSE
    if 'mse' in metrics and 'rmse' not in metrics:
        rmse = np.sqrt(metrics['mse'])
        print(f"RMSE: {rmse:.4f}")
    
    return metrics

def generate_anomaly_detection_plot(trainer, crypto_id, save_path=None):
    """
    Generate a plot for anomaly detection in the model predictions.
    
    Parameters:
    - trainer: The trainer object containing the model and data
    - crypto_id: The ID of the cryptocurrency to plot
    - save_path: Optional path to save the plot
    
    Returns:
    - None
    """
    # Anomaly detection plotting logic here...
    pass

def predict_future(trainer, crypto_id, days=30, plot=True, save_path=None):
    """
    Predict future values for a specific cryptocurrency.
    
    Parameters:
    - trainer: The trainer object containing the model and data
    - crypto_id: The ID of the cryptocurrency to predict
    - days: Number of days to predict into the future
    - plot: Whether to plot the predictions
    - save_path: Optional path to save the plot
    
    Returns:
    - None
    """
    # Future prediction logic here...
    pass

def compare_models(trainer, metric='direction_accuracy'):
    """
    Compare the performance of different models.
    
    Parameters:
    - trainer: The trainer object containing the models and data
    - metric: The metric to use for comparison
    
    Returns:
    - None
    """
    # Get cryptocurrencies that have metrics
    cryptos_with_metrics = [crypto_id for crypto_id in trainer.crypto_ids 
                           if crypto_id in trainer.test_metrics]
    
    if not cryptos_with_metrics:
        print("No cryptocurrencies have metrics available for comparison")
        return
    
    # Collect all unique metrics across all cryptocurrencies
    all_metrics = set()
    for crypto_id in cryptos_with_metrics:
        all_metrics.update(trainer.test_metrics[crypto_id].keys())
    all_metrics = sorted(list(all_metrics))
    
    # Print a nice formatted table of metrics for comparison
    print(f"\n{'='*80}")
    print(f"COMPARING MODEL PERFORMANCE ACROSS CRYPTOCURRENCIES")
    print(f"{'='*80}")
    
    # Print the header
    header = "METRIC".ljust(25)
    for crypto_id in cryptos_with_metrics:
        header += crypto_id.upper().ljust(20)
    print(header)
    print("-" * 80)
    
    # Print each metric for each cryptocurrency
    for metric_name in all_metrics:
        row = metric_name.upper().ljust(25)
        for crypto_id in cryptos_with_metrics:
            if metric_name in trainer.test_metrics[crypto_id]:
                value = trainer.test_metrics[crypto_id][metric_name]
                row += f"{value:.4f}".ljust(20)
            else:
                row += "N/A".ljust(20)
        print(row)
    
    print(f"{'='*80}\n")
    
    # Highlight the best performing cryptocurrency for the specified metric
    if metric in all_metrics:
        best_crypto = None
        best_value = float('-inf')
        
        for crypto_id in cryptos_with_metrics:
            if metric in trainer.test_metrics[crypto_id]:
                value = trainer.test_metrics[crypto_id][metric]
                
                # For error metrics like MSE, MAE, RMSE, lower is better
                if metric in ['mse', 'mae', 'rmse', 'mape', 'smape', 'mdape']:
                    if best_crypto is None or value < best_value:
                        best_crypto = crypto_id
                        best_value = value
                # For accuracy metrics, higher is better
                else:
                    if value > best_value:
                        best_crypto = crypto_id
                        best_value = value
        
        if best_crypto:
            print(f"Best performing cryptocurrency for {metric.upper()}: {best_crypto.upper()} ({best_value:.4f})")
    
    return 