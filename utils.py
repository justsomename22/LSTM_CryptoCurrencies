#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility Functions for Cryptocurrency Price Prediction

This module provides various utility functions used throughout the cryptocurrency
price prediction system, including data visualization, logging setup, model saving
and loading, and miscellaneous helper functions.
"""

import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Configure the logging system for the application.
    
    Args:
        log_level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file (str, optional): Path to the log file. If None, logs to console only.
    """
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplicate log entries
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    logging.info(f"Logging initialized with level {log_level}")
    if log_file:
        logging.info(f"Logs will be saved to {log_file}")
        
def plot_training_history(history: Dict[str, List[float]], title: str = 'Training History', 
                        output_path: Optional[str] = None) -> None:
    """
    Plot training and validation loss over epochs.
    
    Args:
        history (dict): Dictionary containing 'train_loss' and 'val_loss' lists
        title (str): Title for the plot
        output_path (str, optional): Path to save the plot. If None, plot is displayed but not saved.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(history['train_loss'], label='Training Loss')
    
    # Plot validation loss if available
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')
        
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save or display the plot
    if output_path:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Training history plot saved to {output_path}")
    else:
        plt.show()
        
def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = 'Price Predictions',
                   output_path: Optional[str] = None) -> None:
    """
    Plot actual vs. predicted cryptocurrency prices.
    
    Args:
        y_true (numpy.ndarray): Array of actual price values
        y_pred (numpy.ndarray): Array of predicted price values
        title (str): Title for the plot
        output_path (str, optional): Path to save the plot. If None, plot is displayed but not saved.
    """
    plt.figure(figsize=(12, 6))
    
    # Handle different dimensions of predictions
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        # Multi-step prediction, plot only the first sequence for clarity
        plt.plot(y_true[0], label='Actual')
        plt.plot(y_pred[0], label='Predicted', linestyle='--')
    else:
        # Single-step prediction
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted', linestyle='--')
        
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Save or display the plot
    if output_path:
        # Ensure directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_path)
        plt.close()
        logging.info(f"Prediction plot saved to {output_path}")
    else:
        plt.show()
        
def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate various performance metrics between actual and predicted values.
    
    Args:
        y_true (numpy.ndarray): Array of actual values
        y_pred (numpy.ndarray): Array of predicted values
        
    Returns:
        dict: Dictionary containing calculated metrics
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    metrics = {}
    
    # Mean Squared Error
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    
    # Root Mean Squared Error
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # Mean Absolute Error
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    
    # R-squared
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    # Adding a small epsilon to avoid division by zero
    epsilon = 1e-10
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    return metrics

def create_directory_if_not_exists(directory_path: str) -> None:
    """
    Create a directory if it doesn't already exist.
    
    Args:
        directory_path (str): Path to the directory to create
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")
        
def format_time(seconds: float) -> str:
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds (float): Time in seconds
        
    Returns:
        str: Formatted time string (e.g., "2h 30m 45s")
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        time_parts.append(f"{minutes}m")
    time_parts.append(f"{seconds}s")
    
    return " ".join(time_parts)

# Example usage when run directly
if __name__ == "__main__":
    # Setup logging
    setup_logging(log_level='INFO', log_file='example.log')
    
    # Example training history
    history = {
        'train_loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'val_loss': [0.55, 0.45, 0.35, 0.3, 0.25]
    }
    
    # Plot training history
    plot_training_history(history, title='Example Training History')
    
    # Example predictions
    y_true = np.array([10, 11, 12, 13, 14, 15])
    y_pred = np.array([10.2, 10.8, 12.3, 12.9, 14.1, 15.2])
    
    # Plot predictions
    plot_predictions(y_true, y_pred, title='Example Predictions')
    
    # Calculate and print metrics
    metrics = calculate_metrics(y_true, y_pred)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}") 