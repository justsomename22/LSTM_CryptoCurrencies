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
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions).flatten()
    y_test = np.array(y_test).flatten()
    
    # Ensure we have at least 2 elements for diff calculation
    if len(predictions) < 2 or len(y_test) < 2:
        print("Warning: Not enough data points to calculate all metrics")
        return {
            'mse': float('nan'),
            'mae': float('nan'),
            'rmse': float('nan'),
            'direction_accuracy': float('nan'),
            'r2_score': float('nan'),
            'mape': float('nan'),
            'smape': float('nan'),
            'mdape': float('nan')
        }
    
    # Basic metrics that don't require special handling
    metrics = {
        'mse': float(np.mean((predictions - y_test) ** 2)),
        'mae': float(np.mean(np.abs(predictions - y_test))),
        'rmse': float(np.sqrt(np.mean((predictions - y_test) ** 2))),
        'r2_score': float(r2_score(y_test, predictions))
    }
    
    # Direction accuracy requires diff calculation
    try:
        pred_dir = np.diff(predictions) > 0
        actual_dir = np.diff(y_test) > 0
        metrics['direction_accuracy'] = float(np.mean(pred_dir == actual_dir))
    except Exception as e:
        print(f"Warning: Error calculating direction accuracy: {str(e)}")
        metrics['direction_accuracy'] = float('nan')
    
    # Calculate MAPE, but avoid division by zero
    try:
        mask = y_test != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
            metrics['mape'] = float(mape)
        else:
            metrics['mape'] = float('inf')
    except Exception as e:
        print(f"Warning: Error calculating MAPE: {str(e)}")
        metrics['mape'] = float('nan')
    
    # Symmetric MAPE (SMAPE)
    try:
        denominator = np.abs(y_test) + np.abs(predictions)
        mask = denominator != 0
        if np.any(mask):
            smape = np.mean(2.0 * np.abs(predictions[mask] - y_test[mask]) / denominator[mask]) * 100
            metrics['smape'] = float(smape)
        else:
            metrics['smape'] = float('inf')
    except Exception as e:
        print(f"Warning: Error calculating SMAPE: {str(e)}")
        metrics['smape'] = float('nan')
    
    # Median Absolute Percentage Error (MDAPE)
    try:
        mask = y_test != 0
        if np.any(mask):
            mdape = np.median(np.abs((y_test[mask] - predictions[mask]) / y_test[mask])) * 100
            metrics['mdape'] = float(mdape)
        else:
            metrics['mdape'] = float('inf')
    except Exception as e:
        print(f"Warning: Error calculating MDAPE: {str(e)}")
        metrics['mdape'] = float('nan')
    
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
        return None
    
    X_test = trainer.data_by_crypto[crypto_id]['X_test']
    y_test = trainer.data_by_crypto[crypto_id]['y_test']
    
    # Get predictions
    predictions = trainer.predict(crypto_id, X_test)
    
    # Check for empty predictions or shape issues
    if len(predictions) == 0:
        print(f"Error: No valid predictions generated for {crypto_id}")
        return None
        
    # Ensure shapes match for evaluation
    if len(predictions) != len(y_test):
        print(f"Warning: Prediction length ({len(predictions)}) doesn't match ground truth length ({len(y_test)})")
        # Match lengths - use shorter length
        min_len = min(len(predictions), len(y_test))
        predictions = predictions[:min_len]
        y_test = y_test[:min_len]
        
        if min_len == 0:
            print(f"Error: No valid data for evaluation for {crypto_id}")
            return None
    
    # Compute metrics
    metrics = calculate_standard_metrics(predictions, y_test)
    trainer.test_metrics[crypto_id] = metrics
    
    # Print R² score explicitly
    print(f"R² score for {crypto_id}: {metrics['r2_score']:.4f}")
    
    return metrics

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
    if crypto_id not in trainer.best_models:
        print(f"No trained model found for {crypto_id}")
        return
    
    X_test = trainer.data_by_crypto[crypto_id]['X_test']
    y_test = trainer.data_by_crypto[crypto_id]['y_test']
    
    # Get predictions
    try:
        predictions = trainer.predict(crypto_id, X_test)
        
        # Check for empty predictions
        if len(predictions) == 0:
            print(f"Error: No valid predictions generated for {crypto_id}")
            return
            
        # Ensure shapes match for plotting
        if len(predictions) != len(y_test):
            print(f"Warning: Prediction length ({len(predictions)}) doesn't match ground truth length ({len(y_test)})")
            # Match lengths - use shorter length
            min_len = min(len(predictions), len(y_test))
            predictions = predictions[:min_len]
            y_test = y_test[:min_len]
            
            if min_len == 0:
                print(f"Error: No valid data for plotting for {crypto_id}")
                return
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual', color='blue', marker='o', alpha=0.7, markersize=3)
        plt.plot(predictions, label='Predicted', color='red', marker='x', alpha=0.7, markersize=3)
        plt.title(f'Actual vs Predicted Values for {crypto_id.upper()}')
        plt.xlabel('Time Steps')
        plt.ylabel('Price (Normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² score annotation
        try:
            r2 = r2_score(y_test, predictions)
            plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        except Exception as e:
            print(f"Warning: Could not calculate R² score: {str(e)}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        print(f"Error plotting for {crypto_id}: {str(e)}")
        import traceback
        traceback.print_exc()

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
    if crypto_id not in trainer.best_models:
        print(f"No trained model found for {crypto_id}")
        return
    
    X_test = trainer.data_by_crypto[crypto_id]['X_test']
    y_test = trainer.data_by_crypto[crypto_id]['y_test']
    predictions = trainer.predict(crypto_id, X_test)
    
    # Calculate residuals
    residuals = y_test - predictions
    
    plt.figure(figsize=(12, 6))
    plt.scatter(predictions, residuals, alpha=0.6, color='blue')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title(f'Residual Plot for {crypto_id.upper()}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.grid(True, alpha=0.3)
    
    # Add mean and std of residuals
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    plt.annotate(f'Mean: {mean_residual:.4f}\nStd: {std_residual:.4f}', 
                 xy=(0.05, 0.05), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Residual plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

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
    if crypto_id not in trainer.best_models:
        print(f"No trained model found for {crypto_id}")
        return
    
    X_test = trainer.data_by_crypto[crypto_id]['X_test']
    y_test = trainer.data_by_crypto[crypto_id]['y_test']
    predictions = trainer.predict(crypto_id, X_test)
    
    # Calculate errors
    errors = y_test - predictions
    
    plt.figure(figsize=(12, 6))
    sns.histplot(errors, kde=True, color='blue')
    plt.axvline(x=0, color='r', linestyle='-')
    plt.title(f'Error Distribution for {crypto_id.upper()}')
    plt.xlabel('Error (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    plt.annotate(f'Mean: {mean_error:.4f}\nStd: {std_error:.4f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Error distribution plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

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
        print(f"No metrics available for {crypto_id}. Run evaluate_model first.")
        return None
    
    metrics = trainer.test_metrics[crypto_id]
    print(f"\nPerformance for {crypto_id.upper()}:")
    for k, v in metrics.items():
        print(f"{k.upper()}: {v:.4f}")
    
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
    if crypto_id not in trainer.best_models:
        print(f"No trained model found for {crypto_id}")
        return
    
    try:
        X_test = trainer.data_by_crypto[crypto_id]['X_test']
        y_test = trainer.data_by_crypto[crypto_id]['y_test']
        predictions = trainer.predict(crypto_id, X_test)
        
        # Check for empty predictions
        if len(predictions) == 0:
            print(f"Error: No valid predictions generated for {crypto_id}")
            return
            
        # Ensure shapes match for plotting
        if len(predictions) != len(y_test):
            print(f"Warning: Prediction length ({len(predictions)}) doesn't match ground truth length ({len(y_test)})")
            # Match lengths - use shorter length
            min_len = min(len(predictions), len(y_test))
            predictions = predictions[:min_len]
            y_test = y_test[:min_len]
            
            if min_len == 0:
                print(f"Error: No valid data for plotting for {crypto_id}")
                return
        
        # Calculate absolute errors
        abs_errors = np.abs(y_test - predictions)
        
        # Determine error threshold (e.g., mean + 2*std)
        threshold = np.mean(abs_errors) + 2 * np.std(abs_errors)
        
        # Identify anomalies
        anomalies = abs_errors > threshold
        
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Actual', color='blue', alpha=0.7)
        plt.plot(predictions, label='Predicted', color='green', alpha=0.7)
        plt.scatter(np.where(anomalies)[0], y_test[anomalies], color='red', 
                    label='Anomalies', s=50, zorder=5)
        
        plt.title(f'Anomaly Detection for {crypto_id.upper()}')
        plt.xlabel('Time Steps')
        plt.ylabel('Price (Normalized)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.annotate(f'Anomalies: {np.sum(anomalies)} ({np.mean(anomalies)*100:.2f}%)', 
                     xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Anomaly detection plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    except Exception as e:
        print(f"Error generating anomaly detection plot for {crypto_id}: {str(e)}")
        import traceback
        traceback.print_exc()

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
    if crypto_id not in trainer.best_models:
        print(f"No trained model found for {crypto_id}")
        return
    
    print(f"Predicting future {days} days for {crypto_id}...")
    # Future prediction implementation here
    
    # Only print this message if plotting is enabled but implementation is incomplete
    if plot:
        print(f"Future prediction plotting not fully implemented in this version")
    else:
        print(f"Future prediction completed for {crypto_id} (no plots generated)")
    
    return

def compare_models(trainer, metric='mae', plot=True):
    """
    Compare the performance of different models.
    
    Parameters:
    - trainer: The trainer object containing the models and data
    - metric: The metric to use for comparison
    - plot: Whether to generate and save plots
    
    Returns:
    - None
    """
    cryptos = [cid for cid in trainer.crypto_ids if cid in trainer.test_metrics]
    if not cryptos:
        print("No models evaluated. Run evaluate_model first.")
        return
    
    metric_values = {}
    
    for cid in cryptos:
        value = trainer.test_metrics[cid].get(metric, 'N/A')
        if isinstance(value, (int, float)):
            print(f"{cid.upper()}: {value:.4f}")
            metric_values[cid] = value
        else:
            print(f"{cid.upper()}: {value}")
    
    # For ensemble models, we can also add a comparison with the base models
    if trainer.model_type.startswith('ensemble') and hasattr(trainer, 'best_models'):
        for cid in cryptos:
            if cid in trainer.best_models:
                model = trainer.best_models[cid]
                # Check if it's an ensemble model
                if hasattr(model, 'base_models'):
                    print(f"\nEnsemble Performance Analysis for {cid.upper()}")
                    
                    # Evaluate each base model separately and compare
                    X_test = trainer.data_by_crypto[cid]['X_test']
                    y_test = trainer.data_by_crypto[cid]['y_test']
                    X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)
                    y_test_tensor = torch.FloatTensor(y_test).to(trainer.device)
                    
                    # Define loss function (same as trainer)
                    if trainer.target_column == 'price':
                        criterion = torch.nn.HuberLoss()
                    else:
                        criterion = torch.nn.BCELoss()
                    
                    # Evaluate ensemble model (already done before, just get the metrics)
                    ensemble_performance = trainer.test_metrics[cid][metric]
                    
                    # Evaluate base models
                    base_performances = []
                    base_names = ['LSTM', 'GRU', 'Transformer']  # Default names based on standard config
                    
                    # Get predictions from each base model
                    model.eval()
                    with torch.no_grad():
                        for i, base_model in enumerate(model.base_models):
                            base_model.eval()
                            try:
                                # Get predictions
                                base_predictions = base_model(X_test_tensor).cpu().numpy()
                                
                                # Make sure predictions have right shape
                                if len(base_predictions.shape) > 1 and base_predictions.shape[1] == 1:
                                    base_predictions = base_predictions.flatten()
                                
                                # Calculate metrics
                                base_metrics = calculate_standard_metrics(base_predictions, y_test)
                                base_performances.append(base_metrics[metric])
                                
                                # Print comparisons
                                print(f"Base Model {i+1} ({base_names[i]}): {base_metrics[metric]:.4f}")
                            except Exception as e:
                                print(f"Error evaluating base model {i+1}: {str(e)}")
                    
                    # Print ensemble vs best base model
                    if base_performances:
                        best_base = min(base_performances)
                        improvement = ((best_base - ensemble_performance) / best_base) * 100
                        print(f"Ensemble ({model.ensemble_method.capitalize()}): {ensemble_performance:.4f}")
                        print(f"Best Base Model: {min(base_performances):.4f}")
                        print(f"Improvement over best base model: {improvement:.2f}%")
                        
                        # Print weights for weighted ensemble
                        if model.ensemble_method == 'weighted':
                            weights = model.weights.data.cpu().numpy()
                            for i, w in enumerate(weights):
                                print(f"Weight for Model {i+1} ({base_names[i]}): {w:.4f}")

    # Find best performing model
    if metric_values and plot:
        best_crypto = min(metric_values, key=metric_values.get)
        
        # Create comparison bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(metric_values.keys(), metric_values.values(), color='skyblue')
        plt.title(f'Comparison of {metric.upper()} across Cryptocurrencies')
        plt.xlabel('Cryptocurrency')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(metric_values.values()):
            plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        save_path = f"model_evaluation/comparison_{metric}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Comparison plot saved to {save_path}")
        plt.close()
    elif metric_values:
        best_crypto = min(metric_values, key=metric_values.get)
    
    return 

def visualize_ensemble_predictions(trainer, crypto_id, save_path=None):
    """
    Visualize the predictions of the ensemble model and its base models.
    
    Parameters:
        trainer (ImprovedCryptoTrainer): The trainer object with trained models.
        crypto_id (str): The ID of the cryptocurrency to visualize.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Check if a trained model exists for this crypto
    if crypto_id not in trainer.best_models:
        print(f"No trained model found for {crypto_id}")
        return
    
    # Check if this is a feature ensemble model
    if not trainer.model_type.startswith('feature'):
        print(f"Model for {crypto_id} is not a feature ensemble model")
        return
    
    # Load model
    model = trainer.best_models[crypto_id]
    
    # Get test data
    X_test = trainer.data_by_crypto[crypto_id]['X_test']
    y_test = trainer.data_by_crypto[crypto_id]['y_test']
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)
    
    # Get predictions from the ensemble and each base model
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Get predictions from each base model with its respective feature mask
        base_predictions = []
        feature_names = model.feature_sets
        
        with torch.no_grad():
            for i, base_model in enumerate(model.base_models):
                # Apply feature mask
                mask = model.feature_masks[i]
                if mask is not None:
                    mask_expanded = mask.expand(X_test_tensor.shape[0], X_test_tensor.shape[1], -1).to(trainer.device)
                    masked_X = X_test_tensor * mask_expanded
                    
                    # Add small values to maintain tensor structure
                    small_value = 1e-8
                    inverted_mask = 1.0 - mask_expanded
                    masked_X = masked_X + (small_value * inverted_mask)
                    
                    # Get predictions
                    base_pred = base_model(masked_X).cpu().numpy()
                else:
                    # Use all features
                    base_pred = base_model(X_test_tensor).cpu().numpy()
                
                base_predictions.append(base_pred)
                
            # Get ensemble predictions
            ensemble_pred = model(X_test_tensor).cpu().numpy()
    
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Convert predictions back to original scale if necessary
    if trainer.target_column == 'price':
        # Inverse transform the predictions
        scaler = trainer.scalers[crypto_id]
        transform = trainer.target_transforms.get(crypto_id)
        
        # Apply the inverse transformations
        y_test_orig = y_test
        if transform is not None:
            ensemble_pred = transform.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
            y_test_orig = transform.inverse_transform(y_test.reshape(-1, 1)).flatten()
            base_predictions = [transform.inverse_transform(pred.reshape(-1, 1)).flatten() for pred in base_predictions]
    
    # Calculate RMSE for each model
    ensemble_rmse = np.sqrt(mean_squared_error(y_test_orig, ensemble_pred))
    base_rmses = [np.sqrt(mean_squared_error(y_test_orig, pred)) for pred in base_predictions]
    
    # Calculate MAE for each model
    ensemble_mae = mean_absolute_error(y_test_orig, ensemble_pred)
    base_maes = [mean_absolute_error(y_test_orig, pred) for pred in base_predictions]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Plot actual vs predicted values
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(y_test_orig, label='Actual', color='black', linewidth=2)
    ax1.plot(ensemble_pred, label=f'Ensemble ({trainer.model_type})', color='blue', linewidth=2)
    
    # Add base model predictions
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, pred in enumerate(base_predictions):
        ax1.plot(pred, label=f'{feature_names[i]}', color=colors[i % len(colors)], alpha=0.7)
    
    ax1.set_title(f'{crypto_id.capitalize()} Price Prediction Comparison')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Plot feature importance (based on MAE)
    ax2 = fig.add_subplot(2, 2, 2)
    feature_importance = [(feature_names[i], base_maes[i]) for i in range(len(base_maes))]
    # Sort by performance (lower MAE is better)
    feature_importance.sort(key=lambda x: x[1])
    
    feature_names_sorted = [x[0] for x in feature_importance]
    feature_maes_sorted = [x[1] for x in feature_importance]
    
    y_pos = np.arange(len(feature_names_sorted))
    ax2.barh(y_pos, feature_maes_sorted, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names_sorted)
    ax2.invert_yaxis()  # Labels read top-to-bottom
    ax2.set_xlabel('Mean Absolute Error (lower is better)')
    ax2.set_title('Feature Set Performance')
    
    # 3. Plot error distribution
    ax3 = fig.add_subplot(2, 2, 3)
    ensemble_errors = y_test_orig - ensemble_pred
    
    # Plot histogram of errors
    ax3.hist(ensemble_errors, bins=20, alpha=0.7, color='blue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_title(f'Error Distribution (RMSE={ensemble_rmse:.4f}, MAE={ensemble_mae:.4f})')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    
    # 4. Plot comparison of ensemble vs. best base model
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Find the best base model
    best_model_idx = np.argmin(base_maes)
    best_model_name = feature_names[best_model_idx]
    best_model_pred = base_predictions[best_model_idx]
    best_model_mae = base_maes[best_model_idx]
    
    # Calculate improvement
    improvement = (best_model_mae - ensemble_mae) / best_model_mae * 100
    
    # Plot ensemble vs best base model
    ind = np.arange(2)
    width = 0.35
    ax4.bar(ind, [ensemble_mae, best_model_mae], width, label=['Ensemble', best_model_name])
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title(f'Ensemble vs Best Base Model: {improvement:.2f}% Improvement')
    ax4.set_xticks(ind)
    ax4.set_xticklabels(['Ensemble', f'Best Base Model\n({best_model_name})'])
    
    # Add ensemble method information
    ensemble_method = trainer.model_type.split('_')[1] if '_' in trainer.model_type else 'unknown'
    plt.figtext(0.5, 0.01, f"Ensemble Method: {ensemble_method}", ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature ensemble visualization saved to {save_path}")
    else:
        plt.show()
        
    plt.close()

def visualize_feature_ensemble(trainer, crypto_id, save_path=None):
    """
    Visualize the predictions of the feature ensemble model and compare performance
    across different feature sets.
    
    Parameters:
        trainer (ImprovedCryptoTrainer): The trainer object with trained models.
        crypto_id (str): The ID of the cryptocurrency to visualize.
        save_path (str, optional): Path to save the plot. If None, the plot is displayed.
    """
    # Check if a trained model exists for this crypto
    if crypto_id not in trainer.best_models:
        print(f"No trained model found for {crypto_id}")
        return
    
    # Check if this is a feature ensemble model
    if not trainer.model_type.startswith('feature'):
        print(f"Model for {crypto_id} is not a feature ensemble model")
        return
    
    # Load model
    model = trainer.best_models[crypto_id]
    
    # Get test data
    X_test = trainer.data_by_crypto[crypto_id]['X_test']
    y_test = trainer.data_by_crypto[crypto_id]['y_test']
    
    # Convert to PyTorch tensors
    X_test_tensor = torch.FloatTensor(X_test).to(trainer.device)
    
    # Get predictions from the ensemble and each base model
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Get predictions from each base model with its respective feature mask
        base_predictions = []
        feature_names = model.feature_sets
        
        with torch.no_grad():
            for i, base_model in enumerate(model.base_models):
                # Apply feature mask
                mask = model.feature_masks[i]
                if mask is not None:
                    mask_expanded = mask.expand(X_test_tensor.shape[0], X_test_tensor.shape[1], -1).to(trainer.device)
                    masked_X = X_test_tensor * mask_expanded
                    
                    # Add small values to maintain tensor structure
                    small_value = 1e-8
                    inverted_mask = 1.0 - mask_expanded
                    masked_X = masked_X + (small_value * inverted_mask)
                    
                    # Get predictions
                    base_pred = base_model(masked_X).cpu().numpy()
                else:
                    # Use all features
                    base_pred = base_model(X_test_tensor).cpu().numpy()
                
                base_predictions.append(base_pred)
                
            # Get ensemble predictions
            ensemble_pred = model(X_test_tensor).cpu().numpy()
    
    except Exception as e:
        print(f"Error generating predictions: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # Convert predictions back to original scale if necessary
    if trainer.target_column == 'price':
        # Inverse transform the predictions
        scaler = trainer.scalers[crypto_id]
        transform = trainer.target_transforms.get(crypto_id)
        
        # Apply the inverse transformations
        y_test_orig = y_test
        if transform is not None:
            ensemble_pred = transform.inverse_transform(ensemble_pred.reshape(-1, 1)).flatten()
            y_test_orig = transform.inverse_transform(y_test.reshape(-1, 1)).flatten()
            base_predictions = [transform.inverse_transform(pred.reshape(-1, 1)).flatten() for pred in base_predictions]
    
    # Calculate RMSE for each model
    ensemble_rmse = np.sqrt(mean_squared_error(y_test_orig, ensemble_pred))
    base_rmses = [np.sqrt(mean_squared_error(y_test_orig, pred)) for pred in base_predictions]
    
    # Calculate MAE for each model
    ensemble_mae = mean_absolute_error(y_test_orig, ensemble_pred)
    base_maes = [mean_absolute_error(y_test_orig, pred) for pred in base_predictions]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Plot actual vs predicted values
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(y_test_orig, label='Actual', color='black', linewidth=2)
    ax1.plot(ensemble_pred, label=f'Ensemble ({trainer.model_type})', color='blue', linewidth=2)
    
    # Add base model predictions
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, pred in enumerate(base_predictions):
        ax1.plot(pred, label=f'{feature_names[i]}', color=colors[i % len(colors)], alpha=0.7)
    
    ax1.set_title(f'{crypto_id.capitalize()} Price Prediction Comparison')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Plot feature importance (based on MAE)
    ax2 = fig.add_subplot(2, 2, 2)
    feature_importance = [(feature_names[i], base_maes[i]) for i in range(len(base_maes))]
    # Sort by performance (lower MAE is better)
    feature_importance.sort(key=lambda x: x[1])
    
    feature_names_sorted = [x[0] for x in feature_importance]
    feature_maes_sorted = [x[1] for x in feature_importance]
    
    y_pos = np.arange(len(feature_names_sorted))
    ax2.barh(y_pos, feature_maes_sorted, align='center')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names_sorted)
    ax2.invert_yaxis()  # Labels read top-to-bottom
    ax2.set_xlabel('Mean Absolute Error (lower is better)')
    ax2.set_title('Feature Set Performance')
    
    # 3. Plot error distribution
    ax3 = fig.add_subplot(2, 2, 3)
    ensemble_errors = y_test_orig - ensemble_pred
    
    # Plot histogram of errors
    ax3.hist(ensemble_errors, bins=20, alpha=0.7, color='blue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.set_title(f'Error Distribution (RMSE={ensemble_rmse:.4f}, MAE={ensemble_mae:.4f})')
    ax3.set_xlabel('Error')
    ax3.set_ylabel('Frequency')
    
    # 4. Plot comparison of ensemble vs. best base model
    ax4 = fig.add_subplot(2, 2, 4)
    
    # Find the best base model
    best_model_idx = np.argmin(base_maes)
    best_model_name = feature_names[best_model_idx]
    best_model_pred = base_predictions[best_model_idx]
    best_model_mae = base_maes[best_model_idx]
    
    # Calculate improvement
    improvement = (best_model_mae - ensemble_mae) / best_model_mae * 100
    
    # Plot ensemble vs best base model
    ind = np.arange(2)
    width = 0.35
    ax4.bar(ind, [ensemble_mae, best_model_mae], width, label=['Ensemble', best_model_name])
    
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Mean Absolute Error')
    ax4.set_title(f'Ensemble vs Best Base Model: {improvement:.2f}% Improvement')
    ax4.set_xticks(ind)
    ax4.set_xticklabels(['Ensemble', f'Best Base Model\n({best_model_name})'])
    
    # Add ensemble method information
    ensemble_method = trainer.model_type.split('_')[1] if '_' in trainer.model_type else 'unknown'
    plt.figtext(0.5, 0.01, f"Ensemble Method: {ensemble_method}", ha='center', fontsize=12, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Feature ensemble visualization saved to {save_path}")
    else:
        plt.show()
        
    plt.close() 