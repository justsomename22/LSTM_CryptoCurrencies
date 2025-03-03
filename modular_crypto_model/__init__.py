#__init__.py
# Modular Cryptocurrency Prediction Package
# This module initializes the package and imports necessary functions and classes for use in the project.

from data_processing import add_technical_indicators, normalize_and_fill_data, add_advanced_features, find_balanced_threshold
from models import ImprovedCryptoLSTM, PositionalEncoding
from utils import EarlyStopping
from trainer import ImprovedCryptoTrainer
from evaluation import (
    calculate_standard_metrics,
    evaluate_model,
    plot_actual_vs_predicted,
    generate_residual_plot,
    generate_error_distribution_plot,
    analyze_model_performance,
    predict_future,
    compare_models
) 