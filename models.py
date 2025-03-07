#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Neural Network Models for Cryptocurrency Price Prediction

This module defines various neural network architectures for time series forecasting,
specifically designed for cryptocurrency price prediction. It includes implementations
of LSTM, GRU, RNN, Transformer, and CNN-LSTM hybrid models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import numpy as np
from typing import Tuple, Dict, Optional, Union, List

class LSTMModel(nn.Module):
    """
    Long Short-Term Memory (LSTM) model for time series forecasting.
    
    LSTM networks are well-suited for time series prediction as they can learn
    long-term dependencies in sequential data through their specialized memory cell
    structure with input, output, and forget gates.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units in LSTM layers
        num_layers (int): Number of stacked LSTM layers
        output_dim (int): Number of output features (prediction horizon)
        dropout (float): Dropout rate for regularization
        lstm (nn.LSTM): LSTM layer
        fc (nn.Linear): Fully connected output layer
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units in LSTM layers
            num_layers (int): Number of stacked LSTM layers
            output_dim (int): Number of output features (prediction horizon)
            dropout (float): Dropout rate for regularization
        """
        super(LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Initialize hidden state with zeros
        batch_size = x.size(0)
        
        # LSTM forward pass
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, _ = self.lstm(x)
        
        # We take the output from the last time step
        # lstm_out shape: (batch_size, sequence_length, hidden_dim)
        # lstm_out[:, -1, :] shape: (batch_size, hidden_dim)
        y_pred = self.fc(lstm_out[:, -1, :])
        
        return y_pred
        
class GRUModel(nn.Module):
    """
    Gated Recurrent Unit (GRU) model for time series forecasting.
    
    GRU is a simplified version of LSTM with fewer parameters but similar performance
    on many tasks. It uses a gating mechanism to control the flow of information.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units in GRU layers
        num_layers (int): Number of stacked GRU layers
        output_dim (int): Number of output features (prediction horizon)
        dropout (float): Dropout rate for regularization
        gru (nn.GRU): GRU layer
        fc (nn.Linear): Fully connected output layer
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        """
        Initialize the GRU model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units in GRU layers
            num_layers (int): Number of stacked GRU layers
            output_dim (int): Number of output features (prediction horizon)
            dropout (float): Dropout rate for regularization
        """
        super(GRUModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GRU model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # GRU forward pass
        # x shape: (batch_size, sequence_length, input_dim)
        gru_out, _ = self.gru(x)
        
        # We take the output from the last time step
        # gru_out shape: (batch_size, sequence_length, hidden_dim)
        # gru_out[:, -1, :] shape: (batch_size, hidden_dim)
        y_pred = self.fc(gru_out[:, -1, :])
        
        return y_pred
        
class RNNModel(nn.Module):
    """
    Simple Recurrent Neural Network (RNN) model for time series forecasting.
    
    Basic RNN with vanilla recurrent units. While simple, it often suffers from
    the vanishing gradient problem when dealing with long sequences.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units in RNN layers
        num_layers (int): Number of stacked RNN layers
        output_dim (int): Number of output features (prediction horizon)
        dropout (float): Dropout rate for regularization
        rnn (nn.RNN): RNN layer
        fc (nn.Linear): Fully connected output layer
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, dropout: float = 0.2):
        """
        Initialize the RNN model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units in RNN layers
            num_layers (int): Number of stacked RNN layers
            output_dim (int): Number of output features (prediction horizon)
            dropout (float): Dropout rate for regularization
        """
        super(RNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout = dropout
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RNN model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # RNN forward pass
        # x shape: (batch_size, sequence_length, input_dim)
        rnn_out, _ = self.rnn(x)
        
        # We take the output from the last time step
        # rnn_out shape: (batch_size, sequence_length, hidden_dim)
        # rnn_out[:, -1, :] shape: (batch_size, hidden_dim)
        y_pred = self.fc(rnn_out[:, -1, :])
        
        return y_pred
        
class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer models.
    
    Since Transformer models don't have recurrence or convolution, they need
    positional information to understand the order of the sequence.
    This class implements sinusoidal positional encoding.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize the positional encoding.
        
        Args:
            d_model (int): Dimension of the model
            max_len (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model)
            
        Returns:
            torch.Tensor: Tensor with positional encoding added
        """
        # Add positional encoding to the input
        # x shape: (batch_size, sequence_length, d_model)
        # pe shape: (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x
        
class TransformerModel(nn.Module):
    """
    Transformer model for time series forecasting.
    
    Transformers use self-attention mechanisms to capture relationships between
    all positions in a sequence, making them effective for many sequence-based tasks.
    
    Attributes:
        input_dim (int): Number of input features
        d_model (int): Dimension of the model (embedding dimension)
        nhead (int): Number of attention heads
        num_encoder_layers (int): Number of encoder layers
        dim_feedforward (int): Dimension of the feedforward network
        output_dim (int): Number of output features (prediction horizon)
        dropout (float): Dropout rate for regularization
        input_projection (nn.Linear): Projects input features to model dimension
        positional_encoding (PositionalEncoding): Adds positional information
        transformer_encoder (nn.TransformerEncoder): Transformer encoder layers
        output_projection (nn.Linear): Projects hidden states to output dimension
    """
    
    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, 
                 num_encoder_layers: int = 2, dim_feedforward: int = 256, 
                 output_dim: int = 1, dropout: float = 0.2):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim (int): Number of input features
            d_model (int): Dimension of the model (embedding dimension)
            nhead (int): Number of attention heads
            num_encoder_layers (int): Number of encoder layers
            dim_feedforward (int): Dimension of the feedforward network
            output_dim (int): Number of output features (prediction horizon)
            dropout (float): Dropout rate for regularization
        """
        super(TransformerModel, self).__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layers,
            num_layers=num_encoder_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Transformer model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        # Project input to d_model dimensions
        # x shape: (batch_size, sequence_length, input_dim)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        # No mask is needed for encoder-only transformer
        transformer_output = self.transformer_encoder(x)
        
        # Use the output from the last position
        # transformer_output shape: (batch_size, sequence_length, d_model)
        # transformer_output[:, -1, :] shape: (batch_size, d_model)
        y_pred = self.output_projection(transformer_output[:, -1, :])
        
        return y_pred
        
class CNN_LSTMModel(nn.Module):
    """
    Hybrid CNN-LSTM model for time series forecasting.
    
    This model uses convolutional layers to extract features from input data,
    followed by LSTM layers to capture temporal dependencies. The combination
    can be effective for time series with both local and long-range patterns.
    
    Attributes:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units in LSTM layers
        num_layers (int): Number of stacked LSTM layers
        output_dim (int): Number of output features (prediction horizon)
        kernel_size (int): Size of the convolutional kernel
        dropout (float): Dropout rate for regularization
        conv1 (nn.Conv1d): First convolutional layer
        conv2 (nn.Conv1d): Second convolutional layer
        lstm (nn.LSTM): LSTM layer
        fc (nn.Linear): Fully connected output layer
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, 
                 output_dim: int, kernel_size: int = 3, dropout: float = 0.2):
        """
        Initialize the CNN-LSTM model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of hidden units in LSTM layers
            num_layers (int): Number of stacked LSTM layers
            output_dim (int): Number of output features (prediction horizon)
            kernel_size (int): Size of the convolutional kernel
            dropout (float): Dropout rate for regularization
        """
        super(CNN_LSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        
        # CNN layers for feature extraction
        # Note: For 1D convolution in PyTorch, the input shape is (batch_size, channels, length)
        # So we will need to transpose the input
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        self.conv2 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN-LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim)
        """
        batch_size, seq_len, _ = x.size()
        
        # Transpose for CNN: (batch_size, sequence_length, input_dim) -> (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Transpose back for LSTM: (batch_size, hidden_dim, sequence_length) -> (batch_size, sequence_length, hidden_dim)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        y_pred = self.fc(lstm_out[:, -1, :])
        
        return y_pred
        
def create_model(model_type: str, input_dim: int, hidden_dim: int, 
                num_layers: int, output_dim: int, dropout: float = 0.2) -> nn.Module:
    """
    Factory function to create a model of the specified type.
    
    Args:
        model_type (str): Type of model to create ('LSTM', 'GRU', 'SimpleRNN',
                          'Transformer', or 'CNN_LSTM')
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden units
        num_layers (int): Number of layers
        output_dim (int): Number of output features (prediction horizon)
        dropout (float): Dropout rate for regularization
        
    Returns:
        nn.Module: The created model
        
    Raises:
        ValueError: If the model type is not recognized
    """
    if model_type == 'LSTM':
        return LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    elif model_type == 'GRU':
        return GRUModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    elif model_type == 'SimpleRNN':
        return RNNModel(input_dim, hidden_dim, num_layers, output_dim, dropout)
    elif model_type == 'Transformer':
        return TransformerModel(input_dim, hidden_dim, nhead=4, num_encoder_layers=num_layers, 
                               dim_feedforward=hidden_dim*4, output_dim=output_dim, dropout=dropout)
    elif model_type == 'CNN_LSTM':
        return CNN_LSTMModel(input_dim, hidden_dim, num_layers, output_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
def save_model(model: nn.Module, path: str) -> None:
    """
    Save a model to disk.
    
    Args:
        model (nn.Module): The model to save
        path (str): Path where the model will be saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the model
    torch.save(model.state_dict(), path)
    logging.info(f"Model saved to {path}")
    
def load_model(path: str, device: torch.device, 
              model_type: str = 'LSTM', input_dim: int = None, 
              hidden_dim: int = 64, num_layers: int = 2, 
              output_dim: int = 1) -> nn.Module:
    """
    Load a model from disk.
    
    Args:
        path (str): Path to the saved model
        device (torch.device): Device to load the model onto
        model_type (str): Type of model to load
        input_dim (int): Number of input features (required if creating a new model)
        hidden_dim (int): Number of hidden units
        num_layers (int): Number of layers
        output_dim (int): Number of output features (prediction horizon)
        
    Returns:
        nn.Module: The loaded model
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        ValueError: If input_dim is not provided and is needed
    """
    # Check if model file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
        
    # Create a new model instance
    if input_dim is None:
        raise ValueError("input_dim must be provided to load the model")
        
    model = create_model(model_type, input_dim, hidden_dim, num_layers, output_dim)
    
    # Load the model state
    model.load_state_dict(torch.load(path, map_location=device))
    
    # Move model to the specified device
    model = model.to(device)
    
    logging.info(f"Model loaded from {path}")
    return model

# Example usage when run directly
if __name__ == "__main__":
    # Create a sample LSTM model
    input_dim = 3  # Example: Close, Volume, Market_Cap
    hidden_dim = 64
    num_layers = 2
    output_dim = 1  # Predicting the next closing price
    
    lstm_model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    
    # Create a sample input
    batch_size = 16
    sequence_length = 60
    sample_input = torch.randn(batch_size, sequence_length, input_dim)
    
    # Forward pass
    output = lstm_model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}") 