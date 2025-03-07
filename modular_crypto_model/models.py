#models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class AttentionMechanism(nn.Module):
    """
    Basic attention mechanism to focus on important parts of the sequence.
    
    Attributes:
        hidden_dim (int): Number of hidden units in the attention layer.
    """
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),  # First linear layer
            nn.Tanh(),  # Activation function
            nn.Linear(hidden_dim // 2, 1)  # Output layer for attention weights
        )
        
    def forward(self, lstm_output):
        """
        Forward pass through the attention mechanism.

        Parameters:
            lstm_output (Tensor): Output from the LSTM layer.

        Returns:
            Tuple[Tensor, Tensor]: Context vector and attention weights.
        """
        # lstm_output shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = lstm_output.size()
        
        # Calculate attention weights
        attn_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # Normalize weights across sequence
        
        # Apply attention to get context vector
        context = torch.bmm(lstm_output.transpose(1, 2), attn_weights)  # [batch_size, hidden_dim, 1]
        context = context.squeeze(2)  # [batch_size, hidden_dim]
        
        return context, attn_weights


class ImprovedCryptoLSTM(nn.Module):
    """
    Improved LSTM model for cryptocurrency prediction with attention mechanism and residual connections.
    
    This model can effectively leverage GARCH volatility features by incorporating them
    as additional input features, allowing the network to learn relationships between
    historical volatility patterns and price movements.
    
    Attributes:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in the LSTM layer.
        num_layers (int): Number of LSTM layers.
        dropout (float): Dropout rate for regularization.
        output_dim (int): Number of output features.
        sequence_length (int): Length of input sequences.
        prediction_type (str): Type of prediction ('price' or 'direction').
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, 
                 output_dim=1, sequence_length=10, prediction_type='price'):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.prediction_type = prediction_type
        
        # Initial feature processing (normalization layer)
        self.feature_norm = nn.BatchNorm1d(input_dim)
        
        # Use a simple unidirectional LSTM first to ensure stability
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False  # Change to unidirectional for stability
        )
        
        # Output layer with different structures based on prediction type
        if prediction_type == 'direction':
            # For direction prediction (binary classification)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            # For price prediction (regression)
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output predictions.
        """
        # Input shape: [batch_size, sequence_length, input_dim]
        batch_size = x.size(0)
        
        try:
            # Apply batch normalization to features
            # Reshape for BatchNorm1d which expects [N, C, L]
            x_reshaped = x.transpose(1, 2)  # [batch_size, input_dim, sequence_length]
            x_normed = self.feature_norm(x_reshaped).transpose(1, 2)  # back to [batch_size, sequence_length, input_dim]
            
            # Initialize hidden state
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
            
            # Forward propagate LSTM
            # lstm_out shape: [batch_size, sequence_length, hidden_dim]
            lstm_out, _ = self.lstm(x_normed, (h0, c0))
            
            # Use only the last time step's output instead of attention
            last_hidden = lstm_out[:, -1, :]
            
            # Pass through the fully connected layers
            out = self.fc(last_hidden)
            
            return out
        except Exception as e:
            print(f"Error in ImprovedCryptoLSTM.forward: {str(e)}")
            # Fall back to a simpler approach if the above fails
            try:
                # Just use the raw LSTM on the input without batch norm
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
                
                lstm_out, _ = self.lstm(x, (h0, c0))
                last_hidden = lstm_out[:, -1, :]
                
                return self.fc(last_hidden)
            except Exception as e2:
                print(f"Error in fallback code: {str(e2)}")
                # Last resort: return zeros of appropriate shape
                if self.prediction_type == 'direction':
                    return torch.zeros(batch_size, 1).to(x.device)
                else:
                    return torch.zeros(batch_size, self.output_dim).to(x.device)

# Additional model classes could be added here for ensemble methods
class CryptoGRU(nn.Module):
    """
    GRU-based model for cryptocurrency prediction.
    This could be used as part of an ensemble.
    
    Attributes:
        input_dim (int): Number of input features.
        hidden_dim (int): Number of hidden units in the GRU layer.
        num_layers (int): Number of GRU layers.
        dropout (float): Dropout rate for regularization.
        output_dim (int): Number of output features.
        sequence_length (int): Length of input sequences.
        prediction_type (str): Type of prediction ('price' or 'direction').
    """
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3, 
                output_dim=1, sequence_length=10, prediction_type='price'):
        super(CryptoGRU, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.prediction_type = prediction_type
        
        # Feature normalization
        self.feature_norm = nn.BatchNorm1d(input_dim)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        if prediction_type == 'direction':
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim)
            )
    
    def forward(self, x):
        """
        Forward pass through the GRU model.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

        Returns:
            Tensor: Output predictions.
        """
        batch_size = x.size(0)
        
        # Apply batch normalization
        x_normed = self.feature_norm(x.transpose(1, 2)).transpose(1, 2)  # Normalize features
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        gru_out, _ = self.gru(x_normed, h0)
        
        # Get the output from the last time step
        out = self.fc(gru_out[:, -1, :])
        
        return out

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    Adds information about the position of tokens in the sequence.
    
    Attributes:
        d_model (int): Dimension of the model.
        dropout (float): Dropout rate.
        max_len (int): Maximum length of the input sequences.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # Initialize positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # Position indices
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # Divisor for scaling
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)  # Register as buffer to avoid being a model parameter

    def forward(self, x):
        """
        Add positional encoding to the input tensor.

        Parameters:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            Tensor: Tensor with positional encoding added.
        """
        x = x + self.pe[:, :x.size(1), :]  # Add positional encoding
        return self.dropout(x)  # Apply dropout

# Only Transformer kept here for simplicity; SimpleLSTM moved to trainer.py
class CryptoTransformer(nn.Module):
    """
    Transformer-based model for cryptocurrency prediction.
    
    Transformer models with self-attention mechanisms are particularly suited for 
    capturing complex dependencies between GARCH volatility features and price movements.
    The multi-head attention allows the model to focus on different aspects of volatility
    and price patterns simultaneously.
    
    Attributes:
        input_dim (int): Number of input features.
        d_model (int): Dimension of the model.
        n_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        dropout (float): Dropout rate for regularization.
        sequence_length (int): Length of input sequences.
        prediction_type (str): Type of prediction ('price' or 'direction').
    """
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=1, dropout=0.1, sequence_length=10, prediction_type='price'):
        super().__init__()
        self.d_model = d_model if d_model % n_heads == 0 else ((d_model + n_heads - 1) // n_heads) * n_heads
        self.prediction_type = prediction_type
        self.input_embedding = nn.Linear(input_dim, self.d_model)
        self.pos_encoder = nn.Dropout(dropout)  # Simplified positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=n_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(self.d_model, 1)

    def forward(self, x):
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x[:, -1, :])
        return torch.sigmoid(x.squeeze(-1)) if self.prediction_type == 'direction' else x.squeeze(-1) 

class CryptoEnsemble(nn.Module):
    """
    Ensemble model that combines predictions from multiple base models for cryptocurrency prediction.
    
    This model implements several ensemble techniques:
    1. Averaging: Simple average of all model predictions
    2. Weighted Averaging: Weighted average based on model performance
    3. Stacking: Using a meta-model to learn the optimal combination of base models
    
    Attributes:
        base_models (nn.ModuleList): List of base models for the ensemble.
        ensemble_method (str): Method for combining predictions ('average', 'weighted', 'stacking').
        weights (list): Weights for each model if using weighted averaging.
        meta_model (nn.Module): Meta-model for stacking ensemble method.
        input_dim (int): Input dimension for base models.
        output_dim (int): Output dimension for base models.
    """
    def __init__(self, input_dim, hidden_dim=64, models_config=None, ensemble_method='average', 
                 weights=None, prediction_type='price'):
        """
        Initialize the ensemble model.
        
        Args:
            input_dim (int): Input dimension for base models
            hidden_dim (int): Hidden dimension for base models
            models_config (list): List of model configs to use in ensemble
                                  [{'type': 'lstm', 'layers': 2, 'dropout': 0.1}, ...]
            ensemble_method (str): Method for combining predictions ('average', 'weighted', 'stacking')
            weights (list): Weights for each model if using weighted averaging
            prediction_type (str): Type of prediction ('price' or 'direction')
        """
        super().__init__()
        self.input_dim = input_dim
        self.ensemble_method = ensemble_method
        self.prediction_type = prediction_type
        self.output_dim = 1
        
        # Import SimpleLSTM from trainer module
        from trainer import SimpleLSTM
        
        # Default configuration if none provided
        if models_config is None:
            models_config = [
                {'type': 'lstm', 'layers': 1, 'dropout': 0.1},  # Simple LSTM
                {'type': 'gru', 'layers': 2, 'dropout': 0.2},   # GRU
                {'type': 'transformer', 'layers': 1, 'heads': 4, 'dropout': 0.1}  # Transformer
            ]
            
        # Initialize base models
        self.base_models = nn.ModuleList()
        for config in models_config:
            if config['type'].lower() == 'lstm':
                model = SimpleLSTM(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=config.get('layers', 1),
                    dropout=config.get('dropout', 0.1),
                    output_dim=self.output_dim,
                    is_direction=prediction_type == 'direction'
                )
            elif config['type'].lower() == 'gru':
                model = CryptoGRU(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    num_layers=config.get('layers', 1),
                    dropout=config.get('dropout', 0.1),
                    output_dim=self.output_dim,
                    prediction_type=prediction_type
                )
            elif config['type'].lower() == 'transformer':
                model = CryptoTransformer(
                    input_dim=input_dim,
                    d_model=hidden_dim,
                    n_heads=config.get('heads', 4),
                    num_layers=config.get('layers', 1),
                    dropout=config.get('dropout', 0.1),
                    prediction_type=prediction_type
                )
            else:
                raise ValueError(f"Unknown model type: {config['type']}")
                
            self.base_models.append(model)
            
        # Set up ensemble weights (for weighted averaging)
        if weights is not None and len(weights) == len(self.base_models):
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float), requires_grad=False)
        else:
            # Initialize with equal weights
            self.weights = nn.Parameter(torch.ones(len(self.base_models)) / len(self.base_models), requires_grad=False)
            
        # Initialize meta-model for stacking ensemble
        if ensemble_method == 'stacking':
            self.meta_model = nn.Sequential(
                nn.Linear(len(self.base_models), 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            
        self.is_trained = False
            
    def forward(self, x):
        """
        Forward pass through the ensemble model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Predictions
        """
        batch_size = x.size(0)
        
        # Get predictions from each base model
        base_predictions = []
        for model in self.base_models:
            try:
                # Get predictions from the base model
                pred = model(x)
                
                # Reshape if needed
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)
                
                base_predictions.append(pred.view(batch_size, 1))
            except Exception as e:
                print(f"Error in base model prediction: {str(e)}")
                # Add zeros as fallback
                base_predictions.append(torch.zeros(batch_size, 1).to(x.device))
        
        # Stack predictions along the second dimension
        all_preds = torch.cat(base_predictions, dim=1)  # Shape: (batch_size, num_models)
        
        # Combine predictions according to ensemble method
        if self.ensemble_method == 'average':
            # Simple average
            final_preds = torch.mean(all_preds, dim=1)
        
        elif self.ensemble_method == 'weighted':
            # Weighted average
            weights = self.weights.to(x.device)
            final_preds = torch.matmul(all_preds, weights)
        
        elif self.ensemble_method == 'stacking':
            # Meta-model stacking
            if self.is_trained:
                final_preds = self.meta_model(all_preds).squeeze(1)
            else:
                # During initial training, use simple average
                final_preds = torch.mean(all_preds, dim=1)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        return final_preds
        
    def update_weights(self, val_performances):
        """
        Update ensemble weights based on validation performance.
        
        Args:
            val_performances (list): Validation performance metrics for each model (lower is better)
        """
        if len(val_performances) != len(self.base_models):
            raise ValueError("Number of performance metrics must match number of models")
            
        # Convert to inverse performance (higher is better)
        performances = np.array(val_performances)
        if np.all(performances > 0):  # Ensure all values are positive
            inv_performances = 1.0 / performances
        else:
            # Add a small epsilon to avoid division by zero
            epsilon = 1e-10
            inv_performances = 1.0 / (performances + epsilon)
            
        # Normalize to sum to 1
        weights = inv_performances / inv_performances.sum()
        
        # Update weights
        self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float), requires_grad=False)
        
    def freeze_base_models(self):
        """Freeze the parameters of all base models"""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False
                
    def unfreeze_base_models(self):
        """Unfreeze the parameters of all base models"""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = True

class FeatureEnsemble(nn.Module):
    """
    Feature Ensemble model that combines predictions from models trained on different feature sets.
    
    This model tests different technical indicators by training base models on different feature subsets.
    Each base model focuses on a specific set of technical indicators:
    1. Price and GARCH features only
    2. Price and Bollinger Bands features only
    3. Price and MACD features only
    4. Price and Moving Average features only
    5. All features combined (optional)
    
    Attributes:
        base_models (nn.ModuleList): List of base models for the ensemble.
        feature_sets (list): List of feature set names for each base model.
        ensemble_method (str): Method for combining predictions ('average', 'weighted', 'stacking').
        weights (list): Weights for each model if using weighted averaging.
        meta_model (nn.Module): Meta-model for stacking ensemble method.
        input_dim (int): Full input dimension for base models.
        feature_masks (list): Binary masks for selecting features for each base model.
        output_dim (int): Output dimension for base models.
    """
    def __init__(self, input_dim, hidden_dim=64, model_type='lstm', ensemble_method='average', 
                 weights=None, prediction_type='price', use_full_feature_model=True):
        """
        Initialize the feature ensemble model.
        
        Args:
            input_dim (int): Full input dimension including all features
            hidden_dim (int): Hidden dimension for base models
            model_type (str): Base model architecture ('lstm', 'gru', or 'transformer')
            ensemble_method (str): Method for combining predictions ('average', 'weighted', 'stacking')
            weights (list): Weights for each model if using weighted averaging
            prediction_type (str): Type of prediction ('price' or 'direction')
            use_full_feature_model (bool): Whether to include a model trained on all features
        """
        super().__init__()
        self.input_dim = input_dim
        self.ensemble_method = ensemble_method
        self.prediction_type = prediction_type
        self.output_dim = 1
        self.model_type = model_type
        
        # Import SimpleLSTM from trainer module
        from trainer import SimpleLSTM
        
        # Define feature sets and approximate masks
        # These will be properly set by the trainer based on actual features
        self.feature_sets = [
            "Price + GARCH",
            "Price + Bollinger",
            "Price + MACD",
            "Price + Moving Avg"
        ]
        
        if use_full_feature_model:
            self.feature_sets.append("All Features")
        
        # These masks will be properly set by the trainer
        self.feature_masks = [None] * len(self.feature_sets)
        
        # Initialize base models - one for each feature set
        self.base_models = nn.ModuleList()
        for _ in range(len(self.feature_sets)):
            if model_type.lower() == 'lstm':
                model = SimpleLSTM(
                    input_dim=input_dim,  # Will be adjusted in forward pass
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=0.1,
                    output_dim=self.output_dim,
                    is_direction=prediction_type == 'direction'
                )
            elif model_type.lower() == 'gru':
                model = CryptoGRU(
                    input_dim=input_dim,  # Will be adjusted in forward pass
                    hidden_dim=hidden_dim,
                    num_layers=1,
                    dropout=0.1,
                    output_dim=self.output_dim,
                    prediction_type=prediction_type
                )
            elif model_type.lower() == 'transformer':
                model = CryptoTransformer(
                    input_dim=input_dim,  # Will be adjusted in forward pass
                    d_model=hidden_dim,
                    n_heads=4,
                    num_layers=1,
                    dropout=0.1,
                    prediction_type=prediction_type
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
            self.base_models.append(model)
            
        # Set up ensemble weights (for weighted averaging)
        if weights is not None and len(weights) == len(self.base_models):
            self.weights = nn.Parameter(torch.tensor(weights, dtype=torch.float), requires_grad=False)
        else:
            # Initialize with equal weights
            self.weights = nn.Parameter(torch.ones(len(self.base_models)) / len(self.base_models), requires_grad=False)
            
        # Initialize meta-model for stacking ensemble
        if ensemble_method == 'stacking':
            self.meta_model = nn.Sequential(
                nn.Linear(len(self.base_models), 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            
        self.is_trained = False
    
    def set_feature_masks(self, feature_masks):
        """
        Set feature masks for each base model.
        
        Args:
            feature_masks (list): List of binary masks for selecting features for each base model
        """
        if len(feature_masks) != len(self.base_models):
            raise ValueError(f"Expected {len(self.base_models)} feature masks, got {len(feature_masks)}")
        
        self.feature_masks = feature_masks
    
    def forward(self, x):
        """
        Forward pass through the feature ensemble model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Predictions
        """
        batch_size = x.size(0)
        
        # Get predictions from each base model using appropriate feature mask
        base_predictions = []
        for i, model in enumerate(self.base_models):
            try:
                # Apply feature mask if available
                if self.feature_masks[i] is not None:
                    # Create a version of x with only the relevant features
                    # We use broadcasting to apply the mask to all samples and all time steps
                    mask = self.feature_masks[i].to(x.device)
                    # Expand mask to match batch and time dimensions
                    expanded_mask = mask.expand(x.shape[0], x.shape[1], -1)
                    # Apply mask - keeping only the relevant features
                    masked_x = x * expanded_mask
                    
                    # To avoid issues with zero features (which might cause problems in some models),
                    # we will replace zeros with a small value where the mask is zero
                    # This way the model still sees all input dimensions, but irrelevant features are near zero
                    small_value = 1e-8
                    inverted_mask = 1.0 - expanded_mask
                    masked_x = masked_x + (small_value * inverted_mask)
                    
                    # Get predictions from the base model
                    pred = model(masked_x)
                else:
                    # Use all features
                    pred = model(x)
                
                # Reshape if needed
                if len(pred.shape) > 1 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)
                
                base_predictions.append(pred.view(batch_size, 1))
            except Exception as e:
                print(f"Error in feature model prediction: {str(e)}")
                # Add zeros as fallback
                base_predictions.append(torch.zeros(batch_size, 1).to(x.device))
        
        # Stack predictions along the second dimension
        all_preds = torch.cat(base_predictions, dim=1)  # Shape: (batch_size, num_models)
        
        # Combine predictions according to ensemble method
        if self.ensemble_method == 'average':
            # Simple average
            final_preds = torch.mean(all_preds, dim=1)
        
        elif self.ensemble_method == 'weighted':
            # Weighted average
            weights = self.weights.to(x.device)
            # Normalize weights to sum to 1
            weights = weights / weights.sum()
            # Apply weights to each model's predictions
            weighted_preds = all_preds * weights.view(1, -1)
            final_preds = torch.sum(weighted_preds, dim=1)
        
        elif self.ensemble_method == 'stacking':
            # Use a meta-model to combine predictions
            final_preds = self.meta_model(all_preds).squeeze(-1)
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
            
        return final_preds
        
    def update_weights(self, val_performances):
        """
        Update weights based on validation performance.
        
        Args:
            val_performances (list): List of validation performance metrics (lower is better)
        """
        if len(val_performances) != len(self.base_models):
            raise ValueError(f"Expected {len(self.base_models)} performance values, got {len(val_performances)}")
        
        # Invert performances since lower loss is better
        # Add a small constant to avoid division by zero
        epsilon = 1e-8
        inverted_perfs = [1.0 / (perf + epsilon) for perf in val_performances]
        
        # Normalize to get weights that sum to 1
        total = sum(inverted_perfs)
        new_weights = [perf / total for perf in inverted_perfs]
        
        # Update the weights parameter
        self.weights.data = torch.tensor(new_weights, dtype=torch.float)
        
    def freeze_base_models(self):
        """Freeze parameters of all base models for training the meta-model."""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False
                
    def unfreeze_base_models(self):
        """Unfreeze parameters of all base models."""
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = True 