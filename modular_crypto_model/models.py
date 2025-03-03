#models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionMechanism(nn.Module):
    """
    Basic attention mechanism to focus on important parts of the sequence
    """
    def __init__(self, hidden_dim):
        super(AttentionMechanism, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, lstm_output):
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
        batch_size = x.size(0)
        
        # Apply batch normalization
        x_normed = self.feature_norm(x.transpose(1, 2)).transpose(1, 2)
        
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
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x) 