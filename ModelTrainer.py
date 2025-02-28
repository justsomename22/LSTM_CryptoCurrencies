#ModelTrainer.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from copy import deepcopy
from ta.trend import SMAIndicator
import optuna
from ta.momentum import RSIIndicator

class Attention(nn.Module):
    """
    Attention mechanism for the LSTM model.
    This class computes attention weights and applies them to the LSTM output.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)  # Linear layer for attention
        self.v = nn.Parameter(torch.rand(hidden_size))  # Parameter for attention weights

    def forward(self, lstm_output):
        """
        Forward pass for the attention layer.
        
        Args:
            lstm_output: Output from the LSTM layer.
        
        Returns:
            Weighted sum of the LSTM outputs based on attention weights.
        """
        # Compute energy scores using a tanh activation function
        energy = torch.tanh(self.attn(lstm_output))  
        # Compute attention weights using softmax
        attention_weights = torch.softmax(torch.matmul(energy, self.v), dim=1)  
        # Apply attention weights to the LSTM outputs and return the result
        return torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)  

class CryptoLSTM(nn.Module):
    """
    LSTM model for cryptocurrency price prediction.
    This model includes an LSTM layer followed by an attention mechanism and a fully connected layer.
    """
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout, bidirectional=True)  # Bidirectional LSTM
        self.attention = Attention(hidden_size)  # Attention layer
        self.fc = nn.Linear(hidden_size * 2, 1)  # Fully connected layer for output
        self.relu = nn.ReLU()  # ReLU activation function
    
    def forward(self, x):
        """
        Forward pass for the LSTM model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size).
        
        Returns:
            Output of the model after applying LSTM, attention, and activation.
        """
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)  # Initialize hidden state
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)  # Initialize cell state
        out, _ = self.lstm(x, (h0, c0))  # LSTM forward pass
        out = self.attention(out)  # Apply attention
        return self.fc(out)  # Removed self.relu here

class EarlyStopping:
    """
    Early stopping utility to prevent overfitting during training.
    This class monitors validation loss and stops training if it doesn't improve for a specified number of epochs.
    """
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience  # Number of epochs to wait for improvement
        self.min_delta = min_delta  # Minimum change to qualify as an improvement
        self.counter = 0  # Counter for epochs without improvement
        self.best_score = None  # Best score observed
        self.early_stop = False  # Flag to indicate if training should stop
        self.val_loss_min = np.inf  # Minimum validation loss observed
        self.best_model = None  # Best model state

    def __call__(self, val_loss, model):
        """
        Call method to check if early stopping should be triggered.
        
        Args:
            val_loss: Current validation loss.
            model: Current model to save if it is the best.
        """
        # Invert loss for maximization
        score = -val_loss  
        # Check if the current score is better than the best score
        if self.best_score is None or score >= self.best_score + self.min_delta:
            self.best_score = score  # Update best score
            self.save_checkpoint(val_loss, model)  # Save the model
            self.counter = 0  # Reset counter for epochs without improvement
        else:
            self.counter += 1  # Increment counter for epochs without improvement
            # Check if patience has been exceeded
            if self.counter >= self.patience:
                self.early_stop = True  # Set flag to stop training
    
    def save_checkpoint(self, val_loss, model):
        """
        Save the model state if validation loss has improved.
        
        Args:
            val_loss: Current validation loss.
            model: Current model to save.
        """
        self.best_model = deepcopy(model.state_dict())  # Save model state
        self.val_loss_min = val_loss  # Update minimum validation loss

class CryptoTrainer:
    """
    Trainer class for the cryptocurrency LSTM model.
    This class handles data preparation, model training, hyperparameter tuning, and evaluation.
    """
    def __init__(self, data_path, sequence_length=10, batch_size=32, epochs=50, 
                 validation_split=0.1, patience=7):
        self.data_path = data_path  # Path to the dataset
        self.sequence_length = sequence_length  # Length of input sequences
        self.batch_size = batch_size  # Batch size for training
        self.epochs = epochs  # Number of training epochs
        self.validation_split = validation_split  # Fraction of data for validation
        self.patience = patience  # Patience for early stopping
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.scalers = {}  # Dictionary to store scalers for each cryptocurrency
        self.models = {}  # Dictionary to store models
        self.best_models = {}  # To store best model per crypto
        
    def prepare_data(self):
        """
        Load and preprocess the cryptocurrency data.
        This method reads the dataset, computes technical indicators, and scales the features.
        """
        try:
            df = pd.read_csv(self.data_path).sort_values('date')
        except Exception as e:
            raise ValueError(f"Error reading the data file: {e}")

        # Interpolate missing values for price and volume
        df[['price', 'volume']] = df[['price', 'volume']].interpolate(method='linear')
        self.data_by_crypto = {}
        
        for crypto_id in df['crypto_id'].unique():
            crypto_df = df[df['crypto_id'] == crypto_id].copy()
            
            # Feature Engineering
            crypto_df['sma_7'] = SMAIndicator(crypto_df['price'], window=7).sma_indicator()
            crypto_df['price_diff'] = crypto_df['price'].diff()  # Price change
            crypto_df['volatility'] = crypto_df['price'].rolling(window=7).std()  # 7-day volatility
            crypto_df['rsi'] = RSIIndicator(crypto_df['price'], window=14).rsi()  # 14-day RSI
            crypto_df = crypto_df.dropna()  # Drop rows with NaN values
            
            # Select features for modeling
            features = crypto_df[['price', 'volume', 'sma_7', 'price_diff', 'volatility', 'rsi']].values
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Create sequences for training
            X, y = self.create_sequences(scaled_features)
            self.data_by_crypto[crypto_id] = (X, y, crypto_df)
            self.scalers[crypto_id] = scaler
    
    def create_sequences(self, data):
        """
        Create input-output sequences from the scaled data.
        
        Args:
            data: Scaled feature data.
        
        Returns:
            Tuple of input sequences (X) and output values (y).
        """
        X, y = [], []
        # Loop through the data to create sequences
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])  # Append input sequence
            y.append(data[i + self.sequence_length, 0])  # Append output value (price)
        return np.array(X), np.array(y)  # Return sequences as numpy arrays
    
    def create_dataloader(self, X, y, shuffle=True):
        """
        Create a DataLoader for the training or validation data.
        
        Args:
            X: Input features.
            y: Output values.
            shuffle: Whether to shuffle the data.
        
        Returns:
            DataLoader for the dataset.
        """
        tensor_X = torch.FloatTensor(X).to(self.device)  # Convert to tensor and move to device
        tensor_y = torch.FloatTensor(y).unsqueeze(1).to(self.device)  # Convert to tensor and move to device
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)  # Create TensorDataset
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)  # Return DataLoader
    
    def objective(self, trial):
        """
        Objective function for hyperparameter tuning using Optuna.
        
        Args:
            trial: Optuna trial object.
        
        Returns:
            Validation loss for the current trial.
        """
        hidden_size = trial.suggest_int("hidden_size", 32, 128, step=32)  # Suggest hidden size
        num_layers = trial.suggest_int("num_layers", 1, 3)  # Suggest number of layers
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)  # Suggest learning rate
        
        model = CryptoLSTM(input_size=6, hidden_size=hidden_size, 
                         num_layers=num_layers).to(self.device)  # Initialize model
        criterion = nn.MSELoss()  # Loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimizer
        
        for crypto_id, (X, y, _) in self.data_by_crypto.items():
            # Split data into training and testing sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=self.validation_split/(1-0.2), random_state=42
            )
            
            train_loader = self.create_dataloader(X_train, y_train)  # Create training DataLoader
            val_loader = self.create_dataloader(X_val, y_val, shuffle=False)  # Create validation DataLoader
            
            for epoch in range(self.epochs):
                model.train()  # Set model to training mode
                train_loss = self._train_epoch(model, train_loader, criterion, optimizer)  # Train for one epoch
                val_loss = self._eval_epoch(model, val_loader, criterion)  # Evaluate on validation set
                trial.report(val_loss, epoch)  # Report validation loss to Optuna
                if trial.should_prune():  # Check if trial should be pruned
                    raise optuna.exceptions.TrialPruned()
        
        return val_loss  # Return final validation loss

    def _train_epoch(self, model, loader, criterion, optimizer):
        """
        Train the model for one epoch.
        
        Args:
            model: The model to train.
            loader: DataLoader for the training data.
            criterion: Loss function.
            optimizer: Optimizer.
        
        Returns:
            Average training loss for the epoch.
        """
        model.train()  # Set model to training mode
        total_loss = 0  # Initialize total loss
        for batch_X, batch_y in loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_X)  # Forward pass
            loss = criterion(outputs, batch_y)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            total_loss += loss.item()  # Accumulate loss
        return total_loss / len(loader)  # Return average loss

    def _eval_epoch(self, model, loader, criterion):
        """
        Evaluate the model on the validation set.
        
        Args:
            model: The model to evaluate.
            loader: DataLoader for the validation data.
            criterion: Loss function.
        
        Returns:
            Average validation loss for the epoch.
        """
        model.eval()  # Set model to evaluation mode
        total_loss = 0  # Initialize total loss
        with torch.no_grad():  # Disable gradient calculation
            for batch_X, batch_y in loader:
                outputs = model(batch_X)  # Forward pass
                total_loss += criterion(outputs, batch_y).item()  # Accumulate loss
        return total_loss / len(loader)  # Return average loss
    
    def tune_hyperparameters(self, n_trials=50):
        """
        Tune hyperparameters using Optuna.
        
        Args:
            n_trials: Number of trials for hyperparameter tuning.
        
        Returns:
            Best hyperparameters found during tuning.
        """
        study = optuna.create_study(direction="minimize")  # Create Optuna study
        study.optimize(self.objective, n_trials=n_trials)  # Optimize the objective function
        print("Best hyperparameters:", study.best_params)  # Print best hyperparameters
        return study.best_params  # Return best hyperparameters

    def train_model(self, n_folds=5):
        """
        Train the model using K-Fold cross-validation.
        
        Args:
            n_folds: Number of folds for cross-validation.
        """
        all_metrics = {}  # Dictionary to store metrics for all models and folds
        training_history = {}  # Store training history for each model/fold
        
        # Loop through each cryptocurrency's data
        for crypto_id, (X, y, _) in self.data_by_crypto.items():
            print(f"Training {crypto_id}")
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)  # Initialize K-Fold
            
            best_metrics = {'val_loss': float('inf')}  # Initialize best metrics
            
            # Loop through each fold
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                train_loader = self.create_dataloader(X[train_idx], y[train_idx])  # Create training DataLoader
                val_loader = self.create_dataloader(X[val_idx], y[val_idx], shuffle=False)  # Create validation DataLoader
                
                model, metrics, history = self._train_fold(train_loader, val_loader, crypto_id)  # Train the model for this fold
                self.models[f"{crypto_id}_fold{fold + 1}"] = model  # Store model
                training_history[f"{crypto_id}_fold{fold + 1}"] = history  # Store training history
                torch.save(model.state_dict(), f"{crypto_id}_model_fold{fold + 1}.pth")  # Save model state
                
                # Store metrics for this fold
                all_metrics[f"{crypto_id}_fold{fold + 1}"] = metrics
                
                # Update best metrics if current fold's metrics are better
                if metrics['val_loss'] < best_metrics['val_loss']:
                    best_metrics = {**metrics, 'fold': fold + 1}  # Update best metrics
                    self.best_models[crypto_id] = (model, fold + 1, history)  # Store model, fold, and history
            
            print(f"Best fold for {crypto_id}: {best_metrics['fold']} (Val Loss: {best_metrics['val_loss']:.4f})")
        
        # Print all metrics for each model/fold
        print("\nAll metrics for each model/fold:")
        for key, metrics in all_metrics.items():
            print(f"{key}: {metrics}")

        # Save plots for the best models
        self.save_best_model_plots(training_history)

    def save_best_model_plots(self, training_history):
        """
        Save plots for the best models of each cryptocurrency.
        
        Args:
            training_history: Dictionary containing training history for each model/fold.
        """
        for crypto_id, (model, fold, history) in self.best_models.items():
            # Generate a plot for the best model using actual training history
            plt.figure(figsize=(10, 5))
            plt.title(f"Best Model for {crypto_id} - Fold {fold}")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            
            # Get actual training history
            epochs = np.arange(1, len(history['train_loss']) + 1)
            train_losses = history['train_loss']
            val_losses = history['val_loss']
            
            plt.plot(epochs, train_losses, label='Training Loss', color='blue')
            plt.plot(epochs, val_losses, label='Validation Loss', color='red')
            plt.legend()
            plt.grid()
            plt.savefig(f"{crypto_id}_best_model_fold_{fold}.png")  # Save the plot as a PNG file
            plt.close()  # Close the plot to free memory
            print(f"Saved plot for {crypto_id} - Fold {fold} as '{crypto_id}_best_model_fold_{fold}.png'")

    def plot_actual_vs_predicted(self, X_val, y_true, y_pred, crypto_id):
        """
        Plot actual vs predicted prices for the given cryptocurrency.
        
        Args:
            X_val: Validation input features.
            y_true: True output values.
            y_pred: Predicted output values.
            crypto_id: Cryptocurrency ID for labeling the plot.
        """
        last_prices_scaled = X_val[:, -1, 0]  # Get last scaled prices
        feature_dim = X_val.shape[2]  # Get the number of features
        dummy_features = np.zeros((len(last_prices_scaled), feature_dim))  # Create dummy features for inverse transformation
        dummy_features[:, 0] = last_prices_scaled  # Set last prices in dummy features
        
        # Inverse transform to get actual last prices
        last_prices = self.scalers[crypto_id].inverse_transform(dummy_features)[:, 0]  
        
        # Ensure y_true and y_pred are flattened and have the correct shape
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Check if the shapes match
        if y_true_flat.shape[0] != y_pred_flat.shape[0]:
            raise ValueError(f"Shape mismatch: y_true has {y_true_flat.shape[0]} elements, y_pred has {y_pred_flat.shape[0]} elements.")
        
        # Create dummy arrays for inverse transformation
        # Ensure the shape of the dummy array matches the expected input for inverse_transform
        actual_prices = self.scalers[crypto_id].inverse_transform(
            np.column_stack((y_true_flat, np.zeros((y_true_flat.shape[0], feature_dim - 1)))))[:, 0]
        predicted_prices = self.scalers[crypto_id].inverse_transform(
            np.column_stack((y_pred_flat, np.zeros((y_pred_flat.shape[0], feature_dim - 1)))))[:, 0]
        
        plt.figure(figsize=(10, 5))
        plt.plot(actual_prices, label='Actual Prices', color='blue', marker='o')  # Plot actual prices
        plt.plot(predicted_prices, label='Predicted Prices', color='red', marker='x')  # Plot predicted prices
        plt.title(f'Actual vs Predicted Prices for {crypto_id}')  # Set plot title
        plt.xlabel('Time Steps')  # Set x-axis label
        plt.ylabel('Price (USD)')  # Set y-axis label
        plt.legend()  # Show legend
        plt.grid()  # Show grid
        plt.savefig(f"{crypto_id}_actual_vs_predicted.png")  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory
        print(f"Saved actual vs predicted plot for {crypto_id} as '{crypto_id}_actual_vs_predicted.png'")

    def _train_fold(self, train_loader, val_loader, crypto_id):
        model = CryptoLSTM().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=self.patience)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.epochs):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._eval_epoch(model, val_loader, criterion)
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                break
        
        if early_stopping.best_model:
            model.load_state_dict(early_stopping.best_model)
        
        X_val, y_val = val_loader.dataset.tensors
        with torch.no_grad():
            val_pred = model(X_val)
            metrics = self.calculate_metrics(y_val.cpu().numpy(), val_pred.cpu().numpy())
            metrics['val_loss'] = early_stopping.val_loss_min
            
            # Pass X_val along with y_val and val_pred to plot absolute prices
            self.plot_actual_vs_predicted(X_val.cpu().numpy(), y_val.cpu().numpy(), val_pred.cpu().numpy(), crypto_id)
        
        return model, metrics, history

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics for the model predictions.
        
        Args:
            y_true: True values.
            y_pred: Predicted values.
        
        Returns:
            Dictionary of calculated metrics.
        """
        mse = np.mean((y_true - y_pred) ** 2)  # Mean Squared Error
        mask = y_true != 0  # Mask for non-zero true values
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('inf')  # Mean Absolute Percentage Error
        return {
            'MSE': mse,
            'RMSE': np.sqrt(mse),  # Root Mean Squared Error
            'MAE': np.mean(np.abs(y_true - y_pred)),  # Mean Absolute Error
            'MAPE': mape,  # Mean Absolute Percentage Error
            'R2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))  # R-squared
        }

    def predict(self, crypto_id, new_data):
        # Compute technical indicators for new_data
        new_data['sma_7'] = SMAIndicator(new_data['price'], window=7).sma_indicator()
        new_data['price_diff'] = new_data['price'].diff()
        new_data['volatility'] = new_data['price'].rolling(window=7).std()
        new_data['rsi'] = RSIIndicator(new_data['price'], window=14).rsi()
        
        # Use the last sequence_length rows and ensure no NaNs
        features = new_data[['price', 'volume', 'sma_7', 'price_diff', 'volatility', 'rsi']].dropna()
        features = features.tail(self.sequence_length).values
        
        scaled_features = self.scalers[crypto_id].transform(features)
        X = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)
        
        model, fold, _ = self.best_models[crypto_id]
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X).cpu().numpy().flatten()[0]
            # Inverse-scale the prediction
            last_scaled_price = scaled_features[-1, 0]  # Last price in scaled form
            dummy_scaled = np.array([[last_scaled_price + pred_scaled, 0, 0, 0, 0, 0]])  # Dummy for inverse transform
            dummy_unscaled = self.scalers[crypto_id].inverse_transform(dummy_scaled)
            pred = dummy_unscaled[0, 0] - new_data['price'].iloc[-1]  # Difference from last price
            last_price = new_data['price'].iloc[-1]
            print(f"Last price: {last_price}, Predicted difference: {pred}")
            return last_price + pred

if __name__ == "__main__":
    # Main execution block to train the model
    trainer = CryptoTrainer("cryptocurrency_data.csv", sequence_length=20, 
                           batch_size=32, epochs=50, validation_split=0.1, patience=15)
    trainer.prepare_data()  # Prepare the data
    trainer.tune_hyperparameters()  # Tune hyperparameters
    trainer.train_model()  # Train the model
    
    print("\nBest models saved:")
    for crypto_id, model_info in trainer.best_models.items():
        # Adjusted structure check for model_info
        if isinstance(model_info, tuple) and len(model_info) == 3:
            model, fold, history = model_info  # Unpack model_info correctly
        else:
            raise ValueError(f"Unexpected model_info structure for {crypto_id}: {model_info}")
        
        print(f"{crypto_id}: Fold {fold}")
    
    crypto_id = "BTC"  # Example cryptocurrency ID
    if crypto_id in trainer.best_models:
        new_data = pd.DataFrame({
            'price': [60000, 60200, 60500, 60300, 60400, 60600, 60700, 60800, 61000, 61200],
            'volume': [1000, 1200, 900, 1100, 1300, 1100, 1000, 1200, 1400, 1500]
        })
        next_price = trainer.predict(crypto_id, new_data)  # Make prediction
        print(f"Predicted next price for {crypto_id}: ${next_price:.2f}")  # Print predicted price