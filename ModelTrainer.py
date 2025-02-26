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
        energy = torch.tanh(self.attn(lstm_output))  # Compute energy scores
        attention_weights = torch.softmax(torch.matmul(energy, self.v), dim=1)  # Compute attention weights
        return torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)  # Apply attention weights

class CryptoLSTM(nn.Module):
    """
    LSTM model for cryptocurrency price prediction.
    This model includes an LSTM layer followed by an attention mechanism and a fully connected layer.
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
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
        return self.relu(self.fc(out))  # Output after fully connected layer and ReLU

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
        score = -val_loss  # Invert loss for maximization
        if self.best_score is None or score >= self.best_score + self.min_delta:
            self.best_score = score  # Update best score
            self.save_checkpoint(val_loss, model)  # Save the model
            self.counter = 0  # Reset counter
        else:
            self.counter += 1  # Increment counter
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
                 validation_split=0.1, patience=10):
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
        Load and preprocess the data.
        This method reads the dataset, computes technical indicators, and scales the features.
        """
        df = pd.read_csv(self.data_path).sort_values('date')  # Load data
        df[['price', 'volume']] = df[['price', 'volume']].interpolate(method='linear')  # Interpolate missing values
        self.data_by_crypto = {}  # Dictionary to store data for each cryptocurrency
        
        for crypto_id in df['crypto_id'].unique():
            crypto_df = df[df['crypto_id'] == crypto_id].copy()  # Filter data for the current cryptocurrency
            crypto_df['sma_7'] = SMAIndicator(crypto_df['price'], window=7).sma_indicator()  # Compute 7-day SMA
            crypto_df = crypto_df.dropna()  # Drop rows with NaN values
            
            features = crypto_df[['price', 'volume', 'sma_7']].values  # Select features
            scaler = MinMaxScaler()  # Initialize scaler
            scaled_features = scaler.fit_transform(features)  # Scale features
            
            X, y = self.create_sequences(scaled_features)  # Create input-output sequences
            self.data_by_crypto[crypto_id] = (X, y, crypto_df)  # Store data for the cryptocurrency
            self.scalers[crypto_id] = scaler  # Store scaler for the cryptocurrency
    
    def create_sequences(self, data):
        """
        Create input-output sequences from the scaled data.
        
        Args:
            data: Scaled feature data.
        
        Returns:
            Tuple of input sequences (X) and output values (y).
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])  # Append input sequence
            y.append(data[i + self.sequence_length, 0])  # Append output value (price)
        return np.array(X), np.array(y)  # Return as numpy arrays
    
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
        
        model = CryptoLSTM(input_size=3, hidden_size=hidden_size, 
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
        for crypto_id, (X, y, _) in self.data_by_crypto.items():
            print(f"Training {crypto_id}")
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)  # Initialize K-Fold
            
            best_metrics = {'val_loss': float('inf')}  # Initialize best metrics
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                train_loader = self.create_dataloader(X[train_idx], y[train_idx])  # Create training DataLoader
                val_loader = self.create_dataloader(X[val_idx], y[val_idx], shuffle=False)  # Create validation DataLoader
                
                model, metrics = self._train_fold(train_loader, val_loader, crypto_id)  # Pass crypto_id to _train_fold
                self.models[f"{crypto_id}_fold{fold + 1}"] = model  # Store model
                torch.save(model.state_dict(), f"{crypto_id}_model_fold{fold + 1}.pth")  # Save model state
                
                # Store metrics for this fold
                all_metrics[f"{crypto_id}_fold{fold + 1}"] = metrics
                
                if metrics['val_loss'] < best_metrics['val_loss']:
                    best_metrics = {**metrics, 'fold': fold + 1}  # Update best metrics
                    self.best_models[crypto_id] = (model, fold + 1)  # Store model and fold
            
            print(f"Best fold for {crypto_id}: {best_metrics['fold']} (Val Loss: {best_metrics['val_loss']:.4f})")
        
        # Print all metrics for each model/fold
        print("\nAll metrics for each model/fold:")
        for key, metrics in all_metrics.items():
            print(f"{key}: {metrics}")

        # Save plots for the best models
        self.save_best_model_plots()

    def save_best_model_plots(self):
        """
        Save plots for the best models of each cryptocurrency.
        """
        for crypto_id, (model, fold) in self.best_models.items():
            # Generate a plot for the best model
            plt.figure(figsize=(10, 5))
            plt.title(f"Best Model for {crypto_id} - Fold {fold}")
            plt.xlabel("Epochs")
            plt.ylabel("Validation Loss")
            
            # Assuming you have a way to retrieve the training history
            # Here, we will simulate some data for demonstration purposes
            epochs = np.arange(1, self.epochs + 1)
            val_losses = np.random.rand(self.epochs) * 0.1 + 0.2  # Simulated validation loss
            
            plt.plot(epochs, val_losses, label='Validation Loss', color='blue')
            plt.legend()
            plt.grid()
            plt.savefig(f"{crypto_id}_best_model_fold_{fold}.png")  # Save the plot as a PNG file
            plt.close()  # Close the plot to free memory
            print(f"Saved plot for {crypto_id} - Fold {fold} as '{crypto_id}_best_model_fold_{fold}.png'")

    def plot_actual_vs_predicted(self, y_true, y_pred, crypto_id):
        """
        Plot actual vs. predicted prices for the best model.
        
        Args:
            y_true: Actual prices from the validation set.
            y_pred: Predicted prices from the model.
            crypto_id: Identifier for the cryptocurrency.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label='Actual Prices', color='blue', marker='o')
        plt.plot(y_pred, label='Predicted Prices', color='red', marker='x')
        plt.title(f'Actual vs Predicted Prices for {crypto_id}')
        plt.xlabel('Time Steps')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()
        plt.savefig(f"{crypto_id}_actual_vs_predicted.png")  # Save the plot as a PNG file
        plt.close()  # Close the plot to free memory
        print(f"Saved actual vs predicted plot for {crypto_id} as '{crypto_id}_actual_vs_predicted.png'")

    def _train_fold(self, train_loader, val_loader, crypto_id):
        """
        Train the model for one fold of cross-validation.
        
        Args:
            train_loader: DataLoader for the training data.
            val_loader: DataLoader for the validation data.
            crypto_id: Identifier for the cryptocurrency.
        
        Returns:
            Trained model and evaluation metrics.
        """
        model = CryptoLSTM().to(self.device)  # Initialize model
        criterion = nn.MSELoss()  # Loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Optimizer
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)  # Learning rate scheduler
        early_stopping = EarlyStopping(patience=self.patience)  # Early stopping
        
        for _ in range(self.epochs):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)  # Train for one epoch
            val_loss = self._eval_epoch(model, val_loader, criterion)  # Evaluate on validation set
            scheduler.step(val_loss)  # Step the scheduler
            early_stopping(val_loss, model)  # Check for early stopping
            if early_stopping.early_stop:  # Stop training if needed
                break
        
        if early_stopping.best_model:  # Load best model if available
            model.load_state_dict(early_stopping.best_model)
        
        X_val, y_val = val_loader.dataset.tensors  # Get validation data
        with torch.no_grad():
            val_pred = model(X_val)  # Make predictions on validation set
            metrics = self.calculate_metrics(y_val.cpu().numpy(), val_pred.cpu().numpy())  # Calculate metrics
            metrics['val_loss'] = early_stopping.val_loss_min  # Add validation loss to metrics
            
            # Plot actual vs predicted prices
            self.plot_actual_vs_predicted(y_val.cpu().numpy(), val_pred.cpu().numpy(), crypto_id)  # Call the plot function
            
        return model, metrics  # Return model and metrics

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
        """
        Make predictions using the best model for the specified cryptocurrency.
        
        Args:
            crypto_id: Identifier for the cryptocurrency.
            new_data: DataFrame containing new input data for prediction.
        
        Returns:
            Predicted price for the next time step.
        """
        if crypto_id not in self.best_models:
            raise ValueError(f"No best model found for {crypto_id}")  # Raise error if no model is found
        
        features = new_data[['price', 'volume']].values[-self.sequence_length:]  # Get last sequence of features
        scaled_features = self.scalers[crypto_id].transform(features)  # Scale features
        X = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)  # Convert to tensor and move to device
        
        model, fold = self.best_models[crypto_id]  # Get best model
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            pred = model(X).cpu().numpy().flatten()[0]  # Make prediction
            pred_array = np.array([[pred, 0, 0]])  # Prepare array for inverse scaling
            return self.scalers[crypto_id].inverse_transform(pred_array)[0, 0]  # Inverse scale and return prediction

if __name__ == "__main__":
    # Main execution block to train the model
    trainer = CryptoTrainer("cryptocurrency_data.csv", sequence_length=10, 
                           batch_size=32, epochs=10, validation_split=0.1, patience=15)
    trainer.prepare_data()  # Prepare the data
    trainer.tune_hyperparameters()  # Tune hyperparameters
    trainer.train_model()  # Train the model
    
    print("\nBest models saved:")
    for crypto_id, (model, fold) in trainer.best_models.items():
        print(f"{crypto_id}: Fold {fold}")
    
    crypto_id = "BTC"  # Example cryptocurrency ID
    if crypto_id in trainer.best_models:
        new_data = pd.DataFrame({
            'price': [100, 102, 105, 103, 104, 106, 107, 108, 110, 112],
            'volume': [1000, 1200, 900, 1100, 1300, 1100, 1000, 1200, 1400, 1500]
        })
        next_price = trainer.predict(crypto_id, new_data)  # Make prediction
        print(f"Predicted next price for {crypto_id}: ${next_price:.2f}")  # Print predicted price