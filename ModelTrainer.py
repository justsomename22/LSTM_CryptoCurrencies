import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from copy import deepcopy
from ta.trend import SMAIndicator  # Importing the SMAIndicator
import optuna  # Import Optuna

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)  # Adjust for bidirectional
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, lstm_output):
        energy = torch.tanh(self.attn(lstm_output))
        attention_weights = torch.softmax(torch.matmul(energy, self.v), dim=1)
        context = torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)
        return context

class CryptoLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Double hidden_size for bidirectional
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Adjust for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # Adjust for bidirectional
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        out = self.attention(out)  # Apply attention
        out = self.fc(out)
        out = self.relu(out)
        return out

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience 
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_model = None
    
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        self.best_model = deepcopy(model.state_dict())
        self.val_loss_min = val_loss

class CryptoTrainer:
    def __init__(self, data_path, sequence_length=10, batch_size=32, epochs=50, 
                 validation_split=0.1, patience=10):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scalers = {}  # Store scaler for each cryptocurrency
        self.models = {}   # Store trained model for each cryptocurrency
        
    def prepare_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.data_path)
        
        # Interpolate missing values in 'price' and 'volume' columns
        df[['price', 'volume']] = df[['price', 'volume']].interpolate(method='linear')
        
        # Group by crypto and process each separately
        self.data_by_crypto = {}
        for crypto_id in df['crypto_id'].unique():
            crypto_df = df[df['crypto_id'] == crypto_id].sort_values('date')
            
            # Calculate the 7-day Simple Moving Average
            crypto_df['sma_7'] = SMAIndicator(crypto_df['price'], window=7).sma_indicator()
            crypto_df = crypto_df.dropna()  # Drop rows with NaN values after SMA calculation
            
            # Features: price, volume, and SMA
            features = crypto_df[['price', 'volume', 'sma_7']].values
            
            # Create a scaler for this crypto and store it
            self.scalers[crypto_id] = MinMaxScaler()
            scaled_features = self.scalers[crypto_id].fit_transform(features)
            
            # Create sequences
            X, y = self.create_sequences(scaled_features)
            self.data_by_crypto[crypto_id] = (X, y, crypto_df)
            
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])  # Predicting price
        return np.array(X), np.array(y)
    
    def objective(self, trial):
        # Hyperparameters to tune
        hidden_size = trial.suggest_int("hidden_size", 32, 128, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)

        # Initialize model
        model = CryptoLSTM(input_size=3, hidden_size=hidden_size, num_layers=num_layers).to(self.device)
        criterion = nn.MSELoss()

        # Prepare data for training
        for crypto_id, (X, y, crypto_df) in self.data_by_crypto.items():
            # Split data into train, validation, and test sets
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=self.validation_split/(1-0.2), random_state=42
            )
            
            # Convert to PyTorch tensors
            X_train = torch.FloatTensor(X_train).to(self.device)
            y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
            X_val = torch.FloatTensor(X_val).to(self.device)
            y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            # Training loop (simplified for the objective function)
            for epoch in range(self.epochs):
                model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)

                # Validation phase
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)

                # Report the intermediate result to Optuna
                trial.report(val_loss, epoch)

                # Handle pruning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return val_loss  # Return the final validation loss

    def tune_hyperparameters(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=50)  # Number of trials can be adjusted
        print("Best hyperparameters: ", study.best_params)

    def train_model(self, verbosity=1):
        # Training loop for each cryptocurrency
        for crypto_id, (X, y, crypto_df) in self.data_by_crypto.items():
            if verbosity > 0:
                print(f"Training model for {crypto_id}")
            
            # Initialize KFold
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
            # Store best metrics for this cryptocurrency
            best_metrics = {
                'fold': None,
                'val_loss': float('inf'),
                'MSE': None,
                'RMSE': None,
                'MAE': None,
                'MAPE': None,
                'R2': None
            }
            
            # K-Fold Cross-Validation
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                if verbosity > 0 and fold == 0:  # Print only for the first fold
                    print(f"Starting K-Fold Cross-Validation for {crypto_id} - Fold {fold + 1}/{kfold.n_splits}")
                
                # Split data into train and validation sets
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Convert to PyTorch tensors
                X_train = torch.FloatTensor(X_train).to(self.device)
                y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
                X_val = torch.FloatTensor(X_val).to(self.device)
                y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
                
                # Create data loaders
                train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=self.batch_size, shuffle=True
                )
                
                val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset, batch_size=self.batch_size, shuffle=False
                )
                
                # Initialize model
                model = CryptoLSTM().to(self.device)
                criterion = nn.MSELoss()
                
                # Initialize optimizer with weight decay
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                
                # Initialize learning rate scheduler
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

                # Initialize early stopping
                early_stopping = EarlyStopping(patience=self.patience)
                
                # Training
                history = {'train_loss': [], 'val_loss': []}
                for epoch in range(self.epochs):
                    # Training phase
                    model.train()
                    train_loss = 0
                    for batch_X, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        train_loss += loss.item()
                    
                    train_loss = train_loss / len(train_loader)
                    history['train_loss'].append(train_loss)
                    
                    # Validation phase
                    model.eval()
                    val_loss = 0
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)
                            val_loss += loss.item()
                    
                    val_loss = val_loss / len(val_loader)
                    history['val_loss'].append(val_loss)

                    # Print loss every few epochs based on verbosity
                    if verbosity > 0 and (epoch + 1) % 5 == 0:
                        print(f"Epoch {epoch + 1}/{self.epochs}, "
                              f"Train Loss: {train_loss:.4f}, "
                              f"Val Loss: {val_loss:.4f}")
                    
                    # Step the scheduler
                    scheduler.step(val_loss)

                    # Early stopping
                    early_stopping(val_loss, model)
                    if early_stopping.early_stop:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break
                
                # Load the best model
                if early_stopping.best_model is not None:
                    model.load_state_dict(early_stopping.best_model)
                
                # Save the model for the current fold
                self.models[f"{crypto_id}_fold{fold + 1}"] = model
                torch.save(model.state_dict(), f"{crypto_id}_model_fold{fold + 1}.pth")
                if verbosity > 0:
                    print(f"Saved model for {crypto_id} - Fold {fold + 1}")
                
                # Evaluate on validation set
                model.eval()
                with torch.no_grad():
                    val_pred = model(X_val)
                    val_loss = criterion(val_pred, y_val)
                    
                    # Calculate evaluation metrics using calculate_metrics
                    metrics = self.calculate_metrics(y_val.cpu().numpy(), val_pred.cpu().numpy())
                    if verbosity > 0:
                        print(f"Validation Loss for {crypto_id} - Fold {fold + 1}: {val_loss.item():.4f}, "
                              f"MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, "
                              f"MAE: {metrics['MAE']:.4f}, MAPE: {metrics['MAPE']:.4f}, "
                              f"R-squared: {metrics['R2']:.4f}")

                    # Check if this fold has the best validation loss
                    if val_loss < best_metrics['val_loss']:
                        best_metrics['fold'] = fold + 1
                        best_metrics['val_loss'] = val_loss
                        best_metrics['MSE'] = metrics['MSE']
                        best_metrics['RMSE'] = metrics['RMSE']
                        best_metrics['MAE'] = metrics['MAE']
                        best_metrics['MAPE'] = metrics['MAPE']
                        best_metrics['R2'] = metrics['R2']

            # Print best metrics for this cryptocurrency after all folds
            print(f"Best metrics for {crypto_id} - Fold {best_metrics['fold']}: "
                  f"Val Loss: {best_metrics['val_loss']:.4f}, "
                  f"MSE: {best_metrics['MSE']:.4f}, "
                  f"RMSE: {best_metrics['RMSE']:.4f}, "
                  f"MAE: {best_metrics['MAE']:.4f}, "
                  f"MAPE: {best_metrics['MAPE']:.4f}, "
                  f"R-squared: {best_metrics['R2']:.4f}")

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate evaluation metrics.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE in percentage
        r2 = self.r_squared(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2
        }

    def plot_results(self, model, X_test, y_test, crypto_id, history):
        model.eval()
        with torch.no_grad():
            # Get predictions
            predictions = model(X_test).cpu().numpy()
            
            # Convert predictions and actual values back to original scale
            pred_array = np.zeros((len(predictions), 3))  # Change to 3 columns
            pred_array[:, 0] = predictions.flatten()  # Set the first column to predictions
            pred_array[:, 1] = 0  # Placeholder for volume
            pred_array[:, 2] = 0  # Placeholder for SMA
            inverse_predictions = self.scalers[crypto_id].inverse_transform(pred_array)[:, 0]
            
            actual_array = np.zeros((len(y_test), 3))  # Change to 3 columns
            actual_array[:, 0] = y_test.cpu().numpy().flatten()  # Set the first column to actual values
            actual_array[:, 1] = 0  # Placeholder for volume
            actual_array[:, 2] = 0  # Placeholder for SMA
            inverse_actuals = self.scalers[crypto_id].inverse_transform(actual_array)[:, 0]
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot training and validation loss
            ax1.plot(history['train_loss'], label='Training Loss')
            ax1.plot(history['val_loss'], label='Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Training and Validation Loss for {crypto_id}')
            ax1.legend()
            ax1.grid(True)
            
            # Plot predictions vs actual values
            ax2.plot(inverse_actuals, label='Actual Prices')
            ax2.plot(inverse_predictions, label='Predicted Prices')
            ax2.set_xlabel('Time Steps')
            ax2.set_ylabel('Price')
            ax2.set_title(f'Actual vs Predicted Prices for {crypto_id}')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{crypto_id}_results.png")
            plt.close()
            
    def predict(self, crypto_id, new_data):
        """
        Make predictions on new data for a specific cryptocurrency
        
        Args:
            crypto_id: ID of the cryptocurrency
            new_data: DataFrame with 'price' and 'volume' columns (at least sequence_length rows)
        
        Returns:
            Predicted price for the next time step
        """
        if crypto_id not in self.models:
            raise ValueError(f"No trained model found for {crypto_id}")
        
        # Extract features
        features = new_data[['price', 'volume']].values[-self.sequence_length:]
        
        # Scale the features
        scaled_features = self.scalers[crypto_id].transform(features)
        
        # Convert to tensor
        X = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.models[crypto_id].eval()
        with torch.no_grad():
            prediction = self.models[crypto_id](X).cpu().numpy().flatten()[0]
        
        # Convert back to original scale
        pred_array = np.zeros((1, 2))
        pred_array[0, 0] = prediction
        inverse_prediction = self.scalers[crypto_id].inverse_transform(pred_array)[0, 0]
        
        return inverse_prediction

    def r_squared(self, y_true, y_pred):
        """
        Calculate the R-squared (coefficient of determination) regression score function.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            R-squared value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)  # Residual sum of squares
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Total sum of squares
        return 1 - (ss_res / ss_tot)  # R-squared

    def predict_with_uncertainty(self, crypto_id, new_data, n_samples=100):
        model = self.models[crypto_id]
        model.train()  # Activate dropout during inference
        predictions = []
        for _ in range(n_samples):
            pred = model(torch.FloatTensor(self.scalers[crypto_id].transform(new_data)).unsqueeze(0).to(self.device))
            predictions.append(pred.cpu().numpy())
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        return mean_pred, std_pred

# Example usage
if __name__ == "__main__":
    trainer = CryptoTrainer(
        data_path="cryptocurrency_data.csv",
        sequence_length=10,
        batch_size=32,
        epochs=10,
        validation_split=0.1,
        patience=15
    )
    
    # Prepare data
    trainer.prepare_data()
    
    # Tune hyperparameters
    trainer.tune_hyperparameters()
    
    # Train models with the best hyperparameters
    trainer.train_model()
    
    # Example of making predictions with new data
    # Assume we have new data for a specific cryptocurrency
    new_data = pd.DataFrame({
        'price': [100, 102, 105, 103, 104, 106, 107, 108, 110, 112],
        'volume': [1000, 1200, 900, 1100, 1300, 1100, 1000, 1200, 1400, 1500]
    })
    
    # Predict the next price
    crypto_id = "BTC"  # Example cryptocurrency ID
    if crypto_id in trainer.models:
        next_price = trainer.predict(crypto_id, new_data)
        print(f"Predicted next price for {crypto_id}: ${next_price:.2f}")