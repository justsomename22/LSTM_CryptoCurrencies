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
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, lstm_output):
        energy = torch.tanh(self.attn(lstm_output))
        attention_weights = torch.softmax(torch.matmul(energy, self.v), dim=1)
        return torch.sum(attention_weights.unsqueeze(-1) * lstm_output, dim=1)

class CryptoLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.attention(out)
        return self.relu(self.fc(out))

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_model = None
    
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score >= self.best_score + self.min_delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
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
        self.scalers = {}
        self.models = {}
        self.best_models = {}  # To store best model per crypto
        
    def prepare_data(self):
        df = pd.read_csv(self.data_path).sort_values('date')
        df[['price', 'volume']] = df[['price', 'volume']].interpolate(method='linear')
        self.data_by_crypto = {}
        
        for crypto_id in df['crypto_id'].unique():
            crypto_df = df[df['crypto_id'] == crypto_id].copy()
            crypto_df['sma_7'] = SMAIndicator(crypto_df['price'], window=7).sma_indicator()
            crypto_df = crypto_df.dropna()
            
            features = crypto_df[['price', 'volume', 'sma_7']].values
            scaler = MinMaxScaler()
            scaled_features = scaler.fit_transform(features)
            
            X, y = self.create_sequences(scaled_features)
            self.data_by_crypto[crypto_id] = (X, y, crypto_df)
            self.scalers[crypto_id] = scaler
    
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y)
    
    def create_dataloader(self, X, y, shuffle=True):
        tensor_X = torch.FloatTensor(X).to(self.device)
        tensor_y = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        dataset = torch.utils.data.TensorDataset(tensor_X, tensor_y)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
    
    def objective(self, trial):
        hidden_size = trial.suggest_int("hidden_size", 32, 128, step=32)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        
        model = CryptoLSTM(input_size=3, hidden_size=hidden_size, 
                         num_layers=num_layers).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        for crypto_id, (X, y, _) in self.data_by_crypto.items():
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=self.validation_split/(1-0.2), random_state=42
            )
            
            train_loader = self.create_dataloader(X_train, y_train)
            val_loader = self.create_dataloader(X_val, y_val, shuffle=False)
            
            for epoch in range(self.epochs):
                model.train()
                train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
                val_loss = self._eval_epoch(model, val_loader, criterion)
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        return val_loss

    def _train_epoch(self, model, loader, criterion, optimizer):
        model.train()
        total_loss = 0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def _eval_epoch(self, model, loader, criterion):
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in loader:
                outputs = model(batch_X)
                total_loss += criterion(outputs, batch_y).item()
        return total_loss / len(loader)
    
    def tune_hyperparameters(self, n_trials=50):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", study.best_params)
        return study.best_params

    def train_model(self, n_folds=5):
        all_metrics = {}  # Dictionary to store metrics for all models and folds
        for crypto_id, (X, y, _) in self.data_by_crypto.items():
            print(f"Training {crypto_id}")
            kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            best_metrics = {'val_loss': float('inf')}
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
                train_loader = self.create_dataloader(X[train_idx], y[train_idx])
                val_loader = self.create_dataloader(X[val_idx], y[val_idx], shuffle=False)
                
                model, metrics = self._train_fold(train_loader, val_loader)
                self.models[f"{crypto_id}_fold{fold + 1}"] = model
                torch.save(model.state_dict(), f"{crypto_id}_model_fold{fold + 1}.pth")
                
                # Store metrics for this fold
                all_metrics[f"{crypto_id}_fold{fold + 1}"] = metrics
                
                if metrics['val_loss'] < best_metrics['val_loss']:
                    best_metrics = {**metrics, 'fold': fold + 1}
                    self.best_models[crypto_id] = (model, fold + 1)  # Store model and fold
            
            print(f"Best fold for {crypto_id}: {best_metrics['fold']} (Val Loss: {best_metrics['val_loss']:.4f})")
        
        # Print all metrics for each model/fold
        print("\nAll metrics for each model/fold:")
        for key, metrics in all_metrics.items():
            print(f"{key}: {metrics}")

    def _train_fold(self, train_loader, val_loader):
        model = CryptoLSTM().to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=self.patience)
        
        for _ in range(self.epochs):
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            val_loss = self._eval_epoch(model, val_loader, criterion)
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
        return model, metrics

    def calculate_metrics(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else float('inf')
        return {
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'MAE': np.mean(np.abs(y_true - y_pred)),
            'MAPE': mape,
            'R2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        }

    def predict(self, crypto_id, new_data):
        if crypto_id not in self.best_models:
            raise ValueError(f"No best model found for {crypto_id}")
        
        features = new_data[['price', 'volume']].values[-self.sequence_length:]
        scaled_features = self.scalers[crypto_id].transform(features)
        X = torch.FloatTensor(scaled_features).unsqueeze(0).to(self.device)
        
        model, fold = self.best_models[crypto_id]
        model.eval()
        with torch.no_grad():
            pred = model(X).cpu().numpy().flatten()[0]
            pred_array = np.array([[pred, 0, 0]])
            return self.scalers[crypto_id].inverse_transform(pred_array)[0, 0]

if __name__ == "__main__":
    trainer = CryptoTrainer("cryptocurrency_data.csv", sequence_length=10, 
                           batch_size=32, epochs=10, validation_split=0.1, patience=15)
    trainer.prepare_data()
    trainer.tune_hyperparameters()
    trainer.train_model()
    
    print("\nBest models saved:")
    for crypto_id, (model, fold) in trainer.best_models.items():
        print(f"{crypto_id}: Fold {fold}")
    
    crypto_id = "BTC"
    if crypto_id in trainer.best_models:
        new_data = pd.DataFrame({
            'price': [100, 102, 105, 103, 104, 106, 107, 108, 110, 112],
            'volume': [1000, 1200, 900, 1100, 1300, 1100, 1000, 1200, 1400, 1500]
        })
        next_price = trainer.predict(crypto_id, new_data)
        print(f"Predicted next price for {crypto_id}: ${next_price:.2f}")