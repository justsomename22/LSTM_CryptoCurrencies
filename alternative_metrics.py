import numpy as np

def calculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def calculate_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def calculate_mape(y_true, y_pred, epsilon=1e-10):
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def calculate_smape(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100

# Example usage
y_val_fold3 = np.array([...])  # Actual values for Fold 3
val_pred_fold3 = np.array([...])  # Predicted values for Fold 3

mae = calculate_mae(y_val_fold3, val_pred_fold3)
rmse = calculate_rmse(y_val_fold3, val_pred_fold3)
mse = calculate_mse(y_val_fold3, val_pred_fold3)
mape = calculate_mape(y_val_fold3, val_pred_fold3)
smape = calculate_smape(y_val_fold3, val_pred_fold3)

print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAPE: {mape:.2f}%")
print(f"sMAPE: {smape:.2f}%") 