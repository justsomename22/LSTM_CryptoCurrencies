#utils.py
import torch
import numpy as np

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Parameters:
    - patience (int): Number of epochs to wait after last improvement.
    - min_delta (float): Minimum change to qualify as an improvement.
    - verbose (bool): Whether to print debug information.
    """
    def __init__(self, patience=15, min_delta=0.0005, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.verbose = verbose
        
    def step(self, val_loss):
        """
        Update early stopping state based on validation loss.
        
        Parameters:
        - val_loss (float): Current validation loss.
        
        Returns:
        - bool: True if early stopping should be triggered, False otherwise.
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            if self.verbose:
                print(f"EarlyStopping: First score set to {score:.6f}")
            return False
            
        # Check if score improved by at least min_delta
        delta = score - self.best_score
        if delta >= self.min_delta:
            if self.verbose:
                print(f"EarlyStopping: Score improved from {self.best_score:.6f} to {score:.6f}")
            self.best_score = score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Patience ({self.patience}) exhausted. Stopping early.")
                return True
            return False 