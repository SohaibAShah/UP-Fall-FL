import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class FallDetectionDataset(Dataset):
    """
    Loads pre-cleaned sensor data from .csv files, converts multi-class labels
    to binary (Fall vs. No Fall), and splits data into train, validation, and test sets.
    This version does NOT create sliding windows.
    """
    def __init__(self, data_dir: Path, split="train", val_ratio=0.2, seed=42):
        """
        Args:
            data_dir (Path): Directory where the processed _cleaned.csv files are located.
            split (str): The dataset split to load ("train", "val", or "test").
            val_ratio (float): The proportion of training data to use for validation.
            seed (int): Random seed for reproducibility.
        """
        
        # --- 1. Define Label Mapping ---
        # Labels 1-5 are falls, 6-11 are activities of daily living (ADL).
        # We map any fall label to 1, and any ADL label to 0.
        fall_labels = {1, 2, 3, 4, 5}

        # --- 2. Load and Combine Data from CSV files ---
        all_train_files = sorted(data_dir.glob("*_train.csv"))
        all_test_files = sorted(data_dir.glob("*_test.csv"))

        if not all_train_files or not all_test_files:
            raise FileNotFoundError(f"No processed CSV files found in {data_dir}. Please run the preparation script first.")

        # Load all training data, skipping the first (Time) column before converting to numpy
        train_data_list = [pd.read_csv(f, ).iloc[:, 1:].to_numpy(dtype=np.float32) for f in all_train_files]
        X_train_full = np.concatenate(train_data_list)
        
        # The label is now the last column of the modified array
        y_train_full = X_train_full[:, -1]
        # Features are all columns except the last one (the label)
        X_train_full = X_train_full[:, :-1] 

        # Load all test data, applying the same logic
        test_data_list = [pd.read_csv(f, ).iloc[:, 1:].to_numpy(dtype=np.float32) for f in all_test_files]
        X_test = np.concatenate(test_data_list)
        y_test = X_test[:, -1]
        X_test = X_test[:, :-1]

        # --- 3. Convert Labels to Binary ---
        y_train_full_binary = np.isin(y_train_full, list(fall_labels)).astype(int)
        y_test_binary = np.isin(y_test, list(fall_labels)).astype(int)

        # --- 4. Split Training data into Train and Validation ---
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full_binary,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_train_full_binary # Ensures same class balance in train/val
        )

        # --- 5. Select the correct split ---
        if split == "train":
            self.features, self.labels = X_train, y_train
        elif split == "val":
            self.features, self.labels = X_val, y_val
        elif split == "test":
            self.features, self.labels = X_test, y_test_binary
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        # Get a single row of data
        X = self.features[i]
        y = self.labels[i]
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return X_tensor, y_tensor



class FallDetectionDataset_Window(Dataset):
    """
    Loads pre-cleaned sensor data from .csv files, converts multi-class labels
    to binary (Fall vs. No Fall), splits train data into train/validation,
    and creates sliding windows for model training.
    """
    def __init__(self, data_dir: Path, split="train", val_ratio=0.2, seed=42, window_size=150, stride=75):
        """
        Args:
            data_dir (Path): Directory where the processed _cleaned.csv files are located.
            split (str): The dataset split to load ("train", "val", or "test").
            val_ratio (float): The proportion of training data to use for validation.
            seed (int): Random seed for reproducibility.
            window_size (int): The number of time steps in each window.
            stride (int): The step size to move the window.
        """
        self.samples = []
        
        # --- 1. Define Label Mapping ---
        fall_labels = {1, 2, 3, 4, 5}

        # --- 2. Load and Combine Data from CSV files ---
        all_train_files = sorted(data_dir.glob("*_train.csv"))
        all_test_files = sorted(data_dir.glob("*_test.csv"))

        if not all_train_files or not all_test_files:
            raise FileNotFoundError(f"No cleaned CSV files found in {data_dir}. Please run the preparation script first.")

        train_data_list = [pd.read_csv(f,).iloc[:, 1:].to_numpy(dtype=np.float32) for f in all_train_files]
        X_train_full = np.concatenate(train_data_list)
        
        y_train_full = X_train_full[:, -1]
        X_train_full = X_train_full[:, :-1] 

        test_data_list = [pd.read_csv(f,).iloc[:, 1:].to_numpy(dtype=np.float32) for f in all_test_files]
        X_test = np.concatenate(test_data_list)
        y_test = X_test[:, -1]
        X_test = X_test[:, :-1]
        
        # --- 3. Convert Labels to Binary ---
        y_train_full_binary = np.isin(y_train_full, list(fall_labels)).astype(int)
        y_test_binary = np.isin(y_test, list(fall_labels)).astype(int)

        # --- 4. Split Training data into Train and Validation ---
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full_binary,
            test_size=val_ratio,
            random_state=seed,
            stratify=y_train_full_binary
        )

        # --- START: New Normalization Step ---
        # 5. Calculate mean and std ONLY from the training data
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0) + 1e-6 # Add epsilon to avoid division by zero

        # Apply the same normalization to all splits
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        X_test = (X_test - mean) / std
        # --- END: New Normalization Step ---

        # --- 6. Select the correct split and create windows ---
        if split == "train":
            features, labels = X_train, y_train
        elif split == "val":
            features, labels = X_val, y_val
        elif split == "test":
            features, labels = X_test, y_test_binary
        else:
            raise ValueError("Split must be 'train', 'val', or 'test'")

        self._create_windows(features, labels, window_size, stride)

    def _create_windows(self, X, y, window_size, stride):
        """Helper function to generate windowed samples."""
        for i in range(0, len(X) - window_size + 1, stride):
            window = X[i : i + window_size]
            window_label = 1 if np.any(y[i : i + window_size] == 1) else 0
            self.samples.append((window, window_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        X, y = self.samples[i]
        
        X_tensor = torch.from_numpy(X.transpose(1, 0)).float()
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        return X_tensor, y_tensor
