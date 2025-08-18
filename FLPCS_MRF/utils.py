# utils.py 
# This file contains general utility functions and the custom dataset loader class.

import os
import torch
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
sns.set(style="whitegrid")

# Define dataset loader

class CustomDataset(Dataset):
    """
    This is a custom PyTorch Dataset class.
    • The __init__ method initializes the dataset with features (input data) and labels 
    (corresponding targets).
    • The __len__ method returns the total number of samples in the dataset.
    • The __getitem__ method allows indexing the dataset to retrieve a specific sample 
    (feature-label pair). This is crucial for PyTorch DataLoader to iterate over the dataset.
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def display_result(y_test, y_pred):
    """
    This function prints common classification metrics: accuracy, precision, recall, and F1 score. 
    It uses sklearn.metrics functions with average='weighted' to account for class imbalance
    and provides a detailed classification report and confusion matrix visualization.

    """
    print('Accuracy score : ', accuracy_score(y_test, y_pred))
    print('Precision score : ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall score : ', recall_score(y_test, y_pred, average='weighted'))
    print('F1 score : ', f1_score(y_test, y_pred, average='weighted'))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
    plt.show()


def scaled_data(X_train, X_test, X_val):
    """
    This function performs standardization of numerical data using StandardScaler.
    • It fits the scaler on the training data (X_train) to learn the mean and standard deviation.
    • It then transforms the training, test, and validation data using these learned parameters.
    This ensures that the test and validation sets are scaled consistently with the training data, preventing data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_test_scaled, X_val_scaled

def set_seed(seed=0):
    """
    This function sets the random seed for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)

