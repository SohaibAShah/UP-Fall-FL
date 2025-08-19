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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Define dataset loader
class CustomDataseRest(Dataset):
    def __init__(self, features1, features2, features3, labels):
        self.features1 = features1
        self.features2 = features2
        self.features3 = features3
        self.labels = labels

    def __len__(self):
        return len(self.features1)
    
    def __getitem__(self, index):
        return self.features1[index], self.features2[index], self.features3[index], self.labels[index]
    

class CustomDatasetIMG(Dataset):
    def __init__(self, features1, features2, labels):
        self.features1 = features1
        self.features2 = features2
        self.labels = labels

    def __len__(self):
        return len(self.features1)
    
    def __getitem__(self, index):
        return self.features1[index], self.features2[index], self.labels[index]


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

def scale_data(X_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    return X_train_scaled

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

def train_one_epoch(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_id, batch in enumerate(train_loader):
        data1 = batch[0].to(device).float()
        data2 = batch[1].to(device).float()
        data3 = batch[2].to(device).float()
        target = torch.squeeze(batch[3]).to(device).float()

        output = model(data1, data2, data3)
        loss = criterion(output, target)

        # pred = output.detach().max(1)[1]
        # target_ =  target.detach().max(1)[1]
        # correct += pred.eq(target.detach().max(1)[1]).sum().item()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * data1.size()[0]
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target.max(1)[1]).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_id, batch in enumerate(val_loader):
            data1 = batch[0].to(device).float()
            data2 = batch[1].to(device).float()
            data3 = batch[2].to(device).float()
            target = torch.squeeze(batch[3]).to(device).float()

            output = model(data1, data2, data3)
            loss = criterion(output, target)

            # 统计
            running_loss += loss.item() * data1.size()[0]
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target.max(1)[1]).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc