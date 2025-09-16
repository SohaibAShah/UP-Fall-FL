"""fed-fall: A Flower / pytorch_msg_api app."""

from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

print("[task.py] Module loaded.")


# In task.py

class ConvLSTMNet(nn.Module):
    """
    A hybrid 1D CNN and LSTM model for time-series classification.
    
    The CNN layers act as a feature extractor, and the LSTM layer
    learns the temporal dependencies between those features.
    """
    def __init__(self, num_features: int, num_classes: int = 2):
        super(ConvLSTMNet, self).__init__()
        
        # --- CNN Feature Extractor ---
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=5, padding="same")
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding="same")
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # --- LSTM Sequence Processor ---
        # The input features for the LSTM will be the number of output channels from the last conv layer (128)
        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=128, 
            num_layers=2, 
            batch_first=True, # Makes tensor shapes more intuitive
            dropout=0.2
        )
        
        # --- Classifier Head ---
        self.fc1 = nn.Linear(128, 64) # Takes the last hidden state of the LSTM
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, num_features, 200]
        
        # Pass through CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        # Shape: [batch_size, 64, 100]
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # Shape: [batch_size, 128, 50]

        # --- Prepare for LSTM ---
        # LSTM expects input of shape [batch_size, seq_len, features].
        # We need to swap the last two dimensions.
        x = x.permute(0, 2, 1)
        # Shape: [batch_size, 50, 128]
        
        # Pass through LSTM
        # We only need the output of the last hidden state for classification
        _, (h_n, _) = self.lstm(x)
        
        # Get the hidden state of the last layer
        x = h_n[-1]
        # Shape: [batch_size, 128]

        # Pass through the final classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class Net(nn.Module):
    """
    A deeper 1D CNN model with increased capacity.
    """
    def __init__(self, num_features: int, num_classes: int = 2):
        super(Net, self).__init__()
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=5, padding="same")
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding="same")
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # --- NEW BLOCK 3 ---
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding="same")
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # Recalculate the flattened size after the new pooling layer
        # Input: 200 -> pool1: 100 -> pool2: 50 -> pool3: 25
        # The number of output channels from conv3 is 128.
        self.flattened_size = 128 * 25
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        # --- APPLY NEW BLOCK ---
        x = F.relu(self.conv3(x))
        x = self.pool3(x)

        # Flatten the tensor
        x = x.view(-1, self.flattened_size)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_data(partition_id: int):
    """Load partition data from .pt files for a specific client."""
    # Using a relative path is more robust
    base_dir = os.path.join(os.path.dirname(__file__), "partitions")
    
    # Get sorted client IDs from available files
    client_files = [fname for fname in os.listdir(base_dir) if fname.startswith('client_') and fname.endswith('.pt')]
    client_ids = sorted([int(fname.split('_')[-1].split('.')[0]) for fname in client_files])
    
    if partition_id >= len(client_ids):
        raise ValueError(f"Partition ID {partition_id} is out of range. Only {len(client_ids)} client partitions found.")
        
    cid_to_load = client_ids[partition_id]

    # Load client data (tuple: (X_client, y_client))
    X_client, y_client = torch.load(os.path.join(base_dir, f'client_{cid_to_load}.pt'), weights_only=False)
    
    # The data shape is [num_samples, window_size, num_features].
    # Conv1d expects [num_samples, num_features, window_size], so we transpose.
    X_client = np.transpose(X_client, (0, 2, 1))

    # Read the number of features from the saved file
    with open(os.path.join(base_dir, 'num_features.txt'), 'r') as f:
        num_features = int(f.read())

    train_dataset = TensorDataset(torch.from_numpy(X_client).float(), torch.from_numpy(y_client).long())
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Load the global test set for evaluation
    test_file = os.path.join(base_dir, "test.pt")
    if os.path.exists(test_file):
        X_test, y_test = torch.load(test_file, weights_only=False)
        X_test = np.transpose(X_test, (0, 2, 1))
        test_dataset = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
        testloader = DataLoader(test_dataset, batch_size=32)
    else:
        testloader = None

    return trainloader, testloader, num_features


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    #print("[task.py] train() called")
    net.to(device)
    # The weight should be inversely proportional to class frequency.
    # e.g., weight for Fall (Class 0) > weight for Non-Fall (Class 1)
    class_weights = torch.tensor([1.5, 1.0]).to(device) # Example weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # Lowered learning rate
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for data, labels in trainloader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(data), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader) if len(trainloader) > 0 else 0.0
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set and compute advanced metrics."""
    #print("[task.py] test() called with advanced metrics")

    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    all_labels = []
    all_predicted = []
    
    net.eval()
    with torch.no_grad():
        for data, labels in testloader:
            data, labels = data.to(device), labels.to(device)
            outputs = net(data)
            loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Append batch results for final metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = loss / len(testloader) if len(testloader) > 0 else 0.0
    
    # Calculate Precision, Recall, and F1-Score for the "Fall" class (label 0)
    # zero_division=0 ensures that if a metric is not well-defined (e.g., no 'Fall' predictions), it returns 0 instead of an error.
    precision = precision_score(all_labels, all_predicted, pos_label=0, zero_division=0)
    recall = recall_score(all_labels, all_predicted, pos_label=0, zero_division=0)
    f1 = f1_score(all_labels, all_predicted, pos_label=0, zero_division=0)

    # Return the new metrics
    return avg_loss, accuracy, {"precision": precision, "recall": recall, "f1_score": f1}


def get_weights(net):
    #print("[task.py] get_weights() called")
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    #print("[task.py] set_weights() called")
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)