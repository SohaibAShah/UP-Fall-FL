import os
import numpy as np
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import logging

# =========================
# 1. MODEL ARCHITECTURE
# =========================
class CNN_Attention(nn.Module):
    """1D CNN with Temporal Attention."""
    def __init__(self, input_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 5, padding='same'); self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, 5, padding='same'); self.relu2 = nn.ReLU()
        self.attention = self.TemporalAttention(64); self.fc = nn.Linear(64, 1)
    class TemporalAttention(nn.Module):
        def __init__(self, in_features):
            super().__init__(); self.attention_net = nn.Sequential(nn.Linear(in_features, in_features // 2), nn.Tanh(), nn.Linear(in_features // 2, 1))
        def forward(self, x):
            x_permuted = x.permute(0, 2, 1); attn_weights = torch.softmax(self.attention_net(x_permuted), dim=1)
            return torch.sum(x_permuted * attn_weights, dim=1)
    def forward(self, x):
        x = self.relu1(self.conv1(x)); x = self.relu2(self.conv2(x))
        return self.fc(self.attention(x))

# Global cache for the dataset to avoid reloading for each client
CLIENT_DATA = None
TEST_LOADER = None
NUM_FEATURES = 0

# =========================
# 2. DATA LOADING & PREP
# =========================
def load_data(partition_id: int, num_partitions: int):
    """Load, partition, and cache the dataset."""
    global CLIENT_DATA, TEST_LOADER, NUM_FEATURES
    if CLIENT_DATA is None:
        # --- TRACE ---
        logging.info("[Trace] Data cache is empty. Loading and preprocessing data for the first time...")
        file_path = os.path.join(os.path.dirname(__file__), "..", "/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/", "sensor.csv")
        df = pd.read_csv(file_path, header=[0, 1])
        # Clean column names
        cleaned_columns = []
        last_val = ''
        for c in df.columns:
            col_l1, col_l2 = c
            if 'Unnamed' in col_l1: col_l1 = last_val
            else: last_val = col_l1.strip()
            cleaned_columns.append(f"{col_l1}_{col_l2.strip()}" if col_l1 != col_l2 else col_l1)
        df.columns = cleaned_columns

        # Preprocessing
        df = df[~df['Subject'].isin([5, 9])]
        df = df[~((df['Subject'] == 2) & (df['Activity'] == 5))]
        fall_activity_ids = {2, 3, 4, 5, 6}
        df['Fall'] = df['Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
        
        train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
        test_subjects = [s for s in range(14, 18)]
        train_df = df[df['Subject'].isin(train_subjects)]; test_df = df[df['Subject'].isin(test_subjects)]

        imu_columns = [col for col in train_df.columns if 'Accelerometer' in col or 'AngularVelocity' in col]
        NUM_FEATURES = len(imu_columns)
        scaler = StandardScaler().fit(train_df[imu_columns])
        train_df.loc[:, imu_columns] = scaler.transform(train_df[imu_columns])
        test_df.loc[:, imu_columns] = scaler.transform(test_df[imu_columns])

        # Create windowed test set
        X_test_w, y_test_w = [], []
        for subject in test_df['Subject'].unique():
            d, l = test_df[test_df['Subject'] == subject][imu_columns].values, test_df[test_df['Subject'] == subject]['Fall'].values
            for i in range(0, len(d) - 200, 100):
                X_test_w.append(d[i:i+200]); y_test_w.append(0 if np.any(l[i:i+200] == 0) else 1)
        X_test = np.transpose(np.array(X_test_w), (0, 2, 1)); y_test = np.array(y_test_w)
        TEST_LOADER = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), batch_size=256)

        # Create and cache client data partitions
        CLIENT_DATA = {}
        for cid in sorted(train_df['Subject'].unique()):
            d, l = train_df[train_df['Subject'] == cid][imu_columns].values, train_df[train_df['Subject'] == cid]['Fall'].values
            X_client_w, y_client_w = [], []
            for i in range(0, len(d) - 200, 100):
                X_client_w.append(d[i:i+200]); y_client_w.append(0 if np.any(l[i:i+200] == 0) else 1)
            X_client = np.transpose(np.array(X_client_w), (0, 2, 1)); y_client = np.array(y_client_w)
            CLIENT_DATA[cid] = (X_client, y_client)
        # --- TRACE ---
        logging.info("[Trace] Data loading and preprocessing complete. Caching results.")

    # Return the correct partition for the client
    client_ids = sorted(list(CLIENT_DATA.keys()))
    client_id_for_this_partition = client_ids[partition_id]
    return CLIENT_DATA[client_id_for_this_partition]

def get_test_loader():
    global TEST_LOADER
    return TEST_LOADER

def get_num_features():
    global NUM_FEATURES
    return NUM_FEATURES

# =========================
# 3. TRAINING & TEST LOGIC
# =========================
def train(net, trainloader, device, config, initial_params, control_variate=None):
    net.to(device); net.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["learning-rate"])
    for epoch in range(config["local-epochs"]):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            if config["algorithm"] == 'fedprox':
                proximal_term = 0.0
                for param, initial_param in zip(net.parameters(), initial_params):
                    proximal_term += torch.sum((param - initial_param) ** 2)
                loss += (config["mu"] / 2) * proximal_term
            loss.backward()
            if config["algorithm"] == 'scaffold':
                server_cv = [torch.tensor(p, device=device) for p in config['server_cv']]
                for param, scv, ccv in zip(net.parameters(), server_cv, control_variate):
                    param.grad += (scv - ccv).to(device)
            optimizer.step()
    return net

def test(net, testloader, device):
    net.to(device); net.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = net(inputs.to(device))
            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds.flatten()); all_labels.extend(labels.numpy())
    f1 = f1_score(all_labels, all_preds, pos_label=0, zero_division=0.0)
    return 0.0, {"f1_score": f1}

# =========================
# 4. WEIGHTS HELPERS
# =========================
def get_weights(net) -> list[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters: list[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)