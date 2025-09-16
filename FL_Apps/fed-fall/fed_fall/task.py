"""fedfall: A Flower / PyTorch app for Fall Detection."""

from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# =========================
# 1. MODEL ARCHITECTURE
# =========================
# --- ENCODERS ---
class IMUEncoder(nn.Module):
    def __init__(self, input_channels, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding='same')
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, feature_dim)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).squeeze(2))

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, feature_dim)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)); x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return self.fc(self.pool(x).view(x.size(0), -1))

# --- Main Gated Residual Fusion Model ---
class Net(nn.Module):
    def __init__(self, num_csv_features, num_classes=2):
        super().__init__()
        self.imu_encoder = IMUEncoder(num_csv_features, 128)
        self.img_encoder1 = ImageEncoder(64); self.img_encoder2 = ImageEncoder(64)
        self.img_fusion = nn.Linear(64 + 64, 128)
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fused_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.imu_only_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
    def forward(self, x_csv, x_img1, x_img2, threshold=0.5):
        f_csv = self.imu_encoder(x_csv); gate_prob = torch.sigmoid(self.gate(f_csv))
        if self.training:
            f_img_combined = F.relu(self.img_fusion(torch.cat((self.img_encoder1(x_img1), self.img_encoder2(x_img2)), dim=1)))
            out_fused = self.fused_classifier(f_csv + f_img_combined)
            out_imu_only = self.imu_only_classifier(f_csv)
            return gate_prob * out_fused + (1 - gate_prob) * out_imu_only
        else:
            use_images = (gate_prob > threshold).float()
            out_fused = torch.zeros(f_csv.size(0), 2, device=f_csv.device)
            if use_images.sum() > 0:
                f_img_combined = F.relu(self.img_fusion(torch.cat((self.img_encoder1(x_img1), self.img_encoder2(x_img2)), dim=1)))
                out_fused = self.fused_classifier(f_csv + f_img_combined)
            out_imu_only = self.imu_only_classifier(f_csv)
            return use_images * out_fused + (1 - use_images) * out_imu_only

# =========================
# 2. DATA LOADING & HELPERS
# =========================
def get_num_features():
    with open("/home/syed/PhD/UP-Fall-FL/FL_Apps/fed-fall/fed_fall/UP_Fall_partitions/num_features.txt", 'r') as f:
        return int(f.read())
    
def load_data(partition_id: int, num_partitions: int):
    """Loads a single client's data partition from disk."""
    partitions_dir = "/home/syed/PhD/UP-Fall-FL/FL_Apps/fed-fall/fed_fall/UP_Fall_partitions"
    client_ids = sorted([int(fname.split('_')[-1].split('.')[0]) for fname in os.listdir(partitions_dir) if fname.startswith('client_')])
    cid_to_load = client_ids[partition_id]
    
    X_csv, X_img1, X_img2, y = torch.load(os.path.join(partitions_dir, f'client_{cid_to_load}.pt'), weights_only=False)
    
    trainloader = DataLoader(TensorDataset(
        torch.from_numpy(X_csv), torch.from_numpy(X_img1),
        torch.from_numpy(X_img2), torch.from_numpy(y).long()
    ), batch_size=32, shuffle=True)

    # For client-side evaluation, we can just use its own training data
    valloader = DataLoader(TensorDataset(
        torch.from_numpy(X_csv), torch.from_numpy(X_img1),
        torch.from_numpy(X_img2), torch.from_numpy(y).long()
    ), batch_size=64)
    
    num_features = get_num_features()

    return trainloader, valloader, num_features



# =========================
# 3. TRAINING & TEST LOGIC
# =========================
def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for x_csv, x_img1, x_img2, labels in trainloader:
            x_csv, x_img1, x_img2, labels = x_csv.to(device), x_img1.to(device), x_img2.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(x_csv, x_img1, x_img2), labels)
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for x_csv, x_img1, x_img2, labels in testloader:
            x_csv, x_img1, x_img2, labels = x_csv.to(device), x_img1.to(device), x_img2.to(device), labels.to(device)
            outputs = net(x_csv, x_img1, x_img2)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# =========================
# 4. WEIGHTS HELPERS
# =========================
def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)