import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from collections import OrderedDict
import numpy as np

print("[task.py] Module loaded.")

# =========================
# 1. MODEL ARCHITECTURE
# =========================
class CNN_Attention(nn.Module):
    """1D CNN with Temporal Attention."""
    def __init__(self, input_channels):
        super().__init__()
        print("[task.py] CNN_Attention() called")
        self.conv1 = nn.Conv1d(input_channels, 32, 5, padding='same'); self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, 5, padding='same'); self.relu2 = nn.ReLU()
        self.attention = self.TemporalAttention(64); self.fc = nn.Linear(64, 1)
    
    class TemporalAttention(nn.Module):
        def __init__(self, in_features):
            print("[task.py] TemporalAttention() called")
            super().__init__(); self.attention_net = nn.Sequential(nn.Linear(in_features, in_features // 2), nn.Tanh(), nn.Linear(in_features // 2, 1))
        def forward(self, x):
            x_permuted = x.permute(0, 2, 1); attn_weights = torch.softmax(self.attention_net(x_permuted), dim=1)
            return torch.sum(x_permuted * attn_weights, dim=1)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x)); x = self.relu2(self.conv2(x))
        return self.fc(self.attention(x))

# =========================
# 2. TRAINING & TEST LOGIC
# =========================
def train(net, trainloader, device, config, initial_params, control_variate=None):
    """Train the model on the training set."""
    print("[task.py] train() called")
    net.to(device); net.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=config["learning-rate"])
    for _ in range(config["local-epochs"]):
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
    """Evaluate the model on the test set."""
    print("[task.py] test() called")
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
# 3. WEIGHTS HELPERS
# =========================
def get_weights(net) -> list[np.ndarray]:
    print("[task.py] get_weights() called")
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_weights(net, parameters: list[np.ndarray]):
    print("[task.py] set_weights() called")
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)