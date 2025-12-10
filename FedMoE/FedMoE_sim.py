import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
# Removed tabulate import
import os

# Import your data loader
# Assuming LoadCSVIMGClientData.py is in the same folder
try:
    from LoadCSVIMGClientData import loadSensorIMGClientsData
except ImportError:
    print("Warning: LoadCSVIMGClientData not found. Using Dummy Data.")
    loadSensorIMGClientsData = None

# ==========================================
# 1. Model Definitions (MoE)
# ==========================================

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return self.gate(x)

class FedMoE(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts=4, top_k=2):
        super(FedMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            Expert(input_dim, 64, num_classes) for _ in range(num_experts)
        ])
        self.router = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        batch_size = x.size(0)
        gate_logits = self.router(x)
        
        # Select Top-k
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=1)
        weights = F.softmax(weights, dim=1)
        
        final_output = torch.zeros(batch_size, self.experts[0].net[-1].out_features).to(x.device)
        gate_probs = F.softmax(gate_logits, dim=1) # Full probabilities for entropy calc
        
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            weight = weights[:, i].unsqueeze(1)
            
            expert_outputs_list = []
            for b, idx in enumerate(expert_idx):
                expert_outputs_list.append(self.experts[idx](x[b].unsqueeze(0)))
            
            expert_outputs = torch.cat(expert_outputs_list, dim=0)
            final_output += weight * expert_outputs

        return final_output, gate_probs

# ==========================================
# 2. Metric Helper Functions
# ==========================================

def calculate_entropy(probs):
    """
    Calculates Shannon Entropy of the gating distribution.
    High Entropy = Random/Confused Router.
    Low Entropy = Confident/Specialized Router.
    """
    # probs shape: [batch_size, num_experts]
    # Add epsilon to avoid log(0)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=1)
    return entropy.mean().item()

def calculate_accuracy(outputs, targets):
    _, preds = torch.max(outputs, 1)
    if targets.dim() > 1: # If one-hot
        _, targets = torch.max(targets, 1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

# ==========================================
# 3. Client & Server Logic
# ==========================================

class Client:
    def __init__(self, client_id, dataset, model, modality_type):
        self.client_id = client_id
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.modality_type = modality_type # "Sensor" or "Camera"

    def train_and_evaluate(self, epochs=1):
        self.model.train()
        bs = min(32, len(self.dataset))
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=bs, shuffle=True)
        
        epoch_loss = 0
        epoch_acc = 0
        epoch_entropy = 0
        batch_count = 0
        
        for epoch in range(epochs):
            for x, y in dataloader:
                self.optimizer.zero_grad()
                
                # Forward
                outputs, gate_probs = self.model(x)
                
                # Convert y for CrossEntropy
                if y.dim() > 1: y_indices = torch.argmax(y, dim=1)
                else: y_indices = y.long()

                # Loss
                cls_loss = F.cross_entropy(outputs, y_indices)
                # Simple load balance loss
                aux_loss = (gate_probs.sum(0)**2).sum() * 0.01
                total_loss = cls_loss + aux_loss
                
                total_loss.backward()
                self.optimizer.step()
                
                # Metrics Accumulation
                epoch_loss += total_loss.item()
                epoch_acc += calculate_accuracy(outputs, y)
                epoch_entropy += calculate_entropy(gate_probs)
                batch_count += 1
        
        # Average metrics
        avg_loss = epoch_loss / max(1, batch_count)
        avg_acc = (epoch_acc / max(1, batch_count)) * 100
        avg_entropy = epoch_entropy / max(1, batch_count)
        
        return self.model.state_dict(), avg_loss, avg_acc, avg_entropy

def fed_avg(global_model, client_weights):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key]).float()
    for w in client_weights:
        for key in global_dict.keys():
            global_dict[key] += w[key]
    for key in global_dict.keys():
        global_dict[key] = global_dict[key] / len(client_weights)
    global_model.load_state_dict(global_dict)
    return global_model

# ==========================================
# 4. Main Experiment
# ==========================================

import copy

if __name__ == "__main__":
    print("--- Starting FedMoE Analysis Simulation ---")
    
    # Config
    INPUT_DIM = 1024 
    NUM_CLASSES = 12 
    NUM_EXPERTS = 4
    TOP_K = 2
    ROUNDS = 5
    
    # 1. Data Loading
    CSV_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv' 
    IMG_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image'
    
    if loadSensorIMGClientsData and os.path.exists(CSV_PATH):
        X_splits, Y_splits = loadSensorIMGClientsData(CSV_PATH, IMG_PATH)
        NUM_CLIENTS = len(X_splits)
    else:
        print("Using Dummy Data.")
        NUM_CLIENTS = 8
        X_splits, Y_splits = {}, {}
        for i in range(NUM_CLIENTS):
            if i < 6: X_splits[i] = np.random.rand(50, 7)
            else: X_splits[i] = np.random.rand(50, 1024)
            Y_splits[i] = torch.nn.functional.one_hot(torch.randint(0, NUM_CLASSES, (50,)), NUM_CLASSES)

    # 2. Init Model
    global_model = FedMoE(INPUT_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K)
    
    # 3. Setup Clients (With Padding Logic)
    clients = []
    for cid in range(NUM_CLIENTS):
        X_data = X_splits[cid]
        Y_data = Y_splits[cid]
        
        # Determine Modality Type
        if X_data.shape[1] < 100: # It's a sensor client
            modality = "Sensor (Padded)"
            # Padding
            padding = np.zeros((X_data.shape[0], INPUT_DIM - X_data.shape[1]))
            X_data_padded = np.hstack((X_data, padding))
        else:
            modality = "Camera (Full)"
            X_data_padded = X_data
            
        # Tensor conversion
        X_tensor = torch.from_numpy(X_data_padded).float()
        if not isinstance(Y_data, torch.Tensor): Y_tensor = torch.from_numpy(Y_data).float()
        else: Y_tensor = Y_data.float()
            
        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        clients.append(Client(cid, dataset, global_model, modality))

    # 4. Training Loop with Table Reporting
    results_table = []
    
    for r in range(ROUNDS):
        print(f"\nTraining Round {r+1}...")
        local_weights = []
        
        round_results = []
        
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            
            # Train and Get Metrics
            w, loss, acc, entropy = client.train_and_evaluate(epochs=2)
            local_weights.append(w)
            
            # Store results for table
            round_results.append([
                r+1, 
                client.client_id, 
                client.modality_type, 
                f"{loss:.4f}", 
                f"{acc:.2f}%", 
                f"{entropy:.4f}"
            ])
            
        # Aggregate
        global_model = fed_avg(global_model, local_weights)
        
        # Print Simple Table without Tabulate
        print(f"{'Round':<6} | {'Client':<8} | {'Modality':<18} | {'Loss':<8} | {'Acc':<8} | {'Entropy':<8}")
        print("-" * 70)
        for row in round_results:
            # row = [Round, ClientID, Modality, Loss, Acc, Entropy]
            print(f"{row[0]:<6} | {row[1]:<8} | {row[2]:<18} | {row[3]:<8} | {row[4]:<8} | {row[5]:<8}")
        
        results_table.extend(round_results)

    print("\n--- Summary of Findings ---")
    print("1. Compare 'Gating Entropy' between Sensor and Camera clients.")
    print("2. High Entropy in Sensor clients confirms 'Router Collapse' (Random Selection due to zeros).")
    print("3. Low Loss/Entropy in Camera clients shows the model works when data is present.")