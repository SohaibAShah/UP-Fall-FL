import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
# Removed tabulate import

# ==========================================
# 1. Configuration & Constants
# ==========================================

# UP-Fall IMU structure: 7 features per sensor
SENSOR_CONFIG = {
    'Ankle': 7,
    'Pocket': 7,
    'Belt': 7,
    'Neck': 7,
    'Wrist': 7
}

# Mapping of sensor names to column prefixes in your CSV
# Based on LoadCSVIMGClientData.py logic
SENSOR_COLUMNS_MAP = {
    'Ankle': ['AnkleAccelerometer', 'AnkleAngularVelocity', 'AnkleLuminosity'],
    'Pocket': ['RightPocketAccelerometer', 'RightPocketAngularVelocity', 'RightPocketLuminosity'],
    'Belt': ['BeltAccelerometer', 'BeltAngularVelocity', 'BeltLuminosity'],
    'Neck': ['NeckAccelerometer', 'NeckAngularVelocity', 'NeckLuminosity'],
    'Wrist': ['WristAccelerometer', 'WristAngularVelocity', 'WristLuminosity']
}

ALL_SENSORS = list(SENSOR_CONFIG.keys())
TOTAL_IMU_DIM = sum(SENSOR_CONFIG.values()) # 35

NUM_CLASSES = 12
NUM_EXPERTS = 4
TOP_K = 2
ROUNDS = 10
# NUM_CLIENTS will be determined by number of subjects

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Data Loading Logic (Subject-Based)
# ==========================================

def get_sensor_columns(df_columns, sensor_name):
    """Helper to find all actual columns for a sensor type."""
    prefixes = SENSOR_COLUMNS_MAP[sensor_name]
    relevant_cols = []
    for col in df_columns:
        for prefix in prefixes:
            if prefix in col:
                relevant_cols.append(col)
                break
    return relevant_cols

def load_upfall_by_subject(file_path):
    """
    Loads UP-Fall data and splits it by SUBJECT.
    Returns a dictionary: {subject_id: (X_full, y_full)}
    where X_full contains ALL sensor data concatenated.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Using Synthetic Data.")
        return None

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=[0, 1])
    
    # Flatten columns
    cleaned_columns = []
    last_val = ''
    for col_l1, col_l2 in df.columns:
        if 'Unnamed' in col_l1: col_l1 = last_val
        else: last_val = col_l1.strip(); col_l1 = last_val
        if col_l1 == col_l2.strip(): cleaned_columns.append(col_l1)
        else: cleaned_columns.append(f"{col_l1}_{col_l2.strip()}")
    df.columns = cleaned_columns

    # Filter Subjects
    subs = [1, 3, 4, 7, 10] # Example subset of subjects
    df = df[df['Subject'].isin(subs)].copy()
    
    # Clean Data
    df_cleaned = df.drop(columns=[col for col in df.columns if 'Infrared' in col] + ['Trial', 'Tag'], errors='ignore')
    df_cleaned.dropna(inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    
    subject_data = {}
    
    for sub in subs:
        sub_df = df_cleaned[df_cleaned['Subject'] == sub]
        y = sub_df['Activity'].values
        
        # Extract features per sensor to maintain order
        sensor_features = []
        for sensor in ALL_SENSORS:
            cols = get_sensor_columns(sub_df.columns, sensor)
            if len(cols) != SENSOR_CONFIG[sensor]:
                # Handle cases where columns might be missing or different count
                # For simulation, we might need strict checking, but let's assume standard UP-Fall
                pass
            
            X_sensor = sub_df[cols].values
            sensor_features.append(X_sensor)
            
        # Concatenate all sensors horizontally: [Ankle, Pocket, Belt, Neck, Wrist]
        X_all = np.hstack(sensor_features)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        subject_data[sub] = (X_scaled, y)
        
    return subject_data

def generate_synthetic_subject_data(num_subjects=5, num_samples=200):
    """Fallback if CSV is missing."""
    subject_data = {}
    for sub in range(num_subjects):
        # Generate base signal per subject
        X = np.random.randn(num_samples, TOTAL_IMU_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, (num_samples,))
        subject_data[sub] = (X, y)
    return subject_data

# ==========================================
# 3. Client Logic (Modified for Sensor Masking)
# ==========================================

class Client:
    def __init__(self, client_id, X_full, y, model, active_sensors):
        self.client_id = client_id
        self.active_sensors = active_sensors
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # --- Apply Sensor Masking ---
        # X_full has shape (N, 35) corresponding to [Ankle(7), Pocket(7), Belt(7), Neck(7), Wrist(7)]
        # We need to zero out sensors NOT in active_sensors
        
        X_masked = X_full.copy()
        
        current_idx = 0
        for sensor in ALL_SENSORS:
            dim = SENSOR_CONFIG[sensor]
            if sensor not in active_sensors:
                # Zero out this sensor's columns
                X_masked[:, current_idx : current_idx + dim] = 0.0
            current_idx += dim
            
        # Convert to Tensor
        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_masked).float(),
            torch.from_numpy(y).long()
        )

    def train(self, epochs=2):
        self.model.train()
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)
        epoch_loss = 0
        epoch_entropy = 0
        batch_count = 0
        
        print(f"[DEBUG] Client {self.client_id}: Training started") # Debug Print
        
        for epoch in range(epochs):
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                
                # Debug Print: Input Shape
                if i == 0 and epoch == 0:
                    print(f"[DEBUG] Client {self.client_id}: Input batch shape: {x.shape}")

                outputs, gate_probs = self.model(x)
                
                # Debug Print: Gating Probs (First batch only)
                if i == 0 and epoch == 0:
                    print(f"[DEBUG] Client {self.client_id}: Gating Probs Sample: {gate_probs[0].detach().cpu().numpy()}")

                loss = F.cross_entropy(outputs, y)
                aux_loss = (gate_probs.sum(0)**2).sum() * 0.01
                (loss + aux_loss).backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-9), dim=1).mean().item()
                epoch_entropy += entropy
                batch_count += 1
                
        print(f"[DEBUG] Client {self.client_id}: Training finished. Avg Loss: {epoch_loss/max(1,batch_count):.4f}") # Debug Print
        return self.model.state_dict(), epoch_loss/max(1,batch_count), epoch_entropy/max(1,batch_count)

# ==========================================
# 4. Model Definitions (Standard FedMoE)
# ==========================================

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Expert, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): return self.net(x)

class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
    def forward(self, x): return self.gate(x)

class FedMoE(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts=4, top_k=2):
        super(FedMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(input_dim, 64, num_classes) for _ in range(num_experts)])
        self.router = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gate_logits = self.router(x)
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=1)
        weights = F.softmax(weights, dim=1)
        final_output = torch.zeros(x.size(0), self.experts[0].net[-1].out_features).to(x.device)
        gate_probs = F.softmax(gate_logits, dim=1)
        
        # Debug Print: Selected Experts (First batch only logic handled by caller context ideally, 
        # but here we print if training)
        # print(f"[DEBUG] Model Forward: Selected Experts Indices: {selected_experts[0].detach().cpu().numpy()}")

        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            weight = weights[:, i].unsqueeze(1)
            for b, idx in enumerate(expert_idx):
                # FIX: Add .squeeze(0) to remove the extra batch dimension [1, 12] -> [12]
                expert_out = self.experts[idx](x[b].unsqueeze(0)).squeeze(0)
                final_output[b] += weight[b] * expert_out
        return final_output, gate_probs

def fed_avg(global_model, client_weights):
    print("[DEBUG] Server: Aggregating weights...") # Debug Print
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key]).float()
    for w in client_weights:
        for key in global_dict.keys(): global_dict[key] += w[key]
    for key in global_dict.keys(): global_dict[key] = global_dict[key] / len(client_weights)
    global_model.load_state_dict(global_dict)
    print("[DEBUG] Server: Aggregation complete.") # Debug Print
    return global_model

# ==========================================
# 5. Main Execution Scenarios
# ==========================================

def run_scenario_with_report(scenario_name, mode, subject_data):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    global_model = FedMoE(TOTAL_IMU_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K).to(device)
    clients = []
    
    # Create Clients based on Subjects
    subjects = list(subject_data.keys())
    
    print("[DEBUG] Creating clients based on subjects...") # Debug Print
    for sub_id in subjects:
        X_full, y_full = subject_data[sub_id]
        
        # Determine Sensors for this Subject-Client
        if mode == 'homogeneous':
            active_sensors = ALL_SENSORS # All 5
        else:
            # Heterogeneous: Random subset of sensors
            num_sensors = np.random.randint(1, 4) 
            active_sensors = np.random.choice(ALL_SENSORS, num_sensors, replace=False)
            
        print(f"[DEBUG] Client {sub_id}: Active Sensors -> {active_sensors}") # Debug Print
        clients.append(Client(sub_id, X_full, y_full, global_model, active_sensors))

    # Data collection for final table
    final_results = []

    for r in range(ROUNDS):
        print(f"\n[DEBUG] --- Round {r+1} Start ---") # Debug Print
        local_weights = []
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            w, loss, entropy = client.train(epochs=1)
            local_weights.append(w)
            
            # Save data for the FINAL round only
            if r == ROUNDS - 1:
                status = "COLLAPSED" if entropy > 1.30 else "STABLE"
                final_results.append([
                    f"Subject {client.client_id}",
                    f"{len(client.active_sensors)} ({', '.join([s[:2] for s in client.active_sensors])})", # Short names
                    f"{loss:.4f}",
                    f"{entropy:.4f}",
                    status
                ])
                
        global_model = fed_avg(global_model, local_weights)
        print(f"Round {r+1} complete...")

    # Print Final Summary Table
    print("\n" + "="*70)
    print(f"FINAL RESULTS: {scenario_name}")
    print("="*70)
    
    # Manual Table Formatting
    headers = ["Client", "Active Sensors", "Final Loss", "Entropy", "Router Status"]
    col_widths = [15, 30, 12, 10, 15]
    
    # Print Header
    header_row = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    print(header_row)
    print("-" * sum(col_widths))
    
    # Print Rows
    for row in final_results:
        row_str = "".join([f"{str(item):<{w}}" for item, w in zip(row, col_widths)])
        print(row_str)

if __name__ == "__main__":
    # 1. Load Data (Real or Synthetic)
    CSV_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv'
    
    # Try loading real data by subject
    data = load_upfall_by_subject(CSV_PATH)
    if data is None:
        print("Falling back to Synthetic Data for demonstration...")
        data = generate_synthetic_subject_data(num_subjects=5)
    
    # 2. Run Scenarios
    run_scenario_with_report("i. Homogeneous (All Subjects have All Sensors)", mode='homogeneous', subject_data=data)
    run_scenario_with_report("ii. Heterogeneous (Subjects have Random Sensors)", mode='heterogeneous', subject_data=data)