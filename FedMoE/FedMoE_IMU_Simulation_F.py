import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

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
SENSOR_COLUMNS_MAP = {
    'Ankle': ['AnkleAccelerometer', 'AnkleAngularVelocity', 'AnkleLuminosity'],
    'Pocket': ['RightPocketAccelerometer', 'RightPocketAngularVelocity', 'RightPocketLuminosity'],
    'Belt': ['BeltAccelerometer', 'BeltAngularVelocity', 'BeltLuminosity'],
    'Neck': ['NeckAccelerometer', 'NeckAngularVelocity', 'NeckLuminosity'],
    'Wrist': ['WristAccelerometer', 'WristAngularVelocity', 'WristLuminosity']
}

ALL_SENSORS = list(SENSOR_CONFIG.keys())
TOTAL_IMU_DIM = sum(SENSOR_CONFIG.values()) # 35 features total

NUM_CLASSES = 12
NUM_EXPERTS = 4
TOP_K = 2
ROUNDS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Using Synthetic Data.")
        return None

    print(f"Loading data from {file_path}...")
    # Read CSV with multi-level header handling
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

    # Filter Subjects (Example subset)
    subs = [1, 3, 4, 7, 10] 
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
            
            # If sensor missing in CSV, pad with zeros immediately
            if not cols:
                print(f"Warning: Sensor {sensor} columns not found for Subject {sub}. Padding.")
                X_sensor = np.zeros((len(sub_df), SENSOR_CONFIG[sensor]))
            else:
                X_sensor = sub_df[cols].values
                # Fix dimension mismatch if any
                target_dim = SENSOR_CONFIG[sensor]
                if X_sensor.shape[1] < target_dim:
                    padding = np.zeros((X_sensor.shape[0], target_dim - X_sensor.shape[1]))
                    X_sensor = np.hstack((X_sensor, padding))
                elif X_sensor.shape[1] > target_dim:
                    X_sensor = X_sensor[:, :target_dim]

            sensor_features.append(X_sensor)
            
        # Concatenate all sensors horizontally
        X_all = np.hstack(sensor_features)
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        # Fix Labels (Label 20 -> 0)
        y = np.where(y == 20, 0, y)
        
        subject_data[sub] = (X_scaled, y)
        
    return subject_data

def generate_synthetic_subject_data(num_subjects=5, num_samples=200):
    """Fallback if CSV is missing."""
    subject_data = {}
    for sub in range(num_subjects):
        X = np.random.randn(num_samples, TOTAL_IMU_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, (num_samples,))
        subject_data[sub] = (X, y)
    return subject_data

# ==========================================
# 3. Model Definitions (Fixed Forward Pass)
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
        
        # Initialize output container on correct device
        final_output = torch.zeros(x.size(0), self.experts[0].net[-1].out_features).to(x.device)
        gate_probs = F.softmax(gate_logits, dim=1)
        
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            weight = weights[:, i].unsqueeze(1) # Shape: [Batch, 1]
            
            for b, idx in enumerate(expert_idx):
                # Input to expert needs to be [1, Input_Dim]
                input_sample = x[b].unsqueeze(0) 
                
                # Get expert output [1, Classes]
                expert_out = self.experts[idx](input_sample)
                
                # CRITICAL FIX: Squeeze expert output to [Classes] to match final_output[b]
                expert_out = expert_out.squeeze(0) 
                
                # weight[b] is a scalar tensor, expert_out is vector [Classes]
                final_output[b] += weight[b] * expert_out
                
        return final_output, gate_probs

def fed_avg(global_model, client_weights):
    global_dict = global_model.state_dict()
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key]).float()
    for w in client_weights:
        for key in global_dict.keys(): global_dict[key] += w[key]
    for key in global_dict.keys(): global_dict[key] = global_dict[key] / len(client_weights)
    global_model.load_state_dict(global_dict)
    return global_model

# ==========================================
# 4. Client Logic
# ==========================================

class Client:
    def __init__(self, client_id, X_full, y, model, active_sensors):
        self.client_id = client_id
        self.active_sensors = active_sensors
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # --- Apply Sensor Masking ---
        X_masked = X_full.copy()
        current_idx = 0
        
        # Iterate through canonical order to zero out missing sensors
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
        # Ensure batch size is safe
        bs = min(32, len(self.dataset))
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=bs, shuffle=True)
        
        epoch_loss = 0
        epoch_entropy = 0
        all_preds = []
        all_targets = []
        batch_count = 0
        
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                
                outputs, gate_probs = self.model(x)
                
                loss = F.cross_entropy(outputs, y)
                aux_loss = (gate_probs.sum(0)**2).sum() * 0.01
                (loss + aux_loss).backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-9), dim=1).mean().item()
                epoch_entropy += entropy
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
                batch_count += 1
        
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        return self.model.state_dict(), epoch_loss/max(1,batch_count), epoch_entropy/max(1,batch_count), f1

# ==========================================
# 5. Main Execution
# ==========================================

def run_scenario_with_report(scenario_name, mode, subject_data):
    print(f"\n{'='*60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"{'='*60}")
    
    global_model = FedMoE(TOTAL_IMU_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K).to(device)
    clients = []
    
    # Create Clients based on Subjects
    subjects = list(subject_data.keys())
    
    print(f"[DEBUG] Creating {len(subjects)} clients...")
    
    for sub_id in subjects:
        X_full, y_full = subject_data[sub_id]
        
        if mode == 'homogeneous':
            active_sensors = ALL_SENSORS 
        else:
            # Random subset for Heterogeneity
            num_sensors = np.random.randint(1, 4) 
            active_sensors = np.random.choice(ALL_SENSORS, num_sensors, replace=False)
            
        clients.append(Client(sub_id, X_full, y_full, global_model, active_sensors))

    # Training Loop
    final_results = []
    for r in range(ROUNDS):
        local_weights = []
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            w, loss, entropy, f1 = client.train(epochs=1)
            local_weights.append(w)
            
            if r == ROUNDS - 1:
                # Logic for status
                if entropy > 1.29 or f1 < 0.4:
                    status = "COLLAPSED/POOR"
                else:
                    status = "STABLE"
                    
                final_results.append([
                    f"Subject {client.client_id}",
                    f"{len(client.active_sensors)} ({', '.join([s[:2] for s in client.active_sensors])})", 
                    f"{loss:.4f}",
                    f"{entropy:.4f}",
                    f"{f1:.4f}",
                    status
                ])
                
        global_model = fed_avg(global_model, local_weights)
        print(f"Round {r+1} complete...")

    # Print Table
    print("\n" + "="*85)
    print(f"FINAL RESULTS: {scenario_name}")
    print("="*85)
    
    headers = ["Client", "Active Sensors", "Final Loss", "Entropy", "F1 Score", "Router Status"]
    col_widths = [12, 30, 12, 10, 10, 15] 
    
    header_row = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    print(header_row)
    print("-" * sum(col_widths))
    
    for row in final_results:
        row_str = "".join([f"{str(item):<{w}}" for item, w in zip(row, col_widths)])
        print(row_str)

if __name__ == "__main__":
    CSV_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv'
    
    # Load Real Data or Fallback
    data = load_upfall_by_subject(CSV_PATH)
    if data is None:
        print("Falling back to Synthetic Data...")
        data = generate_synthetic_subject_data(num_subjects=5)
    
    # Run Both Scenarios
    run_scenario_with_report("i. Homogeneous (All Subjects have All Sensors)", mode='homogeneous', subject_data=data)
    run_scenario_with_report("ii. Heterogeneous (Subjects have Random Sensors)", mode='heterogeneous', subject_data=data)