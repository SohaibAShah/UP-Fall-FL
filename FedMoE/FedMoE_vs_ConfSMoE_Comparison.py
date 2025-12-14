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

SENSOR_CONFIG = {
    'Ankle': 7,
    'Pocket': 7,
    'Belt': 7,
    'Neck': 7,
    'Wrist': 7,
    'Camera1': 1024,
    'Camera2': 1024
}

SENSOR_COLUMNS_MAP = {
    'Ankle': ['AnkleAccelerometer', 'AnkleAngularVelocity', 'AnkleLuminosity'],
    'Pocket': ['RightPocketAccelerometer', 'RightPocketAngularVelocity', 'RightPocketLuminosity'],
    'Belt': ['BeltAccelerometer', 'BeltAngularVelocity', 'BeltLuminosity'],
    'Neck': ['NeckAccelerometer', 'NeckAngularVelocity', 'NeckLuminosity'],
    'Wrist': ['WristAccelerometer', 'WristAngularVelocity', 'WristLuminosity']
}

ALL_SENSORS = list(SENSOR_CONFIG.keys()) 
TOTAL_INPUT_DIM = sum(SENSOR_CONFIG.values()) # 2083

NUM_CLASSES = 12
NUM_EXPERTS = 4
TOP_K = 2
ROUNDS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. Data Loading Logic (Train 1-13, Test 14-17)
# ==========================================

def get_sensor_columns(df_columns, sensor_name):
    if sensor_name not in SENSOR_COLUMNS_MAP: return [] 
    prefixes = SENSOR_COLUMNS_MAP[sensor_name]
    relevant_cols = []
    for col in df_columns:
        for prefix in prefixes:
            if prefix in col:
                relevant_cols.append(col)
                break
    return relevant_cols

def load_upfall_split(file_path, image_path=None):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Using Synthetic Data.")
        return None, None

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

    # Define Subjects
    # Training: 1-13 (Excluding 8 as per original logic)
    train_subs = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13] 
    # Testing: 14-17
    test_subs = [14, 15, 16, 17]
    
    all_subs = train_subs + test_subs
    df = df[df['Subject'].isin(all_subs)].copy()
    
    # Clean Data
    df_cleaned = df.drop(columns=[col for col in df.columns if 'Infrared' in col] + ['Trial', 'Tag'], errors='ignore')
    df_cleaned.dropna(inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    
    # Cam Data Mock/Load
    cam_data = {}
    if image_path:
        for cam_id in [1, 2]:
            name_file = os.path.join(image_path, f'name_{cam_id}.npy')
            img_file = os.path.join(image_path, f'image_{cam_id}.npy')
            if os.path.exists(name_file) and os.path.exists(img_file):
                print(f"Loading Camera {cam_id} metadata/images...")
                timestamps = np.load(name_file)
                # Load images lazily or assume loaded
                # For demo speed, we simulate the structure
                ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
                cam_data[f'Camera{cam_id}'] = (None, ts_to_idx) # Image data loaded on demand in real scenario

    def process_subjects(subject_list):
        data_dict = {}
        for sub in subject_list:
            sub_df = df_cleaned[df_cleaned['Subject'] == sub].copy()
            if sub_df.empty: continue
            
            y = sub_df['Activity'].values
            y = np.where(y == 20, 0, y)

            sensor_features = []
            for sensor in ALL_SENSORS:
                if 'Camera' in sensor:
                    # Simulate Camera Features for speed/compatibility
                    # In real implementation, load from .npy based on timestamp
                    np.random.seed(sub + 100 if '1' in sensor else sub + 200)
                    X_sensor = np.random.randn(len(sub_df), 1024)
                else:
                    cols = get_sensor_columns(sub_df.columns, sensor)
                    if not cols:
                         X_sensor = np.zeros((len(sub_df), SENSOR_CONFIG[sensor]))
                    else:
                        X_sensor = sub_df[cols].values
                        target_dim = SENSOR_CONFIG[sensor]
                        if X_sensor.shape[1] < target_dim:
                            padding = np.zeros((X_sensor.shape[0], target_dim - X_sensor.shape[1]))
                            X_sensor = np.hstack((X_sensor, padding))
                        elif X_sensor.shape[1] > target_dim:
                            X_sensor = X_sensor[:, :target_dim]
                    
                    scaler = StandardScaler()
                    X_sensor = scaler.fit_transform(X_sensor)
                
                sensor_features.append(X_sensor)
            
            X_all = np.hstack(sensor_features)
            data_dict[sub] = (X_all, y)
        return data_dict

    train_data = process_subjects(train_subs)
    test_data = process_subjects(test_subs)
    
    return train_data, test_data

def generate_synthetic_split(num_train=10, num_test=4):
    train_data = {i: (np.random.randn(100, TOTAL_INPUT_DIM).astype(np.float32), np.random.randint(0, NUM_CLASSES, (100,))) for i in range(num_train)}
    test_data = {i+num_train: (np.random.randn(50, TOTAL_INPUT_DIM).astype(np.float32), np.random.randint(0, NUM_CLASSES, (50,))) for i in range(num_test)}
    return train_data, test_data

# ==========================================
# 3. Model Classes (Baseline vs Conf-SMoE)
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

# --- FedMoE Components ---
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
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            weight = weights[:, i].unsqueeze(1)
            for b, idx in enumerate(expert_idx):
                expert_out = self.experts[idx](x[b].unsqueeze(0)).squeeze(0)
                final_output[b] += weight[b] * expert_out
        return final_output, gate_probs, None # No confidence score

# --- Conf-SMoE Components ---
class ConfidenceGatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(ConfidenceGatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        raw_logits = self.gate(x)
        confidence = self.confidence_net(x)
        return raw_logits * confidence, confidence

class ConfSMoE(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts=4, top_k=2):
        super(ConfSMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(input_dim, 64, num_classes) for _ in range(num_experts)])
        self.router = ConfidenceGatingNetwork(input_dim, num_experts)

    def forward(self, x):
        gate_logits, confidence = self.router(x)
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=1)
        weights = F.softmax(weights, dim=1)
        final_output = torch.zeros(x.size(0), self.experts[0].net[-1].out_features).to(x.device)
        gate_probs = F.softmax(gate_logits, dim=1)
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            weight = weights[:, i].unsqueeze(1)
            for b, idx in enumerate(expert_idx):
                expert_out = self.experts[idx](x[b].unsqueeze(0)).squeeze(0)
                final_output[b] += weight[b] * expert_out
        return final_output, gate_probs, confidence

# ==========================================
# 4. Client & Server Logic
# ==========================================

class Client:
    def __init__(self, client_id, X_full, y, model, active_sensors):
        self.client_id = client_id
        self.active_sensors = active_sensors
        self.model = copy.deepcopy(model).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        
        # Masking
        X_masked = X_full.copy()
        current_idx = 0
        for sensor in ALL_SENSORS:
            dim = SENSOR_CONFIG[sensor]
            is_active = False
            if sensor in active_sensors: is_active = True
            elif 'IMU' in active_sensors and sensor not in ['Camera1', 'Camera2']: is_active = True
            elif 'C1' in active_sensors and sensor == 'Camera1': is_active = True
            elif 'C2' in active_sensors and sensor == 'Camera2': is_active = True
            
            if not is_active:
                X_masked[:, current_idx : current_idx + dim] = 0.0
            current_idx += dim
            
        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_masked).float(),
            torch.from_numpy(y).long()
        )

    def train(self, epochs=1):
        self.model.train()
        bs = min(32, len(self.dataset))
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=bs, shuffle=True)
        
        epoch_loss = 0
        batch_count = 0
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                
                out = self.model(x)
                outputs = out[0]
                gate_probs = out[1]
                
                loss = F.cross_entropy(outputs, y)
                aux_loss = (gate_probs.sum(0)**2).sum() * 0.01
                (loss + aux_loss).backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                batch_count += 1
        return self.model.state_dict(), epoch_loss/max(1,batch_count)

    def evaluate(self):
        self.model.eval()
        bs = min(32, len(self.dataset))
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=bs, shuffle=False)
        all_preds, all_targets = [], []
        epoch_entropy = 0
        batch_count = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                out = self.model(x)
                outputs, gate_probs = out[0], out[1]
                
                entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-9), dim=1).mean().item()
                epoch_entropy += entropy
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                batch_count += 1

        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        acc = (np.array(all_preds) == np.array(all_targets)).mean()
        return f1, acc, epoch_entropy/max(1, batch_count)

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
# 5. Simulation Manager
# ==========================================

def run_simulation(model_type, train_data, test_data):
    print(f"\n>>> Running Simulation: {model_type}")
    
    # 1. Init Model
    if model_type == "FedMoE":
        global_model = FedMoE(TOTAL_INPUT_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K).to(device)
    else:
        global_model = ConfSMoE(TOTAL_INPUT_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K).to(device)
        
    # 2. Setup Train Clients (1-13)
    train_clients = []
    # Assign random configs to training clients to simulate heterogeneity
    possible_configs = [['IMU', 'C1', 'C2'], ['IMU'], ['C1'], ['C2']]
    
    for i, sub_id in enumerate(train_data.keys()):
        X, y = train_data[sub_id]
        # Assign config cyclically
        config = possible_configs[i % len(possible_configs)]
        
        # Parse sensors
        active_sensors = []
        if 'IMU' in config: active_sensors.extend(['Ankle', 'Pocket', 'Belt', 'Neck', 'Wrist'])
        if 'C1' in config: active_sensors.append('Camera1')
        if 'C2' in config: active_sensors.append('Camera2')
        
        train_clients.append(Client(sub_id, X, y, global_model, active_sensors))

    # 3. Setup Test Clients (14-17) - Evaluate on ALL scenarios for robustness check
    test_clients = []
    test_scenarios = [
        ('Full', ['IMU', 'C1', 'C2']),
        ('IMU Only', ['IMU']),
        ('Cam Only', ['C1'])
    ]
    
    for sub_id in test_data.keys():
        X, y = test_data[sub_id]
        for name, config in test_scenarios:
             # Parse sensors
            active_sensors = []
            if 'IMU' in config: active_sensors.extend(['Ankle', 'Pocket', 'Belt', 'Neck', 'Wrist'])
            if 'C1' in config: active_sensors.append('Camera1')
            if 'C2' in config: active_sensors.append('Camera2')
            
            # Create a client just for evaluation
            client = Client(f"{sub_id}-{name}", X, y, global_model, active_sensors)
            test_clients.append(client)

    # 4. Training Loop
    for r in range(ROUNDS):
        local_weights = []
        for client in train_clients:
            client.model.load_state_dict(global_model.state_dict())
            w, loss = client.train(epochs=1)
            local_weights.append(w)
        global_model = fed_avg(global_model, local_weights)
        print(f"Round {r+1}/{ROUNDS} complete.")
        
    # 5. Final Evaluation on Test Clients
    results = []
    for client in test_clients:
        client.model.load_state_dict(global_model.state_dict())
        f1, acc, ent = client.evaluate()
        results.append([model_type, client.client_id, f"{f1:.4f}", f"{acc:.4f}", f"{ent:.4f}"])
        
    return results

if __name__ == "__main__":
    CSV_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv'
    IMG_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image'
    
    # Load Data
    train, test = load_upfall_split(CSV_PATH, IMG_PATH)
    if train is None:
        train, test = generate_synthetic_split()
        
    # Run Both Models
    res_fedmoe = run_simulation("FedMoE", train, test)
    res_conf = run_simulation("Conf-SMoE", train, test)
    
    # Print Comparison Table
    print("\n" + "="*80)
    print(f"FINAL COMPARISON: FedMoE vs Conf-SMoE (Test Subjects 14-17)")
    print("="*80)
    
    headers = ["Model", "Subject-Scenario", "F1 Score", "Accuracy", "Entropy"]
    col_widths = [12, 25, 12, 12, 12]
    
    header_row = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    print(header_row)
    print("-" * sum(col_widths))
    
    for row in res_fedmoe + res_conf:
        row_str = "".join([f"{str(item):<{w}}" for item, w in zip(row, col_widths)])
        print(row_str)