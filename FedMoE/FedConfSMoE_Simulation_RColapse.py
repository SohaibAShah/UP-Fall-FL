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
# For Conf-SMoE, we need input dims per modality group (IMU vs Vision)
# We treat IMU as one block (35 feats) and Cameras as separate blocks (1024 each)
# Or for simplicity, we treat the concatenated input as one vector but the confidence net takes the full vector
TOTAL_INPUT_DIM = sum(SENSOR_CONFIG.values()) # 2083

NUM_CLASSES = 12
NUM_EXPERTS = 4
TOP_K = 2
ROUNDS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. Data Loading Logic (Same as before)
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

def load_upfall_by_subject(file_path, image_path=None):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}. Using Synthetic Data.")
        return None

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=[0, 1])
    
    cleaned_columns = []
    last_val = ''
    for col_l1, col_l2 in df.columns:
        if 'Unnamed' in col_l1: col_l1 = last_val
        else: last_val = col_l1.strip(); col_l1 = last_val
        if col_l1 == col_l2.strip(): cleaned_columns.append(col_l1)
        else: cleaned_columns.append(f"{col_l1}_{col_l2.strip()}")
    df.columns = cleaned_columns

    subs = [1, 3, 4, 7, 10, 11, 12] 
    df = df[df['Subject'].isin(subs)].copy()
    
    df_cleaned = df.drop(columns=[col for col in df.columns if 'Infrared' in col] + ['Trial', 'Tag'], errors='ignore')
    df_cleaned.dropna(inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    
    cam_data = {}
    if image_path:
        for cam_id in [1, 2]:
            name_file = os.path.join(image_path, f'name_{cam_id}.npy')
            img_file = os.path.join(image_path, f'image_{cam_id}.npy')
            if os.path.exists(name_file) and os.path.exists(img_file):
                print(f"Loading Camera {cam_id} metadata/images...")
                timestamps = np.load(name_file)
                images = np.load(img_file)
                ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
                cam_data[f'Camera{cam_id}'] = (images, ts_to_idx)
            else:
                print(f"Warning: Camera {cam_id} files not found. Will simulate.")

    subject_data = {}
    
    for sub in subs:
        sub_df = df_cleaned[df_cleaned['Subject'] == sub].copy()
        valid_indices = sub_df.index.tolist()
        aligned_sensor_rows = []
        aligned_cam1_rows = []
        aligned_cam2_rows = []
        
        for idx, row in sub_df.iterrows():
            ts = row['TimeStamps_Time']
            c1_img = None
            if 'Camera1' in cam_data:
                images, lookup = cam_data['Camera1']
                if ts in lookup: c1_img = images[lookup[ts]].reshape(-1) / 255.0 
                else: continue 
            else:
                np.random.seed(int(ts[-5:].replace(':','')) if isinstance(ts, str) else int(ts)) 
                c1_img = np.random.randn(1024) 

            c2_img = None
            if 'Camera2' in cam_data:
                images, lookup = cam_data['Camera2']
                if ts in lookup: c2_img = images[lookup[ts]].reshape(-1) / 255.0
                else: continue 
            else:
                np.random.seed(int(ts[-5:].replace(':','')) + 1 if isinstance(ts, str) else int(ts) + 1)
                c2_img = np.random.randn(1024)

            aligned_sensor_rows.append(row)
            aligned_cam1_rows.append(c1_img)
            aligned_cam2_rows.append(c2_img)

        if not aligned_sensor_rows: continue
            
        sub_df_aligned = pd.DataFrame(aligned_sensor_rows)
        y = sub_df_aligned['Activity'].values
        y = np.where(y == 20, 0, y)

        sensor_features = []
        for sensor in ALL_SENSORS:
            if sensor == 'Camera1': X_sensor = np.array(aligned_cam1_rows)
            elif sensor == 'Camera2': X_sensor = np.array(aligned_cam2_rows)
            else:
                cols = get_sensor_columns(sub_df_aligned.columns, sensor)
                if not cols: X_sensor = np.zeros((len(sub_df_aligned), SENSOR_CONFIG[sensor]))
                else:
                    X_sensor = sub_df_aligned[cols].values
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
        subject_data[sub] = (X_all, y)
        
    return subject_data

def generate_synthetic_subject_data(num_subjects=7, num_samples=200):
    subject_data = {}
    for sub in range(num_subjects):
        X = np.random.randn(num_samples, TOTAL_INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, (num_samples,))
        subject_data[sub] = (X, y)
    return subject_data

# ==========================================
# 3. Conf-SMoE Model Implementation
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

class ConfidenceGatingNetwork(nn.Module):
    """
    Confidence-Guided Gating Network (Conf-SMoE).
    Instead of Raw Gating, it predicts a Confidence Score for the input
    and modulates the gating logits.
    """
    def __init__(self, input_dim, num_experts):
        super(ConfidenceGatingNetwork, self).__init__()
        # Standard Gating Layer
        self.gate = nn.Linear(input_dim, num_experts)
        
        # Confidence Estimator: Small MLP to predict scalar confidence
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() # Output [0, 1] confidence
        )

    def forward(self, x):
        # 1. Compute Raw Logits
        raw_logits = self.gate(x)
        
        # 2. Compute Confidence Score
        # (Is this input "good" or "missing/noise"?)
        confidence = self.confidence_net(x)
        
        # 3. Modulate Logits
        # If confidence is low, suppress the logits (make them uniform/small)
        # If confidence is high, keep logits sharp
        modulated_logits = raw_logits * confidence
        
        return modulated_logits, confidence

class ConfSMoE(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts=4, top_k=2):
        super(ConfSMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([Expert(input_dim, 64, num_classes) for _ in range(num_experts)])
        
        # Using the Confidence-Guided Router
        self.router = ConfidenceGatingNetwork(input_dim, num_experts)

    def forward(self, x):
        # 1. Get Modulated Logits & Confidence
        gate_logits, confidence = self.router(x)
        
        # 2. Top-k Selection
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=1)
        weights = F.softmax(weights, dim=1)
        
        # 3. Execution
        final_output = torch.zeros(x.size(0), self.experts[0].net[-1].out_features).to(x.device)
        gate_probs = F.softmax(gate_logits, dim=1)
        
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            weight = weights[:, i].unsqueeze(1)
            
            for b, idx in enumerate(expert_idx):
                input_sample = x[b].unsqueeze(0) 
                expert_out = self.experts[idx](input_sample).squeeze(0)
                final_output[b] += weight[b] * expert_out
                
        return final_output, gate_probs, confidence

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
# 4. Client Logic (Adapted for Conf-SMoE)
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

    def train(self, epochs=2):
        self.model.train()
        bs = min(32, len(self.dataset))
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=bs, shuffle=True)
        
        epoch_loss = 0
        epoch_entropy = 0
        epoch_conf = 0
        all_preds, all_targets = [], []
        batch_count = 0
        
        for epoch in range(epochs):
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                self.optimizer.zero_grad()
                
                # Forward with Confidence
                outputs, gate_probs, confidence = self.model(x)
                
                loss = F.cross_entropy(outputs, y)
                aux_loss = (gate_probs.sum(0)**2).sum() * 0.01
                
                # Confidence Loss: Encourage high confidence for valid data
                # (Optional: Usually Conf-SMoE has a regularization term)
                conf_loss = 0 # Simplified for demo
                
                (loss + aux_loss + conf_loss).backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-9), dim=1).mean().item()
                epoch_entropy += entropy
                epoch_conf += confidence.mean().item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
                batch_count += 1
        
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        return self.model.state_dict(), epoch_loss/max(1,batch_count), epoch_entropy/max(1,batch_count), f1, epoch_conf/max(1,batch_count)

# ==========================================
# 5. Main Execution: Router Collapse Check
# ==========================================

def run_conf_smoe_scenario(subject_data):
    print(f"\n{'='*60}")
    print(f"SCENARIO: Conf-SMoE Baseline Evaluation")
    print("Testing if Confidence-Guided Gating fixes Collapse")
    print(f"{'='*60}")
    
    global_model = ConfSMoE(TOTAL_INPUT_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K).to(device)
    clients = []
    
    subjects = list(subject_data.keys())
    
    client_configs = [
        ['IMU', 'C1', 'C2'], # Full 
        ['IMU', 'C1', 'C2'], # Full
        ['IMU'],             # Missing Cameras
        ['IMU'],             # Missing Cameras
        ['C1'],              # Missing IMU/C2
        ['C2'],              # Missing IMU/C1
        ['IMU', 'C1', 'C2']  # Full
    ]
    
    for i, sub_id in enumerate(subjects):
        if i >= len(client_configs): break
        X_full, y_full = subject_data[sub_id]
        config = client_configs[i]
        
        active_sensors = []
        if 'IMU' in config: active_sensors.extend(['Ankle', 'Pocket', 'Belt', 'Neck', 'Wrist'])
        if 'C1' in config: active_sensors.append('Camera1')
        if 'C2' in config: active_sensors.append('Camera2')
            
        config_label = " + ".join(config)
        clients.append(Client(sub_id, X_full, y_full, global_model, active_sensors))
        clients[-1].config_label = config_label

    final_results = []

    for r in range(ROUNDS):
        local_weights = []
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            w, loss, entropy, f1, conf = client.train(epochs=1)
            local_weights.append(w)
            
            if r == ROUNDS - 1:
                status = "STABLE" if f1 > 0.8 else "STRUGGLING"
                final_results.append([
                    f"Sub {client.client_id}",
                    client.config_label, 
                    f"{loss:.4f}",
                    f"{entropy:.4f}",
                    f"{conf:.4f}", # Added Confidence Metric
                    f"{f1:.4f}",
                    status
                ])
                
        global_model = fed_avg(global_model, local_weights)
        print(f"Round {r+1} complete...")

    print("\n" + "="*110)
    print(f"FINAL RESULTS: Conf-SMoE")
    print("="*110)
    
    headers = ["Client", "Config", "Loss", "Entropy", "Avg Conf", "F1 Score", "Status"]
    col_widths = [10, 20, 10, 10, 10, 10, 15] 
    
    header_row = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    print(header_row)
    print("-" * sum(col_widths))
    
    for row in final_results:
        row_str = "".join([f"{str(item):<{w}}" for item, w in zip(row, col_widths)])
        print(row_str)

if __name__ == "__main__":
    CSV_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv'
    IMG_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image'
    
    data = load_upfall_by_subject(CSV_PATH, IMG_PATH)
    if data is None:
        data = generate_synthetic_subject_data(num_subjects=7)
    
    run_conf_smoe_scenario(data)