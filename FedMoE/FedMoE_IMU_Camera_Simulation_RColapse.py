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

# ... (Configuration, Constants, Device, Data Loading functions remain the same as previous) ...
# ==========================================
# 1. Configuration & Constants
# ==========================================

# UP-Fall IMU structure: 7 features per sensor
# Added Cameras with 1024 features (32x32 flattened)
SENSOR_CONFIG = {
    'Ankle': 7,
    'Pocket': 7,
    'Belt': 7,
    'Neck': 7,
    'Wrist': 7,
    'Camera1': 1024,
    'Camera2': 1024
}

# Mapping of sensor names to column prefixes in your CSV (for IMUs)
SENSOR_COLUMNS_MAP = {
    'Ankle': ['AnkleAccelerometer', 'AnkleAngularVelocity', 'AnkleLuminosity'],
    'Pocket': ['RightPocketAccelerometer', 'RightPocketAngularVelocity', 'RightPocketLuminosity'],
    'Belt': ['BeltAccelerometer', 'BeltAngularVelocity', 'BeltLuminosity'],
    'Neck': ['NeckAccelerometer', 'NeckAngularVelocity', 'NeckLuminosity'],
    'Wrist': ['WristAccelerometer', 'WristAngularVelocity', 'WristLuminosity']
}

# ALL_SENSORS list includes both IMUs and Cameras
ALL_SENSORS = list(SENSOR_CONFIG.keys()) 
TOTAL_INPUT_DIM = sum(SENSOR_CONFIG.values()) # 35 (IMU) + 2048 (Cams) = 2083 features total

NUM_CLASSES = 12
NUM_EXPERTS = 4
TOP_K = 2
ROUNDS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. Data Loading Logic (Subject-Based & Aligned)
# ==========================================

def get_sensor_columns(df_columns, sensor_name):
    """Helper to find all actual columns for a sensor type."""
    if sensor_name not in SENSOR_COLUMNS_MAP: return [] # Cameras handled separately
    prefixes = SENSOR_COLUMNS_MAP[sensor_name]
    relevant_cols = []
    for col in df_columns:
        for prefix in prefixes:
            if prefix in col:
                relevant_cols.append(col)
                break
    return relevant_cols

def load_upfall_by_subject(file_path, image_path=None):
    """
    Loads UP-Fall data and splits it by SUBJECT.
    Performs alignment between Sensor CSV and Camera Images based on Timestamps.
    Returns a dictionary: {subject_id: (X_full, y_full)}
    X_full contains ALL sensor data concatenated (IMU + Cams).
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
    subs = [1, 3, 4, 7, 10, 11, 12] 
    df = df[df['Subject'].isin(subs)].copy()
    
    # Clean Data
    df_cleaned = df.drop(columns=[col for col in df.columns if 'Infrared' in col] + ['Trial', 'Tag'], errors='ignore')
    df_cleaned.dropna(inplace=True)
    df_cleaned.drop_duplicates(inplace=True)
    
    # --- Load Camera Metadata for Alignment ---
    cam_data = {}
    if image_path:
        for cam_id in [1, 2]:
            name_file = os.path.join(image_path, f'name_{cam_id}.npy')
            img_file = os.path.join(image_path, f'image_{cam_id}.npy')
            
            if os.path.exists(name_file) and os.path.exists(img_file):
                print(f"Loading Camera {cam_id} metadata/images...")
                # Load timestamps (names)
                timestamps = np.load(name_file)
                # Load images (This can be heavy, in production maybe load lazily)
                images = np.load(img_file)
                
                # Create a lookup dictionary for O(1) access: timestamp -> image_index
                # Handling potential duplicate timestamps in images by keeping first/last or checking mapping
                # For simplicity, we assume uniqueness or take the first
                ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}
                cam_data[f'Camera{cam_id}'] = (images, ts_to_idx)
            else:
                print(f"Warning: Camera {cam_id} files not found at {image_path}. Will simulate.")

    subject_data = {}
    
    for sub in subs:
        sub_df = df_cleaned[df_cleaned['Subject'] == sub].copy()
        
        # --- Alignment Step ---
        # We need rows where we have BOTH sensor data AND Camera data (if cameras exist)
        # to create a unified dataset.
        
        valid_indices = sub_df.index.tolist()
        
        # Filter valid indices based on Camera availability
        final_valid_rows = []
        
        # Lists to hold aligned data
        aligned_sensor_rows = []
        aligned_cam1_rows = []
        aligned_cam2_rows = []
        
        # Iterate through sensor rows to find matches in camera data
        for idx, row in sub_df.iterrows():
            ts = row['TimeStamps_Time']
            
            # Check Cam 1
            c1_img = None
            if 'Camera1' in cam_data:
                images, lookup = cam_data['Camera1']
                if ts in lookup:
                    c1_img = images[lookup[ts]].reshape(-1) / 255.0 # Flatten & Normalize
                else:
                    continue # Skip if no image for this timestamp
            else:
                # Simulate if file missing
                np.random.seed(int(ts[-5:].replace(':','')) if isinstance(ts, str) else int(ts)) 
                c1_img = np.random.randn(1024) 

            # Check Cam 2
            c2_img = None
            if 'Camera2' in cam_data:
                images, lookup = cam_data['Camera2']
                if ts in lookup:
                    c2_img = images[lookup[ts]].reshape(-1) / 255.0
                else:
                    continue # Skip
            else:
                # Simulate
                np.random.seed(int(ts[-5:].replace(':','')) + 1 if isinstance(ts, str) else int(ts) + 1)
                c2_img = np.random.randn(1024)

            # If we reached here, we have data for this timestamp
            aligned_sensor_rows.append(row)
            aligned_cam1_rows.append(c1_img)
            aligned_cam2_rows.append(c2_img)

        if not aligned_sensor_rows:
            print(f"Subject {sub}: No aligned data found. Skipping.")
            continue
            
        # Reconstruct DataFrame from aligned rows
        sub_df_aligned = pd.DataFrame(aligned_sensor_rows)
        y = sub_df_aligned['Activity'].values
        
        # Fix Labels (Label 20 -> 0)
        y = np.where(y == 20, 0, y)

        # Extract features per sensor to maintain order
        sensor_features = []
        for sensor in ALL_SENSORS:
            if sensor == 'Camera1':
                X_sensor = np.array(aligned_cam1_rows)
            elif sensor == 'Camera2':
                X_sensor = np.array(aligned_cam2_rows)
            else:
                # IMU Data
                cols = get_sensor_columns(sub_df_aligned.columns, sensor)
                if not cols:
                    # print(f"Warning: Sensor {sensor} columns not found for Subject {sub}. Padding.")
                    X_sensor = np.zeros((len(sub_df_aligned), SENSOR_CONFIG[sensor]))
                else:
                    X_sensor = sub_df_aligned[cols].values
                    # Fix dimension mismatch if any
                    target_dim = SENSOR_CONFIG[sensor]
                    if X_sensor.shape[1] < target_dim:
                        padding = np.zeros((X_sensor.shape[0], target_dim - X_sensor.shape[1]))
                        X_sensor = np.hstack((X_sensor, padding))
                    elif X_sensor.shape[1] > target_dim:
                        X_sensor = X_sensor[:, :target_dim]
                
                # Standardize IMU data only
                scaler = StandardScaler()
                X_sensor = scaler.fit_transform(X_sensor)

            sensor_features.append(X_sensor)
            
        # Concatenate all sensors horizontally: [Ankle, ..., Wrist, Cam1, Cam2]
        X_all = np.hstack(sensor_features)
        
        # Standardize (Only standardize non-zero columns effectively)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        
        subject_data[sub] = (X_scaled, y)
        
    return subject_data

def generate_synthetic_subject_data(num_subjects=7, num_samples=200):
    """Fallback if CSV is missing."""
    subject_data = {}
    for sub in range(num_subjects):
        X = np.random.randn(num_samples, TOTAL_INPUT_DIM).astype(np.float32)
        y = np.random.randint(0, NUM_CLASSES, (num_samples,))
        subject_data[sub] = (X, y)
    return subject_data

# ... (Expert, GatingNetwork, FedMoE classes remain the same as previous) ...
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

# ... (Client class remains the same) ...
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
            
            # Check if this sensor is active for this client
            is_active = False
            if sensor in active_sensors:
                is_active = True
            # Handle group names like 'IMU' which means all IMU sensors
            elif 'IMU' in active_sensors and sensor not in ['Camera1', 'Camera2']:
                is_active = True
            elif 'C1' in active_sensors and sensor == 'Camera1':
                is_active = True
            elif 'C2' in active_sensors and sensor == 'Camera2':
                is_active = True
            
            if not is_active:
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
        correct_preds = 0
        total_samples = 0
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
                
                correct_preds += (preds == y).sum().item()
                total_samples += y.size(0)
                
                batch_count += 1
        
        accuracy = correct_preds / max(1, total_samples)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        return self.model.state_dict(), epoch_loss/max(1,batch_count), epoch_entropy/max(1,batch_count), f1, accuracy


# ==========================================
# 5. Main Execution: Router Collapse Scenario
# ==========================================

def run_router_collapse_scenario(subject_data):
    print(f"\n{'='*60}")
    print(f"SCENARIO: Router Collapse Demonstration")
    print("Strategy: Train on Full-Modality Clients -> Introduce Missing Modalities -> Observe Collapse")
    print(f"{'='*60}")
    
    global_model = FedMoE(TOTAL_INPUT_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K).to(device)
    clients = []
    
    subjects = list(subject_data.keys())
    
    # 1. Define Client Configs (Mix of Full and Missing)
    # We deliberately give some clients missing cameras to see them fail compared to full clients
    client_configs = [
        ['IMU', 'C1', 'C2'], # Full (Control)
        ['IMU', 'C1', 'C2'], # Full (Control)
        ['IMU'],             # Missing Cameras (Test for Collapse)
        ['IMU'],             # Missing Cameras (Test for Collapse)
        ['C1'],              # Missing IMU/C2
        ['C2'],              # Missing IMU/C1
        ['IMU', 'C1', 'C2']  # Full (Control)
    ]
    
    print(f"[DEBUG] creating clients with specific configs...")
    
    for i, sub_id in enumerate(subjects):
        if i >= len(client_configs): break # Just use as many subjects as configs we defined
        
        X_full, y_full = subject_data[sub_id]
        config = client_configs[i]
        
        # Parse Config
        active_sensors = []
        if 'IMU' in config:
            active_sensors.extend(['Ankle', 'Pocket', 'Belt', 'Neck', 'Wrist'])
        if 'C1' in config:
            active_sensors.append('Camera1')
        if 'C2' in config:
            active_sensors.append('Camera2')
            
        config_label = " + ".join(config)
        clients.append(Client(sub_id, X_full, y_full, global_model, active_sensors))
        clients[-1].config_label = config_label

    final_results = []

    # Training Loop
    for r in range(ROUNDS):
        local_weights = []
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            w, loss, entropy, f1, acc = client.train(epochs=1)
            local_weights.append(w)
            
            if r == ROUNDS - 1:
                # Logic for status
                if entropy > 1.0: # Stricter entropy threshold to show subtle collapse
                    status = "COLLAPSED (High Entropy)"
                elif f1 < 0.6:
                     status = "POOR PERF"
                else:
                    status = "STABLE"
                    
                final_results.append([
                    f"Sub {client.client_id}",
                    client.config_label, 
                    f"{loss:.4f}",
                    f"{entropy:.4f}",
                    f"{f1:.4f}",
                    f"{acc:.4f}",
                    status
                ])
                
        global_model = fed_avg(global_model, local_weights)
        print(f"Round {r+1} complete...")

    print("\n" + "="*100)
    print(f"FINAL RESULTS: Router Collapse Demonstration")
    print("="*100)
    
    headers = ["Client", "Config", "Loss", "Entropy", "F1 Score", "Accuracy", "Status"]
    col_widths = [10, 20, 10, 10, 10, 10, 25] 
    
    header_row = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    print(header_row)
    print("-" * sum(col_widths))
    
    for row in final_results:
        row_str = "".join([f"{str(item):<{w}}" for item, w in zip(row, col_widths)])
        print(row_str)

if __name__ == "__main__":
    CSV_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv'
    IMG_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image'
    
    # Load Data
    data = load_upfall_by_subject(CSV_PATH, IMG_PATH)
    if data is None:
        print("Falling back to Synthetic Data...")
        data = generate_synthetic_subject_data(num_subjects=7)
    
    run_router_collapse_scenario(data)