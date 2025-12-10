import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. UP-Fall Data Loading Logic
# ==========================================

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def loadSensorIMGClientsData(file_path, image_data_path):
    """
    Loads and processes UP-Fall data: Sensor clients (0-5) and Camera clients (6-7).
    """
    
    subs = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    print(f"Loading data for subjects: {subs}")

    # --- Part 1: Load and Create Sensor Clients ---
    # Note: Using header=[0, 1] as per your original script logic
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sensor file not found at: {file_path}")
        
    df = pd.read_csv(file_path, header=[0, 1])

    # Cleaning Columns
    cleaned_columns = []
    last_val = ''
    for col_l1, col_l2 in df.columns:
        if 'Unnamed' in col_l1: col_l1 = last_val
        else: last_val = col_l1.strip(); col_l1 = last_val
        if col_l1 == col_l2.strip(): cleaned_columns.append(col_l1)
        else: cleaned_columns.append(f"{col_l1}_{col_l2.strip()}")
    df.columns = cleaned_columns
    
    df = df[df['Subject'].isin(subs)].copy()
    # Filter specific subject/activity exclusion as per original script
    df = df[~((df['Subject'] == 8) & (df['Activity'] == 11) & (df['Trial'].isin([2, 3])))]

    df_cleaned = df.drop(columns=[col for col in df.columns if 'Infrared' in col] + ['Trial', 'Tag'], errors='ignore')
    df_cleaned.dropna(inplace=True)
    df_cleaned.drop_duplicates(inplace=True)

    train_subjects = [s for s in subs if s <= 13]
    test_subjects = [s for s in subs if s >= 14]
    
    train_df_sensor = df_cleaned[df_cleaned['Subject'].isin(train_subjects)].copy()
    test_df_sensor = df_cleaned[df_cleaned['Subject'].isin(test_subjects)].copy()

    sensor_clients = {
        'Ankle_IMU': ['AnkleAccelerometer_x-axis (g)', 'AnkleAccelerometer_y-axis (g)', 'AnkleAccelerometer_z-axis (g)', 'AnkleAngularVelocity_x-axis (deg/s)', 'AnkleAngularVelocity_y-axis (deg/s)', 'AnkleAngularVelocity_z-axis (deg/s)', 'AnkleLuminosity_illuminance (lx)'],
        'Pocket_IMU': ['RightPocketAccelerometer_x-axis (g)', 'RightPocketAccelerometer_y-axis (g)', 'RightPocketAccelerometer_z-axis (g)', 'RightPocketAngularVelocity_x-axis (deg/s)', 'RightPocketAngularVelocity_y-axis (deg/s)', 'RightPocketAngularVelocity_z-axis (deg/s)', 'RightPocketLuminosity_illuminance (lx)'],
        'Belt_IMU': ['BeltAccelerometer_x-axis (g)', 'BeltAccelerometer_y-axis (g)', 'BeltAccelerometer_z-axis (g)', 'BeltAngularVelocity_x-axis (deg/s)', 'BeltAngularVelocity_y-axis (deg/s)', 'BeltAngularVelocity_z-axis (deg/s)', 'BeltLuminosity_illuminance (lx)'],
        'Neck_IMU': ['NeckAccelerometer_x-axis (g)', 'NeckAccelerometer_y-axis (g)', 'NeckAccelerometer_z-axis (g)', 'NeckAngularVelocity_x-axis (deg/s)', 'NeckAngularVelocity_y-axis (deg/s)', 'NeckAngularVelocity_z-axis (deg/s)', 'NeckLuminosity_illuminance (lx)'],
        'Wrist_IMU': ['WristAccelerometer_x-axis (g)', 'WristAccelerometer_y-axis (g)', 'WristAccelerometer_z-axis (g)', 'WristAngularVelocity_x-axis (deg/s)', 'WristAngularVelocity_y-axis (deg/s)', 'WristAngularVelocity_z-axis (deg/s)', 'WristLuminosity_illuminance (lx)'],
        'EEG': ['BrainSensor']
    }
    
    X_train_splits, X_test_splits, Y_train_splits, Y_test_splits = {}, {}, {}, {}
    num_classes = 12 # UP-Fall classes

    for client_index, (client_name, columns) in enumerate(sensor_clients.items()):
        X_train = train_df_sensor[columns].values
        y_train = train_df_sensor['Activity'].values
        X_test = test_df_sensor[columns].values
        y_test = test_df_sensor['Activity'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        set_seed()
        Y_train = torch.nn.functional.one_hot(torch.from_numpy(y_train).long(), num_classes).float()
        Y_test = torch.nn.functional.one_hot(torch.from_numpy(y_test).long(), num_classes).float()
        
        X_train_splits[client_index], X_test_splits[client_index] = X_train_scaled, X_test_scaled
        Y_train_splits[client_index], Y_test_splits[client_index] = Y_train, Y_test

    # --- Part 2: Timestamps for Alignment ---
    final_train_times = set(train_df_sensor['TimeStamps_Time'])
    final_test_times = set(test_df_sensor['TimeStamps_Time'])

    # --- Part 3: Image Clients (6-7) ---
    def align_and_create_camera_client(camera_id):
        img_path = f'{image_data_path}/image_{camera_id}.npy'
        name_path = f'{image_data_path}/name_{camera_id}.npy'
        label_path = f'{image_data_path}/label_{camera_id}.npy'
        
        if not os.path.exists(img_path):
            print(f"Warning: Image data not found at {img_path}. Creating dummy data for simulation.")
            # Dummy data fallback for testing without full dataset
            return np.random.rand(100, 32, 32, 1), torch.zeros(100, 12), np.random.rand(20, 32, 32, 1), torch.zeros(20, 12)

        img = np.load(img_path)
        name = np.load(name_path)
        label = np.load(label_path)

        train_indices = [i for i, t in enumerate(name) if t in final_train_times]
        test_indices = [i for i, t in enumerate(name) if t in final_test_times]
        
        X_train_img, y_train_img = img[train_indices], label[train_indices]
        X_test_img, y_test_img = img[test_indices], label[test_indices]
        
        # Flatten images to (N, 1024) for standard MLP/MoE input
        X_train_scaled = X_train_img.reshape(X_train_img.shape[0], -1) / 255.0
        X_test_scaled = X_test_img.reshape(X_test_img.shape[0], -1) / 255.0
        
        # Fix label 20 issue
        y_train_final = np.where(y_train_img == 20, 0, y_train_img)
        y_test_final = np.where(y_test_img == 20, 0, y_test_img)

        Y_train_final = torch.nn.functional.one_hot(torch.from_numpy(y_train_final.flatten()).long(), num_classes).float()
        Y_test_final = torch.nn.functional.one_hot(torch.from_numpy(y_test_final.flatten()).long(), num_classes).float()
        
        return X_train_scaled, Y_train_final, X_test_scaled, Y_test_final

    # Client 6 (Camera 1)
    X_train_cam1, Y_train_cam1, X_test_cam1, Y_test_cam1 = align_and_create_camera_client(1)
    X_train_splits[6], Y_train_splits[6] = X_train_cam1, Y_train_cam1
    X_test_splits[6], Y_test_splits[6] = X_test_cam1, Y_test_cam1
    
    # Client 7 (Camera 2)
    X_train_cam2, Y_train_cam2, X_test_cam2, Y_test_cam2 = align_and_create_camera_client(2)
    X_train_splits[7], Y_train_splits[7] = X_train_cam2, Y_train_cam2
    X_test_splits[7], Y_test_splits[7] = X_test_cam2, Y_test_cam2

    return X_train_splits, Y_train_splits

# ==========================================
# 2. Define the Expert and Gating Networks
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

# ==========================================
# 3. The FedMoE Model Architecture
# ==========================================

class FedMoE(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts=4, top_k=2):
        super(FedMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Standardize Expert Size
        # Note: input_dim will be the padded dimension (1024 for UP-Fall)
        self.experts = nn.ModuleList([
            Expert(input_dim, 64, num_classes) for _ in range(num_experts)
        ])
        self.router = GatingNetwork(input_dim, num_experts)

    def forward(self, x):
        batch_size = x.size(0)
        gate_logits = self.router(x)
        weights, selected_experts = torch.topk(gate_logits, self.top_k, dim=1)
        weights = F.softmax(weights, dim=1)
        final_output = torch.zeros(batch_size, self.experts[0].net[-1].out_features).to(x.device)
        gate_probs = F.softmax(gate_logits, dim=1)
        
        for i in range(self.top_k):
            expert_idx = selected_experts[:, i]
            weight = weights[:, i].unsqueeze(1)
            
            # Process each sample with its selected expert
            # (Loop is for simulation clarity; optimizations exist)
            expert_outputs_list = []
            for b, idx in enumerate(expert_idx):
                expert_outputs_list.append(self.experts[idx](x[b].unsqueeze(0)))
            
            expert_outputs = torch.cat(expert_outputs_list, dim=0)
            final_output += weight * expert_outputs

        return final_output, gate_probs

def load_balancing_loss(gate_probs, num_experts):
    importance = gate_probs.sum(0)
    loss = (num_experts * torch.sum(importance**2) / (torch.sum(importance)**2))
    return loss

# ==========================================
# 4. Federated Client Logic
# ==========================================

class Client:
    def __init__(self, client_id, dataset, model):
        self.client_id = client_id
        self.dataset = dataset
        self.model = copy.deepcopy(model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def train(self, epochs=1):
        self.model.train()
        # Ensure batch size doesn't exceed dataset size
        bs = min(16, len(self.dataset))
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=bs, shuffle=True)
        
        epoch_loss = 0
        for epoch in range(epochs):
            batch_count = 0
            for x, y in dataloader:
                self.optimizer.zero_grad()
                
                # y comes as one-hot from loader, convert to class indices for CrossEntropy
                if y.dim() > 1:
                    y_indices = torch.argmax(y, dim=1)
                else:
                    y_indices = y.long()

                outputs, gate_probs = self.model(x)
                
                cls_loss = F.cross_entropy(outputs, y_indices)
                aux_loss = load_balancing_loss(gate_probs, self.model.num_experts)
                total_loss = cls_loss + 0.01 * aux_loss 
                
                total_loss.backward()
                self.optimizer.step()
                epoch_loss += total_loss.item()
                batch_count += 1
        
        return self.model.state_dict(), epoch_loss / max(1, batch_count)

# ==========================================
# 5. Server Logic (FedAvg)
# ==========================================

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
# 6. Main Simulation Loop with UP-Fall Data
# ==========================================

if __name__ == "__main__":
    print("--- Starting FedMoE Simulation with UP-Fall Data ---")
    
    # Configuration
    # UP-Fall Max Dimension is 32*32=1024 (Images). Sensor data is ~7.
    # To run a Standard MoE baseline, we must PAD sensor data to 1024.
    INPUT_DIM = 1024 
    NUM_CLASSES = 12 
    NUM_EXPERTS = 4
    TOP_K = 2
    ROUNDS = 5
    
    # 1. Load Real Data
    # Update paths as per your environment
    CSV_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv' 
    IMG_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image'
    
    # Check if files exist, else warn (to prevent crash in dry-run)
    if os.path.exists(CSV_PATH):
        X_splits, Y_splits = loadSensorIMGClientsData(CSV_PATH, IMG_PATH)
        NUM_CLIENTS = len(X_splits)
        print(f"Loaded {NUM_CLIENTS} clients from UP-Fall.")
    else:
        print("Data files not found. Generating Dummy Data for structural verification.")
        NUM_CLIENTS = 8
        X_splits, Y_splits = {}, {}
        for i in range(NUM_CLIENTS):
            if i < 6: # Sensor clients (small dim)
                X_splits[i] = np.random.rand(50, 7)
            else: # Camera clients (large dim)
                X_splits[i] = np.random.rand(50, 1024)
            Y_splits[i] = torch.nn.functional.one_hot(torch.randint(0, NUM_CLASSES, (50,)), NUM_CLASSES)

    # 2. Create Global Model
    global_model = FedMoE(INPUT_DIM, NUM_CLASSES, NUM_EXPERTS, TOP_K)
    
    # 3. Setup Clients with Padding Logic
    clients = []
    for cid in range(NUM_CLIENTS):
        X_data = X_splits[cid]
        Y_data = Y_splits[cid]
        
        # --- PADDING STRATEGY FOR BASELINE B ---
        # If input dim < 1024 (Sensor Client), pad with zeros.
        # This simulates "Missing Modality" (Image is missing/zero).
        if X_data.shape[1] < INPUT_DIM:
            padding = np.zeros((X_data.shape[0], INPUT_DIM - X_data.shape[1]))
            X_data_padded = np.hstack((X_data, padding))
            print(f"Client {cid}: Padded input from {X_data.shape[1]} to {INPUT_DIM}.")
        else:
            X_data_padded = X_data
            
        # Convert to Tensor
        X_tensor = torch.from_numpy(X_data_padded).float()
        # Y is likely already tensor from loader, ensure float
        if not isinstance(Y_data, torch.Tensor):
            Y_tensor = torch.from_numpy(Y_data).float()
        else:
            Y_tensor = Y_data.float()
            
        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        clients.append(Client(cid, dataset, global_model))

    # 4. Training Loop
    for r in range(ROUNDS):
        print(f"\n--- Round {r+1} ---")
        local_weights = []
        
        for client in clients:
            client.model.load_state_dict(global_model.state_dict())
            
            # Train
            w, loss = client.train(epochs=1)
            local_weights.append(w)
            
            print(f"Client {client.client_id} Loss: {loss:.4f}")
            
        global_model = fed_avg(global_model, local_weights)

    print("\nSimulation Finished.")