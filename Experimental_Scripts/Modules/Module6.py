import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os
import logging
import sys
import copy

# --- 1. Setup and Helper Functions ---

def setup_logging(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, "fl_module6_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- 2. Data Loading ---

def load_and_create_clients(data_dir):
    logging.info("Loading and processing data for federated clients...")
    sub = pd.read_csv(os.path.join(data_dir, 'sensor.csv'), header=[0, 1])
    cleaned_columns = [f"{c[0].strip()}_{c[1].strip()}" if 'Unnamed' not in c[0] else last_val + f"_{c[1].strip()}" for c in sub.columns if 'Unnamed' not in c[0] or (last_val := c[0].strip())]
    sub.columns = [c.replace(c.split('_')[0] + '_', '') if c.split('_')[0] == c.split('_')[1] else c for c in cleaned_columns]
    sub.dropna(inplace=True); sub.drop_duplicates(inplace=True)
    
    img1 = np.load(os.path.join(data_dir, 'image_1.npy')); name1 = np.load(os.path.join(data_dir, 'name_1.npy'))
    img2 = np.load(os.path.join(data_dir, 'image_2.npy'))
    
    sensor_ts = set(sub['TimeStamps_Time']); name1_ts = set(name1)
    common_timestamps = sorted(list(sensor_ts.intersection(name1_ts)))
    
    sub = sub[sub['TimeStamps_Time'].isin(common_timestamps)]
    
    # Create a mapping for faster lookup
    img1_map = {ts: img for ts, img in zip(name1, img1)}
    img2_map = {ts: img for ts, img in zip(name1, img2)} # Assume name1 and name2 are aligned
    
    # Filter the DataFrame and align its index
    sub = sub.set_index('TimeStamps_Time').loc[common_timestamps]
    
    fall_activity_ids = {2, 3, 4, 5, 6}
    sub['Fall'] = sub['Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
    
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
    test_subjects = [s for s in range(14, 18)]
    train_df = sub[sub['Subject'].isin(train_subjects)]
    test_df = sub[sub['Subject'].isin(test_subjects)]
    
    feature_cols = [col for col in sub.columns if 'Infrared' not in col and col not in ['Activity', 'Subject', 'Trial', 'Tag', 'Fall']]
    scaler = StandardScaler().fit(train_df[feature_cols])
    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    # --- Create global test set ---
    X_test_csv = test_df[feature_cols].values
    y_test = test_df['Fall'].values

    # **FIX:** Stack the array of objects into a single multi-dimensional array
    img1_series = test_df.index.map(img1_map)
    stacked_img1 = np.stack(img1_series.values)
    X_test_img1 = np.expand_dims((stacked_img1 / 255.0).astype(np.float32), axis=1)

    img2_series = test_df.index.map(img2_map)
    stacked_img2 = np.stack(img2_series.values)
    X_test_img2 = np.expand_dims((stacked_img2 / 255.0).astype(np.float32), axis=1)
    
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_csv).float(), torch.from_numpy(X_test_img1).float(), torch.from_numpy(X_test_img2).float(), torch.from_numpy(y_test).long()), batch_size=128)

    # --- Create clients (subject-wise) ---
    clients = {}
    for client_id in train_subjects:
        client_df = train_df[train_df['Subject'] == client_id]
        X_csv_client = client_df[feature_cols].values
        y_client = client_df['Fall'].values
        
        # **FIX:** Apply the same stacking logic for each client
        client_img1_series = client_df.index.map(img1_map)
        client_stacked_img1 = np.stack(client_img1_series.values)
        X_img1_client = np.expand_dims((client_stacked_img1 / 255.0).astype(np.float32), axis=1)

        client_img2_series = client_df.index.map(img2_map)
        client_stacked_img2 = np.stack(client_img2_series.values)
        X_img2_client = np.expand_dims((client_stacked_img2 / 255.0).astype(np.float32), axis=1)
        
        clients[client_id] = (X_csv_client, X_img1_client, X_img2_client, y_client)
        
    logging.info(f"Data loading complete. {len(clients)} clients created.")
    return clients, test_loader, len(feature_cols)


# --- 3. Model Architecture (GatedResidualFusionModel from Module 5) ---
class IMUEncoder(nn.Module):
    def __init__(self, input_channels, feature_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 3, padding=1); self.conv2 = nn.Conv1d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(64, feature_dim)
    def forward(self, x):
        x = x.unsqueeze(2); x = F.relu(self.conv1(x)); x = F.relu(self.conv2(x))
        return self.fc(self.pool(x).squeeze(2))

class ImageEncoder(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1); self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1)); self.fc = nn.Linear(32, feature_dim)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)); x = F.relu(F.max_pool2d(self.conv2(x), 2))
        return self.fc(self.pool(x).view(x.size(0), -1))

class GatedResidualFusionModel(nn.Module):
    def __init__(self, num_csv_features, num_classes=2):
        super().__init__()
        self.imu_encoder = IMUEncoder(num_csv_features, 128)
        self.img_encoder1 = ImageEncoder(64); self.img_encoder2 = ImageEncoder(64)
        self.img_fusion = nn.Linear(64 + 64, 128)
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.fused_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.imu_only_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x_csv, x_img1, x_img2, threshold=0.5):
        f_csv = self.imu_encoder(x_csv)
        gate_logit = self.gate(f_csv)
        gate_prob = torch.sigmoid(gate_logit)
        if self.training:
            f_img1 = self.img_encoder1(x_img1); f_img2 = self.img_encoder2(x_img2)
            f_img_combined = F.relu(self.img_fusion(torch.cat((f_img1, f_img2), dim=1)))
            fused = f_csv + f_img_combined
            out_fused = self.fused_classifier(fused)
            out_imu_only = self.imu_only_classifier(f_csv)
            return gate_prob * out_fused + (1 - gate_prob) * out_imu_only
        else:
            use_images = (gate_prob > threshold).float()
            if use_images.sum() > 0:
                f_img1 = self.img_encoder1(x_img1); f_img2 = self.img_encoder2(x_img2)
                f_img_combined = F.relu(self.img_fusion(torch.cat((f_img1, f_img2), dim=1)))
                fused = f_csv + f_img_combined
                out_fused = self.fused_classifier(fused)
            else:
                out_fused = torch.zeros(f_csv.size(0), 2, device=f_csv.device)
            out_imu_only = self.imu_only_classifier(f_csv)
            return use_images * out_fused + (1 - use_images) * out_imu_only

# --- 4. Federated Learning Components ---

class Client:
    def __init__(self, client_id, data, device, completeness_prob=1.0):
        self.id = client_id
        self.device = device
        X_csv, X_img1, X_img2, y = data
        self.completeness = 1.0 if np.random.rand() < completeness_prob else 0.5 # Simulate missing camera
        if self.completeness < 1.0:
            X_img1.fill(0); X_img2.fill(0) # Zero out image data if incomplete
        self.loader = DataLoader(TensorDataset(torch.from_numpy(X_csv).float(), torch.from_numpy(X_img1).float(), torch.from_numpy(X_img2).float(), torch.from_numpy(y).long()), batch_size=64, shuffle=True)
        self.data_size = len(y)
        self.diversity = np.mean(np.var(X_csv, axis=0)) # Proxy for feature diversity

    def train(self, model, config):
        local_model = copy.deepcopy(model).to(self.device)
        local_model.train()
        optimizer = torch.optim.Adam(local_model.parameters(), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        
        total_loss, total_correct, total_samples = 0, 0, 0
        for _ in range(config['local_epochs']):
            for x_csv_b, x_img1_b, x_img2_b, y_b in self.loader:
                if np.random.rand() < config['dropout_prob']:
                    x_img1_b.zero_(); x_img2_b.zero_()
                optimizer.zero_grad()
                outputs = local_model(x_csv_b.to(self.device), x_img1_b.to(self.device), x_img2_b.to(self.device))
                loss = criterion(outputs, y_b.to(self.device))
                loss.backward(); optimizer.step()
                total_loss += loss.item() * y_b.size(0)
                total_correct += (torch.argmax(outputs, dim=1) == y_b.to(self.device)).sum().item()
                total_samples += y_b.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        return local_model.state_dict(), avg_loss, avg_acc

def select_pareto_optimal_clients(metrics, num_to_select):
    """Selects clients from the Pareto front."""
    client_ids = list(metrics.keys())
    metric_values = np.array([list(m.values()) for m in metrics.values()])
    
    # Normalize metrics (loss is minimized, others maximized)
    metric_values[:, 0] = 1 - (metric_values[:, 0] / np.max(metric_values[:, 0]))
    for i in range(1, metric_values.shape[1]):
        if np.max(metric_values[:, i]) > 0:
            metric_values[:, i] /= np.max(metric_values[:, i])
            
    # Find the Pareto front
    is_dominated = np.zeros(len(client_ids), dtype=bool)
    for i in range(len(client_ids)):
        for j in range(len(client_ids)):
            if i == j: continue
            if np.all(metric_values[j] >= metric_values[i]) and np.any(metric_values[j] > metric_values[i]):
                is_dominated[i] = True
                break
    
    pareto_front_indices = np.where(~is_dominated)[0]
    pareto_client_ids = [client_ids[i] for i in pareto_front_indices]
    
    # Subsample if the front is larger than the budget
    if len(pareto_client_ids) > num_to_select:
        return np.random.choice(pareto_client_ids, num_to_select, replace=False).tolist()
    return pareto_client_ids

def evaluate_global_model(model, test_loader, device):
    model.to(device); model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_csv, x_img1, x_img2, y in test_loader:
            outputs = model(x_csv.to(device), x_img1.to(device), x_img2.to(device), threshold=0.5)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
    return f1_score(all_labels, all_preds, pos_label=0, zero_division=0)

# --- 5. Main Simulation Runner ---

def run_federated_simulation(clients_data, test_loader, config, num_features):
    set_seed(42)
    global_model = GatedResidualFusionModel(num_features)
    client_objects = {cid: Client(cid, data, config['device'], completeness_prob=0.8) for cid, data in clients_data.items()}
    
    history = {'round': [], 'f1_score': []}
    
    for round_num in range(1, config['comm_rounds'] + 1):
        logging.info(f"--- Round {round_num}/{config['comm_rounds']} ---")
        
        # 1. All clients train locally to generate metrics
        client_metrics = {}
        client_updates = {}
        for cid, client in client_objects.items():
            updated_weights, loss, acc = client.train(global_model, config)
            client_updates[cid] = {'weights': updated_weights, 'data_size': client.data_size}
            client_metrics[cid] = {'loss': loss, 'completeness': client.completeness, 'diversity': client.diversity, 'data_size': client.data_size}
            logging.info(f"  > Pre-selection | Client {cid:2d}: Loss={loss:.4f}, Acc={acc:.4f}, Completeness={client.completeness:.1f}")

        # 2. Server selects clients
        if config['selection_method'] == 'pareto':
            selected_ids = select_pareto_optimal_clients(client_metrics, config['clients_per_round'])
        else: # random
            selected_ids = np.random.choice(list(client_objects.keys()), config['clients_per_round'], replace=False).tolist()
        logging.info(f"  > Selected clients ({config['selection_method']}): {selected_ids}")

        # 3. Server aggregates updates from selected clients
        global_dict = global_model.state_dict()
        total_data_size = sum(client_objects[cid].data_size for cid in selected_ids)
        for k in global_dict.keys():
            if total_data_size > 0:
                global_dict[k] = torch.stack([client_updates[cid]['weights'][k].float() * client_updates[cid]['data_size'] for cid in selected_ids], 0).sum(0) / total_data_size
        global_model.load_state_dict(global_dict)
        
        # 4. Evaluate global model
        f1 = evaluate_global_model(global_model, test_loader, config['device'])
        history['round'].append(round_num); history['f1_score'].append(f1)
        logging.info(f"  > Global F1-Score after aggregation: {f1:.4f}\n")

    return pd.DataFrame(history)

# --- 6. Main Execution Block ---

if __name__ == "__main__":
    OUTPUT_DIR = '/home/syed/PhD/UP-Fall-FL/Experimental_Scripts/Modules/output/Module6'
    os.makedirs(OUTPUT_DIR, exist_ok=True); setup_logging(OUTPUT_DIR)
    
    config = {
        'comm_rounds': 20, 'local_epochs': 3, 'clients_per_round': 4,
        'learning_rate': 0.001, 'dropout_prob': 0.3,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    clients, test_loader, num_features = load_and_create_clients('/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/')

    # Run experiments
    logging.info("\n--- Running Experiment 1: Random Selection ---")
    config_random = {**config, 'selection_method': 'random'}
    history_random = run_federated_simulation(clients, test_loader, config_random, num_features)
    
    logging.info("\n--- Running Experiment 2: Pareto Selection ---")
    config_pareto = {**config, 'selection_method': 'pareto'}
    history_pareto = run_federated_simulation(clients, test_loader, config_pareto, num_features)

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    plt.plot(history_random['round'], history_random['f1_score'], 'o--', label='Random Selection')
    plt.plot(history_pareto['round'], history_pareto['f1_score'], 's-', label='Pareto Selection')
    plt.title('Global Model F1-Score vs. Communication Rounds')
    plt.xlabel('Communication Round'); plt.ylabel('Global Test F1-Score (Fall Class)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'f1_vs_rounds_comparison.png')); plt.show()
    
    logging.info(f"\n✅ All simulations complete. Results saved in '{OUTPUT_DIR}'.")