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
    file_handler = logging.FileHandler(os.path.join(log_dir, "fl_module7_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# --- 2. Data Loading ---
def load_and_create_clients(data_dir):
    logging.info("Step 1: Loading and processing data for federated clients...")
    sub = pd.read_csv(os.path.join(data_dir, 'sensor.csv'), header=[0, 1])
    cleaned_columns = [f"{c[0].strip()}_{c[1].strip()}" if 'Unnamed' not in c[0] else last_val + f"_{c[1].strip()}" for c in sub.columns if 'Unnamed' not in c[0] or (last_val := c[0].strip())]
    sub.columns = [c.replace(c.split('_')[0] + '_', '') if c.split('_')[0] == c.split('_')[1] else c for c in cleaned_columns]
    sub.dropna(inplace=True); sub.drop_duplicates(inplace=True)
    
    fall_activity_ids = {2, 3, 4, 5, 6}; sub['Fall'] = sub['Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
    
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]; test_subjects = [s for s in range(14, 18)]
    train_df = sub[sub['Subject'].isin(train_subjects)]; test_df = sub[sub['Subject'].isin(test_subjects)]

    feature_cols = [col for col in sub.columns if 'Infrared' not in col and col not in ['TimeStamps_Time', 'Activity', 'Subject', 'Trial', 'Tag', 'Fall']]
    
    logging.info("Step 2: Scaling features...")
    scaler = StandardScaler().fit(train_df[feature_cols])
    sub.loc[:, feature_cols] = scaler.transform(sub[feature_cols])
    
    train_df = sub[sub['Subject'].isin(train_subjects)]
    test_df = sub[sub['Subject'].isin(test_subjects)]

    logging.info("Step 3: Creating global test set...")
    X_test_csv = test_df[feature_cols].values; y_test = test_df['Fall'].values
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test_csv).float(), torch.from_numpy(y_test).long()), batch_size=128)

    logging.info("Step 4: Partitioning data into client datasets...")
    clients = {cid: (df[feature_cols].values, df['Fall'].values) for cid, df in train_df.groupby('Subject')}
    logging.info(f"Data loading complete. {len(clients)} clients created.")
    return clients, test_loader, len(feature_cols)

def simulate_concept_drift(client_data):
    logging.info("Simulating concept drift for one client...")
    X_csv, y = client_data
    split_point = len(X_csv) // 2
    X_csv_drifted = X_csv.copy()
    X_csv_drifted[split_point:, [0, 3]] = X_csv_drifted[split_point:, [3, 0]]
    return (X_csv_drifted, y)

# --- 3. Model Architectures ---

class SharedBackbone(nn.Module):
    """The shared, global part of the model - now a more powerful 1D-CNN."""
    def __init__(self, num_csv_features, feature_dim=128):
        super().__init__()
        # **FIX:** Upgraded from a simple Linear model to a 1D-CNN encoder
        self.imu_encoder = nn.Sequential(
            nn.Conv1d(num_csv_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, feature_dim)
        
    def forward(self, x_csv):
        # Reshape from (N, F) to (N, F, 1) for Conv1D
        x = x_csv.unsqueeze(2)
        features = self.imu_encoder(x).squeeze(2)
        return self.fc(features)

class Adapter(nn.Module):
    def __init__(self, input_dim=128, bottleneck_dim=16):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(input_dim, bottleneck_dim), nn.ReLU(), nn.Linear(bottleneck_dim, input_dim))
    def forward(self, x):
        return x + self.block(x)

class PersonalizedAdapterModel(nn.Module):
    def __init__(self, backbone, adapter):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.classifier = nn.Linear(128, 2)
    def forward(self, x_csv):
        return self.classifier(self.adapter(self.backbone(x_csv)))

class ELMHead:
    def __init__(self, input_dim, hidden_dim, lam=0.01):
        self.hidden_weights = torch.randn(input_dim, hidden_dim) * 0.1
        self.output_weights = None; self.lam = lam
    def _get_hidden_features(self, X):
        if self.hidden_weights.device != X.device:
            self.hidden_weights = self.hidden_weights.to(X.device)
        return torch.tanh(X @ self.hidden_weights)
    def train(self, X, y):
        H = self._get_hidden_features(X)
        y_one_hot = F.one_hot(y, num_classes=2).float().to(H.device)
        H_t_H = H.T @ H; H_t_Y = H.T @ y_one_hot
        identity = torch.eye(H.shape[1], device=H.device)
        self.output_weights = torch.inverse(H_t_H + self.lam * identity) @ H_t_Y
    def predict(self, X):
        H = self._get_hidden_features(X)
        return H @ self.output_weights

# --- 4. Federated Learning Components ---

class Client:
    def __init__(self, client_id, data, device):
        self.id = client_id; self.device = device
        if len(data[1]) > 0:
            self.loader = DataLoader(TensorDataset(torch.from_numpy(data[0]).float(), torch.from_numpy(data[1]).long()), batch_size=64, shuffle=True)
        else: self.loader = None
        self.personal_adapter = Adapter().to(device)
        self.personal_elm = ELMHead(128, 256)

    def train_adapter(self, shared_backbone, config):
        if not self.loader: return shared_backbone.state_dict()
        full_model = PersonalizedAdapterModel(shared_backbone, self.personal_adapter).to(self.device)
        for param in full_model.backbone.parameters(): param.requires_grad = False
        for param in full_model.adapter.parameters(): param.requires_grad = True
        for param in full_model.classifier.parameters(): param.requires_grad = True
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, full_model.parameters()), lr=config['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        full_model.train()
        for _ in range(config['local_epochs']):
            for x_csv, y in self.loader:
                optimizer.zero_grad()
                outputs = full_model(x_csv.to(self.device))
                loss = criterion(outputs, y.to(self.device).long())
                loss.backward(); optimizer.step()
        return shared_backbone.state_dict()

def evaluate_model(model, loader, device):
    model.to(device); model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_csv, y in loader:
            outputs = model(x_csv.to(device))
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(y.numpy())
    return f1_score(all_labels, all_preds, pos_label=0, zero_division=0.0)

# --- 5. Main Simulation Runner ---

def run_fl_simulation_for_adapters(clients_data, test_loader, config, num_features):
    logging.info("Initializing global backbone model...")
    global_backbone = SharedBackbone(num_features)
    client_objects = {cid: Client(cid, data, config['device']) for cid, data in clients_data.items()}

    for round_num in range(1, config['comm_rounds'] + 1):
        logging.info(f"--- Starting FL Round {round_num}/{config['comm_rounds']} ---")
        client_updates = []
        for cid, client in client_objects.items():
            logging.info(f"Training on client {cid}...")
            shared_part_update = client.train_adapter(copy.deepcopy(global_backbone), config)
            client_updates.append(shared_part_update)
        
        logging.info("Aggregating shared backbone updates...")
        global_dict = global_backbone.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([upd[k] for upd in client_updates], 0).mean(0)
        global_backbone.load_state_dict(global_dict)
        logging.info(f"Round {round_num} complete.")

    logging.info("Evaluating personalized models on each client's local data...")
    all_client_f1s = []
    for client in client_objects.values():
        if client.loader:
            f1 = evaluate_model(PersonalizedAdapterModel(global_backbone, client.personal_adapter), client.loader, config['device'])
            all_client_f1s.append(f1)
    
    return min(all_client_f1s) if all_client_f1s else 0.0

# --- 6. Main Execution Block ---

if __name__ == "__main__":
    OUTPUT_DIR = '/home/syed/PhD/UP-Fall-FL/Experimental_Scripts/Modules/output/Module7'
    os.makedirs(OUTPUT_DIR, exist_ok=True); setup_logging(OUTPUT_DIR)
    config = {'comm_rounds': 5, 'local_epochs': 3, 'learning_rate': 0.001,
              'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    
    clients_data, test_loader, num_features = load_and_create_clients('/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/')

    logging.info("\n--- Assignment 1: Adapters vs. Shared-Only Model ---")
    worst_f1_personalized = run_fl_simulation_for_adapters(clients_data, test_loader, config, num_features)
    
    logging.info("Evaluating generic shared-only model for comparison...")
    shared_only_model = PersonalizedAdapterModel(SharedBackbone(num_features), Adapter())
    all_shared_f1s = []
    for cid, d in clients_data.items():
        client_for_eval = Client(cid, d, config['device'])
        if client_for_eval.loader:
            f1 = evaluate_model(shared_only_model, client_for_eval.loader, config['device'])
            all_shared_f1s.append(f1)
    worst_f1_shared = min(all_shared_f1s) if all_shared_f1s else 0.0

    adapter_results = pd.DataFrame([
        {'Model Type': 'Shared-Only (Generic)', 'Worst-Client F1-Score': worst_f1_shared},
        {'Model Type': 'Personalized (Adapters)', 'Worst-Client F1-Score': worst_f1_personalized}
    ])
    logging.info("\n--- Adapter Performance Comparison ---")
    logging.info(f"\n{adapter_results.to_string(index=False)}")

    logging.info("\n--- Assignment 2: ELM Head Recovering from Concept Drift ---")
    logging.info("Training a shared backbone for ELM...")
    # In a real scenario, this backbone would be the result of the FL training.
    # Here, we'll use a freshly initialized one for a clear demonstration.
    elm_backbone = SharedBackbone(num_features).to(config['device'])
    
    client_id_to_test = list(clients_data.keys())[0]
    client_data = clients_data[client_id_to_test]
    client_data_drifted = simulate_concept_drift(client_data)
    
    logging.info(f"Extracting features for client {client_id_to_test}...")
    X_full, y_full = torch.from_numpy(client_data_drifted[0]).float().to(config['device']), torch.from_numpy(client_data_drifted[1]).long()
    with torch.no_grad(): H_full = elm_backbone(X_full)
    
    split_point = len(H_full) // 2
    H_before, y_before = H_full[:split_point], y_full[:split_point]
    H_after, y_after = H_full[split_point:], y_full[split_point:]
    
    logging.info("Training ELM head on pre-drift data...")
    client_elm = ELMHead(128, 256)
    client_elm.train(H_before, y_before)
    
    logging.info("Evaluating ELM head on post-drift data (before adaptation)...")
    preds_before_adapt = torch.argmax(client_elm.predict(H_after), dim=1)
    f1_before_adapt = f1_score(y_after.cpu(), preds_before_adapt.cpu(), pos_label=0, zero_division=0.0)
    
    logging.info("Retraining ELM head on post-drift data (online adaptation)...")
    client_elm.train(H_after, y_after)
    preds_after_adapt = torch.argmax(client_elm.predict(H_after), dim=1)
    f1_after_adapt = f1_score(y_after.cpu(), preds_after_adapt.cpu(), pos_label=0, zero_division=0.0)
    
    elm_results = pd.DataFrame([
        {'State': 'Before Adaptation (Trained on old data)', 'F1-Score on New Data': f1_before_adapt},
        {'State': 'After Adaptation (Retrained on new data)', 'F1-Score on New Data': f1_after_adapt}
    ])
    logging.info(f"\n--- ELM Concept Drift Recovery (Client {client_id_to_test}) ---")
    logging.info(f"\n{elm_results.to_string(index=False)}")

    logging.info(f"\n✅ All simulations complete. Results saved in '{OUTPUT_DIR}'.")