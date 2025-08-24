import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
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
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, "fl_module3_log.txt"))
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

# --- 2. Data Loading (Reused from Module 2) ---

def load_and_prep_data(file_path):
    df = pd.read_csv(file_path, header=[0, 1])
    cleaned_columns = []; last_val = ''
    for c in df.columns:
        col_l1, col_l2 = c
        if 'Unnamed' in col_l1: col_l1 = last_val
        else: last_val = col_l1.strip()
        cleaned_columns.append(f"{col_l1}_{col_l2.strip()}" if col_l1 != col_l2 else col_l1)
    df.columns = cleaned_columns
    df = df[~df['Subject'].isin([5, 9])]
    df = df[~((df['Subject'] == 2) & (df['Activity'] == 5))]
    df = df[~((df['Subject'] == 8) & (df['Activity'] == 11) & (df['Trial'].isin([2, 3])))]
    cols_to_drop = [c for c in df.columns if 'Infrared' in c or 'Tag' in c or 'Time' in c or 'Trial' in c]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.dropna(inplace=True); df.drop_duplicates(inplace=True)
    fall_activity_ids = {2, 3, 4, 5, 6}
    df['Fall'] = df['Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
    test_subjects = [s for s in range(14, 18)]
    return df[df['Subject'].isin(train_subjects)], df[df['Subject'].isin(test_subjects)]

def create_clients(train_df, test_df):
    imu_columns = [col for col in train_df.columns if 'Accelerometer' in col or 'AngularVelocity' in col]
    scaler = StandardScaler().fit(train_df[imu_columns])
    train_df.loc[:, imu_columns] = scaler.transform(train_df[imu_columns])
    test_df.loc[:, imu_columns] = scaler.transform(test_df[imu_columns])

    # Create windowed test set
    X_test_w, y_test_w = [], []
    for subject in test_df['Subject'].unique():
        data = test_df[test_df['Subject'] == subject]
        d, l = data[imu_columns].values, data['Fall'].values
        for i in range(0, len(d) - 200, 100):
            X_test_w.append(d[i:i+200]); y_test_w.append(0 if np.any(l[i:i+200] == 0) else 1)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(np.transpose(np.array(X_test_w), (0, 2, 1))).float(),
                                          torch.from_numpy(np.array(y_test_w)).long()), batch_size=256)

    # Create Non-IID clients
    clients = {}
    for client_id in sorted(train_df['Subject'].unique()):
        data = train_df[train_df['Subject'] == client_id]
        d, l = data[imu_columns].values, data['Fall'].values
        X_client_w, y_client_w = [], []
        for i in range(0, len(d) - 200, 100):
            X_client_w.append(d[i:i+200]); y_client_w.append(0 if np.any(l[i:i+200] == 0) else 1)
        clients[client_id] = (np.transpose(np.array(X_client_w), (0, 2, 1)), np.array(y_client_w))

    return clients, test_loader, len(imu_columns)

# --- 3. Model Architecture (Reused) ---

class CNN_Attention(nn.Module):
    def __init__(self, input_channels):
        super(CNN_Attention, self).__init__()
        self.conv1, self.relu1 = nn.Conv1d(input_channels, 32, 5, padding='same'), nn.ReLU()
        self.conv2, self.relu2 = nn.Conv1d(32, 64, 5, padding='same'), nn.ReLU()
        self.attention = self.TemporalAttention(64)
        self.fc = nn.Linear(64, 1)

    class TemporalAttention(nn.Module):
        def __init__(self, in_features):
            super().__init__()
            self.attention_net = nn.Sequential(nn.Linear(in_features, in_features // 2), nn.Tanh(), nn.Linear(in_features // 2, 1))
        def forward(self, x):
            x_permuted = x.permute(0, 2, 1)
            attn_weights = torch.softmax(self.attention_net(x_permuted), dim=1)
            return torch.sum(x_permuted * attn_weights, dim=1)

    def forward(self, x):
        x = self.relu1(self.conv1(x)); x = self.relu2(self.conv2(x))
        return self.fc(self.attention(x))

# --- 4. Federated Learning Components ---

class Client:
    def __init__(self, client_id, data, device):
        self.id = client_id
        self.device = device
        X, y = data
        self.dataset_size = len(y)
        # Use drop_last=True to prevent errors if the last batch has only one sample
        try:
            self.data_loader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()), batch_size=32, shuffle=True, drop_last=True)
        except IndexError:
             self.data_loader = None # Handle cases with very few samples
        
        self.control_variate = None # For SCAFFOLD

    def train(self, model, config):
        if not self.data_loader: # Skip training if client has no data for a batch
            return model.state_dict(), self.dataset_size

        local_model = copy.deepcopy(model).to(self.device)
        local_model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(local_model.parameters(), lr=config['learning_rate'])
        
        # Keep the initial global weights on the CPU
        initial_global_weights = model.state_dict()

        for _ in range(config['local_epochs']):
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)

                if config['algorithm'] == 'fedprox':
                    proximal_term = 0.0
                    for param, initial_param in zip(local_model.parameters(), model.parameters()):
                        # Move initial param to GPU for this calculation only
                        proximal_term += torch.sum(torch.pow(param - initial_param.to(self.device), 2))
                    loss += (config['mu'] / 2) * proximal_term
                
                loss.backward()

                if config['algorithm'] == 'scaffold':
                    # All control variates are on the CPU, move the update to the GPU
                    for param, server_cv, client_cv in zip(local_model.parameters(), config['server_cv'].parameters(), self.control_variate.parameters()):
                         param.grad += (server_cv - client_cv).to(self.device)

                optimizer.step()

        if config['algorithm'] == 'scaffold':
            # **FIX:** Move final weights to CPU before calculating deltas
            final_weights_cpu = {k: v.cpu() for k, v in local_model.state_dict().items()}
            model_delta = {k: final_weights_cpu[k] - initial_global_weights[k] for k in final_weights_cpu}

            # All subsequent operations are now safely on the CPU
            new_cv_state = self.control_variate.state_dict()
            server_cv_state = config['server_cv'].state_dict()
            coef = 1 / (config['local_epochs'] * len(self.data_loader) * config['learning_rate'])
            
            for k in new_cv_state.keys():
                new_cv_state[k] -= server_cv_state[k] - (coef * model_delta[k])

            new_client_cv = CNN_Attention(config['input_channels'])
            new_client_cv.load_state_dict(new_cv_state)
            
            cv_delta = {k: new_client_cv.state_dict()[k] - self.control_variate.state_dict()[k] for k in new_client_cv.state_dict()}
            self.control_variate = new_client_cv
            
            return model_delta, cv_delta, self.dataset_size

        return local_model.state_dict(), self.dataset_size

def evaluate_model(model, data_loader, device):
    """Evaluates a model and returns F1-score for the 'Fall' class."""
    # **FIX:** Work on a copy of the model to avoid modifying the original
    eval_model = copy.deepcopy(model).to(device)
    eval_model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = eval_model(inputs.to(device))
            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds.flatten()); all_labels.extend(labels.numpy())
            
    return f1_score(all_labels, all_preds, pos_label=0, zero_division=0.0)

# --- 5. Main Simulation Runner ---

def run_federated_simulation(clients_data, test_loader, config, num_features):
    set_seed(42)
    global_model = CNN_Attention(input_channels=num_features)
    client_objects = {cid: Client(cid, data, config['device']) for cid, data in clients_data.items()}
    
    config['input_channels'] = num_features
    
    server_control_variate = CNN_Attention(input_channels=num_features) if config['algorithm'] == 'scaffold' else None
    if server_control_variate:
        for param in server_control_variate.parameters(): param.data.zero_()
        for _, client in client_objects.items():
            client.control_variate = copy.deepcopy(server_control_variate)

    history = {'round': [], 'f1_score': []}
    
    for round_num in range(1, config['comm_rounds'] + 1):
        selected_client_ids = np.random.choice(list(clients_data.keys()), config['clients_per_round'], replace=False)
        
        client_updates, total_data_size = [], 0
        all_cv_deltas = []

        config_for_client = config.copy()
        if config['algorithm'] == 'scaffold':
            config_for_client['server_cv'] = server_control_variate

        for client_id in selected_client_ids:
            client = client_objects[client_id]
            
            if config['algorithm'] == 'scaffold':
                model_delta, cv_delta, data_size = client.train(global_model, config_for_client)
                client_updates.append({'delta': model_delta, 'data_size': data_size})
                all_cv_deltas.append(cv_delta)
            else:
                updated_weights, data_size = client.train(global_model, config_for_client)
                client_updates.append({'weights': updated_weights, 'data_size': data_size})
            
            total_data_size += data_size

        global_dict = global_model.state_dict()
        if config['algorithm'] == 'scaffold':
            for k in global_dict.keys():
                weighted_sum_delta = torch.stack([update['delta'][k] * update['data_size'] for update in client_updates], 0).sum(0)
                if total_data_size > 0: global_dict[k] += weighted_sum_delta / total_data_size
            
            # **FIX:** Iterate over state_dict keys instead of .parameters()
            server_cv_state = server_control_variate.state_dict()
            for name in server_cv_state.keys():
                aggregated_delta = torch.stack([delta[name] for delta in all_cv_deltas], 0).sum(0)
                server_cv_state[name] += aggregated_delta / len(selected_client_ids)
            server_control_variate.load_state_dict(server_cv_state)

        else:
            for k in global_dict.keys():
                if total_data_size > 0:
                    global_dict[k] = torch.stack([update['weights'][k].float() * update['data_size'] for update in client_updates], 0).sum(0) / total_data_size
        
        global_model.load_state_dict(global_dict)
        
        f1 = evaluate_model(global_model, test_loader, config['device'])
        history['round'].append(round_num); history['f1_score'].append(f1)
        logging.info(f"Algo: {config['algorithm']:<8} | Round: {round_num:02d} | Global F1-Score: {f1:.4f}")

    worst_f1 = 1.0
    for cid, client in client_objects.items():
        if client.data_loader:
            client_f1 = evaluate_model(global_model, client.data_loader, config['device'])
            if client_f1 < worst_f1:
                worst_f1 = client_f1
    
    rounds_to_target = next((r for r, f in zip(history['round'], history['f1_score']) if f >= config['target_f1']), -1)

    return rounds_to_target, worst_f1, pd.DataFrame(history)


# --- 6. Main Execution Block ---

if __name__ == "__main__":
    OUTPUT_DIR = '/home/syed/PhD/UP-Fall-FL/Experimental_Scripts/Modules/output/Module3'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(OUTPUT_DIR)
    
    FILE_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv' # IMPORTANT: Ensure 'sensor.csv' is in the same directory

    base_config = {
        'comm_rounds': 50, 'local_epochs': 5, 'clients_per_round': 5,
        'learning_rate': 0.001, 'target_f1': 0.75,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    logging.info("--- Preparing Data ---")
    train_df, test_df = load_and_prep_data(FILE_PATH)
    clients, test_loader, num_features = create_clients(train_df, test_df)

    # --- Run Experiments ---
    results = []
    all_histories = {}

    # FedAvg
    config_fedavg = {**base_config, 'algorithm': 'fedavg'}
    rtt, wf1, hist = run_federated_simulation(clients, test_loader, config_fedavg, num_features)
    results.append({'Algorithm': 'FedAvg', 'Rounds-to-Target-F1': rtt, 'Final-Worst-Client-F1': wf1})
    all_histories['FedAvg'] = hist
    
    # FedProx
    config_fedprox = {**base_config, 'algorithm': 'fedprox', 'mu': 0.01}
    rtt, wf1, hist = run_federated_simulation(clients, test_loader, config_fedprox, num_features)
    results.append({'Algorithm': 'FedProx (μ=0.01)', 'Rounds-to-Target-F1': rtt, 'Final-Worst-Client-F1': wf1})
    all_histories['FedProx'] = hist

    # SCAFFOLD
    config_scaffold = {**base_config, 'algorithm': 'scaffold'}
    rtt, wf1, hist = run_federated_simulation(clients, test_loader, config_scaffold, num_features)
    results.append({'Algorithm': 'SCAFFOLD', 'Rounds-to-Target-F1': rtt, 'Final-Worst-Client-F1': wf1})
    all_histories['SCAFFOLD'] = hist

    # --- Display and Save Results ---
    results_df = pd.DataFrame(results)
    logging.info("\n--- Final Comparison ---")
    logging.info(f"\n{results_df.to_string(index=False)}")
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'final_comparison.csv'), index=False)

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    for name, df in all_histories.items():
        plt.plot(df['round'], df['f1_score'], marker='o', linestyle='--', markersize=4, label=name)
    plt.axhline(y=base_config['target_f1'], color='r', linestyle=':', label=f"Target F1 ({base_config['target_f1']})")
    plt.title('Global Model F1-Score vs. Communication Rounds')
    plt.xlabel('Communication Round'); plt.ylabel('Global Test F1-Score (Fall Class)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'f1_vs_rounds.png')); plt.show()
    
    logging.info(f"\n✅ All simulations complete. Results saved in '{OUTPUT_DIR}'.")