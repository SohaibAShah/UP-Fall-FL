import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import logging
import sys
import copy
from io import BytesIO

# --- 1. Setup and Helper Functions ---

def setup_logging(log_dir):
    """Configures logging to save to a file and print to the console."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    # Clear existing handlers to avoid duplicate logs in interactive environments
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "fl_run_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_model_size_mb(model):
    """Calculates the size of a model's state dictionary in megabytes."""
    param_size = 0
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    param_size += buffer.tell()
    return param_size / (1024**2)


# --- 2. Data Loading and Client Creation ---

def load_and_prep_data(file_path):
    """Loads and performs initial cleaning of the dataset."""
    logging.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=[0, 1])
    # Clean column names
    cleaned_columns = []; last_val = ''
    for c in df.columns:
        col_l1, col_l2 = c
        if 'Unnamed' in col_l1: col_l1 = last_val
        else: last_val = col_l1.strip()
        cleaned_columns.append(f"{col_l1}_{col_l2.strip()}" if col_l1 != col_l2 else col_l1)
    df.columns = cleaned_columns
    
    # Exclusions and cleaning
    df = df[~df['Subject'].isin([5, 9])]
    df = df[~((df['Subject'] == 2) & (df['Activity'] == 5))]
    df = df[~((df['Subject'] == 8) & (df['Activity'] == 11) & (df['Trial'].isin([2, 3])))]
    cols_to_drop = [c for c in df.columns if 'Infrared' in c or 'Tag' in c or 'Time' in c or 'Trial' in c]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.dropna(inplace=True); df.drop_duplicates(inplace=True)
    
    # Binary labels
    fall_activity_ids = {2, 3, 4, 5, 6}
    df['Fall'] = df['Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
    
    # Split into train/test subjects
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
    test_subjects = [s for s in range(14, 18)]
    train_df = df[df['Subject'].isin(train_subjects)]
    test_df = df[df['Subject'].isin(test_subjects)]
    
    return train_df, test_df

def create_clients(train_df, test_df, iid=False):
    """Creates windowed datasets for clients and a global test set."""
    # Define all possible IMU feature columns
    imu_columns = ['AnkleAccelerometer_x-axis (g)', 'AnkleAccelerometer_y-axis (g)', 'AnkleAccelerometer_z-axis (g)', 'AnkleAngularVelocity_x-axis (deg/s)', 'AnkleAngularVelocity_y-axis (deg/s)', 'AnkleAngularVelocity_z-axis (deg/s)', 'RightPocketAccelerometer_x-axis (g)', 'RightPocketAccelerometer_y-axis (g)', 'RightPocketAccelerometer_z-axis (g)', 'RightPocketAngularVelocity_x-axis (deg/s)', 'RightPocketAngularVelocity_y-axis (deg/s)', 'RightPocketAngularVelocity_z-axis (deg/s)', 'BeltAccelerometer_x-axis (g)', 'BeltAccelerometer_y-axis (g)', 'BeltAccelerometer_z-axis (g)', 'BeltAngularVelocity_x-axis (deg/s)', 'BeltAngularVelocity_y-axis (deg/s)', 'BeltAngularVelocity_z-axis (deg/s)', 'NeckAccelerometer_x-axis (g)', 'NeckAccelerometer_y-axis (g)', 'NeckAccelerometer_z-axis (g)', 'NeckAngularVelocity_x-axis (deg/s)', 'NeckAngularVelocity_y-axis (deg/s)', 'NeckAngularVelocity_z-axis (deg/s)', 'WristAccelerometer_x-axis (g)', 'WristAccelerometer_y-axis (g)', 'WristAccelerometer_z-axis (g)', 'WristAngularVelocity_x-axis (deg/s)', 'WristAngularVelocity_y-axis (deg/s)', 'WristAngularVelocity_z-axis (deg/s)']
    feature_columns = [col for col in imu_columns if col in train_df.columns]

    # Process and scale data
    scaler = StandardScaler().fit(train_df[feature_columns])
    train_df.loc[:, feature_columns] = scaler.transform(train_df[feature_columns])
    test_df.loc[:, feature_columns] = scaler.transform(test_df[feature_columns])
    
    # Create a unified, windowed test set
    X_test_w, y_test_w = [], []
    for subject in test_df['Subject'].unique():
        subject_df = test_df[test_df['Subject'] == subject]
        data, labels = subject_df[feature_columns].values, subject_df['Fall'].values
        for i in range(0, len(data) - 200, 100):
            X_test_w.append(data[i:i+200])
            y_test_w.append(0 if np.any(labels[i:i+200] == 0) else 1)
            
    X_test = np.transpose(np.array(X_test_w), (0, 2, 1))
    y_test = np.array(y_test_w)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), batch_size=256)

    # Create client datasets
    clients = {}
    client_ids = sorted(train_df['Subject'].unique())
    
    if iid:
        # **CORRECTED IID LOGIC**
        logging.info("Creating IID client partitions...")
        # 1. Create all possible windows from the entire training set
        all_windows_X, all_windows_y = [], []
        for subject_id in client_ids:
            subject_df = train_df[train_df['Subject'] == subject_id]
            data, labels = subject_df[feature_columns].values, subject_df['Fall'].values
            for i in range(0, len(data) - 200, 100):
                all_windows_X.append(data[i:i+200])
                all_windows_y.append(0 if np.any(labels[i:i+200] == 0) else 1)
        
        all_windows_X = np.array(all_windows_X)
        all_windows_y = np.array(all_windows_y)

        # 2. Shuffle the windows
        shuffled_indices = np.random.permutation(len(all_windows_X))
        
        # 3. Distribute the shuffled windows among clients
        split_indices = np.array_split(shuffled_indices, len(client_ids))
        
        for i, client_id in enumerate(client_ids):
            client_indices = split_indices[i]
            X_client = all_windows_X[client_indices]
            y_client = all_windows_y[client_indices]
            clients[client_id] = (np.transpose(X_client, (0, 2, 1)), y_client)
    else:
        # Non-IID: Partition by subject (natural distribution)
        logging.info("Creating Non-IID client partitions...")
        for client_id in client_ids:
            client_df = train_df[train_df['Subject'] == client_id]
            client_data, client_labels = client_df[feature_columns].values, client_df['Fall'].values
            X_client_w, y_client_w = [], []
            for i in range(0, len(client_data) - 200, 100):
                X_client_w.append(client_data[i:i+200])
                y_client_w.append(0 if np.any(client_labels[i:i+200] == 0) else 1)
            clients[client_id] = (np.transpose(np.array(X_client_w), (0, 2, 1)), np.array(y_client_w))
            
    return clients, test_loader, len(feature_columns)


# --- 3. Model Architecture ---

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
        # Use drop_last=True to prevent batch norm errors if the last batch has only one sample
        self.data_loader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long()), batch_size=32, shuffle=True, drop_last=True)
        self.dataset_size = len(self.data_loader.dataset)

    def train(self, model, local_epochs, learning_rate):
        """Trains the model locally and returns weights, data size, loss, and accuracy."""
        model.to(self.device); model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        total_loss, total_correct, total_samples = 0, 0, 0
        for _ in range(local_epochs):
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device).float().view(-1, 1)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                total_correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        avg_acc = total_correct / total_samples if total_samples > 0 else 0
        
        # Return all four values
        return model.state_dict(), self.dataset_size, avg_loss, avg_acc

def server_aggregate(global_model, client_updates, total_data_size):
    """Aggregates client models using weighted averaging (FedAvg)."""
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client['weights'][k].float() * client['data_size'] for client in client_updates], 0).sum(0) / total_data_size
    global_model.load_state_dict(global_dict)
    return global_model

def evaluate_global_model(model, test_loader, device):
    """Evaluates the global model on the test set."""
    model.to(device); model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds.flatten()); all_labels.extend(labels.numpy())
    return accuracy_score(all_labels, all_preds)

# --- 5. Main Simulation Runner ---

def run_federated_simulation(clients_data, test_loader, config, num_features):
    """Runs a full FedAvg simulation and returns the performance history."""
    set_seed(42)
    global_model = CNN_Attention(input_channels=num_features)
    model_size_mb = get_model_size_mb(global_model)
    
    client_objects = {cid: Client(cid, data, config['device']) for cid, data in clients_data.items()}
    client_ids = list(clients_data.keys())
    client_losses = {cid: 1.0 for cid in client_ids}
    
    logging.info(f"Starting simulation with {len(client_ids)} total clients...")
    history = {'round': [], 'accuracy': [], 'comm_mb': []}
    
    for round_num in range(1, config['comm_rounds'] + 1):
        global_model.train()
        
        # Select clients
        if config['sampling_strategy'] == 'loss':
            losses = np.array([client_losses[cid] for cid in client_ids])
            probabilities = losses / losses.sum() if losses.sum() > 0 else None
            selected_client_ids = np.random.choice(client_ids, config['clients_per_round'], replace=False, p=probabilities)
        else:
            selected_client_ids = np.random.choice(client_ids, config['clients_per_round'], replace=False)
        
        logging.info(f"--- Round: {round_num:02d}/{config['comm_rounds']} | Selected clients: {selected_client_ids.tolist()} ---")
        
        client_updates, total_data_size = [], 0
        for client_id in selected_client_ids:
            client = client_objects[client_id]
            local_model = copy.deepcopy(global_model)
            updated_weights, data_size, avg_loss, avg_acc = client.train(local_model, config['local_epochs'], config['learning_rate'])
            
            logging.info(f"  > Client {client_id:2d}: avg_loss={avg_loss:.4f}, local_acc={avg_acc:.4f}")
            client_updates.append({'weights': updated_weights, 'data_size': data_size})
            total_data_size += data_size
            client_losses[client_id] = avg_loss

        # Server aggregation
        global_model = server_aggregate(global_model, client_updates, total_data_size)
        
        # Global evaluation
        accuracy = evaluate_global_model(global_model, test_loader, config['device'])
        comm_cost = history['comm_mb'][-1] if history['comm_mb'] else 0
        comm_cost += config['clients_per_round'] * model_size_mb
        
        history['round'].append(round_num); history['accuracy'].append(accuracy); history['comm_mb'].append(comm_cost)
        logging.info(f"  > Global Accuracy: {accuracy:.4f} | Cumulative Comm (MB): {comm_cost:.2f}")
        
    return pd.DataFrame(history)


# --- 6. Main Execution Block ---

if __name__ == "__main__":
    OUTPUT_DIR = '/home/syed/PhD/UP-Fall-FL/Experimental_Scripts/Modules/output/Module2'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(OUTPUT_DIR)

    FILE_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv' # IMPORTANT: Ensure 'sensor.csv' is in the same directory

    
    # Config
    config = {
        'comm_rounds': 50,
        'local_epochs': 5,
        'clients_per_round': 5, # Number of clients to sample each round
        'learning_rate': 0.001,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }
    
    # Load and prepare data
    train_df, test_df = load_and_prep_data(FILE_PATH)

    # --- Run Experiments ---
    logging.info("\n--- Running Experiment 1: Non-IID Clients (Uniform Sampling) ---")
    non_iid_clients, test_loader, num_features = create_clients(train_df, test_df, iid=False)
    config['sampling_strategy'] = 'uniform'
    history_non_iid = run_federated_simulation(non_iid_clients, test_loader, config, num_features)
    history_non_iid.to_csv(os.path.join(OUTPUT_DIR, 'history_non_iid.csv'), index=False)
    
    logging.info("\n--- Running Experiment 2: IID Clients (Uniform Sampling) ---")
    iid_clients, _, _ = create_clients(train_df, test_df, iid=True)
    history_iid = run_federated_simulation(iid_clients, test_loader, config, num_features)
    history_iid.to_csv(os.path.join(OUTPUT_DIR, 'history_iid.csv'), index=False)
    
    logging.info("\n--- Running Experiment 3: Non-IID Clients (Loss-Based Sampling) ---")
    config['sampling_strategy'] = 'loss'
    history_loss_sampling = run_federated_simulation(non_iid_clients, test_loader, config, num_features)
    history_loss_sampling.to_csv(os.path.join(OUTPUT_DIR, 'history_loss_sampling.csv'), index=False)

    # --- Plotting Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plot 1: Accuracy vs. Communication Rounds
    plt.figure(figsize=(10, 6))
    plt.plot(history_non_iid['round'], history_non_iid['accuracy'], marker='o', linestyle='--', label='Non-IID (Uniform Sampling)')
    plt.plot(history_iid['round'], history_iid['accuracy'], marker='s', linestyle='-', label='IID (Uniform Sampling)')
    plt.plot(history_loss_sampling['round'], history_loss_sampling['accuracy'], marker='^', linestyle=':', label='Non-IID (Loss-Based Sampling)')
    plt.title('Global Model Accuracy vs. Communication Rounds')
    plt.xlabel('Communication Round')
    plt.ylabel('Global Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_rounds.png'))
    plt.show()

    # Plot 2: Accuracy vs. Communication Cost (MB)
    plt.figure(figsize=(10, 6))
    plt.plot(history_non_iid['comm_mb'], history_non_iid['accuracy'], marker='o', linestyle='--', label='Non-IID (Uniform Sampling)')
    plt.plot(history_iid['comm_mb'], history_iid['accuracy'], marker='s', linestyle='-', label='IID (Uniform Sampling)')
    plt.plot(history_loss_sampling['comm_mb'], history_loss_sampling['accuracy'], marker='^', linestyle=':', label='Non-IID (Loss-Based Sampling)')
    plt.title('Global Model Accuracy vs. Communication Cost')
    plt.xlabel('Cumulative Communication (MB)')
    plt.ylabel('Global Test Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_comm_mb.png'))
    plt.show()

    logging.info(f"\n✅ All simulations complete. Results saved in '{OUTPUT_DIR}'.")