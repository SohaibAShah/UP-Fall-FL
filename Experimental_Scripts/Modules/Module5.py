import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
import logging
import sys
import time

# --- 1. Setup and Helper Functions ---

def setup_logging(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, "fl_module5_log.txt"))
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

# --- 2. Data Loading (Unchanged) ---

def load_and_process_data(data_dir):
    # This function is reused from the previous module.
    # It loads, aligns, and splits the data for IMU and two cameras.
    # For brevity, its implementation is assumed to be the same as in Module 4.
    logging.info("Loading and processing data...")
    sub = pd.read_csv(os.path.join(data_dir, 'sensor.csv'), header=[0, 1])
    cleaned_columns = [f"{c[0].strip()}_{c[1].strip()}" if 'Unnamed' not in c[0] else last_val + f"_{c[1].strip()}" for c in sub.columns if 'Unnamed' not in c[0] or (last_val := c[0].strip())]
    sub.columns = [c.replace(c.split('_')[0] + '_', '') if c.split('_')[0] == c.split('_')[1] else c for c in cleaned_columns]
    sub.dropna(inplace=True); sub.drop_duplicates(inplace=True)
    img1 = np.load(os.path.join(data_dir, 'image_1.npy')); name1 = np.load(os.path.join(data_dir, 'name_1.npy'))
    img2 = np.load(os.path.join(data_dir, 'image_2.npy'))
    sensor_ts = set(sub['TimeStamps_Time']); name1_ts = set(name1)
    common_timestamps = sorted(list(sensor_ts.intersection(name1_ts)))
    sub = sub[sub['TimeStamps_Time'].isin(common_timestamps)]
    name1_map = {ts: i for i, ts in enumerate(name1)}; idx1 = [name1_map[ts] for ts in common_timestamps]
    img1, img2 = img1[idx1], img2[idx1]
    sub = sub.set_index('TimeStamps_Time').loc[common_timestamps]
    feature_cols = [col for col in sub.columns if 'Infrared' not in col and col not in ['Activity', 'Subject', 'Trial', 'Tag']]
    X_csv, y = sub[feature_cols].values, sub['Activity'].values
    fall_activity_ids = {2, 3, 4, 5, 6}
    y_binary = np.array([0 if activity_id in fall_activity_ids else 1 for activity_id in y])
    indices = np.arange(len(y_binary))
    train_idx, rem_idx = train_test_split(indices, train_size=0.6, random_state=42, stratify=y_binary)
    val_idx, test_idx = train_test_split(rem_idx, test_size=0.5, random_state=42, stratify=y_binary[rem_idx])
    scaler = StandardScaler().fit(X_csv[train_idx])
    X_csv_scaled = scaler.transform(X_csv)
    X_img1 = np.expand_dims((img1 / 255.0).astype(np.float32), axis=1)
    X_img2 = np.expand_dims((img2 / 255.0).astype(np.float32), axis=1)
    data_splits = {
        'X_train_csv': X_csv_scaled[train_idx], 'X_val_csv': X_csv_scaled[val_idx], 'X_test_csv': X_csv_scaled[test_idx],
        'X_train_img1': X_img1[train_idx], 'X_val_img1': X_img1[val_idx], 'X_test_img1': X_img1[test_idx],
        'X_train_img2': X_img2[train_idx], 'X_val_img2': X_img2[val_idx], 'X_test_img2': X_img2[test_idx],
        'y_train': y_binary[train_idx], 'y_val': y_binary[val_idx], 'y_test': y_binary[test_idx]
    }
    logging.info(f"Data loading complete. Train samples: {len(data_splits['y_train'])}")
    return data_splits


# --- 3. Model Architectures ---

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
        self.img_encoder1 = ImageEncoder(64)
        self.img_encoder2 = ImageEncoder(64)
        self.img_fusion = nn.Linear(64 + 64, 128)
        
        # Gating network: uses ONLY IMU features to decide
        self.gate = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        
        # Two heads: one for fused data, one for IMU-only data
        self.fused_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
        self.imu_only_classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))

    def forward(self, x_csv, x_img1, x_img2, threshold=0.5):
        # Always process the lightweight IMU data
        f_csv = self.imu_encoder(x_csv)
        
        # The gate decides whether to use the cameras
        gate_logit = self.gate(f_csv)
        gate_prob = torch.sigmoid(gate_logit)

        if self.training:
            # During training, always compute both paths for gradient flow
            # The modality dropout happens in the training loop
            f_img1 = self.img_encoder1(x_img1); f_img2 = self.img_encoder2(x_img2)
            f_img_combined = F.relu(self.img_fusion(torch.cat((f_img1, f_img2), dim=1)))
            fused = f_csv + f_img_combined # Simple residual connection
            
            # The final prediction is a mix of the two heads, weighted by the gate
            out_fused = self.fused_classifier(fused)
            out_imu_only = self.imu_only_classifier(f_csv)
            return gate_prob * out_fused + (1 - gate_prob) * out_imu_only
        else:
            # During inference, the gate makes a hard decision
            use_images = (gate_prob > threshold).float()
            
            # Conditionally compute image features
            if use_images.sum() > 0:
                f_img1 = self.img_encoder1(x_img1); f_img2 = self.img_encoder2(x_img2)
                f_img_combined = F.relu(self.img_fusion(torch.cat((f_img1, f_img2), dim=1)))
                fused = f_csv + f_img_combined
                out_fused = self.fused_classifier(fused)
            else: # Create a placeholder if no images are processed
                out_fused = torch.zeros(f_csv.size(0), 2, device=f_csv.device)

            out_imu_only = self.imu_only_classifier(f_csv)
            
            # Select the output from the correct head based on the gate's decision
            return use_images * out_fused + (1 - use_images) * out_imu_only, gate_prob

# --- 4. Training and Evaluation Logic ---

def train_gated_model(model, train_loader, val_loader, config):
    model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config['epochs']):
        model.train()
        for x_csv_b, x_img1_b, x_img2_b, y_b in train_loader:
            # Modality Dropout: Randomly zero out image data
            if model.training and np.random.rand() < config['dropout_prob']:
                x_img1_b.zero_()
                x_img2_b.zero_()

            optimizer.zero_grad()
            outputs = model(x_csv_b.to(config['device']), x_img1_b.to(config['device']), x_img2_b.to(config['device']))
            loss = criterion(outputs, y_b.to(config['device']))
            loss.backward(); optimizer.step()
        
        # Validation
        f1, _ = evaluate_gated_model(model, val_loader, config['device'], 0.5)
        logging.info(f"Epoch {epoch+1}/{config['epochs']} | Val F1 (Fall): {f1:.4f}")

def evaluate_gated_model(model, data_loader, device, threshold):
    model.to(device); model.eval()
    all_preds, all_labels, trigger_count = [], [], 0
    total_latency, imu_latency = 0, 0
    
    with torch.no_grad():
        for x_csv, x_img1, x_img2, y in data_loader:
            start_time = time.time()
            outputs, gate_probs = model(x_csv.to(device), x_img1.to(device), x_img2.to(device), threshold)
            end_time = time.time()
            total_latency += (end_time - start_time)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(y.numpy())
            trigger_count += (gate_probs.cpu().numpy() > threshold).sum()

    f1 = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)
    trigger_rate = trigger_count / len(all_labels)
    return f1, trigger_rate

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    OUTPUT_DIR = '/home/syed/PhD/UP-Fall-FL/Experimental_Scripts/Modules/output/Module4'
    os.makedirs(OUTPUT_DIR, exist_ok=True); setup_logging(OUTPUT_DIR)
    config = {'epochs': 10, 'learning_rate': 0.001, 'dropout_prob': 0.3,
              'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    
    data_splits = load_and_process_data('/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/')
    
    # --- Create DataLoaders ---
    train_loader = DataLoader(TensorDataset(torch.from_numpy(data_splits['X_train_csv']).float(), torch.from_numpy(data_splits['X_train_img1']).float(), torch.from_numpy(data_splits['X_train_img2']).float(), torch.from_numpy(data_splits['y_train']).long()), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(data_splits['X_val_csv']).float(), torch.from_numpy(data_splits['X_val_img1']).float(), torch.from_numpy(data_splits['X_val_img2']).float(), torch.from_numpy(data_splits['y_val']).long()), batch_size=128)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(data_splits['X_test_csv']).float(), torch.from_numpy(data_splits['X_test_img1']).float(), torch.from_numpy(data_splits['X_test_img2']).float(), torch.from_numpy(data_splits['y_test']).long()), batch_size=128)
    
    # Create a test set with images zeroed out
    X_test_img_absent = torch.zeros_like(torch.from_numpy(data_splits['X_test_img1']))
    absent_modality_loader = DataLoader(TensorDataset(torch.from_numpy(data_splits['X_test_csv']).float(), X_test_img_absent, X_test_img_absent, torch.from_numpy(data_splits['y_test']).long()), batch_size=128)
    
    # --- Assignment 1: Robustness to Missing Modality ---
    logging.info("\n--- Assignment 1: Robustness Evaluation ---")
    
    # Train a baseline model WITHOUT modality dropout
    logging.info("Training baseline model (no dropout)...")
    baseline_model = GatedResidualFusionModel(data_splits['X_train_csv'].shape[1])
    train_gated_model(baseline_model, train_loader, val_loader, {**config, 'dropout_prob': 0.0})
    
    # Train the robust model WITH modality dropout
    logging.info("Training robust model (with 30% dropout)...")
    robust_model = GatedResidualFusionModel(data_splits['X_train_csv'].shape[1])
    train_gated_model(robust_model, train_loader, val_loader, config)

    # Evaluate both on the RGB-absent test set
    f1_baseline_absent, _ = evaluate_gated_model(baseline_model, absent_modality_loader, config['device'], threshold=0.0) # Threshold=0 means always try to use images
    f1_robust_absent, _ = evaluate_gated_model(robust_model, absent_modality_loader, config['device'], threshold=0.0)
    
    robustness_results = pd.DataFrame([
        {'Model': 'Baseline (No Dropout)', 'F1 on RGB-Absent Data': f1_baseline_absent},
        {'Model': 'Robust (With Dropout)', 'F1 on RGB-Absent Data': f1_robust_absent}
    ])
    logging.info("\n--- Robustness Comparison ---")
    logging.info(f"\n{robustness_results.to_string(index=False)}")

    # --- Assignment 2: Latency/Energy Savings ---
    logging.info("\n--- Assignment 2: Gating Performance ---")
    f1_gated, trigger_rate = evaluate_gated_model(robust_model, test_loader, config['device'], threshold=0.5)
    
    # Simple energy/latency proxy
    cost_imu = 1; cost_image = 10 # Image processing is 10x more expensive
    cost_before_gating = cost_imu + 2 * cost_image
    cost_after_gating = cost_imu + (trigger_rate * 2 * cost_image)
    
    gating_results = pd.DataFrame([
        {'Metric': 'F1-Score (Fall)', 'Value': f"{f1_gated:.4f}"},
        {'Metric': 'Camera Trigger Rate', 'Value': f"{trigger_rate:.2%}"},
        {'Metric': 'Cost Before Gating (Proxy)', 'Value': f"{cost_before_gating}"},
        {'Metric': 'Cost After Gating (Proxy)', 'Value': f"{cost_after_gating:.2f}"},
        {'Metric': 'Savings', 'Value': f"{(1 - cost_after_gating/cost_before_gating):.2%}"}
    ])
    logging.info("\n--- Gating Performance (Threshold=0.5) ---")
    logging.info(f"\n{gating_results.to_string(index=False)}")

    # --- Stretch Goal: Learn τ to meet a budget ---
    logging.info("\n--- Stretch Goal: Learn Threshold τ for 40% Energy Budget ---")
    target_trigger_rate = 0.40 # Use camera on 40% of samples
    
    # Get all gate probabilities from the validation set
    robust_model.eval()
    all_gate_probs = []
    with torch.no_grad():
        for x_csv, x_img1, x_img2, y in val_loader:
             _, gate_probs = robust_model(x_csv.to(config['device']), x_img1.to(config['device']), x_img2.to(config['device']), threshold=0.0)
             all_gate_probs.extend(gate_probs.cpu().numpy().flatten())
    
    # Find the threshold that corresponds to the target trigger rate
    learned_threshold = np.quantile(all_gate_probs, 1 - target_trigger_rate)
    
    # Evaluate with the new learned threshold
    f1_budget, actual_trigger_rate = evaluate_gated_model(robust_model, test_loader, config['device'], learned_threshold)
    logging.info(f"Learned Threshold τ = {learned_threshold:.4f} to meet {target_trigger_rate:.0%} budget.")
    logging.info(f"Performance with learned τ: F1-Score={f1_budget:.4f}, Actual Trigger Rate={actual_trigger_rate:.2%}")

    logging.info(f"\n✅ All simulations complete. Results saved in '{OUTPUT_DIR}'.")