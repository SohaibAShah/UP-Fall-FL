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
import copy

# --- 1. Setup and Helper Functions ---

def setup_logging(log_dir):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    if root_logger.hasHandlers(): root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(log_dir, "fl_module4_log.txt"))
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

def add_noise_to_images(image_data, noise_level=0.5):
    """Adds Gaussian noise to images to simulate a noisy sensor."""
    noisy_images = image_data + np.random.normal(0, noise_level, image_data.shape)
    return np.clip(noisy_images, 0., 1.)

# --- 2. Data Loading and Preprocessing ---

def load_and_process_data(data_dir):
    """Loads and processes sensor data and data from two cameras."""
    logging.info("Loading and processing data...")
    # Load and clean sensor data
    sub = pd.read_csv(os.path.join(data_dir, 'sensor.csv'), header=[0, 1])
    cleaned_columns = [f"{c[0].strip()}_{c[1].strip()}" if 'Unnamed' not in c[0] else last_val + f"_{c[1].strip()}" for c in sub.columns if 'Unnamed' not in c[0] or (last_val := c[0].strip())]
    sub.columns = [c.replace(c.split('_')[0] + '_', '') if c.split('_')[0] == c.split('_')[1] else c for c in cleaned_columns]
    sub.dropna(inplace=True); sub.drop_duplicates(inplace=True)

    # Load image data for both cameras
    img1 = np.load(os.path.join(data_dir, 'image_1.npy'))
    name1 = np.load(os.path.join(data_dir, 'name_1.npy'))
    img2 = np.load(os.path.join(data_dir, 'image_2.npy'))
    
    # Align timestamps using a faster, set-based approach
    logging.info("Aligning timestamps (optimized method)...")
    sensor_ts = set(sub['TimeStamps_Time'])
    name1_ts = set(name1)
    common_timestamps = sorted(list(sensor_ts.intersection(name1_ts)))

    # Filter all data sources ONCE
    sub = sub[sub['TimeStamps_Time'].isin(common_timestamps)]
    
    name1_map = {ts: i for i, ts in enumerate(name1)}
    idx1 = [name1_map[ts] for ts in common_timestamps]
    img1 = img1[idx1]
    img2 = img2[idx1] # Filter img2 using the same indices as img1

    # Align the sensor dataframe to the exact order
    sub = sub.set_index('TimeStamps_Time').loc[common_timestamps]
    
    feature_cols = [col for col in sub.columns if 'Infrared' not in col and col not in ['Activity', 'Subject', 'Trial', 'Tag']]
    X_csv = sub[feature_cols].values
    y = sub['Activity'].values
    
    fall_activity_ids = {2, 3, 4, 5, 6}
    y_binary = np.array([0 if activity_id in fall_activity_ids else 1 for activity_id in y])
    
    set_seed(42)
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
    def __init__(self, input_channels, feature_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 16, 3, padding=1); self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1); self.fc = nn.Linear(32, feature_dim)
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

class EarlyFusionModel(nn.Module):
    def __init__(self, num_csv_features, num_classes=2):
        super().__init__()
        self.imu_encoder = IMUEncoder(num_csv_features, 32)
        self.img_encoder1 = ImageEncoder(64)
        self.img_encoder2 = ImageEncoder(64)
        self.classifier = nn.Sequential(nn.Linear(32 + 64 + 64, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))
    def forward(self, x_csv, x_img1, x_img2):
        f_csv = self.imu_encoder(x_csv); f_img1 = self.img_encoder1(x_img1); f_img2 = self.img_encoder2(x_img2)
        fused = torch.cat((f_csv, f_img1, f_img2), dim=1)
        return self.classifier(fused)

class LateFusionModel(nn.Module):
    def __init__(self, num_csv_features, num_classes=2):
        super().__init__()
        self.imu_branch = nn.Sequential(IMUEncoder(num_csv_features, 32), nn.Linear(32, num_classes))
        self.img_branch1 = nn.Sequential(ImageEncoder(64), nn.Linear(64, num_classes))
        self.img_branch2 = nn.Sequential(ImageEncoder(64), nn.Linear(64, num_classes))
        self.fusion_layer = nn.Linear(num_classes * 3, num_classes)
    def forward(self, x_csv, x_img1, x_img2):
        p_csv = self.imu_branch(x_csv); p_img1 = self.img_branch1(x_img1); p_img2 = self.img_branch2(x_img2)
        return self.fusion_layer(torch.cat((p_csv, p_img1, p_img2), dim=1))

class ResidualFusionModel(nn.Module):
    def __init__(self, num_csv_features, num_classes=2):
        super().__init__()
        self.imu_encoder = IMUEncoder(num_csv_features, 128)
        self.img_encoder1 = ImageEncoder(64)
        self.img_encoder2 = ImageEncoder(64)
        self.img_fusion = nn.Linear(64 + 64, 128)
        self.gate = nn.Sequential(nn.Linear(128 + 64 + 64, 128), nn.ReLU(), nn.Linear(128, 128), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, num_classes))
    def forward(self, x_csv, x_img1, x_img2, return_gate=False):
        f_csv = self.imu_encoder(x_csv); f_img1 = self.img_encoder1(x_img1); f_img2 = self.img_encoder2(x_img2)
        f_img_combined = F.relu(self.img_fusion(torch.cat((f_img1, f_img2), dim=1)))
        gate_val = self.gate(torch.cat((f_csv, f_img1, f_img2), dim=1))
        fused = f_csv + (gate_val * f_img_combined)
        output = self.classifier(fused)
        return (output, gate_val) if return_gate else output

# --- 4. Training and Evaluation Logic ---

def train_model(model, train_loader, val_loader, config):
    model.to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    for epoch in range(config['epochs']):
        model.train()
        # **FIX:** Unpack all four items from the DataLoader
        for x_csv_b, x_img1_b, x_img2_b, y_b in train_loader:
            optimizer.zero_grad()
            # **FIX:** Pass all three input tensors to the model
            outputs = model(x_csv_b.to(config['device']), x_img1_b.to(config['device']), x_img2_b.to(config['device']))
            loss = criterion(outputs, y_b.to(config['device']))
            loss.backward(); optimizer.step()
        
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            # **FIX:** Unpack all four items from the DataLoader
            for x_csv_b, x_img1_b, x_img2_b, y_b in val_loader:
                # **FIX:** Pass all three input tensors to the model
                outputs = model(x_csv_b.to(config['device']), x_img1_b.to(config['device']), x_img2_b.to(config['device']))
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds); all_labels.extend(y_b.numpy())
        f1 = f1_score(all_labels, all_preds, pos_label=0); logging.info(f"Epoch {epoch+1}/{config['epochs']} | Val F1 (Fall): {f1:.4f}")

def evaluate_model(model, data_loader, device):
    model.to(device); model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        # **FIX:** Unpack all four items from the DataLoader
        for x_csv, x_img1, x_img2, y in data_loader:
            # **FIX:** Pass all three input tensors to the model
            outputs = model(x_csv.to(device), x_img1.to(device), x_img2.to(device))
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(y.numpy())
    return classification_report(all_labels, all_preds, target_names=['Fall', 'No Fall'], output_dict=True, zero_division=0)

# --- 5. Main Execution Block ---

if __name__ == "__main__":
    OUTPUT_DIR = '/home/syed/PhD/UP-Fall-FL/Experimental_Scripts/Modules/output/Module4'
    os.makedirs(OUTPUT_DIR, exist_ok=True); setup_logging(OUTPUT_DIR)
    config = {'epochs': 10, 'learning_rate': 0.001, 'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")}
    
    data_splits = load_and_process_data('/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/')
    
    train_loader = DataLoader(TensorDataset(
        torch.from_numpy(data_splits['X_train_csv']).float(),
        torch.from_numpy(data_splits['X_train_img1']).float(),
        torch.from_numpy(data_splits['X_train_img2']).float(),
        torch.from_numpy(data_splits['y_train']).long()
    ), batch_size=64, shuffle=True)
    
    val_loader = DataLoader(TensorDataset(
        torch.from_numpy(data_splits['X_val_csv']).float(),
        torch.from_numpy(data_splits['X_val_img1']).float(),
        torch.from_numpy(data_splits['X_val_img2']).float(),
        torch.from_numpy(data_splits['y_val']).long()
    ), batch_size=128)
    
    test_loader = DataLoader(TensorDataset(
        torch.from_numpy(data_splits['X_test_csv']).float(),
        torch.from_numpy(data_splits['X_test_img1']).float(),
        torch.from_numpy(data_splits['X_test_img2']).float(),
        torch.from_numpy(data_splits['y_test']).long()
    ), batch_size=128)

    X_test_img1_noisy = add_noise_to_images(data_splits['X_test_img1'])
    X_test_img2_noisy = add_noise_to_images(data_splits['X_test_img2'])
    noisy_test_loader = DataLoader(TensorDataset(
        torch.from_numpy(data_splits['X_test_csv']).float(),
        torch.from_numpy(X_test_img1_noisy).float(),
        torch.from_numpy(X_test_img2_noisy).float(),
        torch.from_numpy(data_splits['y_test']).long()
    ), batch_size=128)
    
    models = {
        'EarlyFusion': EarlyFusionModel(data_splits['X_train_csv'].shape[1]),
        'LateFusion': LateFusionModel(data_splits['X_train_csv'].shape[1]),
        'ResidualFusion': ResidualFusionModel(data_splits['X_train_csv'].shape[1])
    }
    
    results = []
    for name, model in models.items():
        logging.info(f"\n--- Training {name} Model ---")
        train_model(model, train_loader, val_loader, config)
        
        logging.info(f"--- Evaluating {name} on Clean Data ---")
        clean_report = evaluate_model(model, test_loader, config['device'])
        
        logging.info(f"--- Evaluating {name} on Noisy Data ---")
        noisy_report = evaluate_model(model, noisy_test_loader, config['device'])
        
        results.append({'Model': name, 'Clean F1 (Fall)': clean_report['Fall']['f1-score'], 'Noisy F1 (Fall)': noisy_report['Fall']['f1-score'], 'Performance Drop': clean_report['Fall']['f1-score'] - noisy_report['Fall']['f1-score']})
        
    results_df = pd.DataFrame(results)
    logging.info(f"\n--- Final Comparison ---\n\n{results_df.to_string(index=False)}")
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'final_comparison.csv'), index=False)
    
    # Stretch Goal: Visualize Gate Values (No changes needed here, but loop is included for completeness)
    logging.info("\n--- Running Stretch Goal: Visualizing Gate Values ---")
    residual_model = models['ResidualFusion']
    residual_model.eval()
    gate_values, motion_intensities = [], []
    with torch.no_grad():
        # **FIX:** Unpack all four items from the DataLoader
        for x_csv_b, x_img1_b, x_img2_b, y_b in test_loader:
            # **FIX:** Pass all three input tensors to the model
            _, gates = residual_model(x_csv_b.to(config['device']), x_img1_b.to(config['device']), x_img2_b.to(config['device']), return_gate=True)
            motion = torch.mean(torch.abs(x_csv_b[:, 3:6]), dim=1).cpu().numpy()
            gate_values.extend(torch.mean(gates, dim=1).cpu().numpy())
            motion_intensities.extend(motion)
            
    plt.figure(figsize=(10, 6))
    plt.scatter(motion_intensities, gate_values, alpha=0.5)
    plt.title('Residual Fusion: Gate Value vs. Motion Intensity')
    plt.xlabel('Motion Intensity (Avg. Gyro Magnitude)')
    plt.ylabel('Average Gate Value (Trust in Image Modalities)')
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'gate_visualization.png'))
    plt.show()

    logging.info(f"\n✅ All simulations complete. Results saved in '{OUTPUT_DIR}'.")