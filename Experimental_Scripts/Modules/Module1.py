import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. LOGGING & SEED SETUP
# =========================

def setup_logging(log_dir):
    """Configure logging to file and console."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, "run_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =========================
# 2. METRICS & PLOTTING
# =========================

def calculate_specificity(y_true, y_pred, labels=[0,1]):
    """Compute specificity (True Negative Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def plot_confusion_matrix(y_true, y_pred, title, filepath):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fall (0)', 'No Fall (1)'],
                yticklabels=['Fall (0)', 'No Fall (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# =========================
# 3. DATA LOADING & PREP
# =========================

def create_windows(data, labels, window_size, step):
    """Create sliding windows for time-series data."""
    X, y = [], []
    for i in range(0, len(data) - window_size, step):
        window_data = data[i:i + window_size]
        window_label = labels[i:i + window_size]
        # If any label in window is 'fall', label window as fall (0)
        label = 0 if np.any(window_label == 0) else 1
        X.append(window_data)
        y.append(label)
    return np.array(X), np.array(y)

def load_and_preprocess_data(file_path, window_size=200, step=100):
    """Load CSV, clean, window, and scale data."""
    logging.info(f"Loading data from: {file_path}")
    df = pd.read_csv(file_path, header=[0, 1])

    # Clean column names
    cleaned_columns = []
    last_val = ''
    for col_l1, col_l2 in df.columns:
        if 'Unnamed' in col_l1:
            col_l1 = last_val
        else:
            last_val = col_l1.strip()
        cleaned_columns.append(f"{col_l1.strip()}_{col_l2.strip()}" if col_l1 != col_l2 else col_l1)
    df.columns = cleaned_columns

    # Remove problematic subjects/trials
    df = df[~df['Subject'].isin([5, 9])]
    df = df[~((df['Subject'] == 2) & (df['Activity'] == 5))]
    df = df[~((df['Subject'] == 8) & (df['Activity'] == 11) & (df['Trial'].isin([2, 3])))]

    # Drop unnecessary columns
    cols_to_drop = [col for col in df.columns if any(x in col for x in ['Infrared', 'Tag', 'Time', 'Trial'])]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Label: 0 = Fall, 1 = No Fall
    fall_activity_ids = {2, 3, 4, 5, 6}
    df['Fall'] = df['Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)

    # IMU sensor columns
    imu_clients = {
        'Ankle_IMU': [
            'AnkleAccelerometer_x-axis (g)', 'AnkleAccelerometer_y-axis (g)', 'AnkleAccelerometer_z-axis (g)',
            'AnkleAngularVelocity_x-axis (deg/s)', 'AnkleAngularVelocity_y-axis (deg/s)', 'AnkleAngularVelocity_z-axis (deg/s)'
        ],
        'Pocket_IMU': [
            'RightPocketAccelerometer_x-axis (g)', 'RightPocketAccelerometer_y-axis (g)', 'RightPocketAccelerometer_z-axis (g)',
            'RightPocketAngularVelocity_x-axis (deg/s)', 'RightPocketAngularVelocity_y-axis (deg/s)', 'RightPocketAngularVelocity_z-axis (deg/s)'
        ],
        'Belt_IMU': [
            'BeltAccelerometer_x-axis (g)', 'BeltAccelerometer_y-axis (g)', 'BeltAccelerometer_z-axis (g)',
            'BeltAngularVelocity_x-axis (deg/s)', 'BeltAngularVelocity_y-axis (deg/s)', 'BeltAngularVelocity_z-axis (deg/s)'
        ],
        'Neck_IMU': [
            'NeckAccelerometer_x-axis (g)', 'NeckAccelerometer_y-axis (g)', 'NeckAccelerometer_z-axis (g)',
            'NeckAngularVelocity_x-axis (deg/s)', 'NeckAngularVelocity_y-axis (deg/s)', 'NeckAngularVelocity_z-axis (deg/s)'
        ],
        'Wrist_IMU': [
            'WristAccelerometer_x-axis (g)', 'WristAccelerometer_y-axis (g)', 'WristAccelerometer_z-axis (g)',
            'WristAngularVelocity_x-axis (deg/s)', 'WristAngularVelocity_y-axis (deg/s)', 'WristAngularVelocity_z-axis (deg/s)'
        ]
    }

    # Split subjects for train/test
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
    test_subjects = [s for s in range(14, 18)]

    X_train_w, y_train_w, X_test_w, y_test_w = [], [], [], []

    logging.info("Creating training windows...")
    for subject in train_subjects:
        subject_df = df[df['Subject'] == subject]
        for columns in imu_clients.values():
            X_subject, y_subject = create_windows(subject_df[columns].values, subject_df['Fall'].values, window_size, step)
            if len(X_subject) > 0:
                X_train_w.append(X_subject)
                y_train_w.append(y_subject)

    logging.info("Creating testing windows...")
    for subject in test_subjects:
        subject_df = df[df['Subject'] == subject]
        for columns in imu_clients.values():
            X_subject, y_subject = create_windows(subject_df[columns].values, subject_df['Fall'].values, window_size, step)
            if len(X_subject) > 0:
                X_test_w.append(X_subject)
                y_test_w.append(y_subject)

    # Stack and scale
    X_train = np.vstack(X_train_w)
    y_train = np.concatenate(y_train_w)
    X_test = np.vstack(X_test_w)
    y_test = np.concatenate(y_test_w)

    scaler = StandardScaler()
    num_instances, window_len, num_features = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(num_instances, window_len, num_features)
    num_instances_test = X_test.shape[0]
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(num_instances_test, window_len, num_features)

    # Transpose for PyTorch (batch, channels, seq_len)
    X_train_final = np.transpose(X_train_scaled, (0, 2, 1))
    X_test_final = np.transpose(X_test_scaled, (0, 2, 1))

    logging.info(f"Train shape: {X_train_final.shape}, Test shape: {X_test_final.shape}")
    logging.info(f"Fall events (0) in train: {np.sum(y_train == 0)}, in test: {np.sum(y_test == 0)}")
    return X_train_final, y_train, X_test_final, y_test

# =========================
# 4. MODEL DEFINITIONS
# =========================

class CNN_GAP(nn.Module):
    """1D CNN with Global Average Pooling."""
    def __init__(self, input_channels=6):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 5, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, 5, padding='same')
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for sequence data."""
    def __init__(self, in_features):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x_permuted = x.permute(0, 2, 1)  # (batch, seq_len, channels)
        attn_weights = torch.softmax(self.attention_net(x_permuted), dim=1)
        attended = torch.sum(x_permuted * attn_weights, dim=1)
        return attended

class CNN_Attention(nn.Module):
    """1D CNN with Temporal Attention."""
    def __init__(self, input_channels=6):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, 5, padding='same')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, 5, padding='same')
        self.relu2 = nn.ReLU()
        self.attention = TemporalAttention(64)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.attention(x)
        return self.fc(x)

# =========================
# 5. TRAINING & EVALUATION
# =========================

def train_model(model, train_loader, epochs, learning_rate, device):
    """Train model and log progress."""
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = total_correct / total_samples
        logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    logging.info("Training complete.")

def evaluate_model(model, test_loader, device):
    """Evaluate model and return metrics."""
    model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy())

    metrics = {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "F1-Score (Fall)": f1_score(all_labels, all_preds, pos_label=0),
        "Sensitivity (Fall)": recall_score(all_labels, all_preds, pos_label=0),
        "Specificity (No Fall)": calculate_specificity(all_labels, all_preds, labels=[0, 1])
    }
    return metrics, all_labels, all_preds

# =========================
# 6. MAIN EXECUTION
# =========================

if __name__ == "__main__":
    # Output directory
    OUTPUT_DIR = '/home/syed/PhD/UP-Fall-FL/Experimental_Scripts/Modules/output/Module1'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(OUTPUT_DIR)

    # Hyperparameters
    set_seed(42)
    FILE_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv'
    WINDOW_SIZE = 200
    STEP = 100
    BATCH_SIZE = 128
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")

    try:
        # Data loading
        X_train, y_train, X_test, y_test = load_and_preprocess_data(FILE_PATH, WINDOW_SIZE, STEP)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()), shuffle=True, batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), shuffle=False, batch_size=BATCH_SIZE)

        # --- Model 1: CNN-GAP ---
        logging.info("Training Model 1: CNN with Global Average Pooling")
        model_gap = CNN_GAP(input_channels=X_train.shape[1])
        train_model(model_gap, train_loader, EPOCHS, LEARNING_RATE, DEVICE)

        logging.info("Evaluating Model 1...")
        metrics_gap, labels_gap, preds_gap = evaluate_model(model_gap, test_loader, DEVICE)
        with open(os.path.join(OUTPUT_DIR, 'metrics_gap.json'), 'w') as f:
            json.dump(metrics_gap, f, indent=4)
        for name, value in metrics_gap.items():
            logging.info(f"[CNN-GAP] {name}: {value:.4f}")
        plot_confusion_matrix(labels_gap, preds_gap, "Confusion Matrix (CNN-GAP)", os.path.join(OUTPUT_DIR, 'cm_gap.png'))

        # --- Model 2: CNN-Attention ---
        logging.info("Training Model 2: CNN with Temporal Attention")
        model_attn = CNN_Attention(input_channels=X_train.shape[1])
        train_model(model_attn, train_loader, EPOCHS, LEARNING_RATE, DEVICE)

        logging.info("Evaluating Model 2...")
        metrics_attn, labels_attn, preds_attn = evaluate_model(model_attn, test_loader, DEVICE)
        with open(os.path.join(OUTPUT_DIR, 'metrics_attn.json'), 'w') as f:
            json.dump(metrics_attn, f, indent=4)
        for name, value in metrics_attn.items():
            logging.info(f"[CNN-Attention] {name}: {value:.4f}")
        plot_confusion_matrix(labels_attn, preds_attn, "Confusion Matrix (CNN-Attention)", os.path.join(OUTPUT_DIR, 'cm_attn.png'))

        # --- Results Comparison ---
        logging.info("Final Comparison:")
        results_df = pd.DataFrame([metrics_gap, metrics_attn], index=['CNN-GAP', 'CNN-Attention'])
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'final_comparison.csv'))
        logging.info(f"\n{results_df.to_string()}")
        logging.info(f"All results, logs, and plots saved to: {OUTPUT_DIR}")

    except FileNotFoundError:
        logging.error(f"ERROR: '{FILE_PATH}' not found. Please check the file path.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)