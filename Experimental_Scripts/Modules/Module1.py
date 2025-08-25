# Imports
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
import os
import json
import logging
import sys

# --- 1. Setup and Helper Functions ---

def setup_logging(log_dir):
    """Configures logging to save to a file and print to the console."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # File handler to save logs
    file_handler = logging.FileHandler(os.path.join(log_dir, "run_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Stream handler to print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def calculate_specificity(y_true, y_pred, labels=[0,1]):
    """Calculates the specificity metric."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0

def plot_confusion_matrix(y_true, y_pred, title, filepath):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Fall (0)', 'No Fall (1)'],
                yticklabels=['Fall (0)', 'No Fall (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.savefig(filepath) # Save the figure
    plt.show()
    plt.close() # Close the figure to free up memory

# --- 2. Data Loading and Preprocessing (Unchanged) ---
def create_windows(data, labels, window_size, step):
    X, y = [], []
    for i in range(0, len(data) - window_size, step):
        window_data, window_label = data[i:i + window_size], labels[i:i + window_size]
        label = 0 if np.any(window_label == 0) else 1
        X.append(window_data)
        y.append(label)
    return np.array(X), np.array(y)

def load_and_preprocess_data(file_path, window_size=200, step=100):
    logging.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=[0, 1])
    cleaned_columns = []
    last_val = ''
    for col_l1, col_l2 in df.columns:
        if 'Unnamed' in col_l1: col_l1 = last_val
        else: last_val = col_l1.strip()
        cleaned_columns.append(f"{col_l1.strip()}_{col_l2.strip()}" if col_l1 != col_l2 else col_l1)
    df.columns = cleaned_columns
    df = df[~df['Subject'].isin([5, 9])]
    df = df[~((df['Subject'] == 2) & (df['Activity'] == 5))]
    df = df[~((df['Subject'] == 8) & (df['Activity'] == 11) & (df['Trial'].isin([2, 3])))]
    cols_to_drop = [col for col in df.columns if 'Infrared' in col or 'Tag' in col or 'Time' in col or 'Trial' in col]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    df.dropna(inplace=True); df.drop_duplicates(inplace=True)
    fall_activity_ids = {2, 3, 4, 5, 6}
    df['Fall'] = df['Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
    imu_clients = {
        'Ankle_IMU': ['AnkleAccelerometer_x-axis (g)', 'AnkleAccelerometer_y-axis (g)', 'AnkleAccelerometer_z-axis (g)', 'AnkleAngularVelocity_x-axis (deg/s)', 'AnkleAngularVelocity_y-axis (deg/s)', 'AnkleAngularVelocity_z-axis (deg/s)'],
        'Pocket_IMU': ['RightPocketAccelerometer_x-axis (g)', 'RightPocketAccelerometer_y-axis (g)', 'RightPocketAccelerometer_z-axis (g)', 'RightPocketAngularVelocity_x-axis (deg/s)', 'RightPocketAngularVelocity_y-axis (deg/s)', 'RightPocketAngularVelocity_z-axis (deg/s)'],
        'Belt_IMU': ['BeltAccelerometer_x-axis (g)', 'BeltAccelerometer_y-axis (g)', 'BeltAccelerometer_z-axis (g)', 'BeltAngularVelocity_x-axis (deg/s)', 'BeltAngularVelocity_y-axis (deg/s)', 'BeltAngularVelocity_z-axis (deg/s)'],
        'Neck_IMU': ['NeckAccelerometer_x-axis (g)', 'NeckAccelerometer_y-axis (g)', 'NeckAccelerometer_z-axis (g)', 'NeckAngularVelocity_x-axis (deg/s)', 'NeckAngularVelocity_y-axis (deg/s)', 'NeckAngularVelocity_z-axis (deg/s)'],
        'Wrist_IMU': ['WristAccelerometer_x-axis (g)', 'WristAccelerometer_y-axis (g)', 'WristAccelerometer_z-axis (g)', 'WristAngularVelocity_x-axis (deg/s)', 'WristAngularVelocity_y-axis (deg/s)', 'WristAngularVelocity_z-axis (deg/s)']
    }
    train_subjects, test_subjects = [s for s in range(1, 14) if s not in [5, 9]], [s for s in range(14, 18)]
    X_train_w, y_train_w, X_test_w, y_test_w = [], [], [], []
    logging.info("Creating training windows...")
    for subject in train_subjects:
        subject_df = df[df['Subject'] == subject]
        for _, columns in imu_clients.items():
            X_subject, y_subject = create_windows(subject_df[columns].values, subject_df['Fall'].values, window_size, step)
            if len(X_subject) > 0: X_train_w.append(X_subject); y_train_w.append(y_subject)
    logging.info("Creating testing windows...")
    for subject in test_subjects:
        subject_df = df[df['Subject'] == subject]
        for _, columns in imu_clients.items():
            X_subject, y_subject = create_windows(subject_df[columns].values, subject_df['Fall'].values, window_size, step)
            if len(X_subject) > 0: X_test_w.append(X_subject); y_test_w.append(y_subject)
    X_train, y_train, X_test, y_test = np.vstack(X_train_w), np.concatenate(y_train_w), np.vstack(X_test_w), np.concatenate(y_test_w)
    scaler = StandardScaler()
    num_instances, window_len, num_features = X_train.shape
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(num_instances, window_len, num_features)
    num_instances_test, _, _ = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, num_features)).reshape(num_instances_test, window_len, num_features)
    X_train_final, X_test_final = np.transpose(X_train_scaled, (0, 2, 1)), np.transpose(X_test_scaled, (0, 2, 1))
    logging.info(f"Train data shape: {X_train_final.shape}, Test data shape: {X_test_final.shape}")
    logging.info(f"Fall events (0) in train: {np.sum(y_train == 0)}, in test: {np.sum(y_test == 0)}")
    return X_train_final, y_train, X_test_final, y_test

# --- 3. Model Architectures (Unchanged) ---
class CNN_GAP(nn.Module):
    def __init__(self, input_channels=6):
        super(CNN_GAP, self).__init__()
        self.conv1, self.relu1 = nn.Conv1d(input_channels, 32, 5, padding='same'), nn.ReLU()
        self.conv2, self.relu2 = nn.Conv1d(32, 64, 5, padding='same'), nn.ReLU()
        self.pool, self.fc = nn.AdaptiveAvgPool1d(1), nn.Linear(64, 1)
    def forward(self, x):
        x = self.relu1(self.conv1(x)); x = self.relu2(self.conv2(x)); x = self.pool(x)
        return self.fc(x.view(x.size(0), -1))

class TemporalAttention(nn.Module):
    def __init__(self, in_features):
        super(TemporalAttention, self).__init__()
        self.attention_net = nn.Sequential(nn.Linear(in_features, in_features // 2), nn.Tanh(), nn.Linear(in_features // 2, 1))
    def forward(self, x):
        x_permuted = x.permute(0, 2, 1)
        attn_weights = torch.softmax(self.attention_net(x_permuted), dim=1)
        return torch.sum(x_permuted * attn_weights, dim=1)

class CNN_Attention(nn.Module):
    def __init__(self, input_channels=6):
        super(CNN_Attention, self).__init__()
        self.conv1, self.relu1 = nn.Conv1d(input_channels, 32, 5, padding='same'), nn.ReLU()
        self.conv2, self.relu2 = nn.Conv1d(32, 64, 5, padding='same'), nn.ReLU()
        self.attention, self.fc = TemporalAttention(64), nn.Linear(64, 1)
    def forward(self, x):
        x = self.relu1(self.conv1(x)); x = self.relu2(self.conv2(x))
        return self.fc(self.attention(x))

# --- 4. Training and Evaluation Logic ---

def train_model(model, train_loader, epochs, learning_rate, device):
    """Trains the model and prints loss and accuracy per epoch."""
    model.to(device)
    criterion, optimizer = nn.BCEWithLogitsLoss(), optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss, total_correct, total_samples = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate training accuracy for the batch
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
        epoch_loss = running_loss / total_samples
        epoch_acc = total_correct / total_samples
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        
    logging.info("Training finished.")

def evaluate_model(model, test_loader, device):
    """Evaluates the model and returns metrics."""
    model.to(device); model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(device))
            preds = (torch.sigmoid(outputs).cpu().numpy() > 0.5).astype(int)
            all_preds.extend(preds.flatten()); all_labels.extend(labels.numpy())
            
    return {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "F1-Score (Fall)": f1_score(all_labels, all_preds, pos_label=0),
        "Sensitivity (Fall)": recall_score(all_labels, all_preds, pos_label=0),
        "Specificity (No Fall)": calculate_specificity(all_labels, all_preds, labels=[0, 1])
    }, all_labels, all_preds

# --- 5. Main Execution Block ---
if __name__ == "__main__":
    # Setup output directory and logging
    OUTPUT_DIR = '/home/syed/PhD/UP-Fall-FL/Experimental_Scripts/Modules/output/Module1'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging(OUTPUT_DIR)

    # Hyperparameters
    set_seed(42)
    FILE_PATH = '/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/sensor.csv'sensor.csv' # IMPORTANT: Ensure 'sensor.csv' is in the same directory
    WINDOW_SIZE, STEP, BATCH_SIZE, EPOCHS, LEARNING_RATE = 200, 100, 128, 50, 0.001
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {DEVICE}")

    try:
        X_train, y_train, X_test, y_test = load_and_preprocess_data(FILE_PATH, WINDOW_SIZE, STEP)
        train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long()), shuffle=True, batch_size=BATCH_SIZE)
        test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), shuffle=False, batch_size=BATCH_SIZE)

        # --- Experiment 1: 1D-CNN with Global Average Pooling ---
        logging.info("\n--- Training Model 1: CNN with Global Average Pooling ---")
        model_gap = CNN_GAP(input_channels=X_train.shape[1])
        train_model(model_gap, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
        
        logging.info("\n--- Evaluating Model 1 ---")
        metrics_gap, labels_gap, preds_gap = evaluate_model(model_gap, test_loader, DEVICE)
        with open(os.path.join(OUTPUT_DIR, 'metrics_gap.json'), 'w') as f:
            json.dump(metrics_gap, f, indent=4)
        logging.info("Metrics (CNN-GAP):")
        for name, value in metrics_gap.items(): logging.info(f"  - {name}: {value:.4f}")
        plot_confusion_matrix(labels_gap, preds_gap, "Confusion Matrix (CNN-GAP)", os.path.join(OUTPUT_DIR, 'cm_gap.png'))

        # --- Experiment 2: 1D-CNN with Temporal Attention ---
        logging.info("\n--- Training Model 2: CNN with Temporal Attention ---")
        model_attn = CNN_Attention(input_channels=X_train.shape[1])
        train_model(model_attn, train_loader, EPOCHS, LEARNING_RATE, DEVICE)
        
        logging.info("\n--- Evaluating Model 2 ---")
        metrics_attn, labels_attn, preds_attn = evaluate_model(model_attn, test_loader, DEVICE)
        with open(os.path.join(OUTPUT_DIR, 'metrics_attn.json'), 'w') as f:
            json.dump(metrics_attn, f, indent=4)
        logging.info("Metrics (CNN-Attention):")
        for name, value in metrics_attn.items(): logging.info(f"  - {name}: {value:.4f}")
        plot_confusion_matrix(labels_attn, preds_attn, "Confusion Matrix (CNN-Attention)", os.path.join(OUTPUT_DIR, 'cm_attn.png'))
        
        # --- Comparison ---
        logging.info("\n--- Final Comparison ---")
        results_df = pd.DataFrame([metrics_gap, metrics_attn], index=['CNN-GAP', 'CNN-Attention'])
        results_df.to_csv(os.path.join(OUTPUT_DIR, 'final_comparison.csv'))
        logging.info(f"\n{results_df.to_string()}")
        logging.info(f"\nAll results, logs, and plots have been saved to the '{OUTPUT_DIR}' directory.")

    except FileNotFoundError:
        logging.error(f"\nERROR: '{FILE_PATH}' not found. Please place it in the same directory.")
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred: {e}", exc_info=True)