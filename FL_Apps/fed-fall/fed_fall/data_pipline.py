import os
import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch

# --- Constants for data processing ---
WINDOW_SIZE = 200
STEP_SIZE = 100

def create_multimodal_windows(df, feature_cols, img1_map, img2_map):
    """
    Creates correctly aligned windows of IMU data with their corresponding
    single images from the center of the window.
    """
    X_csv_w, X_img1_w, X_img2_w, y_w = [], [], [], []
    timestamps = df.index
    
    for i in range(0, len(df) - WINDOW_SIZE, STEP_SIZE):
        # Define the window for sensor data and labels
        imu_window = df.iloc[i:i + WINDOW_SIZE][feature_cols].values
        labels_in_window = df.iloc[i:i + WINDOW_SIZE]['Fall'].values
        
        # Use the timestamp from the center of the window to find the corresponding image
        center_timestamp = timestamps[i + WINDOW_SIZE // 2]
        
        # Ensure the timestamp exists in our image maps
        if center_timestamp in img1_map and center_timestamp in img2_map:
            X_csv_w.append(imu_window)
            X_img1_w.append(img1_map[center_timestamp])
            X_img2_w.append(img2_map[center_timestamp])
            # A window is a "Fall" (0) if any fall event is present within it
            y_w.append(0 if np.any(labels_in_window == 0) else 1)

    if not X_csv_w: # Handle cases where no windows could be created
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Reshape for PyTorch: (N, C, L) for IMU and (N, C, H, W) for images
    X_csv_w = np.transpose(np.array(X_csv_w), (0, 2, 1)).astype(np.float32)
    X_img1_w = np.expand_dims((np.array(X_img1_w) / 255.0).astype(np.float32), axis=1)
    X_img2_w = np.expand_dims((np.array(X_img2_w) / 255.0).astype(np.float32), axis=1)
    
    return X_csv_w, X_img1_w, X_img2_w, np.array(y_w)

def load_and_partition_upfall_data(data_dir: str):
    """
    The main function to load, clean, align, and partition the entire dataset.
    """
    logging.info("--- Step 1: Loading and Aligning Raw Data ---")
    
    # Load raw data files
    sensor_df = pd.read_csv(os.path.join(data_dir, 'sensor.csv'), header=[0, 1])
    # Clean column names robustly
    sensor_df.columns = ["_".join(col).strip() if col[0] != col[1] else col[0] for col in sensor_df.columns.values]
    sensor_df.columns = sensor_df.columns.str.replace(r'Unnamed: \d+_level_0_', '', regex=True)
    sensor_df.dropna(inplace=True); sensor_df.drop_duplicates(inplace=True)

    img1 = np.load(os.path.join(data_dir, 'image_1.npy'))
    name1 = np.load(os.path.join(data_dir, 'name_1.npy'))
    img2 = np.load(os.path.join(data_dir, 'image_2.npy'))

    # Align timestamps using a fast, set-based method
    sensor_ts = set(sensor_df['Time'])
    name1_ts = set(name1)
    common_timestamps = sorted(list(sensor_ts.intersection(name1_ts)))
    
    sensor_df = sensor_df[sensor_df['Time'].isin(common_timestamps)].set_index('Time').loc[common_timestamps]
    
    img1_map = {ts: img for ts, img in zip(name1, img1)}
    img2_map = {ts: img for ts, img in zip(name1, img2)} # Assumes name1 and name2 are aligned

    logging.info("--- Step 2: Creating Binary Labels and Splitting by Subject ---")
    fall_activity_ids = {2, 3, 4, 5, 6}
    sensor_df['Fall'] = sensor_df['Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
    
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
    test_subjects = [s for s in range(14, 18)]
    train_df = sensor_df[sensor_df['Subject'].isin(train_subjects)]
    test_df = sensor_df[sensor_df['Subject'].isin(test_subjects)]

    feature_cols = [col for col in sensor_df.columns if 'Accelerometer' in col or 'AngularVelocity' in col]
    
    logging.info("--- Step 3: Scaling Features ---")
    scaler = StandardScaler().fit(train_df[feature_cols])
    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    logging.info("--- Step 4: Creating Global Test Set ---")
    test_data = create_multimodal_windows(test_df, feature_cols, img1_map, img2_map)
    
    logging.info("--- Step 5: Partitioning Data into Federated Clients ---")
    client_partitions = {}
    for cid in train_subjects:
        client_df = train_df[train_df['Subject'] == cid]
        if client_df.empty: continue
        
        client_data = create_multimodal_windows(client_df, feature_cols, img1_map, img2_map)
        if len(client_data[0]) > 0:
            client_partitions[str(cid)] = client_data

    logging.info(f"Data loading complete. {len(client_partitions)} clients created.")
    return client_partitions, test_data, len(feature_cols)