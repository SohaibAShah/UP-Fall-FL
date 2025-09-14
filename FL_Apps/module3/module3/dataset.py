import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os
import logging

WINDOW_SIZE = 200
STEP_SIZE = 100

def create_windows(data, labels):
    """Creates overlapping windows from time-series data."""
    X, y = [], []
    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        X.append(data[i:i + WINDOW_SIZE])
        # A window is a "Fall" if any sample within it is a fall
        y.append(0 if np.any(labels[i:i + WINDOW_SIZE] == 0) else 1)
    return np.array(X), np.array(y)

def load_and_partition_data(data_dir: str):
    """
    Loads, cleans, aligns, and partitions the UP-FALL dataset for both
    centralized and federated learning experiments.
    """
    logging.info("--- Step 1: Loading and Aligning Raw Data ---")
    
    # Load raw data
    sensor_df = pd.read_csv(os.path.join(data_dir, 'sensor.csv'), header=[0, 1])
    # Clean column names
    sensor_df.columns = ['_'.join(col).strip() for col in sensor_df.columns.values]
    sensor_df.dropna(inplace=True); sensor_df.drop_duplicates(inplace=True)

    img1 = np.load(os.path.join(data_dir, 'image_1.npy'))
    name1 = np.load(os.path.join(data_dir, 'name_1.npy'))
    img2 = np.load(os.path.join(data_dir, 'image_2.npy'))

    # Robust timestamp alignment
    sensor_ts = set(sensor_df['TimeStamps_Time'])
    name1_ts = set(name1)
    common_timestamps = sorted(list(sensor_ts.intersection(name1_ts)))

    sensor_df = sensor_df[sensor_df['TimeStamps_Time'].isin(common_timestamps)].set_index('TimeStamps_Time').loc[common_timestamps]
    
    name1_map = {ts: i for i, ts in enumerate(name1)}
    idx1 = [name1_map[ts] for ts in common_timestamps]
    img1, img2 = img1[idx1], img2[idx1]

    logging.info("--- Step 2: Creating Binary Labels and Splitting Data by Subject ---")
    fall_activity_ids = {2, 3, 4, 5, 6}
    sensor_df['Fall'] = sensor_df['Activity_Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
    
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
    test_subjects = [s for s in range(14, 18)]
    train_df = sensor_df[sensor_df['Subject_Subject'].isin(train_subjects)]
    test_df = sensor_df[sensor_df['Subject_Subject'].isin(test_subjects)]

    feature_cols = [col for col in sensor_df.columns if 'Accelerometer' in col or 'AngularVelocity' in col]
    
    logging.info("--- Step 3: Scaling Features ---")
    scaler = StandardScaler().fit(train_df[feature_cols])
    train_df.loc[:, feature_cols] = scaler.transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    logging.info("--- Step 4: Creating Global Test Set ---")
    # Align images with the test dataframe
    test_img1_map = {ts: img for ts, img in zip(common_timestamps, img1)}
    test_img2_map = {ts: img for ts, img in zip(common_timestamps, img2)}
    
    X_test_csv, y_test = create_windows(test_df[feature_cols].values, test_df['Fall'].values)
    
    # Create windowed image data for the test set
    X_test_img1, _ = create_windows(np.stack(test_df.index.map(test_img1_map).values), test_df['Fall'].values)
    X_test_img2, _ = create_windows(np.stack(test_df.index.map(test_img2_map).values), test_df['Fall'].values)
    
    # Reshape for PyTorch (N, C, L or N, C, H, W)
    X_test_csv = np.transpose(X_test_csv, (0, 2, 1))
    X_test_img1 = np.expand_dims((X_test_img1 / 255.0).astype(np.float32), axis=1)
    X_test_img2 = np.expand_dims((X_test_img2 / 255.0).astype(np.float32), axis=1)

    test_data = (X_test_csv, X_test_img1, X_test_img2, y_test)
    
    logging.info("--- Step 5: Partitioning Data into Federated Clients ---")
    client_partitions = {}
    train_img1_map = {ts: img for ts, img in zip(common_timestamps, img1)}
    train_img2_map = {ts: img for ts, img in zip(common_timestamps, img2)}

    for cid in train_subjects:
        client_df = train_df[train_df['Subject_Subject'] == cid]
        if client_df.empty: continue
        
        X_client_csv, y_client = create_windows(client_df[feature_cols].values, client_df['Fall'].values)
        if len(X_client_csv) == 0: continue

        X_client_img1, _ = create_windows(np.stack(client_df.index.map(train_img1_map).values), client_df['Fall'].values)
        X_client_img2, _ = create_windows(np.stack(client_df.index.map(train_img2_map).values), client_df['Fall'].values)
        
        # Reshape for PyTorch
        X_client_csv = np.transpose(X_client_csv, (0, 2, 1))
        X_client_img1 = np.expand_dims((X_client_img1 / 255.0).astype(np.float32), axis=1)
        X_client_img2 = np.expand_dims((X_client_img2 / 255.0).astype(np.float32), axis=1)

        client_partitions[str(cid)] = (X_client_csv, X_client_img1, X_client_img2, y_client)

    logging.info(f"Data loading complete. {len(client_partitions)} clients created.")
    return client_partitions, test_data, len(feature_cols)
