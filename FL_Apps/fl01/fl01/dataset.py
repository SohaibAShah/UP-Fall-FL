import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import logging

WINDOW_SIZE = 200
STEP_SIZE = 100

def create_windows(data, labels):
    """Creates overlapping windows from time-series data."""
    X, y = [], []
    for i in range(0, len(data) - WINDOW_SIZE, STEP_SIZE):
        X.append(data[i:i + WINDOW_SIZE])
        y.append(0 if np.any(labels[i:i + WINDOW_SIZE] == 0) else 1)
    return np.array(X), np.array(y)

def prepare_partitions(data_path: str, partitions_dir: str):
    """
    Loads the full dataset, processes it, and saves each client's
    data partition to disk.
   
    if os.path.exists(partitions_dir):
        logging.info("Client partitions already exist. Skipping preparation.")
        return
    """
    
    logging.info("--- Preparing and saving client data partitions ---")
    os.makedirs(partitions_dir)
    
    df = pd.read_csv(os.path.join(data_path, 'sensor.csv'), header=[0, 1])
    # ... (cleaning logic) ...
    cleaned_columns = ['_'.join(col).strip() for col in df.columns.values]
    df.columns = cleaned_columns
    df.dropna(inplace=True); df.drop_duplicates(inplace=True)
    df = df[~df['Subject_Subject'].isin([5, 9])]
    fall_activity_ids = {2, 3, 4, 5, 6}
    df['Fall'] = df['Activity_Activity'].apply(lambda x: 0 if x in fall_activity_ids else 1)
    
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
    test_subjects = [s for s in range(14, 18)]
    train_df = df[df['Subject_Subject'].isin(train_subjects)]
    test_df = df[df['Subject_Subject'].isin(test_subjects)]

    imu_columns = [col for col in train_df.columns if 'Accelerometer' in col or 'AngularVelocity' in col]
    scaler = StandardScaler().fit(train_df[imu_columns])
    train_df.loc[:, imu_columns] = scaler.transform(train_df[imu_columns])
    test_df.loc[:, imu_columns] = scaler.transform(test_df[imu_columns])

    # Save global test set
    X_test, y_test = create_windows(test_df[imu_columns].values, test_df['Fall'].values)
    torch.save((X_test, y_test), os.path.join(partitions_dir, 'test.pt'))
    
    # Create and save each client's partition
    for cid in sorted(train_df['Subject_Subject'].unique()):
        client_df = train_df[train_df['Subject_Subject'] == cid]
        X_client, y_client = create_windows(client_df[imu_columns].values, client_df['Fall'].values)
        if len(X_client) > 0:
            torch.save((X_client, y_client), os.path.join(partitions_dir, f'client_{cid}.pt'))
    
    # Save feature count
    with open(os.path.join(partitions_dir, 'num_features.txt'), 'w') as f:
        f.write(str(len(imu_columns)))
    
    logging.info(f"Partitions saved to '{partitions_dir}' directory.")