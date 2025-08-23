import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

file_path = '/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image/sensor.csv'

def loadSensorClientsData_from_csv(file_path):
    """
    Loads, processes, and splits sensor data from a single combined CSV file,
    treating each sensor location as a separate client and handling specified
    data exclusions.
    """
    
    # --- 1. Load Data and Clean Column Headers ---
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, header=[0, 1])

    # Clean the multi-level column names
    cleaned_columns = []
    last_val = ''
    for col_l1, col_l2 in df.columns:
        if 'Unnamed' in col_l1:
            col_l1 = last_val
        else:
            last_val = col_l1.strip()
            col_l1 = last_val
        if col_l1 == col_l2:
            cleaned_columns.append(col_l1)
        else:
            cleaned_columns.append(f"{col_l1}_{col_l2.strip()}")
    df.columns = cleaned_columns
    print("Column headers cleaned successfully.")
    print(f"Data shape after loading: {df.shape}\n")
    print("First few rows of the dataset:")
    print(df.head())
    # --- 2. Apply All Data Exclusions ---
    print("Applying data exclusion rules...")
    
    # Rule 1: Skip all data from subjects 5 and 9
    initial_rows = len(df)
    df = df[~df['Subject'].isin([5, 9])]
    print(f"  - Removed {initial_rows - len(df)} rows for Subjects 5 and 9.")
    
    # Rule 2: Skip all data from Activity 5 of Subject 2
    initial_rows = len(df)
    df = df[~((df['Subject'] == 2) & (df['Activity'] == 5))]
    print(f"  - Removed {initial_rows - len(df)} rows for Subject 2, Activity 5.")

    # Rule 3: Skip the two missing trials in Activity 11 of Subject 8
    initial_rows = len(df)
    df = df[~((df['Subject'] == 8) & (df['Activity'] == 11) & (df['Trial'].isin([2, 3])))]
    print(f"  - Removed {initial_rows - len(df)} rows for Subject 8, Activity 11, Trials 2 & 3.")

    # --- 3. Preprocess and Split Data ---
    # Drop columns that are not needed for modeling
    cols_to_drop = [col for col in df.columns if 'Infrared' in col]
    cols_to_drop.extend(['TimeStamps_Time', 'Trial', 'Tag'])
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')
    
    # Handle any remaining missing values
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Split data by the remaining subjects for training and testing
    # Note: Subjects 5 and 9 will not be in either set.
    train_subjects = [s for s in range(1, 14) if s not in [5, 9]]
    test_subjects = [s for s in range(14, 18)]
    
    train_df = df[df['Subject'].isin(train_subjects)].copy()
    test_df = df[df['Subject'].isin(test_subjects)].copy()

    # --- 4. Define Clients and Process Data ---
    sensor_clients = {
        'Ankle_IMU': [
            'AnkleAccelerometer_x-axis (g)', 'AnkleAccelerometer_y-axis (g)', 'AnkleAccelerometer_z-axis (g)',
            'AnkleAngularVelocity_x-axis (deg/s)', 'AnkleAngularVelocity_y-axis (deg/s)', 'AnkleAngularVelocity_z-axis (deg/s)',
            'AnkleLuminosity_illuminance (lx)'
        ],
        'Pocket_IMU': [
            'RightPocketAccelerometer_x-axis (g)', 'RightPocketAccelerometer_y-axis (g)', 'RightPocketAccelerometer_z-axis (g)',
            'RightPocketAngularVelocity_x-axis (deg/s)', 'RightPocketAngularVelocity_y-axis (deg/s)', 'RightPocketAngularVelocity_z-axis (deg/s)',
            'RightPocketLuminosity_illuminance (lx)'
        ],
        'Belt_IMU': [
            'BeltAccelerometer_x-axis (g)', 'BeltAccelerometer_y-axis (g)', 'BeltAccelerometer_z-axis (g)',
            'BeltAngularVelocity_x-axis (deg/s)', 'BeltAngularVelocity_y-axis (deg/s)', 'BeltAngularVelocity_z-axis (deg/s)',
            'BeltLuminosity_illuminance (lx)'
        ],
        'Neck_IMU': [
            'NeckAccelerometer_x-axis (g)', 'NeckAccelerometer_y-axis (g)', 'NeckAccelerometer_z-axis (g)',
            'NeckAngularVelocity_x-axis (deg/s)', 'NeckAngularVelocity_y-axis (deg/s)', 'NeckAngularVelocity_z-axis (deg/s)',
            'NeckLuminosity_illuminance (lx)'
        ],
        'Wrist_IMU': [
            'WristAccelerometer_x-axis (g)', 'WristAccelerometer_y-axis (g)', 'WristAccelerometer_z-axis (g)',
            'WristAngularVelocity_x-axis (deg/s)', 'WristAngularVelocity_y-axis (deg/s)', 'WristAngularVelocity_z-axis (deg/s)',
            'WristLuminosity_illuminance (lx)'
        ],
        'EEG': ['BrainSensor']
    }
    
    X_train_splits, X_test_splits = {}, {}
    Y_train_splits, Y_test_splits = {}, {}
    
    num_classes = 11 # 11 activities

    print("\nProcessing data for each sensor client...")
    for client_index, (client_name, columns) in enumerate(sensor_clients.items()):
        print(f"  - Client {client_index}: {client_name}")
        
        # Select data for the current client
        X_train = train_df[columns].values
        # ActivityIDs are 1-11, map to 0-10 for zero-based indexing
        y_train = train_df['Activity'].values - 1 
        
        X_test = test_df[columns].values
        y_test = test_df['Activity'].values - 1

        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        set_seed() # Set seed for reproducibility
        
        # One-hot encode the labels
        Y_train = torch.nn.functional.one_hot(torch.from_numpy(y_train).long(), num_classes).float()
        Y_test = torch.nn.functional.one_hot(torch.from_numpy(y_test).long(), num_classes).float()
        
        # Store the results
        X_train_splits[client_index] = X_train_scaled
        X_test_splits[client_index] = X_test_scaled
        Y_train_splits[client_index] = Y_train
        Y_test_splits[client_index] = Y_test

    return X_train_splits, X_test_splits, Y_train_splits, Y_test_splits, sensor_clients

# --- How to Use ---
if __name__ == '__main__':
    try:
        # Make sure 'sensor.csv' is in the same directory as this script
        X_train, X_test, Y_train, Y_test, sensor_clients = loadSensorClientsData_from_csv(file_path)
        print("\nData loaded successfully for all sensor clients.")
        # You can now access the data for each client using its index (0 to 5)
        print(sensor_clients)
        for client_index, (client_name, columns) in enumerate(sensor_clients.items()):
            print(f"\nData for Client {client_index} ({client_name}):")
            print(f"  - X_train shape: {X_train[client_index].shape}")
            print(f"  - Y_train shape: {Y_train[client_index].shape}")
            print(f"  - X_test shape:  {X_test[client_index].shape}")
            print(f"  - Y_test shape:  {Y_test[client_index].shape}")

    except FileNotFoundError:
        print("\nError: 'sensor.csv' not found.")
        print("Please make sure the dataset file is in the same directory as the script.")
    except Exception as e:
        print(f"An error occurred: {e}")