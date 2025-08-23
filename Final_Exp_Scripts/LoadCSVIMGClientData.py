import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
image_data_path = '/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image'


def loadSensorIMGClientsData(file_path, image_data_path):
    """
    Loads and processes data by first creating sensor clients, then using their
    timestamps to find and align corresponding images for camera clients.
    """
    
    subs = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    print(f"Loading data for a specific subset of {len(subs)} subjects: {subs}")

    # --- Part 1: Load and Create Sensor Clients (0-5) ---
    print("\n--- Starting Part 1: Loading and Processing Sensor Clients ---")
    df = pd.read_csv(file_path, header=[0, 1])

    cleaned_columns = []
    last_val = ''
    for col_l1, col_l2 in df.columns:
        if 'Unnamed' in col_l1: col_l1 = last_val
        else: last_val = col_l1.strip(); col_l1 = last_val
        if col_l1 == col_l2.strip(): cleaned_columns.append(col_l1)
        else: cleaned_columns.append(f"{col_l1}_{col_l2.strip()}")
    df.columns = cleaned_columns
    
    df = df[df['Subject'].isin(subs)].copy()
    df = df[~((df['Subject'] == 8) & (df['Activity'] == 11) & (df['Trial'].isin([2, 3])))]

    df_cleaned = df.drop(columns=[col for col in df.columns if 'Infrared' in col] + ['Trial', 'Tag'], errors='ignore')
    df_cleaned.dropna(inplace=True)
    df_cleaned.drop_duplicates(inplace=True)

    train_subjects = [s for s in subs if s <= 13]
    test_subjects = [s for s in subs if s >= 14]
    
    train_df_sensor = df_cleaned[df_cleaned['Subject'].isin(train_subjects)].copy()
    test_df_sensor = df_cleaned[df_cleaned['Subject'].isin(test_subjects)].copy()

    sensor_clients = {
        'Ankle_IMU': ['AnkleAccelerometer_x-axis (g)', 'AnkleAccelerometer_y-axis (g)', 'AnkleAccelerometer_z-axis (g)', 'AnkleAngularVelocity_x-axis (deg/s)', 'AnkleAngularVelocity_y-axis (deg/s)', 'AnkleAngularVelocity_z-axis (deg/s)', 'AnkleLuminosity_illuminance (lx)'],
        'Pocket_IMU': ['RightPocketAccelerometer_x-axis (g)', 'RightPocketAccelerometer_y-axis (g)', 'RightPocketAccelerometer_z-axis (g)', 'RightPocketAngularVelocity_x-axis (deg/s)', 'RightPocketAngularVelocity_y-axis (deg/s)', 'RightPocketAngularVelocity_z-axis (deg/s)', 'RightPocketLuminosity_illuminance (lx)'],
        'Belt_IMU': ['BeltAccelerometer_x-axis (g)', 'BeltAccelerometer_y-axis (g)', 'BeltAccelerometer_z-axis (g)', 'BeltAngularVelocity_x-axis (deg/s)', 'BeltAngularVelocity_y-axis (deg/s)', 'BeltAngularVelocity_z-axis (deg/s)', 'BeltLuminosity_illuminance (lx)'],
        'Neck_IMU': ['NeckAccelerometer_x-axis (g)', 'NeckAccelerometer_y-axis (g)', 'NeckAccelerometer_z-axis (g)', 'NeckAngularVelocity_x-axis (deg/s)', 'NeckAngularVelocity_y-axis (deg/s)', 'NeckAngularVelocity_z-axis (deg/s)', 'NeckLuminosity_illuminance (lx)'],
        'Wrist_IMU': ['WristAccelerometer_x-axis (g)', 'WristAccelerometer_y-axis (g)', 'WristAccelerometer_z-axis (g)', 'WristAngularVelocity_x-axis (deg/s)', 'WristAngularVelocity_y-axis (deg/s)', 'WristAngularVelocity_z-axis (deg/s)', 'WristLuminosity_illuminance (lx)'],
        'EEG': ['BrainSensor']
    }
    
    X_train_splits, X_test_splits, Y_train_splits, Y_test_splits = {}, {}, {}, {}
    num_classes = 12

    for client_index, (client_name, columns) in enumerate(sensor_clients.items()):
        X_train = train_df_sensor[columns].values
        y_train = train_df_sensor['Activity'].values
        X_test = test_df_sensor[columns].values
        y_test = test_df_sensor['Activity'].values
        scaler = StandardScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
        set_seed()
        Y_train, Y_test = torch.nn.functional.one_hot(torch.from_numpy(y_train).long(), num_classes).float(), torch.nn.functional.one_hot(torch.from_numpy(y_test).long(), num_classes).float()
        X_train_splits[client_index], X_test_splits[client_index] = X_train_scaled, X_test_scaled
        Y_train_splits[client_index], Y_test_splits[client_index] = Y_train, Y_test

    print("--- Finished Part 1. Sensor clients 0-5 created. ---")

    # --- Part 2: Extract Final Timestamps from Processed Sensor Data ---
    final_train_times = set(train_df_sensor['TimeStamps_Time'])
    final_test_times = set(test_df_sensor['TimeStamps_Time'])
    print(f"\nExtracted {len(final_train_times)} unique training and {len(final_test_times)} unique testing timestamps from sensor clients.")

    # --- Part 3: Load and Align Image Data using Final Timestamps (Clients 6-7) ---
    print("\n--- Starting Part 3: Aligning Image Data for Camera Clients ---")

    def align_and_create_camera_client(camera_id):
        # Load complete, combined image files
        img = np.load(f'{image_data_path}/image_{camera_id}.npy')
        name = np.load(f'{image_data_path}/name_{camera_id}.npy')
        label = np.load(f'{image_data_path}/label_{camera_id}.npy')

        # Find indices where image timestamps match the final sensor timestamps
        train_indices = [i for i, t in enumerate(name) if t in final_train_times]
        test_indices = [i for i, t in enumerate(name) if t in final_test_times]
        
        # Select the aligned data
        X_train_img, y_train_img = img[train_indices], label[train_indices]
        X_test_img, y_test_img = img[test_indices], label[test_indices]
        
        print(f"  - Camera {camera_id}: Found {len(X_train_img)} training images and {len(X_test_img)} testing images.")

        # Process the aligned data
        X_train_scaled = X_train_img.reshape(-1, 32, 32, 1) / 255.0
        X_test_scaled = X_test_img.reshape(-1, 32, 32, 1) / 255.0
        
        y_train_final = np.where(y_train_img == 20, 0, y_train_img)
        y_test_final = np.where(y_test_img == 20, 0, y_test_img)

        set_seed()
        Y_train_final = torch.nn.functional.one_hot(torch.from_numpy(y_train_final.flatten()).long(), num_classes).float()
        Y_test_final = torch.nn.functional.one_hot(torch.from_numpy(y_test_final.flatten()).long(), num_classes).float()
        
        return X_train_scaled, Y_train_final, X_test_scaled, Y_test_final

    # Create Client 6 (Camera 1)
    X_train_cam1, Y_train_cam1, X_test_cam1, Y_test_cam1 = align_and_create_camera_client(1)
    X_train_splits[6], Y_train_splits[6] = X_train_cam1, Y_train_cam1
    X_test_splits[6], Y_test_splits[6] = X_test_cam1, Y_test_cam1
    sensor_clients['Camera_1'] = ['image_data']
    
    # Create Client 7 (Camera 2)
    X_train_cam2, Y_train_cam2, X_test_cam2, Y_test_cam2 = align_and_create_camera_client(2)
    X_train_splits[7], Y_train_splits[7] = X_train_cam2, Y_train_cam2
    X_test_splits[7], Y_test_splits[7] = X_test_cam2, Y_test_cam2
    sensor_clients['Camera_2'] = ['image_data']

    print("--- Finished. All 8 clients created with the new alignment logic. ---")
    
    return X_train_splits, X_test_splits, Y_train_splits, Y_test_splits, sensor_clients


def plot_class_distributions(Y_train_splits, Y_test_splits, clients_info):
    """
    Generates and saves bar charts for the class distribution of each client.
    """
    print("\n--- Generating Class Distribution Plots ---")
    
    # Define class names for plotting, corresponding to labels 0-11
    class_names = [
        'Special Fall', 'Falling Hands', 'Falling Knees', 'Falling Backwards', 
        'Falling Sideward', 'Falling off Chair', 'Walking', 'Standing', 
        'Sitting', 'Picking Object', 'Jumping', 'Laying'
    ]
    
    # Get a list of client names from the info dictionary
    for client_index, (client_name, columns) in enumerate(sensor_clients.items()):
        # Loop through each client to create a separate plot

        # Check if data exists for this client
        if client_index not in Y_train_splits or client_index not in Y_test_splits:
            print(f"  - Skipping Client {client_index} ({client_name}): No label data found.")
            continue

        y_train_indices = torch.argmax(Y_train_splits[client_index], dim=1).cpu().numpy()
        y_test_indices = torch.argmax(Y_test_splits[client_index], dim=1).cpu().numpy()

        train_counts = pd.Series(y_train_indices).value_counts()
        test_counts = pd.Series(y_test_indices).value_counts()
        
        df = pd.DataFrame({'Train': train_counts, 'Test': test_counts}).fillna(0)
        
        # --- BUG FIX: Ensure the DataFrame has an entry for all 12 classes ---
        # Reindex the DataFrame with all possible class labels (0-11), filling any missing classes with 0.
        full_class_index = np.arange(len(class_names))
        df = df.reindex(full_class_index, fill_value=0)
        
        # --- Plotting ---
        plt.figure(figsize=(14, 7))
        
        x = np.arange(len(class_names))
        width = 0.35

        rects1 = plt.bar(x - width/2, df['Train'], width, label='Train Set')
        rects2 = plt.bar(x + width/2, df['Test'], width, label='Test Set')

        plt.ylabel('Number of Samples')
        plt.xlabel('Activity Class')
        plt.title(f'Class Distribution for Client {client_index}: {client_name}')
        plt.xticks(x, class_names, rotation=45, ha="right")
        plt.legend()
        
        plt.bar_label(rects1, padding=3)
        plt.bar_label(rects2, padding=3)
        
        plt.tight_layout()

        plot_filename = f"./class_dist_client_{client_index}_{client_name}.png"
        plt.savefig(plot_filename)
        plt.close()

        print(f"  - Saved plot for Client {client_index} ({client_name}) to {plot_filename}")

def plot_aggregated_class_distributions(Y_train_splits, Y_test_splits, clients_info):
    """
    Generates two charts showing the aggregated class distribution for all clients,
    one for the training set and one for the testing set.
    """
    print("\n--- Generating Aggregated Class Distribution Plots ---")

    class_names = [
        'Special Fall', 'Falling Hands', 'Falling Knees', 'Falling Backwards',
        'Falling Sideward', 'Falling off Chair', 'Walking', 'Standing',
        'Sitting', 'Picking Object', 'Jumping', 'Laying'
    ]

     # --- 1. Aggregate Data ---
    train_counts_per_client = {}
    test_counts_per_client = {}
    
    # Get a list of client names from the info dictionary
    for client_id, (client_name, columns) in enumerate(sensor_clients.items()):
        # Loop through each client to create a separate plot

        # Check if data exists for this client
        if client_id not in Y_train_splits or client_id not in Y_test_splits:
            print(f"  - Skipping Client {client_id} ({client_name}): No label data found.")
            continue

       

        # Convert one-hot tensors to class indices
        y_train_indices = torch.argmax(Y_train_splits[client_id], dim=1).cpu().numpy()
        y_test_indices = torch.argmax(Y_test_splits[client_id], dim=1).cpu().numpy()

        # Store the value counts for each client
        train_counts_per_client[client_name] = pd.Series(y_train_indices).value_counts()
        test_counts_per_client[client_name] = pd.Series(y_test_indices).value_counts()

    # Create DataFrames, ensuring all classes (0-11) and clients are included
    full_class_index = np.arange(len(class_names))
    train_df = pd.DataFrame(train_counts_per_client).reindex(full_class_index, fill_value=0)
    test_df = pd.DataFrame(test_counts_per_client).reindex(full_class_index, fill_value=0)

    # --- 2. Plot Training Data Chart ---
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # **BUG FIX: Remove the .T to plot with classes on the x-axis**
    train_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    ax.set_title('Aggregated Class Distribution (Training Set)', fontsize=16)
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Activity Class')
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(title='Clients', bbox_to_anchor=(1.02, 1), loc='upper left')
    
    plt.tight_layout()
    train_plot_filename = "./aggregated_class_dist_train.png"
    plt.savefig(train_plot_filename)
    plt.close()
    print(f"  - Saved training data distribution plot to {train_plot_filename}")

    # --- 3. Plot Testing Data Chart ---
    fig, ax = plt.subplots(figsize=(16, 8))

    # **FIX: Remove the .T to plot with classes on the x-axis**
    test_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

    ax.set_title('Aggregated Class Distribution (Test Set)', fontsize=16)
    ax.set_ylabel('Number of Samples')
    ax.set_xlabel('Activity Class')
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(title='Clients', bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    test_plot_filename = "./aggregated_class_dist_test.png"
    plt.savefig(test_plot_filename)
    plt.close()
    print(f"  - Saved testing data distribution plot to {test_plot_filename}")


def plot_aggregated_class_percent_distributions(Y_train_splits, Y_test_splits, clients_info):
    """
    Generates two 100% percentage stacked bar charts for the aggregated class distribution,
    one for the training set and one for the testing set.
    """
    print("\n--- Generating Aggregated Percentage Class Distribution Plots ---")

    class_names = [
        'Special Fall', 'Falling Hands', 'Falling Knees', 'Falling Backwards',
        'Falling Sideward', 'Falling off Chair', 'Walking', 'Standing',
        'Sitting', 'Picking Object', 'Jumping', 'Laying'
    ]
     # --- 1. Aggregate Data ---
    train_counts_per_client = {}
    test_counts_per_client = {}
    
    # Get a list of client names from the info dictionary
    for client_id, (client_name, columns) in enumerate(sensor_clients.items()):
        # Loop through each client to create a separate plot

        # Check if data exists for this client
        if client_id not in Y_train_splits or client_id not in Y_test_splits:
            print(f"  - Skipping Client {client_id} ({client_name}): No label data found.")
            continue

        y_train_indices = torch.argmax(Y_train_splits[client_id], dim=1).cpu().numpy()
        y_test_indices = torch.argmax(Y_test_splits[client_id], dim=1).cpu().numpy()

        train_counts_per_client[client_name] = pd.Series(y_train_indices).value_counts()
        test_counts_per_client[client_name] = pd.Series(y_test_indices).value_counts()

    full_class_index = np.arange(len(class_names))
    train_df = pd.DataFrame(train_counts_per_client).reindex(full_class_index, fill_value=0)
    test_df = pd.DataFrame(test_counts_per_client).reindex(full_class_index, fill_value=0)

    # --- 2. Convert Counts to Percentages ---
    # Calculate the total samples for each class (row-wise sum)
    train_totals = train_df.sum(axis=1)
    test_totals = test_df.sum(axis=1)

    # Divide each client's count by the class total and multiply by 100
    # Add a small epsilon to totals to avoid division by zero if a class has no samples
    train_df_percent = train_df.div(train_totals + 1e-9, axis=0) * 100
    test_df_percent = test_df.div(test_totals + 1e-9, axis=0) * 100

    # --- 3. Plot Training Data Chart ---
    fig, ax = plt.subplots(figsize=(16, 8))
    # Plot the percentage dataframe
    train_df_percent.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    
    ax.set_title('Client Contribution per Class (Training Set)', fontsize=16)
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Activity Class')
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(title='Clients', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 100) # Set y-axis to go from 0 to 100
    
    plt.tight_layout()
    train_plot_filename = "./aggregated_percentage_dist_train.png"
    plt.savefig(train_plot_filename)
    plt.close()
    print(f"  - Saved training data percentage plot to {train_plot_filename}")

    # --- 4. Plot Testing Data Chart ---
    fig, ax = plt.subplots(figsize=(16, 8))
    # Plot the transposed percentage dataframe
    test_df_percent.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

    ax.set_title('Client Contribution per Class (Test Set)', fontsize=16)
    ax.set_ylabel('Percentage (%)')
    ax.set_xlabel('Activity Class')
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend(title='Clients', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.set_ylim(0, 100) # Set y-axis to go from 0 to 100

    plt.tight_layout()
    test_plot_filename = "./aggregated_percentage_dist_test.png"
    plt.savefig(test_plot_filename)
    plt.close()
    print(f"  - Saved testing data percentage plot to {test_plot_filename}")

# --- How to Use ---
if __name__ == '__main__':
    try:
        # Make sure 'sensor.csv' is in the same directory as this script
        X_train, X_test, Y_train, Y_test, sensor_clients = loadSensorIMGClientsData(file_path, image_data_path)
        print("\nData loaded successfully for all sensor clients.")
        # You can now access the data for each client using its index (0 to 5)
        print(sensor_clients)
        for client_index, (client_name, columns) in enumerate(sensor_clients.items()):
            print(f"\nData for Client {client_index} ({client_name}):")
            print(f"  - X_train shape: {X_train[client_index].shape}")
            print(f"  - Y_train shape: {Y_train[client_index].shape}")
            print(f"  - X_test shape:  {X_test[client_index].shape}")
            print(f"  - Y_test shape:  {Y_test[client_index].shape}")

        # --- NEW: Call the new visualization function ---
        # plot_aggregated_class_distributions(Y_train, Y_test, sensor_clients)

        # --- NEW: Call the new visualization function ---
        plot_aggregated_class_percent_distributions(Y_train, Y_test, sensor_clients)

        # Generate class distribution plots
        # plot_class_distributions(Y_train, Y_test, sensor_clients)

    except FileNotFoundError:
        print("\nError: 'sensor.csv' not found.")
        print("Please make sure the dataset file is in the same directory as the script.")
    #except Exception as e:
     #   print(f"An error occurred: {e}")