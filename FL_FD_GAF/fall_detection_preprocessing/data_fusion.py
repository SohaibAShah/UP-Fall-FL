import os
import pickle
import numpy as np

def fuse_data(GAF_data, Camera_data, output_dir='dataset'):
    """Combine GAF-transformed sensor data and camera data.
    
    Args:
        GAF_data (dict): Sensor data with GAF transformations.
        Camera_data (dict): Processed camera data.
        output_dir (str): Directory to save GAF_Camera_data.pkl.
    
    Returns:
        dict: Fused data with keys (Subject, Activity, Trial) and values as lists of fused arrays.
    """
    gaf_camera_data_path = os.path.join(output_dir, 'GAF_Camera_data.pkl')
    
    # Check for existing GAF_Camera_data.pkl
    if os.path.exists(gaf_camera_data_path):
        print(f"Loading fused data from {gaf_camera_data_path}...")
        with open(gaf_camera_data_path, 'rb') as f:
            GAF_Camera_data = pickle.load(f)
        return GAF_Camera_data
    
    combined_keys = list(set(GAF_data.keys()) & set(Camera_data.keys()))  # Ensure matching keys
    combined_keys.sort()
    GAF_Camera_data = {}
    
    for key in combined_keys:
        print(f"Fusing data for Subject {key[0]}, Activity {key[1]}, Trial {key[2]}")
        Ankle = np.concatenate((GAF_data[key]['GAF_Ankle'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        RightPocket = np.concatenate((GAF_data[key]['GAF_RightPocket'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        Belt = np.concatenate((GAF_data[key]['GAF_Belt'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        Neck = np.concatenate((GAF_data[key]['GAF_Neck'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        Wrist = np.concatenate((GAF_data[key]['GAF_Wrist'].transpose(2, 0, 1), Camera_data[key]), axis=0)
        l = [Ankle, RightPocket, Belt, Neck, Wrist]
        GAF_Camera_data[key] = l
    
    # Save GAF_Camera_data
    os.makedirs(output_dir, exist_ok=True)
    with open(gaf_camera_data_path, 'wb') as f:
        pickle.dump(GAF_Camera_data, f)
    print(f"Saved fused data to {gaf_camera_data_path}")
    
    return GAF_Camera_data