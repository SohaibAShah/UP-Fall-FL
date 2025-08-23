import os
import pickle
from sensor_processing import process_sensor_data
from camera_processing import process_camera_data
from data_fusion import fuse_data
from dataset_splitting import split_and_save_data

def main():
    # Configuration
    sensor_path = 'dataset'
    camera_zip_path = 'dataset/downloaded_camera_files'
    temp_camera_path = 'dataset/camera'
    output_dir = 'dataset'
    gaf_data_path = os.path.join(sensor_path, 'GAF_sensor_data.pkl')
    
    # Load or process sensor data
    print("Checking for existing GAF data...")
    if os.path.exists(gaf_data_path):
        print(f"Loading GAF data from {gaf_data_path}...")
        with open(gaf_data_path, 'rb') as f:
            GAF_data = pickle.load(f)
    else:
        print("GAF data not found. Processing sensor data...")
        GAF_data = process_sensor_data(sensor_path)
        with open(gaf_data_path, 'wb') as f:
            pickle.dump(GAF_data, f)
        print(f"Saved GAF data to {gaf_data_path}")
    
    # Process camera data
    print("Processing camera data...")
    Camera_data = process_camera_data(camera_zip_path, temp_camera_path, output_dir)

    # Fuse data
    print("Fusing sensor and camera data...")
    GAF_Camera_data = fuse_data(GAF_data, Camera_data, output_dir)
    
    # Split and save data
    print("Splitting and saving data...")
    Train_data, Test_data = split_and_save_data(GAF_Camera_data, output_dir)

if __name__ == "__main__":
    main()