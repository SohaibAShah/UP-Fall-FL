import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sensor_processing import process_sensor_data
from camera_processing import process_camera_data
from data_fusion import fuse_data

def visualize_camera_data(Camera_data, output_dir='visualizations/camera', max_samples=5):
    """Visualize camera data (average difference images).
    
    Args:
        Camera_data (dict): Dictionary with keys (Subject, Activity, Trial) and values as 1x140x140 images.
        output_dir (str): Directory to save visualization images.
        max_samples (int): Maximum number of samples to visualize.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    keys = list(Camera_data.keys())[:max_samples]
    
    for key in keys:
        subject, activity, trial = key
        camera_image = Camera_data[key][0]  # Shape: (140, 140)
        
        # Normalize to [0, 1] for visualization
        camera_image = (camera_image - camera_image.min()) / (camera_image.max() - camera_image.min() + 1e-8)
        
        plt.figure(figsize=(5, 5), dpi=150)
        plt.imshow(camera_image, cmap='gray')
        plt.title(f'Camera Data\nSubject {subject}, Activity {activity}, Trial {trial}')
        plt.axis('off')
        
        filename = f'S{subject}_A{activity}_T{trial}_Camera.png'
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.close()
        print(f"Saved camera visualization: {filename}")

def visualize_fused_data(GAF_Camera_data, output_dir='visualizations/fused', max_samples=5):
    """Visualize fused GAF and camera data for each sensor.
    
    Args:
        GAF_Camera_data (dict): Dictionary with keys (Subject, Activity, Trial) and values as lists of 3x140x140 arrays.
        output_dir (str): Directory to save visualization images.
        max_samples (int): Maximum number of samples to visualize.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sensors = ['Ankle', 'RightPocket', 'Belt', 'Neck', 'Wrist']
    keys = list(GAF_Camera_data.keys())[:max_samples]
    
    for key in keys:
        subject, activity, trial = key
        for i, sensor in enumerate(sensors):
            fused_image = GAF_Camera_data[key][i]  # Shape: (3, 140, 140)
            
            # Normalize channels to [0, 1]
            gaf_acc = (fused_image[0] + 1) / 2  # GAF Accelerometer
            gaf_gyro = (fused_image[1] + 1) / 2  # GAF Angular Velocity
            camera = (fused_image[2] - fused_image[2].min()) / (fused_image[2].max() - fused_image[2].min() + 1e-8)  # Camera
            
            # Create figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
            
            ax1.imshow(gaf_acc, cmap='viridis')
            ax1.set_title(f'{sensor} (GAF Accelerometer)\nSubject {subject}, Activity {activity}, Trial {trial}')
            ax1.axis('off')
            
            ax2.imshow(gaf_gyro, cmap='viridis')
            ax2.set_title(f'{sensor} (GAF Angular Velocity)\nSubject {subject}, Activity {activity}, Trial {trial}')
            ax2.axis('off')
            
            ax3.imshow(camera, cmap='gray')
            ax3.set_title(f'{sensor} (Camera)\nSubject {subject}, Activity {activity}, Trial {trial}')
            ax3.axis('off')
            
            plt.tight_layout()
            
            filename = f'S{subject}_A{activity}_T{trial}_{sensor}_Fused.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()
            print(f"Saved fused visualization: {filename}")

def main():
    # Configuration
    sensor_path = 'dataset'
    camera_zip_path = 'dataset/downloaded_camera_files'
    temp_camera_path = 'dataset/camera'
    output_dir = 'dataset'
    camera_vis_dir = 'visualizations/camera'
    fused_vis_dir = 'visualizations/fused'
    max_samples = 5
    gaf_data_path = os.path.join(sensor_path, 'GAF_sensor_data.pkl')
    camera_data_path = os.path.join(sensor_path, 'Camera_data.pkl')
    gaf_camera_data_path = os.path.join(sensor_path, 'GAF_Camera_data.pkl')
    
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
    
    # Load or process camera data
    print("Checking for existing Camera data...")
    if os.path.exists(camera_data_path):
        print(f"Loading Camera data from {camera_data_path}...")
        with open(camera_data_path, 'rb') as f:
            Camera_data = pickle.load(f)
    else:
        print("Processing camera data...")
        Camera_data = process_camera_data(camera_zip_path, temp_camera_path, output_dir)

    # Visualize camera data
    print("Visualizing camera data...")
    visualize_camera_data(Camera_data, camera_vis_dir, max_samples)
    
    # Load or fuse data
    print("Checking for existing fused data...")
    if os.path.exists(gaf_camera_data_path):
        print(f"Loading fused data from {gaf_camera_data_path}...")
        with open(gaf_camera_data_path, 'rb') as f:
            GAF_Camera_data = pickle.load(f)
    else:
        print("Fusing sensor and camera data...")
        GAF_Camera_data = fuse_data(GAF_data, Camera_data, output_dir)
    
    # Visualize fused data
    print("Visualizing fused data...")
    visualize_fused_data(GAF_Camera_data, fused_vis_dir, max_samples)

if __name__ == "__main__":
    main()