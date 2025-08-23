import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from sensor_processing import process_sensor_data

def visualize_gaf(GAF_data, output_dir='gaf_visualizations', max_samples=5):
    """Visualize GAF images from sensor data.
    
    Args:
        GAF_data (dict): Dictionary with keys (Subject, Activity, Trial) and values as sensor GAFs.
        output_dir (str): Directory to save visualization images.
        max_samples (int): Maximum number of samples to visualize.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Sensor names
    sensors = ['GAF_Ankle', 'GAF_RightPocket', 'GAF_Belt', 'GAF_Neck', 'GAF_Wrist']
    
    # Limit the number of samples to visualize
    keys = list(GAF_data.keys())[:max_samples]
    
    for key in keys:
        subject, activity, trial = key
        for sensor in sensors:
            gaf_image = GAF_data[key][sensor]  # Shape: (140, 140, 2)
            
            # Normalize GAF data to [0, 1] for visualization
            gaf_acc = (gaf_image[:, :, 0] + 1) / 2  # Accelerometer channel
            gaf_gyro = (gaf_image[:, :, 1] + 1) / 2  # Angular velocity channel
            
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
            
            # Plot accelerometer GAF
            ax1.imshow(gaf_acc, cmap='viridis')
            ax1.set_title(f'{sensor} (Accelerometer)\nSubject {subject}, Activity {activity}, Trial {trial}')
            ax1.axis('off')
            
            # Plot angular velocity GAF
            ax2.imshow(gaf_gyro, cmap='viridis')
            ax2.set_title(f'{sensor} (Angular Velocity)\nSubject {subject}, Activity {activity}, Trial {trial}')
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Save the figure
            filename = f'S{int(subject)}_A{int(activity)}_T{int(trial)}_{sensor}.png'
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
            plt.close()
            print(f"Saved visualization: {filename}")

def main():
    # Configuration
    sensor_path = 'dataset'  # Directory containing CompleteDataSet.csv
    output_dir = 'gaf_visualizations'
    max_samples = 5  # Number of samples to visualize
    
    # Process sensor data to get GAF_data
    print("Processing sensor data...")
    GAF_data = process_sensor_data(sensor_path)
    
    # Visualize GAF images
    print("Visualizing GAF images...")
    visualize_gaf(GAF_data, output_dir, max_samples)

if __name__ == "__main__":
    main()