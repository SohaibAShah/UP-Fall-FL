# main.py

import os
import numpy as np
import pandas as pd

# Import functions and configurations from local modules
import config
import data_loader

def main():
    """
    Main function to load sensor data, process camera images,
    and save the processed data.
    """

    # --- Load Sensor Data ---
    print("Loading sensor data...")
    sensor_data_path = config.SENSOR_DATA_PATH
    if not os.path.exists(sensor_data_path):
        print(f"Error: Sensor data CSV not found at {sensor_data_path}. Please check your dataset path.")
        return

    SUB = pd.read_csv(sensor_data_path) #cite: 1
    print("Sensor data head:")
    print(SUB.head()) #cite: 1

    times = SUB.iloc[:, 0].values #cite: 1
    labels = SUB.iloc[:, -1].values #cite: 1
    Time_Label = pd.DataFrame(labels, index=times) #cite: 1
    print("\nTime_Label head:")
    print(Time_Label.head()) #cite: 1

    # --- Load and Process Camera 1 Images ---
    print("\nLoading and processing Camera 1 images...")
    img_1, path_1 = data_loader.load_img(
        config.START_SUBJECT, config.END_SUBJECT,
        config.START_ACTIVITY, config.END_ACTIVITY,
        1, 1, # Only Camera 1
        config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT
    ) #cite: 1

    name_1 = data_loader.handle_name(path_1) #cite: 1

    # --- Save Processed Camera 1 Data ---
    print(f"\nSaving processed Camera 1 data ({len(img_1)} images)...")
    np.save(os.path.join(config.DATASET_BASE_DIR, config.IMAGE_FILE_PREFIX + '1' + config.NPY_EXTENSION), img_1) #cite: 1
    np.save(os.path.join(config.DATASET_BASE_DIR, config.NAME_FILE_PREFIX + '1' + config.NPY_EXTENSION), name_1) #cite: 1

    # --- Load and Process Camera 2 Images ---
    print("\nLoading and processing Camera 2 images...")
    img_2, path_2 = data_loader.load_img(
        config.START_SUBJECT, config.END_SUBJECT,
        config.START_ACTIVITY, config.END_ACTIVITY,
        2, 2, # Only Camera 2
        config.DEFAULT_IMAGE_WIDTH, config.DEFAULT_IMAGE_HEIGHT
    ) #cite: 1

    name_2 = data_loader.handle_name(path_2)

    # --- Save Processed Camera 2 Data ---
    print(f"\nSaving processed Camera 2 data ({len(img_2)} images)...")
    np.save(os.path.join(config.DATASET_BASE_DIR, config.IMAGE_FILE_PREFIX + '2' + config.NPY_EXTENSION), img_2) #cite: 1
    np.save(os.path.join(config.DATASET_BASE_DIR, config.NAME_FILE_PREFIX + '2' + config.NPY_EXTENSION), name_2) #cite: 1

    # --- Data Synchronization (Matching timestamps) ---
    print("\nSynchronizing image data based on timestamps...")
    # Find indices where names in name_1 and name_2 do not match
    # Assuming the original notebook's logic of finding non-matching
    # elements and deleting the first found non-matching index is intentional.
    # This is a very specific indexing operation; ensure it aligns with your data structure.
    
    # Create index arrays for deletion
    ind1 = np.arange(len(name_1)) #cite: 1
    red_in1 = ind1[~np.isin(name_1, name_2)] #cite: 1

    # Perform deletion if red_in1 is not empty
    if len(red_in1) > 0:
        # Note: The original code only deletes the first element `red_in1[0]`
        # This might be an oversight if multiple mismatches exist.
        # If there are multiple mismatches and you want to remove all of them,
        # you would need to iterate or use a more advanced set operation.
        name_1d = np.delete(name_1, red_in1[0]) #cite: 1
        img_1d = np.delete(img_1, red_in1[0], axis=0) #cite: 1
    else:
        name_1d = np.copy(name_1)
        img_1d = np.copy(img_1)

    ind2 = np.arange(len(name_2)) #cite: 1
    red_in2 = ind2[~np.isin(name_2, name_1)] #cite: 1
    
    # Perform deletion if red_in2 is not empty
    if len(red_in2) > 0:
        # Similar note for red_in2[0] deletion
        name_2d = np.delete(name_2, red_in2[0]) #cite: 1
        img_2d = np.delete(img_2, red_in2[0], axis=0) #cite: 1
    else:
        name_2d = np.copy(name_2)
        img_2d = np.copy(img_2)
    
    # Verify synchronization
    if (name_1d == name_2d).all(): #cite: 1
        print("Image names for Camera 1 and Camera 2 are synchronized.") #cite: 1
        print(f"Number of synchronized images: {len(name_1d)}") #cite: 1
    else:
        print("Warning: Image names for Camera 1 and Camera 2 are NOT fully synchronized after deletion.")

    # --- Load Labels based on synchronized names ---
    print("\nLoading labels for synchronized images...")
    label_1 = Time_Label.loc[name_1d].values #cite: 1
    label_2 = Time_Label.loc[name_2d].values #cite: 1

    print(f"Length of synchronized img_1: {len(img_1d)}") #cite: 1
    print(f"Length of synchronized name_1: {len(name_1d)}") #cite: 1
    print(f"Length of labels for img_1: {len(label_1)}") #cite: 1
    print(f"Length of synchronized img_2: {len(img_2d)}") #cite: 1
    print(f"Length of synchronized name_2: {len(name_2d)}") #cite: 1
    print(f"Length of labels for img_2: {len(label_2)}") #cite: 1

    # --- Save Synchronized and Labeled Data ---
    print("\nSaving synchronized and labeled data...")
    # Save Camera 1 data with labels
    np.save(os.path.join(config.DATASET_BASE_DIR, config.IMAGE_FILE_PREFIX + '1' + config.NPY_EXTENSION), img_1d) #cite: 1
    np.save(os.path.join(config.DATASET_BASE_DIR, config.NAME_FILE_PREFIX + '1' + config.NPY_EXTENSION), name_1d) #cite: 1
    np.save(os.path.join(config.DATASET_BASE_DIR, config.LABEL_FILE_PREFIX + '1' + config.NPY_EXTENSION), label_1) #cite: 1

    # Save Camera 2 data with labels
    np.save(os.path.join(config.DATASET_BASE_DIR, config.IMAGE_FILE_PREFIX + '2' + config.NPY_EXTENSION), img_2d) #cite: 1
    np.save(os.path.join(config.DATASET_BASE_DIR, config.NAME_FILE_PREFIX + '2' + config.NPY_EXTENSION), name_2d) #cite: 1
    np.save(os.path.join(config.DATASET_BASE_DIR, config.LABEL_FILE_PREFIX + '2' + config.NPY_EXTENSION), label_2) #cite: 1
    
    print("\nData loading, processing, and saving complete!")

if __name__ == "__main__":
    main()