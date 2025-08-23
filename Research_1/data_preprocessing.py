# data_preprocessing.py

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import os
import random

import config
from utils import set_seed # Import set_seed from utils

def load_and_preprocess_data():
    """
    Loads raw sensor and image data, performs preprocessing steps
    (handling NaNs, duplicates, scaling, reshaping), and splits data
    into training, validation, and test sets.

    Returns:
        dict: A dictionary containing preprocessed and split data.
              Keys include 'X_train_csv_scaled', 'Y_train_csv', etc.
    """
    set_seed(config.RANDOM_SEED)

    # --- Load Sensor Data ---
    print("Loading sensor data...")
    SUB = pd.read_csv(config.SENSOR_CSV_PATH)
    print("Original Sensor Data shape:", SUB.shape)

    # Data Cleaning for Sensor Data
    NA_cols = SUB.columns[SUB.isnull().any()]
    SUB.dropna(inplace=True)
    SUB.drop_duplicates(inplace=True)
    print(f"Sensor Data shape after dropping NaN and redundant samples: {SUB.shape}")

    times = SUB['Time']
    list_DROP = [
        'Infrared 1', 'Infrared 2', 'Infrared 3',
        'Infrared 4', 'Infrared 5', 'Infrared 6'
    ]
    SUB.drop(list_DROP, axis=1, inplace=True)
    SUB.drop(NA_cols, axis=1, inplace=True) # drop NAN COLS
    SUB.set_index('Time', inplace=True)
    print(f"Sensor Data shape after dropping columns containing NaN values: {SUB.shape}")

    # --- Load Image Data ---
    print("\nLoading image data...")
    img_1 = np.load(config.IMAGE_CAM1_NPY)
    label_1 = np.load(config.LABEL_CAM1_NPY)
    name_1 = np.load(config.NAME_CAM1_NPY)

    img_2 = np.load(config.IMAGE_CAM2_NPY)
    label_2 = np.load(config.LABEL_CAM2_NPY)
    name_2 = np.load(config.NAME_CAM2_NPY)

    print(f"Initial img_1 shape: {img_1.shape}, name_1 length: {len(name_1)}")
    print(f"Initial img_2 shape: {img_2.shape}, name_2 length: {len(name_2)}")

    # Remove NaN values corresponding to index sample in csv file
    # This logic is directly from the notebook and assumes specific indexing behavior.
    # It might only remove the first encountered redundant entry if multiple exist.
    # For a more robust solution, consider set operations or filtering.
    redundant_1 = list(set(name_1) - set(times))
    if redundant_1:
        red_in1 = np.where(np.isin(name_1, redundant_1))[0]
        name_1 = np.delete(name_1, red_in1)
        img_1 = np.delete(img_1, red_in1, axis=0)
        label_1 = np.delete(label_1, red_in1)

    redundant_2 = list(set(name_2) - set(times))
    if redundant_2:
        red_in2 = np.where(np.isin(name_2, redundant_2))[0]
        name_2 = np.delete(name_2, red_in2)
        img_2 = np.delete(img_2, red_in2, axis=0)
        label_2 = np.delete(label_2, red_in2)

    print(f"img_1 shape after NaN removal: {img_1.shape}, name_1 length: {len(name_1)}")
    print(f"img_2 shape after NaN removal: {img_2.shape}, name_2 length: {len(name_2)}")

    # Data synchronization (matching timestamps)
    # This part assumes that after initial NaN removal, `name_1` and `name_2`
    # should ideally be identical or have minimal differences that need reconciliation.
    # The original notebook's logic of deleting the *first* non-matching index
    # is preserved but might not be ideal for all synchronization scenarios.
    
    # Identify names unique to cam1 or cam2
    unique_to_name1 = np.setdiff1d(name_1, name_2)
    unique_to_name2 = np.setdiff1d(name_2, name_1)

    # Filter arrays to keep only common names
    common_names = np.intersect1d(name_1, name_2)

    # Re-index data based on common names
    # This is a more robust way to synchronize compared to deleting single indices.
    # It ensures that only samples present in *both* camera streams (by name) are kept.
    img_1_synced = img_1[np.isin(name_1, common_names)]
    label_1_synced = label_1[np.isin(name_1, common_names)]
    name_1_synced = name_1[np.isin(name_1, common_names)]

    img_2_synced = img_2[np.isin(name_2, common_names)]
    label_2_synced = label_2[np.isin(name_2, common_names)]
    name_2_synced = name_2[np.isin(name_2, common_names)]

    # Ensure labels are aligned with the synchronized names from SUB DataFrame
    # This assumes SUB's index (times) is the ground truth for labels.
    # Sort common_names to ensure consistent order for .loc indexing
    common_names_sorted = np.sort(common_names)
    
    # Use .reindex to align labels based on common_names_sorted
    label_csv_synced = SUB.loc[common_names_sorted, 'Label'].values

    # Verify synchronization
    if (label_1_synced == label_csv_synced).all() and \
       (label_2_synced == label_csv_synced).all() and \
       (name_1_synced == common_names_sorted).all() and \
       (name_2_synced == common_names_sorted).all():
        print("Image names and labels for Camera 1 and Camera 2 are synchronized with CSV.")
    else:
        print("Warning: Synchronization issues remain or data order differs.")
        # For debugging, you might want to print differences:
        # print("Labels 1 vs CSV:", (label_1_synced == label_csv_synced).all())
        # print("Labels 2 vs CSV:", (label_2_synced == label_csv_synced).all())
        # print("Names 1 vs Sorted Common:", (name_1_synced == common_names_sorted).all())
        # print("Names 2 vs Sorted Common:", (name_2_synced == common_names_sorted).all())


    # --- Prepare Data for Models ---
    data = SUB.loc[common_names_sorted].values # Use synchronized names for CSV data
    X_csv, y_csv = data[:, :-1], data[:, -1]

    # Handle class 20 (if it exists) by mapping it to 0
    y_csv = np.where(y_csv == 20, 0, y_csv)
    label_1_synced = np.where(label_1_synced == 20, 0, label_1_synced)
    label_2_synced = np.where(label_2_synced == 20, 0, label_2_synced)

    # Split data
    X_train_csv, X_rem_csv, y_train_csv, y_rem_csv = train_test_split(
        X_csv, y_csv, train_size=0.6, random_state=config.RANDOM_SEED, stratify=y_csv # Stratify for balanced classes
    )
    X_val_csv, X_test_csv, y_val_csv, y_test_csv = train_test_split(
        X_rem_csv, y_rem_csv, test_size=0.5, random_state=config.RANDOM_SEED, stratify=y_rem_csv # Stratify for balanced classes
    )

    X_train_1, X_rem_1, y_train_1, y_rem_1 = train_test_split(
        img_1_synced, label_1_synced, train_size=0.6, random_state=config.RANDOM_SEED, stratify=label_1_synced
    )
    X_val_1, X_test_1, y_val_1, y_test_1 = train_test_split(
        X_rem_1, y_rem_1, test_size=0.5, random_state=config.RANDOM_SEED, stratify=y_rem_1
    )

    X_train_2, X_rem_2, y_train_2, y_rem_2 = train_test_split(
        img_2_synced, label_2_synced, train_size=0.6, random_state=config.RANDOM_SEED, stratify=label_2_synced
    )
    X_val_2, X_test_2, y_val_2, y_test_2 = train_test_split(
        X_rem_2, y_rem_2, test_size=0.5, random_state=config.RANDOM_SEED, stratify=y_rem_2
    )

    # Scale CSV data
    scaler = StandardScaler()
    X_train_csv_scaled = scaler.fit_transform(X_train_csv)
    X_test_csv_scaled = scaler.transform(X_test_csv)
    X_val_csv_scaled = scaler.transform(X_val_csv)

    # Convert labels to categorical (one-hot encoding) for Keras models
    Y_train_csv = to_categorical(y_train_csv, config.NUM_CLASSES)
    Y_test_csv = to_categorical(y_test_csv, config.NUM_CLASSES)
    Y_val_csv = to_categorical(y_val_csv, config.NUM_CLASSES)

    Y_train_1 = to_categorical(y_train_1, config.NUM_CLASSES)
    Y_test_1 = to_categorical(y_test_1, config.NUM_CLASSES)
    Y_val_1 = to_categorical(y_val_1, config.NUM_CLASSES)

    Y_train_2 = to_categorical(y_train_2, config.NUM_CLASSES)
    Y_test_2 = to_categorical(y_test_2, config.NUM_CLASSES)
    Y_val_2 = to_categorical(y_val_2, config.NUM_CLASSES)

    # Reshape image data for CNNs (add channel dimension)
    shape1, shape2 = 32, 32 # Assuming images are 32x32
    X_train_1_scaled = X_train_1.reshape(X_train_1.shape[0], shape1, shape2, 1) / 255.0
    X_train_2_scaled = X_train_2.reshape(X_train_2.shape[0], shape1, shape2, 1) / 255.0
    X_val_1_scaled = X_val_1.reshape(X_val_1.shape[0], shape1, shape2, 1) / 255.0
    X_val_2_scaled = X_val_2.reshape(X_val_2.shape[0], shape1, shape2, 1) / 255.0
    X_test_1_scaled = X_test_1.reshape(X_test_1.shape[0], shape1, shape2, 1) / 255.0
    X_test_2_scaled = X_test_2.reshape(X_test_2.shape[0], shape1, shape2, 1) / 255.0

    print("\nData preprocessing complete. Shapes of final datasets:")
    print(f"X_train_csv_scaled: {X_train_csv_scaled.shape}, Y_train_csv: {Y_train_csv.shape}")
    print(f"X_val_csv_scaled: {X_val_csv_scaled.shape}, Y_val_csv: {Y_val_csv.shape}")
    print(f"X_test_csv_scaled: {X_test_csv_scaled.shape}, Y_test_csv: {Y_test_csv.shape}")
    print(f"X_train_1_scaled: {X_train_1_scaled.shape}, Y_train_1: {Y_train_1.shape}")
    print(f"X_val_1_scaled: {X_val_1_scaled.shape}, Y_val_1: {Y_val_1.shape}")
    print(f"X_test_1_scaled: {X_test_1_scaled.shape}, Y_test_1: {Y_test_1.shape}")
    print(f"X_train_2_scaled: {X_train_2_scaled.shape}, Y_train_2: {Y_train_2.shape}")
    print(f"X_val_2_scaled: {X_val_2_scaled.shape}, Y_val_2: {Y_val_2.shape}")
    print(f"X_test_2_scaled: {X_test_2_scaled.shape}, Y_test_2: {Y_test_2.shape}")

    return {
        'X_train_csv_scaled': X_train_csv_scaled, 'Y_train_csv': Y_train_csv,
        'X_val_csv_scaled': X_val_csv_scaled, 'Y_val_csv': Y_val_csv,
        'X_test_csv_scaled': X_test_csv_scaled, 'Y_test_csv': Y_test_csv,
        'y_test_csv': y_test_csv, # Keep original integer labels for XGBoost/CatBoost evaluation

        'X_train_1_scaled': X_train_1_scaled, 'Y_train_1': Y_train_1,
        'X_val_1_scaled': X_val_1_scaled, 'Y_val_1': Y_val_1,
        'X_test_1_scaled': X_test_1_scaled, 'Y_test_1': Y_test_1,
        'y_test_1': y_test_1,

        'X_train_2_scaled': X_train_2_scaled, 'Y_train_2': Y_train_2,
        'X_val_2_scaled': X_val_2_scaled, 'Y_val_2': Y_val_2,
        'X_test_2_scaled': X_test_2_scaled, 'Y_test_2': Y_test_2,
        'y_test_2': y_test_2,
    }