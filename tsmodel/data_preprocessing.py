# data_preprocessing.py 
# This file handles the loading, cleaning, and splitting of the sensor and image data.

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import set_seed, scaled_data # Import from local utils.py


def loadData():
    print("------------------------------Sensor ---------------------------- ")
    Subjects = pd.read_csv('./dataset/Sensor + Image/sensor.csv', skiprows=1)
    print("Sensor Data Shape:", Subjects.shape)
    Subjects.isnull().sum()  # Check for missing values
    NA_columns = Subjects.columns[Subjects.isnull().any()].tolist()
    print("Columns with missing values:", NA_columns)
    Subjects.dropna(inplace=True)  # Drop rows with any missing values
    Subjects.drop_duplicates(inplace=True)  # Drop duplicate rows
    print("Subjects.shape after dropping NaN and duplicates:", Subjects.shape)

    times = Subjects['Time']
    list_DROP = ['Infrared1', 'Infrared2', 'Infrared3', 'Infrared4', 'Infrared5', 'Infrared6']
    Subjects.drop(columns=list_DROP, axis=1, inplace=True)  # Drop unnecessary columns
    Subjects.drop(NA_columns, axis=1, inplace=True)
    print("Sensor Data Shape after dropping unnecessary columns:", Subjects.shape)
    Subjects.set_index('Time', inplace=True)  # Set 'Time' as index

    print("------------------------------Camera1 ---------------------------- ")
    cam = 1
    image_1 = np.load('./dataset/Sensor + Image' + '/' + 'image_' + str(cam) + '.npy')
    print("Image 1 Shape:", image_1.shape)
    name_1 = np.load('./dataset/Sensor + Image' + '/' + 'name_' + str(cam) + '.npy')
    print("Name 1 Shape:", name_1.shape)
    label_1 = np.load('./dataset/Sensor + Image' + '/' + 'label_' + str(cam) + '.npy')
    print("Label 1 Shape:", label_1.shape)

    print("------------------------------Camera2 ---------------------------- ")
    cam = 2
    image_2 = np.load('./dataset/Sensor + Image' + '/' + 'image_' + str(cam) + '.npy')
    print("Image 2 Shape:", image_2.shape)
    name_2 = np.load('./dataset/Sensor + Image' + '/' + 'name_' + str(cam) + '.npy')
    print("Name 2 Shape:", name_2.shape)
    label_2 = np.load('./dataset/Sensor + Image' + '/' + 'label_' + str(cam) + '.npy')
    print("Label 2 Shape:", label_2.shape)

    # remove NaN values corresponding to index sample in csv file
    redundant_1 = list(set(name_1) - set(times))
    redundant_2 = list(set(name_2) - set(times))

    ind = np.arange(0, len(image_1))
    red_in1 = ind[np.isin(name_1, redundant_1)]
    name_1 = np.delete(name_1, red_in1)
    image_1 = np.delete(image_1, red_in1, axis=0)
    label_1 = np.delete(label_1, red_in1)

    red_in2 = ind[np.isin(name_2, redundant_2)]
    name_2 = np.delete(name_2, red_in2)
    image_2 = np.delete(image_2, red_in2, axis=0)
    label_2 = np.delete(label_2, red_in2)

    print("Image 1 Shape after removing NaN:", image_1.shape)
    print("Image 2 Shape after removing NaN:", image_2.shape)

    class_name = ['?????', 
                  'Falling hands', 
                  'Falling knees', 
                  'Falling backwards', 
                  'Falling sideward', 
                  'Falling chair', 
                  'Walking', 
                  'Standing', 
                  'Sitting', 
                  'Picking object', 
                  'Jumping', 
                  'Laying'
                  ]
    
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_1[i], cmap='gray')
        plt.xlabel(class_name[label_1[i]])
    plt.show()

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image_2[i], cmap='gray')
        plt.xlabel(class_name[label_2[i]])
    plt.show()

    print("------------------------------Sensor ---------------------------- ")

    data = Subjects.loc[name_1].values
    print("Image_1 Shape:", image_1.shape)
    print("Name_1 Shape:", name_1.shape)
    print("Label_1 Shape:", label_1.shape)
    print("Sensor Data Shape after indexing with Camera 1 names:", data.shape)

    set_seed(0)  # Set seed for reproducibility

    X_csv, y_csv = data[:, :-1], data[:, -1]
    y_csv = np.where(y_csv == 20, 0, y_csv)
    label_1 = np.where(label_1 == 20, 0, label_1)
    label_2 = np.where(label_2 == 20, 0, label_2)

    X_train_csv, X_rem_csv, y_train_csv, y_rem_csv = train_test_split(
        X_csv, y_csv, test_size=0.4, random_state=42
    )

    X_val_csv, X_test_csv, y_val_csv, y_test_csv = train_test_split(
        X_rem_csv, y_rem_csv, test_size=0.5, random_state=42
    )

    print("X_train_csv Shape:", X_train_csv.shape)
    print("X_val_csv Shape:", X_val_csv.shape)
    print("X_test_csv Shape:", X_test_csv.shape)
    print("y_train_csv Shape:", y_train_csv.shape)
    print("y_val_csv Shape:", y_val_csv.shape)
    print("y_test_csv Shape:", y_test_csv.shape)

    Y_train_csv = torch.nn.functional.one_hot(torch.from_numpy(y_train_csv), num_classes=12).float()
    Y_val_csv = torch.nn.functional.one_hot(torch.from_numpy(y_val_csv), num_classes=12).float()
    Y_test_csv = torch.nn.functional.one_hot(torch.from_numpy(y_test_csv), num_classes=12).float()

    print("----------------------------Scaling Sensor Data -----------------------")
    X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled = scaled_data(X_train_csv, X_test_csv, X_val_csv)

    print("Y_train_csv Shape:", Y_train_csv.shape)
    print("Y_val_csv Shape:", Y_val_csv.shape)
    print("Y_test_csv Shape:", Y_test_csv.shape)

    print("------------------------------Camera1 ---------------------------- ")
    X_train_1, X_rem_1, y_train_1, y_rem_1 = train_test_split(
        image_1, label_1, train_size=0.6, random_state=42
        )
    X_val_1, X_test_1, y_val_1, y_test_1 = train_test_split(
        X_rem_1, y_rem_1, test_size=0.5, random_state=42
        )

    print("X_train_1 Shape:", X_train_1.shape)
    print("X_val_1 Shape:", X_val_1.shape)
    print("X_test_1 Shape:", X_test_1.shape)
    print("y_train_1 Shape:", y_train_1.shape)
    print("y_val_1 Shape:", y_val_1.shape)
    print("y_test_1 Shape:", y_test_1.shape)

    Y_train_1 = torch.nn.functional.one_hot(torch.from_numpy(y_train_1), num_classes=12).float()
    Y_val_1 = torch.nn.functional.one_hot(torch.from_numpy(y_val_1), num_classes=12).float()
    Y_test_1 = torch.nn.functional.one_hot(torch.from_numpy(y_test_1), num_classes=12).float()

    print("Y_train_1 Shape:", Y_train_1.shape)
    print("Y_val_1 Shape:", Y_val_1.shape)
    print("Y_test_1 Shape:", Y_test_1.shape)

    print("------------------------------Camera 2 ---------------------------- ")
    X_train_2, X_rem_2, y_train_2, y_rem_2 = train_test_split(
        image_2, label_2, train_size=0.6, random_state=42
    )
    X_val_2, X_test_2, y_val_2, y_test_2 = train_test_split(
        X_rem_2, y_rem_2, test_size=0.5, random_state=42
    )

    print("X_train_2 Shape:", X_train_2.shape)
    print("X_val_2 Shape:", X_val_2.shape)
    print("X_test_2 Shape:", X_test_2.shape)
    print("y_train_2 Shape:", y_train_2.shape)
    print("y_val_2 Shape:", y_val_2.shape)
    print("y_test_2 Shape:", y_test_2.shape)

    Y_train_2 = torch.nn.functional.one_hot(torch.from_numpy(y_train_2), num_classes=12).float()
    Y_val_2 = torch.nn.functional.one_hot(torch.from_numpy(y_val_2), num_classes=12).float()
    Y_test_2 = torch.nn.functional.one_hot(torch.from_numpy(y_test_2), num_classes=12).float()

    print("Y_train_2 Shape:", Y_train_2.shape)
    print("Y_val_2 Shape:", Y_val_2.shape)
    print("Y_test_2 Shape:", Y_test_2.shape)

    print("----------------------------Testing Data -------------------------------")
    print((y_train_1 == y_train_csv).all())
    print((y_train_2 == y_train_csv).all())
    print((y_val_1 == y_val_csv).all())
    print((y_val_2 == y_val_csv).all())
    print((y_test_1 == y_test_csv).all())
    print((y_test_2 == y_test_csv).all())

    print("----------------------------Reshaping Images Data -----------------------")
    shape1 , shape2 = 32, 32
    X_train_1 = X_train_1.reshape(X_train_1.shape[0], 1, shape1, shape2)
    X_val_1 = X_val_1.reshape(X_val_1.shape[0], 1, shape1, shape2)
    X_test_1 = X_test_1.reshape(X_test_1.shape[0], 1, shape1, shape2)

    X_train_2 = X_train_2.reshape(X_train_2.shape[0], 1, shape1, shape2)
    X_val_2 = X_val_2.reshape(X_val_2.shape[0], 1, shape1, shape2)
    X_test_2 = X_test_2.reshape(X_test_2.shape[0], 1, shape1, shape2)

    print("----------------------------Scaling Image Data -----------------------")
    X_train_1_scaled = X_train_1 / 255.0
    X_val_1_scaled = X_val_1 / 255.0
    X_test_1_scaled = X_test_1 / 255.0

    X_train_2_scaled = X_train_2 / 255.0
    X_val_2_scaled = X_val_2 / 255.0
    X_test_2_scaled = X_test_2 / 255.0

    print("X_train_1_scaled Shape:", X_train_1_scaled.shape)
    print("X_val_1_scaled Shape:", X_val_1_scaled.shape)
    print("X_test_1_scaled Shape:", X_test_1_scaled.shape)

    print("X_train_2_scaled Shape:", X_train_2_scaled.shape)
    print("X_val_2_scaled Shape:", X_val_2_scaled.shape)
    print("X_test_2_scaled Shape:", X_test_2_scaled.shape)

    return (X_train_csv_scaled, X_val_csv_scaled, X_test_csv_scaled,
            Y_train_csv, Y_val_csv, Y_test_csv,
            X_train_1_scaled, X_val_1_scaled, X_test_1_scaled,
            Y_train_1, Y_val_1, Y_test_1,
            X_train_2_scaled, X_val_2_scaled, X_test_2_scaled,
            Y_train_2, Y_val_2, Y_test_2)


def splitforClients(total_clients, ratios,
                    X_train_csv_scaled, X_val_csv_scaled, X_test_csv_scaled,
                    Y_train_csv, Y_val_csv, Y_test_csv,
                    X_train_1_scaled, X_val_1_scaled, X_test_1_scaled,
                    Y_train_1, Y_val_1, Y_test_1,
                    X_train_2_scaled, X_val_2_scaled, X_test_2_scaled,
                    Y_train_2, Y_val_2, Y_test_2
                    ):
    
    # Split train data
    total_samples = X_train_csv_scaled.shape[0]
    indices = np.random.permutation(total_samples)
    split_size = [int(total_samples * ratio) for ratio in ratios]
    
    X_train_csv_scaled_splits = {}
    Y_train_csv_splits = {}
    X_train_1_scaled_splits = {}
    Y_train_1_splits = {}
    X_train_2_scaled_splits = {}
    Y_train_2_splits = {}

    start_idx = 0
    client_id = 0
    for size in split_size:
        end_idx = start_idx + size
        client_indices = indices[start_idx:end_idx]
        
        X_train_csv_scaled_splits[client_id] = X_train_csv_scaled[client_indices]
        Y_train_csv_splits[client_id] = Y_train_csv[client_indices]
        
        X_train_1_scaled_splits[client_id] = X_train_1_scaled[client_indices]
        Y_train_1_splits[client_id] = Y_train_1[client_indices]
        
        X_train_2_scaled_splits[client_id] = X_train_2_scaled[client_indices]
        Y_train_2_splits[client_id] = Y_train_2[client_indices]
        
        start_idx = end_idx
        client_id += 1

    # Split Validation data
    total_samples = X_val_csv_scaled.shape[0]
    indices = np.random.permutation(total_samples)
    split_size = [int(total_samples * ratio) for ratio in ratios]

    X_val_csv_scaled_splits = {}
    Y_val_csv_splits = {}
    X_val_1_scaled_splits = {}
    Y_val_1_splits = {}
    X_val_2_scaled_splits = {}
    Y_val_2_splits = {}

    start_idx = 0
    client_id = 0
    for size in split_size:
        end_idx = start_idx + size
        client_indices = indices[start_idx:end_idx]

        X_val_csv_scaled_splits[client_id] = X_val_csv_scaled[client_indices]
        Y_val_csv_splits[client_id] = Y_val_csv[client_indices]

        X_val_1_scaled_splits[client_id] = X_val_1_scaled[client_indices]
        Y_val_1_splits[client_id] = Y_val_1[client_indices]

        X_val_2_scaled_splits[client_id] = X_val_2_scaled[client_indices]
        Y_val_2_splits[client_id] = Y_val_2[client_indices]

        start_idx = end_idx
        client_id += 1

    # Split Test data
    total_samples = X_test_csv_scaled.shape[0]
    indices = np.random.permutation(total_samples)
    split_size = [int(total_samples * ratio) for ratio in ratios]

    X_test_csv_scaled_splits = {}
    Y_test_csv_splits = {}
    X_test_1_scaled_splits = {}
    Y_test_1_splits = {}
    X_test_2_scaled_splits = {}
    Y_test_2_splits = {}

    start_idx = 0
    client_id = 0
    for size in split_size:
        end_idx = start_idx + size
        client_indices = indices[start_idx:end_idx]

        X_test_csv_scaled_splits[client_id] = X_test_csv_scaled[client_indices]
        Y_test_csv_splits[client_id] = Y_test_csv[client_indices]

        X_test_1_scaled_splits[client_id] = X_test_1_scaled[client_indices]
        Y_test_1_splits[client_id] = Y_test_1[client_indices]

        X_test_2_scaled_splits[client_id] = X_test_2_scaled[client_indices]
        Y_test_2_splits[client_id] = Y_test_2[client_indices]

        start_idx = end_idx
        client_id += 1

    return (X_train_csv_scaled_splits, Y_train_csv_splits,
            X_train_1_scaled_splits, Y_train_1_splits,
            X_train_2_scaled_splits, Y_train_2_splits,
            X_val_csv_scaled_splits, Y_val_csv_splits,
            X_val_1_scaled_splits, Y_val_1_splits,
            X_val_2_scaled_splits, Y_val_2_splits,
            X_test_csv_scaled_splits, Y_test_csv_splits,
            X_test_1_scaled_splits, Y_test_1_splits,
            X_test_2_scaled_splits, Y_test_2_splits)


