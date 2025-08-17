# data_preprocessing.py 
# This file handles the loading, cleaning, and splitting of the sensor and image data.

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from zmq import SUB
from utils import set_seed, scaled_data # Import from local utils.py


def loadData():
    SUB = pd.read_csv('./dataset/Sensor + Image/sensor.csv', skiprows=1)
    SUB.head()
    print(SUB.shape)

    SUB.isnull().sum()
    NA_cols = SUB.columns[SUB.isnull().any()]
    print('Columns contain NULL values : \n', NA_cols)
    SUB.dropna(inplace=True)
    SUB.drop_duplicates(inplace=True)
    print('Sensor Data shape after dropping NaN and redudant samples :', SUB.shape)
    times = SUB['Time']
    list_DROP = ['Infrared1',
                 'Infrared2',
                 'Infrared3',
                 'Infrared4',
                 'Infrared5',
                 'Infrared6']
    SUB.drop(list_DROP, axis=1, inplace=True)
    SUB.drop(NA_cols, axis=1, inplace=True)  # drop NAN COLS

    print('Sensor Data shape after dropping columns contain NaN values :', SUB.shape)

    SUB.set_index('Time', inplace=True)
    SUB.head()

    cam = '1'

    image = './dataset/Sensor + Image' + '/' + 'image_' + cam + '.npy'
    name = './dataset/Sensor + Image' + '/' + 'name_' + cam + '.npy'
    label = './dataset/Sensor + Image' + '/' + 'label_' + cam + '.npy'

    img_1 = np.load(image)
    label_1 = np.load(label)
    name_1 = np.load(name)

    cam = '2'

    image = './dataset/Sensor + Image' + '/' + 'image_' + cam + '.npy'
    name = './dataset/Sensor + Image' + '/' + 'name_' + cam + '.npy'
    label = './dataset/Sensor + Image' + '/' + 'label_' + cam + '.npy'

    img_2 = np.load(image)
    label_2 = np.load(label)
    name_2 = np.load(name)


    print(len(img_1))
    print(len(name_1))
    print(len(img_2))
    print(len(name_2))


    # remove NaN values corresponding to index sample in csv file
    redundant_1 = list(set(name_1) - set(times))
    redundant_2 = list(set(name_2) - set(times))
    # ind = np.arange(0, 294677)
    ind = np.arange(0, len(img_1))

    red_in1 = ind[np.isin(name_1, redundant_1)]
    name_1 = np.delete(name_1, red_in1)
    img_1 = np.delete(img_1, red_in1, axis=0)
    label_1 = np.delete(label_1, red_in1)

    red_in2 = ind[np.isin(name_2, redundant_2)]
    name_2 = np.delete(name_2, red_in2)
    img_2 = np.delete(img_2, red_in2, axis=0)
    label_2 = np.delete(label_2, red_in2)

    print(len(name_2))
    print(len(name_1))

    class_name = ['?????',
                  'Falling hands',
                  'Falling knees',
                  'Falling backwards',
                  'Falling sideward',
                  ' Falling chair',
                  ' Walking',
                  'Standing',
                  'Sitting',
                  'Picking object',
                  'Jumping',
                  'Laying']

    """ plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_1[i], cmap='gray')
        plt.xlabel(class_name[label_1[i]])
    plt.show()


    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img_2[i], cmap='gray')
        plt.xlabel(class_name[label_2[i]])
    plt.show()"""


    data = SUB.loc[name_1].values
    print(img_1.shape)
    print(img_2.shape)
    print(data.shape)

    print((label_2 == data[:, -1]).all())
    print((label_1 == data[:, -1]).all())

    set_seed()
    X_csv, y_csv = data[:, :-1], data[:, -1]

    y_csv = np.where(y_csv == 20, 0, y_csv)
    label_1 = np.where(label_1 == 20, 0, label_1)
    label_2 = np.where(label_2 == 20, 0, label_2)
    X_train_csv, X_rem_csv, y_train_csv, y_rem_csv = train_test_split(X_csv, y_csv,
                                                                      train_size=0.6,
                                                                      random_state=42)

    X_val_csv, X_test_csv, y_val_csv, y_test_csv = train_test_split(X_rem_csv, y_rem_csv,
                                                                    test_size=0.5,
                                                                    random_state=42)

    print('X_train_csv shape : ', X_train_csv.shape)
    print('X_test_csv shape : ', X_test_csv.shape)
    print('X_val_csv shape : ', X_val_csv.shape)
    print('y_train_csv shape : ', y_train_csv.shape)
    print('y_test_csv shape : ', y_test_csv.shape)
    print('y_val_csv shape : ', y_val_csv.shape)

    Y_train_csv = torch.nn.functional.one_hot(torch.from_numpy(y_train_csv).long(), 12).float()
    Y_test_csv = torch.nn.functional.one_hot(torch.from_numpy(y_test_csv).long(), 12).float()
    Y_val_csv = torch.nn.functional.one_hot(torch.from_numpy(y_val_csv).long(), 12).float()

    X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled = scaled_data(X_train_csv, X_test_csv, X_val_csv)

    print('Y_train_csv shape : ', Y_train_csv.shape)
    print('Y_test_csv shape : ', Y_test_csv.shape)
    print('Y_val_csv shape : ', Y_val_csv.shape)

    X_train_1, X_rem_1, y_train_1, y_rem_1 = train_test_split(img_1, label_1,
                                                              train_size=0.6,
                                                              random_state=42,
                                                              )

    X_val_1, X_test_1, y_val_1, y_test_1 = train_test_split(X_rem_1, y_rem_1,
                                                            test_size=0.5,
                                                            random_state=42,
                                                            )
    print('*' * 20)
    print('X_train_1 shape : ', X_train_1.shape)
    print('X_test_1 shape : ', X_test_1.shape)
    print('X_val_1 shape : ', X_val_1.shape)
    print('y_train_1 shape : ', y_train_1.shape)
    print('y_test_1 shape : ', y_test_1.shape)
    print('y_val_1 shape : ', y_val_1.shape)

    Y_train_1 = torch.nn.functional.one_hot(torch.from_numpy(y_train_1).long(), 12).float()
    Y_test_1 = torch.nn.functional.one_hot(torch.from_numpy(y_test_1).long(), 12).float()
    Y_val_1 = torch.nn.functional.one_hot(torch.from_numpy(y_val_1).long(), 12).float()

    print('Y_train_1 shape : ', Y_train_1.shape)
    print('Y_test_1 shape : ', Y_test_1.shape)
    print('Y_val_1 shape : ', Y_val_1.shape)

    X_train_2, X_rem_2, y_train_2, y_rem_2 = train_test_split(img_2, label_2,
                                                              train_size=0.6,
                                                              random_state=42,
                                                              )

    X_val_2, X_test_2, y_val_2, y_test_2 = train_test_split(X_rem_2, y_rem_2,
                                                            test_size=0.5,
                                                            random_state=42,
                                                            )

    print('*' * 20)
    print('X_train_2 shape : ', X_train_2.shape)
    print('X_test_2 shape : ', X_test_2.shape)
    print('X_val_2 shape : ', X_val_2.shape)
    print('y_train_2 shape : ', y_train_2.shape)
    print('y_test_2 shape : ', y_test_2.shape)
    print('y_val_2 shape : ', y_val_2.shape)

    Y_train_2 = torch.nn.functional.one_hot(torch.from_numpy(y_train_2).long(), 12).float()
    Y_test_2 = torch.nn.functional.one_hot(torch.from_numpy(y_test_2).long(), 12).float()
    Y_val_2 = torch.nn.functional.one_hot(torch.from_numpy(y_val_2).long(), 12).float()

    print('Y_train_2 shape : ', Y_train_2.shape)
    print('Y_test_2 shape : ', Y_test_2.shape)
    print('Y_val_2 shape : ', Y_val_2.shape)


    print((y_train_1 == y_train_csv).all())
    print((y_train_2 == y_train_csv).all())

    print((y_val_1 == y_val_csv).all())
    print((y_val_2 == y_val_csv).all())

    print((y_test_1 == y_test_csv).all())
    print((y_test_2 == y_test_csv).all())


    shape1, shape2 = 32, 32
    X_train_1 = X_train_1.reshape(X_train_1.shape[0], shape1, shape2, 1)
    X_train_2 = X_train_2.reshape(X_train_2.shape[0], shape1, shape2, 1)
    X_val_1 = X_val_1.reshape(X_val_1.shape[0], shape1, shape2, 1)
    X_val_2 = X_val_2.reshape(X_val_2.shape[0], shape1, shape2, 1)
    X_test_1 = X_test_1.reshape(X_test_1.shape[0], shape1, shape2, 1)
    X_test_2 = X_test_2.reshape(X_test_2.shape[0], shape1, shape2, 1)


    X_train_1_scaled = X_train_1 / 255.0
    X_train_2_scaled = X_train_2 / 255.0

    X_val_1_scaled = X_val_1 / 255.0
    X_val_2_scaled = X_val_2 / 255.0

    X_test_1_scaled = X_test_1 / 255.0
    X_test_2_scaled = X_test_2 / 255.0

    print(X_train_1_scaled.shape)
    print(X_test_1_scaled.shape)
    print(X_val_1_scaled.shape)

    print(X_train_2_scaled.shape)
    print(X_test_2_scaled.shape)
    print(X_val_2_scaled.shape)

    return X_train_csv_scaled,X_test_csv_scaled,X_val_csv_scaled,\
        Y_train_csv,Y_test_csv,Y_val_csv,\
        X_train_1_scaled,X_test_1_scaled,X_val_1_scaled,\
        Y_train_1,Y_test_1,Y_val_1,\
        X_train_2_scaled,X_test_2_scaled,X_val_2_scaled,\
        Y_train_2,Y_test_2,Y_val_2

def splitForClients(total_client,ratios,
                    X_train_csv_scaled,X_test_csv_scaled,X_val_csv_scaled,
                    Y_train_csv,Y_test_csv,Y_val_csv,
                    X_train_1_scaled,X_test_1_scaled,X_val_1_scaled,
                    Y_train_1,Y_test_1,Y_val_1,
                    X_train_2_scaled,X_test_2_scaled,X_val_2_scaled,
                    Y_train_2,Y_test_2,Y_val_2):
    # split train data
    # 样本数量
    total_samples = X_train_csv_scaled.shape[0]
    # 生成随机索引
    indices = np.random.permutation(total_samples)
    # 计算每个部分的样本数量
    split_sizes = [int(r * total_samples) for r in ratios]
    # # 确保总样本数量与分割大小匹配
    # split_sizes[-1] += total_samples - sum(split_sizes)
    # 切分数据
    X_train_csv_scaled_splits = {}
    Y_train_csv_splits = {}
    X_train_1_scaled_splits = {}
    Y_train_1_splits = {}
    X_train_2_scaled_splits = {}
    Y_train_2_splits = {}

    start_index = 0
    clientId = 0
    for size in split_sizes:
        end_index = start_index + size
        X_train_csv_scaled_splits[clientId] = X_train_csv_scaled[indices[start_index:end_index]]
        Y_train_csv_splits[clientId] = Y_train_csv[indices[start_index:end_index]]
        X_train_1_scaled_splits[clientId] = X_train_1_scaled[indices[start_index:end_index]]
        Y_train_1_splits[clientId] = Y_train_1[indices[start_index:end_index]]
        X_train_2_scaled_splits[clientId] = X_train_2_scaled[indices[start_index:end_index]]
        Y_train_2_splits[clientId] = Y_train_2[indices[start_index:end_index]]
        start_index = end_index
        clientId += 1

    # split val data=============================================================
    # 样本数量
    total_samples = X_val_csv_scaled.shape[0]
    # 生成随机索引
    indices = np.random.permutation(total_samples)
    # 计算每个部分的样本数量
    split_sizes = [int(r * total_samples) for r in ratios]
    # # 确保总样本数量与分割大小匹配
    # split_sizes[-1] += total_samples - sum(split_sizes)
    # 切分数据
    X_val_csv_scaled_splits = {}
    Y_val_csv_splits = {}
    X_val_1_scaled_splits = {}
    Y_val_1_splits = {}
    X_val_2_scaled_splits = {}
    Y_val_2_splits = {}

    start_index = 0
    clientId = 0
    for size in split_sizes:
        end_index = start_index + size
        X_val_csv_scaled_splits[clientId] = X_val_csv_scaled[indices[start_index:end_index]]
        Y_val_csv_splits[clientId] = Y_val_csv[indices[start_index:end_index]]
        X_val_1_scaled_splits[clientId] = X_val_1_scaled[indices[start_index:end_index]]
        Y_val_1_splits[clientId] = Y_val_1[indices[start_index:end_index]]
        X_val_2_scaled_splits[clientId] = X_val_2_scaled[indices[start_index:end_index]]
        Y_val_2_splits[clientId] = Y_val_2[indices[start_index:end_index]]
        start_index = end_index
        clientId += 1

    # split test data=====================================================
    # 样本数量
    total_samples = X_test_csv_scaled.shape[0]
    # 生成随机索引
    indices = np.random.permutation(total_samples)
    # 计算每个部分的样本数量
    split_sizes = [int(r * total_samples) for r in ratios]
    # # 确保总样本数量与分割大小匹配
    # split_sizes[-1] += total_samples - sum(split_sizes)
    # 切分数据
    X_test_csv_scaled_splits = {}
    Y_test_csv_splits = {}
    X_test_1_scaled_splits = {}
    Y_test_1_splits = {}
    X_test_2_scaled_splits = {}
    Y_test_2_splits = {}

    start_index = 0
    clientId = 0
    for size in split_sizes:
        end_index = start_index + size
        X_test_csv_scaled_splits[clientId] = X_test_csv_scaled[indices[start_index:end_index]]
        Y_test_csv_splits[clientId] = Y_test_csv[indices[start_index:end_index]]
        X_test_1_scaled_splits[clientId] = X_test_1_scaled[indices[start_index:end_index]]
        Y_test_1_splits[clientId] = Y_test_1[indices[start_index:end_index]]
        X_test_2_scaled_splits[clientId] = X_test_2_scaled[indices[start_index:end_index]]
        Y_test_2_splits[clientId] = Y_test_2[indices[start_index:end_index]]
        start_index = end_index
        clientId += 1
    return X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits, \
        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits, \
        X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits, \
        Y_train_1_splits, Y_test_1_splits, Y_val_1_splits, \
        X_train_2_scaled_splits, X_test_2_scaled_splits, X_val_2_scaled_splits, \
        Y_train_2_splits, Y_test_2_splits, Y_val_2_splits