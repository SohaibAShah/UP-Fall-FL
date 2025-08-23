# data_preprocessing.py 
# This file handles the loading, cleaning, and splitting of the sensor and image data.

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from zmq import SUB
from utils import set_seed, scale_data # Import from local utils.py


def loadClientsData():
    subs = [1, 3, 4, 7, 10, 11, 12, 13, 14, 15, 16, 17]
    X_train_csv_scaled_splits = {}
    X_test_csv_scaled_splits = {}
    Y_train_csv_splits = {}
    Y_test_csv_splits = {}
    X_train_1_scaled_splits = {}
    X_test_1_scaled_splits = {}
    Y_train_1_splits = {}
    Y_test_1_splits = {}
    X_train_2_scaled_splits = {}
    X_test_2_scaled_splits = {}
    Y_train_2_splits = {}
    Y_test_2_splits = {}
    clint_index = 0
    for sub_ in subs:
        SUB_train = pd.read_csv('/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image/{}_sensor_train.csv'.format(sub_), skiprows=1)
        SUB_train.head()
        # print('{}_SUB.shap'.format(sub_),SUB_train.shape)

        SUB_train.isnull().sum()
        NA_cols = SUB_train.columns[SUB_train.isnull().any()]
        # print('Columns contain NULL values : \n', NA_cols)
        SUB_train.dropna(inplace=True)
        SUB_train.drop_duplicates(inplace=True)
        # print('Sensor Data shape after dropping NaN and redudant samples :', SUB_train.shape)
        times_train = SUB_train['Time']
        list_DROP = ['Infrared 1',
                     'Infrared 2',
                     'Infrared 3',
                     'Infrared 4',
                     'Infrared 5',
                     'Infrared 6']
        SUB_train.drop(list_DROP, axis=1, inplace=True)
        SUB_train.drop(NA_cols, axis=1, inplace=True)  # drop NAN COLS

        # print('{}_train_Sensor Data shape after dropping columns contain NaN values :'.format(sub_), SUB_train.shape)

        SUB_train.set_index('Time', inplace=True)
        SUB_train.head()

        cam = '1'
        image_train = '/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image' + '/' + '{}_image_1_train.npy'.format(sub_)
        name_train = '/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image' + '/' + '{}_name_1_train.npy'.format(sub_)
        label_train = '/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image' + '/' + '{}_label_1_train.npy'.format(sub_)

        img_1_train = np.load(image_train)
        label_1_train = np.load(label_train)
        name_1_train = np.load(name_train)

        cam = '2'

        image_train = '/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image' + '/' + '{}_image_2_train.npy'.format(sub_)
        name_train = '/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image' + '/' + '{}_name_2_train.npy'.format(sub_)
        label_train = '/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image' + '/' + '{}_label_2_train.npy'.format(sub_)

        img_2_train = np.load(image_train)
        label_2_train = np.load(label_train)
        name_2_train = np.load(name_train)

        # print('{}_len(img_1_train)'.format(sub_),len(img_1_train))
        # print('{}_len(name_1_train)'.format(sub_),len(name_1_train))
        # print('{}_len(img_2_train)'.format(sub_),len(img_2_train))
        # print('{}_len(name_2_train)'.format(sub_),len(name_2_train))

        # remove NaN values corresponding to index sample in csv file
        redundant_1 = list(set(name_1_train) - set(times_train))
        redundant_2 = list(set(name_2_train) - set(times_train))
        # ind = np.arange(0, 294677)
        ind = np.arange(0, len(img_1_train))

        red_in1 = ind[np.isin(name_1_train, redundant_1)]
        name_1_train = np.delete(name_1_train, red_in1)
        img_1_train = np.delete(img_1_train, red_in1, axis=0)
        label_1_train = np.delete(label_1_train, red_in1)

        red_in2 = ind[np.isin(name_2_train, redundant_2)]
        name_2_train = np.delete(name_2_train, red_in2)
        img_2_train = np.delete(img_2_train, red_in2, axis=0)
        label_2_train = np.delete(label_2_train, red_in2)

        # print('{}_len(name_1_train)'.format(sub_),len(name_1_train))
        # print('{}_len(name_2_train)'.format(sub_),len(name_2_train))

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


        data_train = SUB_train.loc[name_1_train].values
        # print(img_1_train.shape)
        # print(img_2_train.shape)
        # print(data_train.shape)

        # print((label_2_train == data_train[:, -1]).all())
        # print((label_1_train == data_train[:, -1]).all())

        set_seed()
        X_csv_train, y_csv_train = data_train[:, :-1], data_train[:, -1]
        y_csv_train = np.where(y_csv_train == 20, 0, y_csv_train)

        label_1_train = np.where(label_1_train == 20, 0, label_1_train)
        label_2_train = np.where(label_2_train == 20, 0, label_2_train)

        # print('X_csv_train shape : ', X_csv_train.shape)
        # print('y_csv_train shape : ', y_csv_train.shape)

        Y_csv_train = torch.nn.functional.one_hot(torch.from_numpy(y_csv_train).long(), 12).float()

        X_csv_train_scaled = scale_data(X_csv_train)

        print('X_csv_train_scaled shape : ', X_csv_train_scaled.shape)
        print('Y_csv_train shape : ', Y_csv_train.shape)

        X_train_1 = img_1_train
        y_train_1 = label_1_train

        # print('*' * 20)
        # print('X_train_1 shape : ', X_train_1.shape)
        # print('y_train_1 shape : ', y_train_1.shape)

        Y_train_1 = torch.nn.functional.one_hot(torch.from_numpy(y_train_1).long(), 12).float()

        print('X_train_1 shape : ', X_train_1.shape)
        print('y_train_1 shape : ', Y_train_1.shape)

        X_train_2 = img_2_train
        y_train_2 = label_2_train

        # print('*' * 20)
        # print('X_train_2 shape : ', X_train_2.shape)
        # print('y_train_2 shape : ', y_train_2.shape)

        Y_train_2 = torch.nn.functional.one_hot(torch.from_numpy(y_train_2).long(), 12).float()

        print('X_train_2 shape : ', X_train_2.shape)
        print('y_train_2 shape : ', Y_train_2.shape)


        # print('(y_train_1 == y_csv_train).all():',(y_train_1 == y_csv_train).all())
        # print('(y_train_2 == y_csv_train).all()',(y_train_2 == y_csv_train).all())


        shape1, shape2 = 32, 32
        X_train_1 = X_train_1.reshape(X_train_1.shape[0], shape1, shape2, 1)
        X_train_2 = X_train_2.reshape(X_train_2.shape[0], shape1, shape2, 1)


        X_train_1_scaled = X_train_1 / 255.0
        X_train_2_scaled = X_train_2 / 255.0

        SUB_test = pd.read_csv('/home/syed/PhD/UP-Fall-FL/dataset/Sensor + Image/{}_sensor_test.csv'.format(sub_), skiprows=1)
        SUB_test.head()
        # print('{}_SUB.shap'.format(sub_), SUB_test.shape)

        SUB_test.isnull().sum()
        NA_cols = SUB_test.columns[SUB_test.isnull().any()]
        # print('Columns contain NULL values : \n', NA_cols)
        SUB_test.dropna(inplace=True)
        SUB_test.drop_duplicates(inplace=True)
        # print('Sensor Data shape after dropping NaN and redudant samples :', SUB_test.shape)
        times_test = SUB_test['Time']
        list_DROP = ['Infrared 1',
                     'Infrared 2',
                     'Infrared 3',
                     'Infrared 4',
                     'Infrared 5',
                     'Infrared 6']
        SUB_test.drop(list_DROP, axis=1, inplace=True)
        SUB_test.drop(NA_cols, axis=1, inplace=True)  # drop NAN COLS

        # print('{}_test_Sensor Data shape after dropping columns contain NaN values :'.format(sub_), SUB_test.shape)

        SUB_test.set_index('Time', inplace=True)
        SUB_test.head()

        cam = '1'
        image_test = './dataset/Sensor + Image' + '/' + '{}_image_1_test.npy'.format(sub_)
        name_test = './dataset/Sensor + Image' + '/' + '{}_name_1_test.npy'.format(sub_)
        label_test = './dataset/Sensor + Image' + '/' + '{}_label_1_test.npy'.format(sub_)

        img_1_test = np.load(image_test)
        label_1_test = np.load(label_test)
        name_1_test = np.load(name_test)

        cam = '2'

        image_test = './dataset/Sensor + Image' + '/' + '{}_image_2_test.npy'.format(sub_)
        name_test = './dataset/Sensor + Image' + '/' + '{}_name_2_test.npy'.format(sub_)
        label_test = './dataset/Sensor + Image' + '/' + '{}_label_2_test.npy'.format(sub_)

        img_2_test = np.load(image_test)
        label_2_test = np.load(label_test)
        name_2_test = np.load(name_test)


        # print('{}_len(img_1_test)'.format(sub_), len(img_1_test))
        # print('{}_len(name_1_test)'.format(sub_), len(name_1_test))
        # print('{}_len(img_2_test)'.format(sub_), len(img_2_test))
        # print('{}_len(name_2_test)'.format(sub_), len(name_2_test))

        # remove NaN values corresponding to index sample in csv file
        redundant_1 = list(set(name_1_test) - set(times_test))
        redundant_2 = list(set(name_2_test) - set(times_test))
        # ind = np.arange(0, 294677)
        ind = np.arange(0, len(img_1_test))

        red_in1 = ind[np.isin(name_1_test, redundant_1)]
        name_1_test = np.delete(name_1_test, red_in1)
        img_1_test = np.delete(img_1_test, red_in1, axis=0)
        label_1_test = np.delete(label_1_test, red_in1)

        red_in2 = ind[np.isin(name_2_test, redundant_2)]
        name_2_test = np.delete(name_2_test, red_in2)
        img_2_test = np.delete(img_2_test, red_in2, axis=0)
        label_2_test = np.delete(label_2_test, red_in2)

        # print('{}_len(name_1_test)'.format(sub_), len(name_1_test))
        # print('{}_len(name_2_test)'.format(sub_), len(name_2_test))

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

        data_test = SUB_test.loc[name_1_test].values
        # print(img_1_test.shape)
        # print(img_2_test.shape)
        # print(data_test.shape)

        # print((label_2_test == data_test[:, -1]).all())
        # print((label_1_test == data_test[:, -1]).all())

        set_seed()
        X_csv_test, y_csv_test = data_test[:, :-1], data_test[:, -1]
        y_csv_test = np.where(y_csv_test == 20, 0, y_csv_test)

        label_1_test = np.where(label_1_test == 20, 0, label_1_test)
        label_2_test = np.where(label_2_test == 20, 0, label_2_test)

        # print('X_csv_test shape : ', X_csv_test.shape)
        # print('y_csv_test shape : ', y_csv_test.shape)

        Y_csv_test = torch.nn.functional.one_hot(torch.from_numpy(y_csv_test).long(), 12).float()

        X_csv_test_scaled = scale_data(X_csv_test)

        # print('X_csv_test_scaled shape : ', X_csv_test_scaled.shape)
        # print('Y_csv_test shape : ', Y_csv_test.shape)

        X_test_1 = img_1_test
        y_test_1 = label_1_test

        # print('*' * 20)
        # print('X_test_1 shape : ', X_test_1.shape)
        # print('y_test_1 shape : ', y_test_1.shape)

        Y_test_1 = torch.nn.functional.one_hot(torch.from_numpy(y_test_1).long(), 12).float()

        # print('X_test_1 shape : ', X_test_1.shape)
        # print('y_test_1 shape : ', Y_test_1.shape)

        X_test_2 = img_2_test
        y_test_2 = label_2_test

        # print('*' * 20)
        # print('X_test_2 shape : ', X_test_2.shape)
        # print('y_test_2 shape : ', y_test_2.shape)

        Y_test_2 = torch.nn.functional.one_hot(torch.from_numpy(y_test_2).long(), 12).float()

        # print('X_test_2 shape : ', X_test_2.shape)
        # print('y_test_2 shape : ', Y_test_2.shape)


        # print('(y_test_1 == y_csv_test).all():', (y_test_1 == y_csv_test).all())
        # print('(y_test_2 == y_csv_test).all()', (y_test_2 == y_csv_test).all())


        X_test_1 = X_test_1.reshape(X_test_1.shape[0], shape1, shape2, 1)
        X_test_2 = X_test_2.reshape(X_test_2.shape[0], shape1, shape2, 1)

        X_test_1_scaled = X_test_1 / 255.0
        X_test_2_scaled = X_test_2 / 255.0

        # print(X_train_1_scaled.shape)
        # print(X_test_1_scaled.shape)
        #
        # print(X_train_2_scaled.shape)
        # print(X_test_2_scaled.shape)

        X_train_csv_scaled_splits[clint_index] = X_csv_train_scaled
        X_test_csv_scaled_splits[clint_index] = X_csv_test_scaled
        Y_train_csv_splits[clint_index] = Y_csv_train
        Y_test_csv_splits[clint_index] = Y_csv_test
        X_train_1_scaled_splits[clint_index] = X_train_1_scaled
        X_test_1_scaled_splits[clint_index] = X_test_1_scaled
        Y_train_1_splits[clint_index] = Y_train_1
        Y_test_1_splits[clint_index] = Y_test_1
        X_train_2_scaled_splits[clint_index] = X_train_2
        X_test_2_scaled_splits[clint_index] = X_test_2_scaled
        Y_train_2_splits[clint_index] = Y_train_2
        Y_test_2_splits[clint_index] = Y_test_2
        clint_index += 1
    return (X_train_csv_scaled_splits,X_test_csv_scaled_splits, 
            Y_train_csv_splits,Y_test_csv_splits,
            X_train_1_scaled_splits,X_test_1_scaled_splits,
            Y_train_1_splits,Y_test_1_splits,
            X_train_2_scaled_splits,X_test_2_scaled_splits,
            Y_train_2_splits,Y_test_2_splits
        )