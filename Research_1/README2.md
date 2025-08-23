# Fall Detection using Multimodal Sensor and Image Data

This repository contains the codebase for a PhD research project focused on developing robust fall detection systems using multimodal data, specifically combining sensor readings and camera imagery. The project explores various machine learning and deep learning models to classify human activities, with a particular emphasis on identifying fall events. This repository implements and extends the work from the following repositories:

- [Fall-Detection-Research-1](https://github.com/hoangNguyen210/Fall-Detection-Research-1)
- [Fall-Detection-Research-2](https://github.com/hoangNguyen210/Fall-Detection-Research-2)

It further incorporates **Federated Averaging (FedAvg)** for distributed learning and explores **edge-based machine learning (ML) and deep learning (DL)** models to enhance the scalability and efficiency of fall detection systems for the PhD project.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architectures](#model-architectures)
  - [Multi-Layer Perceptron (MLP)](#multi-layer-perceptron-mlp)
  - [XGBoost](#xgboost)
  - [CatBoost](#catboost)
  - [Convolutional Neural Network (CNN) - Camera 1](#convolutional-neural-network-cnn---camera-1)
  - [Convolutional Neural Network (CNN) - Camera 2](#convolutional-neural-network-cnn---camera-2)
  - [Concatenated CNN (Camera 1 & 2)](#concatenated-cnn-camera-1--2)
  - [Concatenated CSV + Images (Camera 1 & 2)](#concatenated-csv--images-camera-1--2)
  - [Random Forest](#random-forest)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [k-Nearest Neighbors (k-NN)](#k-nearest-neighbors-k-nn)
  - [Mixture of Experts (MoE) Models](#mixture-of-experts-moe-models)
  - [3D-CNN Models](#3d-cnn-models)
  - [Federated Averaging (FedAvg)](#federated-averaging-fedavg)
  - [Edge-Based ML/DL Models](#edge-based-mldl-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Requirements](#requirements)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Introduction
Falls are a significant health concern, particularly among the elderly, leading to injuries, reduced quality of life, and increased healthcare costs. This research project aims to develop and evaluate various machine learning and deep learning models for accurate and timely fall detection. By leveraging multimodal data from both wearable sensors and ambient cameras, the project seeks to build more robust and reliable detection systems compared to unimodal approaches. 

This repository builds upon the work presented at the MMM 2022 Conference (6-10 June 2022) from [Fall-Detection-Research-1](https://github.com/hoangNguyen210/Fall-Detection-Research-1) and extends the advancements in [Fall-Detection-Research-2](https://github.com/hoangNguyen210/Fall-Detection-Research-2), incorporating Federated Averaging (FedAvg) for distributed learning and edge-based ML/DL models to enable efficient deployment on resource-constrained devices.

The codebase is structured to allow for easy experimentation with different models and data modalities, providing a clear pipeline from raw data to trained model evaluation.

## Dataset
The project utilizes the [UP-Fall Dataset](https://sites.google.com/up.edu.mx/har-up/), a publicly available dataset designed for fall detection research. It includes:

- **Sensor Data (Imp_sensor.csv)**: Time-series data from wearable sensors (e.g., accelerometers, gyroscopes, luminosity, infrared, brainwave signals). Each row represents a timestamped reading.
- **Camera Data (zipped image files)**: Video frames captured from two camera perspectives (Camera 1 and Camera 2) during various activities, including fall events, provided as sequences of images within zip archives.

The dataset captures 11 activities performed by 17 young, healthy individuals (aged 18–24), including different types of falls and Activities of Daily Living (ADLs):

| Activity ID | Description                     | Duration (s) |
|-------------|---------------------------------|--------------|
| 1           | Falling forward using hands     | 10           |
| 2           | Falling forward using knees     | 10           |
| 3           | Falling backwards               | 10           |
| 4           | Falling sideward               | 10           |
| 5           | Falling sitting in empty chair  | 10           |
| 6           | Walking                        | 60           |
| 7           | Standing                       | 60           |
| 8           | Sitting                        | 60           |
| 9           | Picking up an object            | 10           |
| 10          | Jumping                        | 30           |
| 11          | Laying                         | 60           |

For more details, refer to the [UP-Fall Dataset paper](https://link.springer.com/article/10.1007/s11042-020-10348-5).

## Data Preprocessing
The `data_preprocessing.py` script handles all necessary steps to prepare the raw dataset for model training, incorporating techniques from both prior repositories.

### Loading Data
- **Sensor Data**: Loaded from `Imp_sensor.csv` into a Pandas DataFrame.
- **Camera Data**: Loaded from pre-saved `.npy` files (generated from raw zipped camera files) or processed using `load_raw.ipynb`, `load_background.ipynb`, and `load_clip.ipynb` for raw, foreground/background, and video frame datasets, respectively.

### Sensor Data Cleaning
- Drop rows with NaN values and duplicate records.
- Remove specific columns related to infrared sensors.
- Set the 'Time' column as the DataFrame index for synchronization.

### Image Data Processing
- Images are loaded as grayscale (`cv2.IMREAD_GRAYSCALE`) and resized to 32x32 resolution.
- Skip problematic image paths to avoid errors.
- Extract timestamps from image filenames for synchronization.
- Apply foreground/background extraction using MOD-NET model (from `load_background.ipynb`) for data augmentation.
- Create video frame datasets by combining images with ascending timestamps (from `load_clip.ipynb`).

### Data Synchronization
- Synchronize sensor and image data based on matching timestamps, retaining only data points with common timestamps for combined models.

### Label Handling
- Remap class 20 to 0 (dataset-specific adjustment).
- Convert labels to one-hot encoded format for Keras-based models (MLP, CNNs).
- Keep integer labels for scikit-learn and XGBoost/CatBoost models.

### Data Splitting
- Split dataset into training (60%), validation (20%), and test (20%) sets using `stratify=y_csv` to ensure proportional class representation.

### Data Scaling and Reshaping
- **CSV Data**: Scale sensor features using `StandardScaler`.
- **Image Data**: Normalize pixel values to [0, 1] by dividing by 255.0. Reshape images to `(height, width, 1)` for TensorFlow's `Conv2D` layers.
- **Data Augmentation**: Apply augmentation techniques (e.g., foreground/background extraction) to minority classes to address dataset imbalance.

## Model Architectures
This repository implements models from both prior repositories and introduces FedAvg and edge-based ML/DL models.

### Multi-Layer Perceptron (MLP)
- **Type**: Deep Learning (Feedforward Neural Network)
- **Input**: Scaled CSV sensor data
- **Architecture** (`models/mlp_model.py`):
  - Input Layer: `Dense(2000, activation='relu')`
  - Hidden Layer 1: `BatchNormalization()`
  - Hidden Layer 2: `Dense(600, activation='relu')`
  - Hidden Layer 3: `BatchNormalization()`
  - Regularization: `Dropout(0.2)`
  - Output Layer: `Dense(12, activation='softmax')`
- **Compilation**:
  - Optimizer: `Adam(learning_rate=0.001)`
  - Loss: `CategoricalCrossentropy`
  - Metrics: `CategoricalAccuracy`, `Precision`, `Recall`, `F1Score(average='weighted')`

### XGBoost
- **Type**: Gradient Boosting (Ensemble Learning)
- **Input**: Scaled CSV sensor data (integer labels)
- **Architecture** (`models/xgboost_model.py`):
  - `XGBClassifier` with `objective="multi:softprob"`
  - Parameters: `learning_rate=0.5`, `n_estimators=60`, `random_state=42`, `eval_metric="mlogloss"`
  - Early stopping: `early_stopping_rounds=5`

### CatBoost
- **Type**: Gradient Boosting (Ensemble Learning)
- **Input**: Scaled CSV sensor data (integer labels)
- **Architecture** (`models/catboost_model.py`):
  - `CatBoostClassifier`
  - Parameters: `n_estimators=500`, `random_seed=42`, `learning_rate=0.25`, `max_depth=12`, `loss_function='MultiClass'`
  - Early stopping: `early_stopping_rounds=10`

### Convolutional Neural Network (CNN) - Camera 1
- **Type**: Deep Learning (CNN)
- **Input**: Reshaped and normalized Camera 1 image data (32x32x1)
- **Architecture** (`models/cnn_camera1_model.py`):
  - Input Layer: `Input(shape=(32, 32, 1))`
  - Convolutional Block: `Conv2D(16, (3,3), activation='relu')`, `BatchNormalization()`, `MaxPooling2D((2,2))`
  - Flatten Layer: `Flatten()`
  - Dense Layers: `Dense(200, activation='relu')`, `Dropout(0.2)`
  - Output Layer: `Dense(12, activation='softmax')`
- **Compilation**: Same as MLP

### Convolutional Neural Network (CNN) - Camera 2
- **Type**: Deep Learning (CNN)
- **Input**: Reshaped and normalized Camera 2 image data (32x32x1)
- **Architecture** (`models/cnn_camera2_model.py`): Identical to Camera 1 CNN, but trained on Camera 2 data
- **Compilation**: Same as MLP

### Concatenated CNN (Camera 1 & 2)
- **Type**: Deep Learning (Multi-Input CNN)
- **Input**: Reshaped and normalized Camera 1 and Camera 2 images
- **Architecture** (`models/cnn_concatenate_model.py`):
  - **Branch 1 (Camera 1)**:
    - Input: `Input(shape=(32, 32, 1))`
    - Conv Block: `Conv2D(16, (3,3), activation='relu')`, `MaxPooling2D((2,2))`, `BatchNormalization()`
    - Flatten: `Flatten()`
  - **Branch 2 (Camera 2)**:
    - Input: `Input(shape=(32, 32, 1))`
    - Conv Block: `Conv2D(16, (3,3), activation='relu')`, `MaxPooling2D((2,2))`, `BatchNormalization()`
    - Flatten: `Flatten()`
  - Concatenation: `Concatenate(axis=1)` of flattened outputs
  - Dense Layers: `Dense(400, activation='relu')`, `Dense(200, activation='relu')`, `Dropout(0.2)`
  - Output Layer: `Dense(12, activation='softmax')`
- **Compilation**: Same as MLP

### Concatenated CSV + Images (Camera 1 & 2)
- **Type**: Deep Learning (Multi-Input Multimodal Model)
- **Input**: Scaled CSV sensor data, Camera 1 images, and Camera 2 images
- **Architecture** (`models/cnn_csv_img_concatenate_model.py`):
  - **Branch 1 (CSV Data)**:
    - Input: `Input(shape=(num_csv_features, 1))`
    - Conv Block: `Conv1D(10, kernel_size=3, activation='relu')`, `MaxPooling1D(pool_size=2)`, `BatchNormalization()`
    - Flatten: `Flatten()`
  - **Branch 2 (Camera 1)**:
    - Input: `Input(shape=(32, 32, 1))`
    - Conv Block: `Conv2D(16, (3,3), activation='relu')`, `MaxPooling2D((2,2))`, `BatchNormalization()`
    - Flatten: `Flatten()`
  - **Branch 3 (Camera 2)**:
    - Input: `Input(shape=(32, 32, 1))`
    - Conv Block: `Conv2D(16, (3,3), activation='relu')`, `MaxPooling2D((2,2))`, `BatchNormalization()`
    - Flatten: `Flatten()`
  - Concatenation: `Concatenate(axis=1)` of all flattened outputs
  - Dense Layers: `Dense(600, activation='relu')`, `Dense(1200, activation='relu')`, `Dropout(0.2)`
  - Output Layer: `Dense(12, activation='softmax')`
- **Compilation**: Same as MLP

### Random Forest
- **Type**: Ensemble Learning (Bagging)
- **Input**: Scaled CSV sensor data (integer labels)
- **Architecture** (`models/random_forest_model.py`):
  - `RandomForestClassifier`
  - Parameters: `n_estimators=10`, `min_samples_split=2`, `min_samples_leaf=1`, `bootstrap=True`, `random_state=42`

### Support Vector Machine (SVM)
- **Type**: Supervised Learning
- **Input**: Scaled CSV sensor data (integer labels)
- **Architecture** (`models/svm_model.py`):
  - `svm.SVC`
  - Parameters: `C=1`, `kernel='rbf'`, `gamma='auto'`, `shrinking=True`, `tol=0.001`, `random_state=42`

### k-Nearest Neighbors (k-NN)
- **Type**: Non-parametric, Lazy Learning
- **Input**: Scaled CSV sensor data (integer labels)
- **Architecture** (`models/knn_model.py`):
  - `KNeighborsClassifier`
  - Parameters: `n_neighbors=5`, `leaf_size=30`, `metric='euclidean'`

### Mixture of Experts (MoE) Models
- **Type**: Deep Learning (Ensemble CNN)
- **Input**: Camera 1 (C1), Camera 2 (C2), or combined (C12) image data
- **Architecture** (from `model/Raw_data.ipynb`, `Augment_data.ipynb`):
  - Variants: `MoE-1` (C1), `MoE-2` (C2), `C-MoE`, `M-MoE`, `A-MoE` (C12)
  - Includes data augmentation (DA) variants for improved generalization
  - Combines multiple CNN experts with a gating mechanism for enhanced performance
- **Performance**:
  - C1: `MoE-1 + DA` achieves 99.50% accuracy
  - C2: `MoE-2 + DA` achieves 99.61% accuracy
  - C12: `A-MoE + DA` achieves 99.67% accuracy

### 3D-CNN Models
- **Type**: Deep Learning (3D Convolutional Neural Network)
- **Input**: Video frame datasets from Camera 1, Camera 2, or combined (C12)
- **Architecture** (from `model/3D-data.ipynb`):
  - Variants: `3DCNN-1` (C1), `3DCNN-2` (C2), `C-3DCNN`, `M-3DCNN`, `A-3DCNN` (C12)
  - Processes temporal sequences of video frames for dynamic activity recognition
- **Performance**:
  - C1: `3DCNN-1` achieves 99.38% accuracy
  - C2: `3DCNN-2` achieves 99.41% accuracy
  - C12: `A-3DCNN` achieves 99.49% accuracy

### Federated Averaging (FedAvg)
- **Type**: Distributed Learning
- **Input**: Multimodal data (sensor, Camera 1, Camera 2) distributed across multiple clients
- **Architecture**:
  - Implements FedAvg to aggregate model updates from edge devices while preserving data privacy
  - Supports MLP, CNN, and MoE models for distributed training
- **Purpose**: Enables scalable fall detection across multiple devices (e.g., wearable sensors, edge cameras) without centralizing sensitive data

### Edge-Based ML/DL Models
- **Type**: Lightweight Machine Learning and Deep Learning
- **Input**: Multimodal data optimized for edge devices
- **Architecture**:
  - Optimized versions of MLP, CNN, and MoE models with reduced computational complexity
  - Techniques include model pruning, quantization, and lightweight architectures (e.g., MobileNet-inspired CNNs)
- **Purpose**: Facilitates real-time fall detection on resource-constrained devices like wearables or IoT cameras

## Evaluation Metrics
All models are evaluated using:
- **Accuracy Score**: Proportion of correctly classified instances.
- **Precision Score (weighted)**: Measures ability to avoid false positives, accounting for class imbalance.
- **Recall Score (weighted)**: Measures ability to find all positive samples, accounting for class imbalance.
- **F1 Score (weighted)**: Harmonic mean of precision and recall.
- **Balanced Accuracy Score**: Average of recall across all classes.
- **Confusion Matrix**: Visualizes true vs. predicted labels.

### Performance Summary
From [Fall-Detection-Research-1](https://github.com/hoangNguyen210/Fall-Detection-Research-1):
| Data | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|--------|----------|
| S    | XGBoost | 99.21 | 99.19 | 99.21 | 99.20 |
| S    | CatBoost | 99.05 | 99.02 | 99.05 | 99.02 |
| S    | MLP | 99.04 | 99.05 | 99.03 | 99.03 |
| C1   | CNN | 99.17 | 99.24 | 99.12 | 99.16 |
| C2   | CNN | 99.39 | 99.40 | 99.39 | 99.40 |
| C1+C2| Combination | 99.47 | 99.46 | 99.47 | 99.46 |
| C2   | Combination | 99.56 | 99.56 | 99.56 | 99.55 |

From [Fall-Detection-Research-2](https://github.com/hoangNguyen210/Fall-Detection-Research-2):
| Data | Model | Accuracy | Precision | Recall | F1-Score |
|------|-------|----------|-----------|--------|----------|
| C1   | MoE-1 + DA | 99.50 | 99.49 | 99.50 | 99.49 |
| C2   | MoE-2 + DA | 99.61 | 99.61 | 99.61 | 99.61 |
| C12  | A-MoE + DA | 99.67 | 99.67 | 99.67 | 99.67 |

## Project Structure