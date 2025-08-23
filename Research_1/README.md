# Fall Detection using Multimodal Sensor and Image Data

This repository contains the codebase for a PhD research project focused on developing robust fall detection systems using multimodal data, specifically combining sensor readings and camera imagery. The project explores various machine learning and deep learning models to classify human activities, with a particular emphasis on identifying fall events.

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
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Requirements](#requirements)
- [Future Work](#future-work)

## Introduction
Falls are a significant health concern, particularly among the elderly, leading to injuries, reduced quality of life, and increased healthcare costs. This research project aims to develop and evaluate various machine learning and deep learning models for accurate and timely fall detection. By leveraging multimodal data from both wearable sensors and ambient cameras, the project seeks to build more robust and reliable detection systems compared to unimodal approaches.

The codebase is structured to allow for easy experimentation with different models and data modalities, providing a clear pipeline from raw data to trained model evaluation.

## Dataset
The project utilizes the [UP-Fall Dataset](https://sites.google.com/up.edu.mx/har-up/), a publicly available dataset specifically designed for fall detection research. It is a multimodal dataset containing:

- **Sensor Data (Imp_sensor.csv)**: Time-series data from multiple wearable sensors (e.g., accelerometers, gyroscopes, luminosity, infrared, brainwave signals). Each row represents a timestamped reading.
- **Camera Data (zipped image files)**: Video frames captured from two different camera perspectives (Camera 1 and Camera 2) during various activities, including fall events. These are provided as sequences of images within zip archives.

The dataset captures a variety of activities, including different types of falls and Activities of Daily Living (ADLs), enabling the training and evaluation of models on diverse human movements.

## Data Preprocessing
The `data_preprocessing.py` script handles all necessary steps to prepare the raw dataset for model training.

### Loading Data
- **Sensor Data**: Loaded from `Imp_sensor.csv` into a Pandas DataFrame.
- **Camera Data**: Loaded from pre-saved `.npy` files (generated from raw zipped camera files), containing images and associated timestamps/labels.

### Sensor Data Cleaning
- Drop rows with NaN values.
- Remove duplicate rows to ensure unique samples.
- Drop specific columns related to infrared sensors.
- Set the 'Time' column as the DataFrame index for easier lookup and synchronization.

### Image Data Processing
- Images are loaded as grayscale (`cv2.IMREAD_GRAYSCALE`).
- Images are resized to a default 32x32 resolution to standardize input dimensions for CNNs.
- Specific problematic image paths are skipped to avoid errors (e.g., NO SHAPE or Invalid image errors).
- Timestamps are extracted from image filenames and formatted for synchronization.

### Data Synchronization
- Synchronize sensor and image data based on matching timestamps across modalities. Only data points with common timestamps are retained for combined models.

### Label Handling
- Extract raw labels and remap class 20 to 0 (dataset-specific adjustment).
- Convert labels to one-hot encoded format for Keras-based models (MLP, CNNs).
- Keep integer labels for scikit-learn and XGBoost/CatBoost models.

### Data Splitting
- Split dataset into training (60%), validation (20%), and test (20%) sets.
- Use `stratify=y_csv` (or `stratify=labels`) to ensure proportional representation of all activity classes in each split.

### Data Scaling and Reshaping
- **CSV Data**: Scale sensor features using `StandardScaler` to normalize their range.
- **Image Data**: Normalize pixel values to [0, 1] by dividing by 255.0. Reshape images to `(height, width, 1)` for TensorFlow's `Conv2D` layers.

## Model Architectures

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

## Evaluation Metrics
All models are evaluated using the following classification metrics:
- **Accuracy Score**: Proportion of correctly classified instances.
- **Precision Score (weighted)**: Measures the classifier's ability to avoid false positives, accounting for class imbalance.
- **Recall Score (weighted)**: Measures the classifier's ability to find all positive samples, accounting for class imbalance.
- **F1 Score (weighted)**: Harmonic mean of precision and recall, suitable for imbalanced datasets.
- **Balanced Accuracy Score**: Average of recall across all classes, useful for imbalanced datasets.
- **Confusion Matrix**: Visualizes true vs. predicted labels for each class.

## Project Structure
```
TEST_UP_FALL/
├── UP-Fall Dataset/
│   ├── Imp_sensor.csv
│   ├── image_1.npy
│   ├── label_1.npy
│   ├── name_1.npy
│   ├── image_2.npy
│   ├── label_2.npy
│   └── name_2.npy
├── Saved Model/
│   └── Experiments/  # Trained models will be saved here
├── config.py           # Global configurations and constants
├── data_preprocessing.py # Scripts for loading, cleaning, and preparing data
├── utils.py            # Utility functions (e.g., set_seed, display_result, plotting)
├── main_training.py    # Main script to run model training and evaluation
└── models/             # Directory containing individual model definitions
    ├── __init__.py
    ├── mlp_model.py
    ├── xgboost_model.py
    ├── catboost_model.py
    ├── cnn_camera1_model.py
    ├── cnn_camera2_model.py
    ├── cnn_concatenate_model.py
    ├── cnn_csv_img_concatenate_model.py
    ├── random_forest_model.py
    ├── svm_model.py
    └── knn_model.py
```

## How to Run the Project
1. Navigate to the project root directory (`your_project_root/`).
2. Execute the `main_training.py` script to run the training and evaluation pipeline.

### Run All Models
```bash
python main_training.py
```

### Run a Specific Model (e.g., MLP)
```bash
python main_training.py --model mlp
```

### Run Multiple Specific Models (e.g., XGBoost and Concatenated CNN)
```bash
python main_training.py --model xgboost cnn_concat
```

### Available Model Choices
- `all` (runs all models)
- `mlp`
- `xgboost`
- `catboost`
- `cnn_cam1`
- `cnn_cam2`
- `cnn_concat`
- `cnn_csv_img_concat`
- `random_forest`
- `svm`
- `knn`

## Requirements
Ensure you have Python 3.8+ installed. Install the required libraries using:
```bash
pip install tensorflow tensorflow-addons scikit-learn pandas numpy opencv-python matplotlib xgboost catboost joblib
```

**Note**: Ensure the `tensorflow-addons` version is compatible with your `tensorflow` version. Refer to the [TensorFlow Addons GitHub page](https://github.com/tensorflow/addons) for the compatibility matrix if you encounter import errors.

## Future Work
- **Hyperparameter Optimization**: Implement systematic hyperparameter tuning (e.g., Grid Search, Random Search, Bayesian Optimization) for all models.
- **Cross-Validation**: Integrate k-fold cross-validation for more robust model evaluation.
- **Advanced Architectures**: Explore LSTMs, Transformers for sensor data, and advanced CNNs (e.g., ResNet, Inception) for image data.
- **Real-time Inference**: Develop a prototype for real-time fall detection.
- **Data Augmentation**: Apply augmentation techniques to image data to improve model generalization.
- **Interpretability**: Investigate methods to interpret model decisions, especially for deep learning models.