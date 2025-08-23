# models/cnn_csv_img_concatenate_model.py

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import numpy as np

import config
from utils import set_seed, display_result, plot_training_history

def build_csv_img_concatenate_model(num_csv_features, img_shape1, img_shape2):
    """
    Builds and compiles a concatenated model for CSV sensor data and two camera image streams.

    Args:
        num_csv_features (int): The number of features in the CSV data.
        img_shape1 (int): Height of the input images.
        img_shape2 (int): Width of the input images.

    Returns:
        tf.keras.Model: The compiled concatenated model.
    """
    set_seed(config.RANDOM_SEED)

    # --- CSV Data Branch (1D CNN) ---
    # Input for CSV data: (num_features, 1) as Conv1D expects 3D input (batch, steps, features)
    # Here, each feature is a 'step' and the '1' is a dummy feature dimension for Conv1D
    inputs1 = Input(shape=(num_csv_features, 1), name='csv_input')
    conv1 = Conv1D(filters=10, kernel_size=3, activation='relu')(inputs1)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    batch1 = BatchNormalization()(pool1)
    flat1 = Flatten()(batch1)

    # --- Camera 1 Image Branch (2D CNN) ---
    inputs2 = Input(shape=(img_shape1, img_shape2, 1), name='camera1_input')
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu)(inputs2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization()(pool2)
    flat2 = Flatten()(batch2)

    # --- Camera 2 Image Branch (2D CNN) ---
    inputs3 = Input(shape=(img_shape1, img_shape2, 1), name='camera2_input')
    conv3 = Conv2D(16, (3, 3), activation='relu')(inputs3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    batch3 = BatchNormalization()(pool3)
    flat3 = Flatten()(batch3)

    # --- Concatenate all branches ---
    merged = Concatenate(axis=1)([flat1, flat2, flat3])

    # --- Dense Layers for combined features ---
    dense1 = Dense(units=600, activation='relu')(merged) # Corrected from 400 to 600 as per your code
    dense2 = Dense(units=1200, activation='relu')(dense1) # Corrected from 200 to 1200 as per your code
    dropout = Dropout(0.2)(dense2)
    
    outputs = Dense(config.NUM_CLASSES, activation='softmax')(dropout)
    
    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(name='categorical_crossentropy'),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(name='f1_score', average='weighted')
        ]
    )
    return model

def train_and_evaluate_csv_img_concatenate(X_train_csv, X_train_1, X_train_2, Y_train,
                                           X_val_csv, X_val_1, X_val_2, Y_val,
                                           X_test_csv, X_test_1, X_test_2, Y_test, y_test_raw):
    """
    Trains, evaluates, and saves the concatenated model for CSV and image data.

    Args:
        X_train_csv (np.ndarray): Training features for CSV data.
        X_train_1 (np.ndarray): Training features for Camera 1 images.
        X_train_2 (np.ndarray): Training features for Camera 2 images.
        Y_train (np.ndarray): One-hot encoded training labels (common for all inputs).
        X_val_csv (np.ndarray): Validation features for CSV data.
        X_val_1 (np.ndarray): Validation features for Camera 1 images.
        X_val_2 (np.ndarray): Validation features for Camera 2 images.
        Y_val (np.ndarray): One-hot encoded validation labels.
        X_test_csv (np.ndarray): Test features for CSV data.
        X_test_1 (np.ndarray): Test features for Camera 1 images.
        X_test_2 (np.ndarray): Test features for Camera 2 images.
        Y_test (np.ndarray): One-hot encoded test labels.
        y_test_raw (np.ndarray): Raw integer test labels (for sklearn metrics).
    """
    print("\n--- Training Concatenated CSV and Image Model ---")
    
    # Reshape CSV data for 1D CNN input: (samples, features, 1)
    X_train_csv_reshaped = X_train_csv.reshape(X_train_csv.shape[0], X_train_csv.shape[1], 1)
    X_val_csv_reshaped = X_val_csv.reshape(X_val_csv.shape[0], X_val_csv.shape[1], 1)
    X_test_csv_reshaped = X_test_csv.reshape(X_test_csv.shape[0], X_test_csv.shape[1], 1)

    # Determine input shapes for the model
    num_csv_features = X_train_csv_reshaped.shape[1]
    img_height, img_width = X_train_1.shape[1], X_train_1.shape[2] # Assuming images are consistent

    model_concat_all = build_csv_img_concatenate_model(num_csv_features, img_height, img_width)
    model_concat_all.summary()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.SAVED_MODELS_DIR + '/model_concatenate_all.keras'), exist_ok=True)

    f1_callback_concat_all = ModelCheckpoint(
        os.path.join(config.SAVED_MODELS_DIR, 'model_concatenate_all.keras'),
        monitor='val_f1_score',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    history_concat_all = model_concat_all.fit(
        x=[X_train_csv_reshaped, X_train_1, X_train_2], # Pass all inputs as a list
        y=Y_train,
        epochs=30, # As per your example
        batch_size=2**10,
        validation_data=([X_val_csv_reshaped, X_val_1, X_val_2], Y_val),
        callbacks=[f1_callback_concat_all]
    )

    print("\n--- Concatenated CSV and Image Model Evaluation ---")
    print("best model: ")
    model_concat_all.load_weights(os.path.join(config.SAVED_MODELS_DIR, 'model_concatenate_all.keras'))

    print('Validation Set')
    val_results = model_concat_all.evaluate([X_val_csv_reshaped, X_val_1, X_val_2], Y_val, verbose=0)
    print(f"Loss: {val_results[0]:.4f}, Accuracy: {val_results[1]:.4f}, "
          f"Precision: {val_results[2]:.4f}, Recall: {val_results[3]:.4f}, F1-Score: {val_results[4]:.4f}")

    print('Test Set')
    test_results = model_concat_all.evaluate([X_test_csv_reshaped, X_test_1, X_test_2], Y_test, verbose=0)
    print(f"Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}, "
          f"Precision: {test_results[2]:.4f}, Recall: {test_results[3]:.4f}, F1-Score: {test_results[4]:.4f}")

    y_pred_concat_all_prob = model_concat_all.predict([X_test_csv_reshaped, X_test_1, X_test_2])
    y_pred_concat_all_labels = tf.argmax(y_pred_concat_all_prob, axis=1).numpy()
    display_result(y_test_raw, y_pred_concat_all_labels)

    plot_training_history(history_concat_all, "Concatenated (CSV + Camera 1 & 2)")