# models/cnn_concatenate_model.py

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint
import os

import config
from utils import set_seed, display_result, plot_training_history

def build_cnn_concatenate_model(input_shape_cam1, input_shape_cam2):
    """
    Builds and compiles a concatenated CNN model for two camera image streams.

    Args:
        input_shape_cam1 (tuple): The input shape for Camera 1 images (height, width, channels).
        input_shape_cam2 (tuple): The input shape for Camera 2 images (height, width, channels).

    Returns:
        tf.keras.Model: The compiled concatenated CNN model.
    """
    set_seed(config.RANDOM_SEED)

    # --- Camera 1 Branch ---
    input1 = Input(shape=input_shape_cam1, name='camera1_input')
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu)(input1)
    conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv1 = BatchNormalization()(conv1)
    flatten1 = Flatten()(conv1)

    # --- Camera 2 Branch ---
    input2 = Input(shape=input_shape_cam2, name='camera2_input')
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu)(input2)
    conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv2 = BatchNormalization()(conv2)
    flatten2 = Flatten()(conv2)

    # --- Concatenation and Dense Layers ---
    concat = Concatenate(axis=1)([flatten1, flatten2]) # Concatenate flattened outputs
    
    concat = Dense(units=400, activation=tf.nn.relu)(concat)
    concat = Dense(units=200, activation=tf.nn.relu)(concat)
    dropout = Dropout(0.2)(concat)
    
    softmax_output = Dense(config.NUM_CLASSES, activation=tf.nn.softmax)(dropout)
    
    model = Model(inputs=[input1, input2], outputs=softmax_output)
    
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

def train_and_evaluate_cnn_concatenate(X_train_1, X_train_2, Y_train,
                                       X_val_1, X_val_2, Y_val,
                                       X_test_1, X_test_2, Y_test, y_test_raw):
    """
    Trains, evaluates, and saves the concatenated CNN model.

    Args:
        X_train_1 (np.ndarray): Training features for Camera 1 (reshaped images).
        X_train_2 (np.ndarray): Training features for Camera 2 (reshaped images).
        Y_train (np.ndarray): One-hot encoded training labels.
        X_val_1 (np.ndarray): Validation features for Camera 1.
        X_val_2 (np.ndarray): Validation features for Camera 2.
        Y_val (np.ndarray): One-hot encoded validation labels.
        X_test_1 (np.ndarray): Test features for Camera 1.
        X_test_2 (np.ndarray): Test features for Camera 2.
        Y_test (np.ndarray): One-hot encoded test labels.
        y_test_raw (np.ndarray): Raw integer test labels (for sklearn metrics).
    """
    print("\n--- Training Concatenated CNN Model (Camera 1 & 2) ---")
    
    # Assuming images are 32x32 with 1 channel (grayscale)
    input_shape_cam = (config.DEFAULT_IMAGE_HEIGHT, config.DEFAULT_IMAGE_WIDTH, 1)
    
    model_img12 = build_cnn_concatenate_model(input_shape_cam, input_shape_cam)
    model_img12.summary()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.SAVED_MODELS_DIR + '/model_img12.keras'), exist_ok=True) # Ensure path is correct

    f1_callback_img12 = ModelCheckpoint(
        os.path.join(config.SAVED_MODELS_DIR, 'model_img12.keras'), # Use os.path.join for path
        monitor='val_f1_score',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    history_img12 = model_img12.fit(
        x=[X_train_1, X_train_2], # Pass inputs as a list for multi-input model
        y=Y_train,
        epochs=30,
        batch_size=2**10,
        validation_data=([X_val_1, X_val_2], Y_val), # Pass validation data as a tuple of lists
        callbacks=[f1_callback_img12]
    )

    print("\n--- Concatenated CNN Model Evaluation ---")
    print("best model: ")
    # Load the best model weights
    model_img12.load_weights(os.path.join(config.SAVED_MODELS_DIR, 'model_img12.keras'))

    print('Validation Set')
    val_results = model_img12.evaluate([X_val_1, X_val_2], Y_val, verbose=0)
    print(f"Loss: {val_results[0]:.4f}, Accuracy: {val_results[1]:.4f}, "
          f"Precision: {val_results[2]:.4f}, Recall: {val_results[3]:.4f}, F1-Score: {val_results[4]:.4f}")

    print('Test Set')
    test_results = model_img12.evaluate([X_test_1, X_test_2], Y_test, verbose=0)
    print(f"Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}, "
          f"Precision: {test_results[2]:.4f}, Recall: {test_results[3]:.4f}, F1-Score: {test_results[4]:.4f}")

    y_pred_img12_prob = model_img12.predict([X_test_1, X_test_2])
    y_pred_img12_labels = tf.argmax(y_pred_img12_prob, axis=1).numpy()
    display_result(y_test_raw, y_pred_img12_labels)

    plot_training_history(history_img12, "Concatenated CNN (Camera 1 & 2 Images)")