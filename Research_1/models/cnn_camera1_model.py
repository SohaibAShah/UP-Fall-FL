# models/cnn_camera1_model.py

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import os

import config
from utils import set_seed, display_result, plot_training_history

def build_cnn_camera1_model(input_shape):
    """
    Builds and compiles a CNN model for Camera 1 image data.

    Args:
        input_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        tf.keras.Model: The compiled CNN model.
    """
    set_seed(config.RANDOM_SEED)

    input_layer = Input(shape=input_shape)
    
    conv = Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.relu)(input_layer)
    conv = BatchNormalization()(conv)
    conv = MaxPooling2D(pool_size=(2, 2))(conv)
    
    flatten = Flatten()(conv)
    
    fc = Dense(units=200, activation=tf.nn.relu)(flatten)
    dropout = Dropout(rate=0.2)(fc)
    
    softmax_output = Dense(units=config.NUM_CLASSES, activation=tf.nn.softmax)(dropout)
    
    model = Model(inputs=input_layer, outputs=softmax_output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, name='Adam'),
        loss='categorical_crossentropy',
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(name='f1_score', average='weighted'),
        ]
    )
    return model

def train_and_evaluate_cnn_camera1(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_test_raw):
    """
    Trains, evaluates, and saves the CNN model for Camera 1.

    Args:
        X_train (np.ndarray): Training features (reshaped images).
        Y_train (np.ndarray): One-hot encoded training labels.
        X_val (np.ndarray): Validation features (reshaped images).
        Y_val (np.ndarray): One-hot encoded validation labels.
        X_test (np.ndarray): Test features (reshaped images).
        Y_test (np.ndarray): One-hot encoded test labels.
        y_test_raw (np.ndarray): Raw integer test labels (for sklearn metrics).
    """
    print("\n--- Training CNN Model for Camera 1 Images ---")
    model_img1 = build_cnn_camera1_model(X_train.shape[1:])
    model_img1.summary()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.CNN_IMG1_MODEL_PATH), exist_ok=True)

    f1_callback1 = ModelCheckpoint(
        config.CNN_IMG1_MODEL_PATH,
        monitor='val_f1_score',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    history_img1 = model_img1.fit(
        X_train, Y_train,
        epochs=30,
        batch_size=2**10,
        validation_data=(X_val, Y_val),
        callbacks=[f1_callback1]
    )

    print("\n--- CNN Camera 1 Model Evaluation ---")
    print("best model: ")
    model_img1.load_weights(config.CNN_IMG1_MODEL_PATH)

    print('Validation Set')
    val_results = model_img1.evaluate(X_val, Y_val, verbose=0)
    print(f"Loss: {val_results[0]:.4f}, Accuracy: {val_results[1]:.4f}, "
          f"Precision: {val_results[2]:.4f}, Recall: {val_results[3]:.4f}, F1-Score: {val_results[4]:.4f}")

    print('Test Set')
    test_results = model_img1.evaluate(X_test, Y_test, verbose=0)
    print(f"Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}, "
          f"Precision: {test_results[2]:.4f}, Recall: {test_results[3]:.4f}, F1-Score: {test_results[4]:.4f}")

    y_pred_img1_prob = model_img1.predict(X_test)
    y_pred_img1_labels = tf.argmax(y_pred_img1_prob, axis=1).numpy()
    display_result(y_test_raw, y_pred_img1_labels)

    plot_training_history(history_img1, "CNN (Camera 1 Images)")