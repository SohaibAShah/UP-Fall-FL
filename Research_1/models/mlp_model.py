# models/mlp_model.py

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import os

import config
from utils import set_seed, display_result, plot_training_history, save_results_to_csv

def build_mlp_model(input_shape):
    """
    Builds and compiles a Multi-Layer Perceptron (MLP) model for CSV data.

    Args:
        input_shape (int): The number of input features for the MLP.

    Returns:
        tf.keras.Model: The compiled MLP model.
    """
    set_seed(config.RANDOM_SEED)
    
    model = Sequential([
        Dense(2000, activation=tf.nn.relu, input_shape=(input_shape,)),
        BatchNormalization(),
        Dense(600, activation=tf.nn.relu),
        BatchNormalization(),
        Dropout(0.2),
        Dense(config.NUM_CLASSES, activation='softmax'),
    ])
    
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

def train_and_evaluate_mlp(X_train, Y_train, X_val, Y_val, X_test, Y_test, y_test_raw):
    """
    Trains, evaluates, saves the MLP model, and logs results.
    """
    print("\n--- Training Multilayer Perceptron (MLP) Model ---")
    model_mlp = build_mlp_model(X_train.shape[1])
    model_mlp.summary()

    # Create directory for saving models and plots
    os.makedirs(os.path.dirname(config.MLP_MODEL_PATH), exist_ok=True)
    results_dir = os.path.dirname(config.MLP_MODEL_PATH)
    results_csv_path = os.path.join(results_dir, 'model_results.csv')

    f1_callback_mlp = ModelCheckpoint(
        config.MLP_MODEL_PATH,
        monitor='val_f1_score',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    history_mlp = model_mlp.fit(
        X_train, Y_train,
        epochs=150,
        batch_size=2**10,
        validation_data=(X_val, Y_val),
        callbacks=[f1_callback_mlp]
    )
    
    # Save the training history plot
    plot_training_history(history_mlp, "MLP_csv", save_path=results_dir)

    print("\n--- MLP Model Evaluation ---")
    print("best model: ")
    model_mlp.load_weights(config.MLP_MODEL_PATH)

    print('Test Set Results:')
    y_pred_mlp_prob = model_mlp.predict(X_test)
    y_pred_mlp_labels = tf.argmax(y_pred_mlp_prob, axis=1).numpy()
    
    # Call display_result which now returns a dictionary
    test_results = display_result(y_test_raw, y_pred_mlp_labels)

    # Save the test results to CSV
    save_results_to_csv('MLP', test_results, results_csv_path)
    print("\n--- MLP Model Training and Evaluation Completed ---")  