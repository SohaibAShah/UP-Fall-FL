# models/knn_model.py

from sklearn.neighbors import KNeighborsClassifier
import joblib
import os
import numpy as np

import config
from utils import set_seed, display_result

def build_knn_model():
    """
    Builds and returns a KNeighborsClassifier model.

    Returns:
        KNeighborsClassifier: The k-NN model.
    """
    set_seed(config.RANDOM_SEED)
    model = KNeighborsClassifier(
        n_neighbors=5,
        leaf_size=30,
        metric='euclidean'
    )
    return model

def train_and_evaluate_knn(X_train, y_train_raw, X_test, y_test_raw):
    """
    Trains, evaluates, and saves the k-NN model.

    Args:
        X_train (np.ndarray): Training features.
        y_train_raw (np.ndarray): Raw integer training labels.
        X_test (np.ndarray): Test features.
        y_test_raw (np.ndarray): Raw integer test labels.
    """
    print("\n--- Training k-Nearest Neighbors (k-NN) Model ---")
    model_knn = build_knn_model()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.XGB_MODEL_PATH), exist_ok=True) # Reusing XGB path for general models

    model_knn.fit(X_train, y_train_raw)

    print("\n--- k-NN Model Evaluation ---")
    print("---------------------Test Set--------------------------")
    y_pred_knn = model_knn.predict(X_test)
    display_result(y_test_raw, y_pred_knn)

    # Save the trained model
    knn_model_path = os.path.join(config.SAVED_MODELS_DIR, 'KNN_model.sav')
    joblib.dump(model_knn, knn_model_path)
    print(f"k-NN model saved to {knn_model_path}")

    # Load and test the saved model
    loaded_knn_model = joblib.load(knn_model_path)
    print("\n--- Loaded k-NN Model Evaluation ---")
    y_pred_loaded_knn = loaded_knn_model.predict(X_test)
    display_result(y_test_raw, y_pred_loaded_knn)