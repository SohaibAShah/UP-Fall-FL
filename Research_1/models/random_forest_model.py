# models/random_forest_model.py

from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import numpy as np

import config
from utils import set_seed, display_result

def build_random_forest_model():
    """
    Builds and returns a RandomForestClassifier model.

    Returns:
        RandomForestClassifier: The Random Forest model.
    """
    set_seed(config.RANDOM_SEED)
    model = RandomForestClassifier(
        n_estimators=10,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=config.RANDOM_SEED # Use the global random seed
    )
    return model

def train_and_evaluate_random_forest(X_train, y_train_raw, X_test, y_test_raw):
    """
    Trains, evaluates, and saves the Random Forest model.

    Args:
        X_train (np.ndarray): Training features.
        y_train_raw (np.ndarray): Raw integer training labels.
        X_test (np.ndarray): Test features.
        y_test_raw (np.ndarray): Raw integer test labels.
    """
    print("\n--- Training Random Forest Model ---")
    model_rf = build_random_forest_model()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.XGB_MODEL_PATH), exist_ok=True) # Reusing XGB path for general models

    model_rf.fit(X_train, y_train_raw)

    print("\n--- Random Forest Model Evaluation ---")
    print("---------------------Test Set--------------------------")
    y_pred_rf = model_rf.predict(X_test)
    display_result(y_test_raw, y_pred_rf)

    # Save the trained model
    rf_model_path = os.path.join(config.SAVED_MODELS_DIR, 'RandomForest_model.sav')
    joblib.dump(model_rf, rf_model_path)
    print(f"Random Forest model saved to {rf_model_path}")

    # Load and test the saved model
    loaded_rf_model = joblib.load(rf_model_path)
    print("\n--- Loaded Random Forest Model Evaluation ---")
    y_pred_loaded_rf = loaded_rf_model.predict(X_test)
    display_result(y_test_raw, y_pred_loaded_rf)