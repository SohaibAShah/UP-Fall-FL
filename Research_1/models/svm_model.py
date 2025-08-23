# models/svm_model.py

from sklearn import svm
import joblib
import os
import numpy as np

import config
from utils import set_seed, display_result

def build_svm_model():
    """
    Builds and returns an SVM Classifier model.

    Returns:
        svm.SVC: The SVM model.
    """
    set_seed(config.RANDOM_SEED)
    model = svm.SVC(
        C=1,
        kernel='rbf',
        gamma='auto',
        shrinking=True,
        tol=0.001,
        random_state=config.RANDOM_SEED # Use the global random seed
    )
    return model

def train_and_evaluate_svm(X_train, y_train_raw, X_test, y_test_raw):
    """
    Trains, evaluates, and saves the SVM model.

    Args:
        X_train (np.ndarray): Training features.
        y_train_raw (np.ndarray): Raw integer training labels.
        X_test (np.ndarray): Test features.
        y_test_raw (np.ndarray): Raw integer test labels.
    """
    print("\n--- Training SVM Model ---")
    model_svm = build_svm_model()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.XGB_MODEL_PATH), exist_ok=True) # Reusing XGB path for general models

    model_svm.fit(X_train, y_train_raw)

    print("\n--- SVM Model Evaluation ---")
    print("---------------------Test Set--------------------------")
    y_pred_svm = model_svm.predict(X_test)
    display_result(y_test_raw, y_pred_svm)

    # Save the trained model
    svm_model_path = os.path.join(config.SAVED_MODELS_DIR, 'SVM_model.sav')
    joblib.dump(model_svm, svm_model_path)
    print(f"SVM model saved to {svm_model_path}")

    # Load and test the saved model
    loaded_svm_model = joblib.load(svm_model_path)
    print("\n--- Loaded SVM Model Evaluation ---")
    y_pred_loaded_svm = loaded_svm_model.predict(X_test)
    display_result(y_test_raw, y_pred_loaded_svm)