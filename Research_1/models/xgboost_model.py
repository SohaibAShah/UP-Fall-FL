# models/xgboost_model.py

import xgboost
from xgboost import XGBClassifier
import numpy as np
import joblib
import os

import config
from utils import set_seed, display_result

def build_xgboost_model():
    """
    Builds and returns an XGBoost Classifier model.

    Returns:
        XGBClassifier: The XGBoost model.
    """
    set_seed(config.RANDOM_SEED)
    model_xgb = XGBClassifier(
        objective="multi:softprob",  # Multi-class classification with softmax probabilities
        learning_rate=0.5,
        random_state=config.RANDOM_SEED,
        n_estimators=60,
        eval_metric="mlogloss",
    )
    return model_xgb

def train_and_evaluate_xgboost(X_train, y_train_raw, X_val, y_val_raw, X_test, y_test_raw):
    """
    Trains, evaluates, and saves the XGBoost model.

    Args:
        X_train (np.ndarray): Training features.
        y_train_raw (np.ndarray): Raw integer training labels.
        X_val (np.ndarray): Validation features.
        y_val_raw (np.ndarray): Raw integer validation labels.
        X_test (np.ndarray): Test features.
        y_test_raw (np.ndarray): Raw integer test labels.
    """
    print("\n--- Training XGBoost Model ---")
    model_xgb = build_xgboost_model()

    # Create directory for saving models if it doesn't exist
    os.makedirs(os.path.dirname(config.XGB_MODEL_PATH), exist_ok=True)

    model_xgb.fit(
        X=X_train,
        y=y_train_raw,
        eval_set=[(X_train, y_train_raw), (X_val, y_val_raw)],
        verbose=1
    )

    print("\n--- XGBoost Model Evaluation ---")
    print("---------------------Test Set--------------------------")
    y_pred_xgb = model_xgb.predict(X_test)
    display_result(y_test_raw, y_pred_xgb)

    # Save the trained model
    joblib.dump(model_xgb, config.XGB_MODEL_PATH)
    print(f"XGBoost model saved to {config.XGB_MODEL_PATH}")

    # Load and test the saved model
    loaded_xgb_model = joblib.load(config.XGB_MODEL_PATH)
    print("\n--- Loaded XGBoost Model Evaluation ---")
    y_pred_loaded_xgb = loaded_xgb_model.predict(X_test)
    display_result(y_test_raw, y_pred_loaded_xgb)