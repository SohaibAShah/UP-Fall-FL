# main_training.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse # Import the argparse module

# Import modules
import config
import data_preprocessing
from utils import set_seed, display_result, plot_training_history

# Import model training functions
from models.mlp_model import train_and_evaluate_mlp
from models.xgboost_model import train_and_evaluate_xgboost
from models.catboost_model import train_and_evaluate_catboost
from models.cnn_camera1_model import train_and_evaluate_cnn_camera1
from models.cnn_camera2_model import train_and_evaluate_cnn_camera2
from models.cnn_concatenate_model import train_and_evaluate_cnn_concatenate
from models.cnn_csv_img_concatenate_model import train_and_evaluate_csv_img_concatenate
from models.random_forest_model import train_and_evaluate_random_forest
from models.svm_model import train_and_evaluate_svm 
from models.knn_model import train_and_evaluate_knn  

def main():
    """
    Main function to orchestrate data loading, preprocessing,
    and training/evaluation of selected models based on arguments.
    """
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Train and evaluate machine learning models for fall detection."
    )

    # Add arguments
    parser.add_argument(
        '--model',
        type=str,
        nargs='+', # Allows multiple model choices, e.g., --model mlp xgb
        default=['all'], # Default to 'all' if no model is specified
        choices=['all', 'mlp', 'xgboost', 'catboost', 'cnn_cam1', 'cnn_cam2', 'cnn_concat', 'cnn_csv_img_concat', 'random_forest', 'svm', 'knn'],
        help="Specify which model(s) to train and evaluate. Choose from 'mlp', 'xgboost', 'catboost', 'cnn_cam1', 'cnn_cam2', 'cnn_concat', 'cnn_csv_img_concat', 'random_forest', 'svm', 'knn', or 'all' for all models."
    )

    # Parse arguments
    args = parser.parse_args()
    models_to_run = args.model

    # Ensure Saved Model/Experiments directory exists
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)

    # --- Data Preprocessing ---
    print("--- Starting Data Preprocessing ---")
    processed_data = data_preprocessing.load_and_preprocess_data()
    print("--- Data Preprocessing Complete ---\n")

    # Extract data for different models
    # CSV Data
    X_train_csv_scaled = processed_data['X_train_csv_scaled']
    Y_train_csv = processed_data['Y_train_csv']
    X_val_csv_scaled = processed_data['X_val_csv_scaled']
    Y_val_csv = processed_data['Y_val_csv']
    X_test_csv_scaled = processed_data['X_test_csv_scaled']
    Y_test_csv = processed_data['Y_test_csv']
    y_train_csv_raw = processed_data['Y_train_csv'] # Raw labels for XGBoost/CatBoost
    y_val_csv_raw = processed_data['Y_val_csv']     # Raw labels for XGBoost/CatBoost
    y_test_csv_raw = processed_data['Y_test_csv']   # Raw labels for XGBoost/CatBoost

    # Camera 1 Image Data
    X_train_1_scaled = processed_data['X_train_1_scaled']
    Y_train_1 = processed_data['Y_train_1']
    X_val_1_scaled = processed_data['X_val_1_scaled']
    Y_val_1 = processed_data['Y_val_1']
    X_test_1_scaled = processed_data['X_test_1_scaled']
    Y_test_1 = processed_data['Y_test_1']
    y_test_1_raw = processed_data['y_test_1']

    # Camera 2 Image Data
    X_train_2_scaled = processed_data['X_train_2_scaled']
    Y_train_2 = processed_data['Y_train_2']
    X_val_2_scaled = processed_data['X_val_2_scaled']
    Y_val_2 = processed_data['Y_val_2']
    X_test_2_scaled = processed_data['X_test_2_scaled']
    Y_test_2 = processed_data['Y_test_2']
    y_test_2_raw = processed_data['y_test_2']

    # --- Train and Evaluate Selected Models ---
    if 'all' in models_to_run or 'mlp' in models_to_run:
        train_and_evaluate_mlp(
            X_train_csv_scaled, Y_train_csv,
            X_val_csv_scaled, Y_val_csv,
            X_test_csv_scaled, Y_test_csv, y_test_csv_raw
        )

    if 'all' in models_to_run or 'xgboost' in models_to_run:
        train_and_evaluate_xgboost(
            X_train_csv_scaled, y_train_csv_raw,
            X_val_csv_scaled, y_val_csv_raw,
            X_test_csv_scaled, y_test_csv_raw
        )

    if 'all' in models_to_run or 'catboost' in models_to_run:
        train_and_evaluate_catboost(
            X_train_csv_scaled, y_train_csv_raw,
            X_val_csv_scaled, y_val_csv_raw,
            X_test_csv_scaled, y_test_csv_raw
        )

    if 'all' in models_to_run or 'cnn_cam1' in models_to_run:
        train_and_evaluate_cnn_camera1(
            X_train_1_scaled, Y_train_1,
            X_val_1_scaled, Y_val_1,
            X_test_1_scaled, Y_test_1, y_test_1_raw
        )

    if 'all' in models_to_run or 'cnn_cam2' in models_to_run:
        train_and_evaluate_cnn_camera2(
            X_train_2_scaled, Y_train_2,
            X_val_2_scaled, Y_val_2,
            X_test_2_scaled, Y_test_2, y_test_2_raw
        )

    if 'all' in models_to_run or 'cnn_concat' in models_to_run:
        train_and_evaluate_cnn_concatenate(
            X_train_1_scaled, X_train_2_scaled, Y_train_1, # Y_train_1 is used as common labels
            X_val_1_scaled, X_val_2_scaled, Y_val_1,
            X_test_1_scaled, X_test_2_scaled, Y_test_1, y_test_1_raw
        )

    if 'all' in models_to_run or 'cnn_csv_img_concat' in models_to_run:
        train_and_evaluate_csv_img_concatenate(
            X_train_csv_scaled, X_train_1_scaled, X_train_2_scaled, Y_train_csv,
            X_val_csv_scaled, X_val_1_scaled, X_val_2_scaled, Y_val_csv,
            X_test_csv_scaled, X_test_1_scaled, X_test_2_scaled, Y_test_csv, y_test_csv_raw
        )
    if 'all' in models_to_run or 'random_forest' in models_to_run:
        train_and_evaluate_random_forest(
            X_train_csv_scaled, y_train_csv_raw,
            X_val_csv_scaled, y_val_csv_raw,
            X_test_csv_scaled, y_test_csv_raw
        )
    if 'all' in models_to_run or 'svm' in models_to_run:
        train_and_evaluate_svm(
            X_train_csv_scaled, y_train_csv_raw,
            X_val_csv_scaled, y_val_csv_raw,
            X_test_csv_scaled, y_test_csv_raw
        )
    if 'all' in models_to_run or 'knn' in models_to_run:
        train_and_evaluate_knn(
            X_train_csv_scaled, y_train_csv_raw,
            X_test_csv_scaled, y_test_csv_raw
        )
    # --- End of Model Training and Evaluation ---

    print("\nSelected model training and evaluation complete!")

if __name__ == "__main__":
    main()