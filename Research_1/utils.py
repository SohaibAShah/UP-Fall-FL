# utils.py

import tensorflow as tf
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import os
import pandas as pd # Import pandas for CSV handling

import config

def set_seed(seed=config.RANDOM_SEED):
    """
    Sets random seeds for reproducibility across TensorFlow, NumPy, and Python's random module.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

def display_result(y_true, y_pred_labels):
    """
    Calculates and returns a dictionary of classification metrics.

    Args:
        y_true (array-like): True labels (can be 1D integer or 2D one-hot encoded).
        y_pred_labels (array-like): Predicted labels (expected to be 1D integer).

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    # Ensure y_true is a 1D numpy array of integer labels
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_processed = np.argmax(y_true, axis=1)
    else:
        y_true_processed = np.asarray(y_true).ravel()

    # Ensure y_pred_labels is a 1D numpy array of integer labels
    y_pred_labels_processed = np.asarray(y_pred_labels).ravel()

    results = {
        'accuracy_score': accuracy_score(y_true_processed, y_pred_labels_processed),
        'precision_score': precision_score(y_true_processed, y_pred_labels_processed, average='weighted', zero_division=0),
        'recall_score': recall_score(y_true_processed, y_pred_labels_processed, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true_processed, y_pred_labels_processed, average='weighted', zero_division=0),
        'balanced_accuracy_score': balanced_accuracy_score(y_true_processed, y_pred_labels_processed),
        'confusion_matrix': confusion_matrix(y_true_processed, y_pred_labels_processed).tolist()
    }

    # Print the results to the console
    print(f"Accuracy score : {results['accuracy_score']:.4f}")
    print(f"Precision score: {results['precision_score']:.4f}")
    print(f"Recall score   : {results['recall_score']:.4f}")
    print(f"F1 score       : {results['f1_score']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy_score']:.4f}")
    print('Confusion Matrix:\n', results['confusion_matrix'])
    
    return results

def plot_training_history(history, model_name, save_path=None):
    """
    Plots and optionally saves training history (accuracy, loss, precision, recall, f1_score).

    Args:
        history (keras.callbacks.History): History object returned by model.fit().
        model_name (str): Name of the model for plot titles and filenames.
        save_path (str, optional): Directory path to save the plots. If None, plots are only shown.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'Training and Validation Metrics for {model_name}', fontsize=16)

    # List of metrics to plot
    metrics = {
        'categorical_accuracy': 'Accuracy',
        'loss': 'Loss',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1_score': 'F1 Score'
    }

    # Plot each metric
    for i, (metric_key, metric_title) in enumerate(metrics.items()):
        ax = axes.flatten()[i]
        if metric_key in history.history:
            ax.plot(history.history[metric_key], label=f'Train {metric_title}')
            if f'val_{metric_key}' in history.history:
                ax.plot(history.history[f'val_{metric_key}'], label=f'Validation {metric_title}')
            ax.set_title(metric_title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_title)
            ax.legend()
        else:
            ax.set_title(f"{metric_title} (Not Logged)")
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plot_filename = os.path.join(save_path, f'{model_name}_training_history.png')
        plt.savefig(plot_filename)
        print(f"Training plots saved to {plot_filename}")
        plt.close() # Close the plot to prevent it from displaying
    else:
        plt.show()

def save_results_to_csv(model_name, results_dict, save_path):
    """
    Saves a dictionary of model evaluation results to a CSV file.

    Args:
        model_name (str): The name of the model.
        results_dict (dict): A dictionary of evaluation metrics.
        save_path (str): The file path to save the CSV.
    """
    # Create a DataFrame from the results dictionary
    df = pd.DataFrame([results_dict], index=[model_name])
    
    # Clean up the confusion matrix to be a single string for CSV compatibility
    if 'confusion_matrix' in df.columns:
        df['confusion_matrix'] = df['confusion_matrix'].apply(lambda x: str(x))

    # Check if the file exists
    if os.path.exists(save_path):
        # If it exists, read the existing data, append the new row, and save
        existing_df = pd.read_csv(save_path, index_col=0)
        updated_df = pd.concat([existing_df, df])
        updated_df.to_csv(save_path)
    else:
        # If not, save the new DataFrame directly
        df.to_csv(save_path)
    
    print(f"Results for {model_name} saved to {save_path}")