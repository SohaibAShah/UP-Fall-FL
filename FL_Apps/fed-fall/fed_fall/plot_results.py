import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
LOG_FILE_NAME = "/home/syed/PhD/UP-Fall-FL/FL_Apps/waooowooo/waooowooo/training_log.txt"  # The name of the log file you saved

def parse_log_file(log_path):
    """Parses a Flower log file to extract evaluation metrics for each round."""
    
    # Regex to find and capture the floating point values for each metric
    patterns = {
        "round": re.compile(r"\[ROUND (\d+)/\d+\]"),
        "eval_loss": re.compile(r"'eval_loss': ([\d.]+)"),
        "eval_acc": re.compile(r"'eval_acc': ([\d.]+)"),
        "f1_score": re.compile(r"'F1-score': ([\d.]+)"),
        "precision": re.compile(r"'Precision': ([\d.]+)"),
        "recall": re.compile(r"'Recall': ([\d.]+)")
    }
    
    metrics_data = []
    current_round = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Check for the start of a new round
            round_match = patterns["round"].search(line)
            if round_match:
                current_round = int(round_match.group(1))
            
            # Check for the line containing aggregated evaluation metrics
            if "aggregate_evaluate" in line and "Aggregated MetricRecord" in line:
                loss_match = patterns["eval_loss"].search(line)
                acc_match = patterns["eval_acc"].search(line)
                f1_match = patterns["f1_score"].search(line)
                precision_match = patterns["precision"].search(line)
                recall_match = patterns["recall"].search(line)
                
                if all([loss_match, acc_match, f1_match, precision_match, recall_match]):
                    metrics_data.append({
                        "Round": current_round,
                        "Loss": float(loss_match.group(1)),
                        "Accuracy": float(acc_match.group(1)),
                        "F1-Score": float(f1_match.group(1)),
                        "Precision": float(precision_match.group(1)),
                        "Recall": float(recall_match.group(1)),
                    })
                    
    if not metrics_data:
        print("Warning: No evaluation metrics found in the log file. Make sure the log is complete.")
        return None
        
    return pd.DataFrame(metrics_data)

def plot_metrics(df):
    """Creates and saves a 2x2 plot of the training metrics."""
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Federated Learning Performance Metrics per Round', fontsize=20)

    # Plot 1: F1-Score and Accuracy
    sns.lineplot(ax=axes[0, 0], x='Round', y='F1-Score', data=df, marker='o', label='F1-Score (Fall Class)')
    sns.lineplot(ax=axes[0, 0], x='Round', y='Accuracy', data=df, marker='o', label='Overall Accuracy')
    axes[0, 0].set_title('F1-Score & Accuracy')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)

    # Plot 2: Evaluation Loss
    sns.lineplot(ax=axes[0, 1], x='Round', y='Loss', data=df, marker='o', color='r')
    axes[0, 1].set_title('Evaluation Loss')
    axes[0, 1].set_ylabel('Loss')

    # Plot 3: Precision
    sns.lineplot(ax=axes[1, 0], x='Round', y='Precision', data=df, marker='o', color='purple')
    axes[1, 0].set_title('Precision (Fall Class)')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_ylim(0, 1)

    # Plot 4: Recall
    sns.lineplot(ax=axes[1, 1], x='Round', y='Recall', data=df, marker='o', color='orange')
    axes[1, 1].set_title('Recall (Fall Class)')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_ylim(0, 1)

    for ax in axes.flat:
        ax.set_xlabel('Round')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_filename = "fl_metrics_plot.png"
    plt.savefig(output_filename)
    print(f"✅ Plot saved successfully as '{output_filename}'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Ensure you have the required libraries installed:
    # pip install pandas matplotlib seaborn
    
    if not os.path.exists(LOG_FILE_NAME):
        print(f"Error: Log file '{LOG_FILE_NAME}' not found.")
        print("Please save your training log to this file and run the script again.")
    else:
        metrics_df = parse_log_file(LOG_FILE_NAME)
        if metrics_df is not None:
            plot_metrics(metrics_df)