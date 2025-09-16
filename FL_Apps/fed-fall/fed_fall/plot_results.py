import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
LOG_FILE_NAME = "output.log"  # The name of the log file to parse

def parse_log_file(log_path: str) -> pd.DataFrame | None:
    """Parses a Flower log file to extract evaluation metrics for each round."""
    
    # Regex to find the round number and capture the floating point values for each metric
    round_pattern = re.compile(r"\[ROUND (\d+)/\d+\]")
    metrics_pattern = re.compile(
        r"'eval_loss': ([\d.e+-]+), 'eval_acc': ([\d.e+-]+), 'F1-score': ([\d.e+-]+), "
        r"'Precision': ([\d.e+-]+), 'Recall': ([\d.e+-]+)"
    )
    
    metrics_data = []
    current_round = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            # Check for the start of a new round
            round_match = round_pattern.search(line)
            if round_match:
                current_round = int(round_match.group(1))
            
            # Check for the line containing aggregated evaluation metrics
            if "Aggregated MetricRecord" in line:
                metrics_match = metrics_pattern.search(line)
                
                if metrics_match:
                    metrics_data.append({
                        "Round": current_round,
                        "Loss": float(metrics_match.group(1)),
                        "Accuracy": float(metrics_match.group(2)),
                        "F1-Score": float(metrics_match.group(3)),
                        "Precision": float(metrics_match.group(4)),
                        "Recall": float(metrics_match.group(5)),
                    })
                    
    if not metrics_data:
        print("Warning: No evaluation metrics found in the log file.")
        return None
        
    return pd.DataFrame(metrics_data)

def plot_metrics(df: pd.DataFrame):
    """Creates and saves a 2x2 plot of the training metrics."""
    
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Federated Learning Performance Metrics per Round', fontsize=20)

    # Flatten the axes array for easier iteration
    ax = axes.flatten()

    # --- Plot 1: F1-Score and Accuracy ---
    sns.lineplot(ax=ax[0], x='Round', y='F1-Score', data=df, marker='o', label='F1-Score (Fall Class)')
    sns.lineplot(ax=ax[0], x='Round', y='Accuracy', data=df, marker='o', label='Overall Accuracy')
    ax[0].set_title('F1-Score & Accuracy')
    ax[0].set_ylabel('Score')
    ax[0].set_ylim(0, max(1.0, df['Accuracy'].max() * 1.1)) # Adjust y-axis if accuracy > 1.0
    ax[0].legend()

    # --- Plot 2: Evaluation Loss ---
    sns.lineplot(ax=ax[1], x='Round', y='Loss', data=df, marker='o', color='r')
    ax[1].set_title('Evaluation Loss')
    ax[1].set_ylabel('Loss')

    # --- Plot 3: Precision ---
    sns.lineplot(ax=ax[2], x='Round', y='Precision', data=df, marker='o', color='purple')
    ax[2].set_title('Precision (Fall Class)')
    ax[2].set_ylabel('Precision')
    ax[2].set_ylim(0, 1)

    # --- Plot 4: Recall ---
    sns.lineplot(ax=ax[3], x='Round', y='Recall', data=df, marker='o', color='orange')
    ax[3].set_title('Recall (Fall Class)')
    ax[3].set_ylabel('Recall')
    ax[3].set_ylim(0, 1)

    # Set common labels and grid for all subplots
    for subplot in ax:
        subplot.set_xlabel('Round')
        subplot.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    output_filename = "training_metrics_plot.png"
    plt.savefig(output_filename)
    print(f"✅ Plot saved successfully as '{output_filename}'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE_NAME):
        print(f"Error: Log file '{LOG_FILE_NAME}' not found.")
        print("Please make sure the log file is in the same directory as this script.")
    else:
        metrics_df = parse_log_file(LOG_FILE_NAME)
        if metrics_df is not None and not metrics_df.empty:
            plot_metrics(metrics_df)