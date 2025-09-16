import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Find all client evaluation metric CSV files in the current directory
csv_files = glob.glob('client_*_eval_metrics.csv')

if not csv_files:
    print("No 'client_*_eval_metrics.csv' files found in this directory.")
else:
    # Create a figure with a 2x2 grid of subplots for better visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comparison of Evaluation Metrics Across All Clients', fontsize=18)

    # Flatten the axes array for easier iteration
    ax = axes.flatten()

    # Define the metrics to plot on each subplot
    metrics_to_plot = ['eval_loss', 'eval_acc', 'F1-score', 'Precision']
    y_labels = ['Loss', 'Accuracy', 'F1-Score', 'Score']

    # Loop through each metric to create a dedicated subplot
    for i, metric in enumerate(metrics_to_plot):
        # Loop through each CSV file and plot the metric on the current subplot
        for file in sorted(csv_files):
            try:
                # Extract client ID from filename for the legend label
                client_id_str = os.path.basename(file).split('_')[1]
                label = f"Client {client_id_str}"

                # Read the data and plot
                df = pd.read_csv(file)
                
                # Check if the dataframe has a 'round' column, if not, use index
                if 'round' in df.columns:
                    x_axis = df['round']
                else:
                    # If there's no 'round' column, we can create one from the index
                    df['round'] = range(1, len(df) + 1)
                    x_axis = df['round']

                ax[i].plot(x_axis, df[metric], label=label, marker='o', linestyle='--')

                # For the last subplot, also plot 'Recall'
                if metric == 'Precision':
                    ax[i].plot(x_axis, df['Recall'], label=f"Recall (Client {client_id_str})", marker='x', linestyle=':')

            except Exception as e:
                print(f"Could not process file {file}. Error: {e}")

        ax[i].set_title(f'{metric.replace("_", " ").title()} vs. Round')
        ax[i].set_xlabel("Round")
        ax[i].set_ylabel(y_labels[i])
        ax[i].grid(True)
        ax[i].legend()

    # Adjust layout to prevent titles and labels from overlapping
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    plt.savefig("all_clients_evaluation_metrics.png")
    print("Plot saved as all_clients_evaluation_metrics.png")

    # Show the plot
    plt.show()