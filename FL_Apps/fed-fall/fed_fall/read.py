import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_class_distribution(file_path):
    """Loads a .pt file and returns the percentage distribution of its labels."""
    try:
        # Load the tuple (features, labels) from the file
        _, y_data = torch.load(file_path, weights_only=False)

        # Count unique labels and their occurrences
        labels, counts = np.unique(y_data, return_counts=True)

        total_samples = np.sum(counts)
        distribution = {label: (count / total_samples) * 100 for label, count in zip(labels, counts)}

        # Ensure both classes 0 and 1 are present in the dictionary for consistency
        distribution.setdefault(0, 0.0)
        distribution.setdefault(1, 0.0)

        return distribution
    except Exception as e:
        print(f"Could not process {os.path.basename(file_path)}: {e}")
        return None

def main():
    """Main function to find .pt files, analyze them, and plot the results."""
    partitions_dir = '/home/syed/PhD/UP-Fall-FL/FL_Apps/fed-fall/fed_fall/UP_Fall_partitions'
    if not os.path.isdir(partitions_dir):
        print(f"Error: Directory '{partitions_dir}' not found. Make sure you are in the correct directory.")
        return

    # Find all client files and the test file
    file_paths = [os.path.join(partitions_dir, f) for f in os.listdir(partitions_dir) if f.startswith('client_') and f.endswith('.pt')]
    test_file = os.path.join(partitions_dir, 'test.pt')
    if os.path.exists(test_file):
        file_paths.append(test_file)

    file_paths.sort()

    results = {}
    for path in file_paths:
        file_name = os.path.basename(path).replace('.pt', '')
        dist = analyze_class_distribution(path)
        if dist:
            results[file_name] = dist

    if not results:
        print("No data to plot. Exiting.")
        return

    # --- Print Results to Console ---
    print("--- Class Distribution Analysis ---")
    print(f"{'File':<12} | {'% Fall (Class 0)':<20} | {'% Non-Fall (Class 1)':<20}")
    print("-" * 58)
    for name, dist in results.items():
        print(f"{name:<12} | {dist[0]:<20.2f} | {dist[1]:<20.2f}")

    # --- Plotting ---
    df = pd.DataFrame(results).T.rename(columns={0: '% Fall (Class 0)', 1: '% Non-Fall (Class 1)'})
    df = df.sort_index()

    ax = df.plot(kind='bar', figsize=(14, 7), rot=45, color=['#d9534f', '#5cb85c'])

    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='edge', padding=3)

    plt.title('Class Distribution Across All Clients and Test Set', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xlabel('Client / Dataset', fontsize=12)
    plt.ylim(0, 100)
    plt.legend(title='Activity Type')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot
    output_filename = 'class_distribution.png'
    plt.savefig(output_filename)
    print(f"\n📈 Plot saved as '{output_filename}'")

if __name__ == '__main__':
    main()