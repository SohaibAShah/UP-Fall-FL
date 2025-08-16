import numpy as np
import pandas as pd
from pathlib import Path

# --- Configuration ---
RAW_DIR = Path("data/upfall/raw/sensor")
# Define the nested output directories as requested
IMUWRIST_DIR = Path("data/upfall/processed/sensor/IMUwrist")
SENSOR_DIR = Path("data/upfall/processed/sensor")

# Create directories if they don't exist
IMUWRIST_DIR.mkdir(parents=True, exist_ok=True)
SENSOR_DIR.mkdir(parents=True, exist_ok=True)

# --- End Configuration ---

def process_subject(file_path: Path, output_base_name: str):
    """
    Reads a raw CSV, removes specified columns, saves the cleaned CSV,
    and then saves the entire content as a .npy file.
    """
    # 1. Read the raw CSV file
    original_df = pd.read_csv(file_path, skiprows=1)
    
    # 2. Define and remove the specified columns by index
    # Note: pandas .columns indexing is 0-based.
    # np.r_ creates a single list of indices from 1-28 and 35-42.
    indices_to_drop = np.r_[1:29, 35:43]

    # Ensure the indices to drop are valid for the current dataframe
    valid_indices_to_drop = [i for i in indices_to_drop if i < len(original_df.columns)]
    
    # Create the new dataframe with columns removed
    cleaned_df = original_df.drop(original_df.columns[valid_indices_to_drop], axis=1)

    # 3. Save the cleaned data to the IMUwrist folder
    cleaned_csv_path = IMUWRIST_DIR / f"{output_base_name}_IMUWrist.csv"
    cleaned_df.to_csv(cleaned_csv_path, index=False)
    
    # Save the array to the sensor folder
    output_npy_path = SENSOR_DIR / f"{output_base_name}.csv"
    cleaned_df.to_csv(output_npy_path, index=False)


def main():
    """Main function to find and process all subject files."""
    # Updated glob pattern to find files like '1_sensor_train.csv'
    files = sorted(RAW_DIR.glob("*_sensor_*.csv"))
    if not files:
        print(f"No files found in {RAW_DIR}. Expected format: [ID]_sensor_[train/test].csv")
        return
        
    for f in files:
        # Extract info from filename, e.g., "1_sensor_train.csv" -> ['1', 'sensor', 'train']
        parts = f.stem.split("_")
        if len(parts) < 3:
            print(f"Skipping file with unexpected format: {f.name}")
            continue
        
        subject_id = parts[0]
        split_type = parts[-1] # 'train' or 'test'
        # Create a new base name for output files, e.g., "1_sensor_train"
        output_base_name = f"{subject_id}_sensor_{split_type}"
        
        print(f"Processing {f.name} -> {output_base_name}")
        process_subject(f, output_base_name)
        
    print("\n--- Processing Complete ---")
    print(f"Cleaned CSVs saved in: {IMUWRIST_DIR}")
    print(f"Final .npy arrays saved in: {SENSOR_DIR}")

if __name__ == "__main__":
    main()
