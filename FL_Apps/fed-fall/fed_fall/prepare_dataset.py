import logging
import os
import sys

# Add the project's source directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from data_pipeline import prepare_partitions

# Configure basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    """
    This script runs the one-time data preprocessing and partitioning step.
    It creates the `partitions/` directory which the Flower simulation needs.
    """
    logging.info("Starting the data preparation and partitioning process...")
    
    # Define paths relative to the project root
    DATA_DIR = "/home/syed/PhD/UP_Fall_Dataset/Sensor + Image"
    PARTITIONS_DIR = "/home/syed/PhD/UP-Fall-FL/FL_Apps/fed-fall/fed_fall/UP_Fall_partitions"

    # Run the preparation function
    prepare_partitions(data_dir=DATA_DIR, partitions_dir=PARTITIONS_DIR)

    logging.info("✅ Data preparation complete.")