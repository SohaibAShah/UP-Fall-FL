# main.py 
# This is the entry point of the program, which orchestrates the data loading, splitting, and training process.

from utils import set_seed # Import from local utils.py
from data_preprocessing import loadData, splitForClients # Import from local data_preprocessing.py
from federated_learning import trainValSensorModel # Import from local federated_learning.py

# Global variable as defined in the original source
model_name = 'tModel'

def main():
    set_seed(42)  # Set random seed for reproducibility

    # hyperparameters
    total_clients = 15
    num_clients = 12
    epoch = 10
    max_acc = 80 # threshold of accuracy (80%), for saving best model

    # Define ratio, default is equal distribution
    ratio = [1/total_clients] * total_clients

    # Load Data

    (X_train_csv_scaled, X_val_csv_scaled, X_test_csv_scaled,
            Y_train_csv, Y_val_csv, Y_test_csv,
            X_train_1_scaled, X_val_1_scaled, X_test_1_scaled,
            Y_train_1, Y_val_1, Y_test_1,
            X_train_2_scaled, X_val_2_scaled, X_test_2_scaled,
            Y_train_2, Y_val_2, Y_test_2) = loadData()  # Load the data
    
    (X_train_csv_scaled_splits, Y_train_csv_splits,
            X_train_1_scaled_splits, Y_train_1_splits,
            X_train_2_scaled_splits, Y_train_2_splits,
            X_val_csv_scaled_splits, Y_val_csv_splits,
            X_val_1_scaled_splits, Y_val_1_splits,
            X_val_2_scaled_splits, Y_val_2_splits,
            X_test_csv_scaled_splits, Y_test_csv_splits,
            X_test_1_scaled_splits, Y_test_1_splits,
            X_test_2_scaled_splits, Y_test_2_splits) = splitForClients(
                total_clients,
                ratio,
                (X_train_csv_scaled, X_val_csv_scaled, X_test_csv_scaled,
                Y_train_csv, Y_val_csv, Y_test_csv,
                X_train_1_scaled, X_val_1_scaled, X_test_1_scaled,
                Y_train_1, Y_val_1, Y_test_1,
                X_train_2_scaled, X_val_2_scaled, X_test_2_scaled,
                Y_train_2, Y_val_2, Y_test_2)
                ) # Split the data
    
    # Train and Validate Sensor Model through Federated Learning
    trainValSensorModel(
        (X_train_csv_scaled_splits, Y_train_csv_splits,
         X_train_1_scaled_splits, Y_train_1_splits,
         X_train_2_scaled_splits, Y_train_2_splits,
         X_val_csv_scaled_splits, Y_val_csv_splits,
         X_val_1_scaled_splits, Y_val_1_splits,
         X_val_2_scaled_splits, Y_val_2_splits,
         X_test_csv_scaled_splits, Y_test_csv_splits,
         X_test_1_scaled_splits, Y_test_1_splits,
         X_test_2_scaled_splits, Y_test_2_splits),
        total_clients,
        num_clients,
        epoch,
        max_acc,
        model_name
    )



if __name__ == "__main__":
    main()  # Run the main function when the script is executed