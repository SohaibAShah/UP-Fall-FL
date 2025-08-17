# main.py 
# This is the entry point of the program, which orchestrates the data loading, splitting, and training process.

from utils import set_seed # Import from local utils.py
from data_preprocessing import loadData, splitForClients # Import from local data_preprocessing.py
from model_trainer import trainValSensorModel # Import from local model_trainer.py
from federated_learning import Server, Client # Import from local federated_learning.py

# Global variable as defined in the original source
model_name = 'SensorModel1'  # Name of the model to be used in training

def main():
    set_seed()
    # hyperparameters
    max_acc = 80  # thorshold of accuracy (80%), for saving best model
    epoch = 200
    total_client = 15  # total number of clients
    num_clients = 12  # number of clients selected per round
    # 定义比例，默认均分
    ratios = [1 / total_client] * total_client
    # load data
    X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled, \
        Y_train_csv, Y_test_csv, Y_val_csv, \
        X_train_1_scaled, X_test_1_scaled, X_val_1_scaled, \
        Y_train_1, Y_test_1, Y_val_1, \
        X_train_2_scaled, X_test_2_scaled, X_val_2_scaled, \
        Y_train_2, Y_test_2, Y_val_2 = loadData()
    # split data according to total_client
    X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits, \
        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits, \
        X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits, \
        Y_train_1_splits, Y_test_1_splits, Y_val_1_splits, \
        X_train_2_scaled_splits, X_test_2_scaled_splits, X_val_2_scaled_splits, \
        Y_train_2_splits, Y_test_2_splits, Y_val_2_splits = splitForClients(total_client,ratios,X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled,
        Y_train_csv, Y_test_csv, Y_val_csv,
        X_train_1_scaled, X_test_1_scaled, X_val_1_scaled,
        Y_train_1, Y_test_1, Y_val_1,
        X_train_2_scaled, X_test_2_scaled, X_val_2_scaled,
        Y_train_2, Y_test_2, Y_val_2 )

    trainValSensorModel(total_client,num_clients,epoch,max_acc,
                        X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits,
                        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits)



if __name__ == "__main__":
    main()  # Run the main function when the script is executed