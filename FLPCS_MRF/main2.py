# main.py 
# This is the entry point of the program, which orchestrates the data loading, splitting, and training process.

import argparse # Import the argparse library
from utils import set_seed # Import from local utils.py
from load_data import loadData, splitForClients # Import from local data_preprocessing.py
from load_Clientdata import loadClientsData # Import from local load_Clientdata.py
from models.timeseries_model import trainValSensorModel 
from models.tscamera1_model import trainValModelCSVIMG1 
from models.tscamera2_model import trainValModelCSVIMG2
from models.camera1_model import trainValImg1Model # Import from local camera1_model.py
from models.camera2_model import trainValImg2Model # Import from local camera2_model.py
from models.tscamera_resmodel import trainValModelRes # Import from local tscamera_resmodel.py

# Removed the global model_name variable, as it's now handled by arguments.

def main():
    # --- 1. Set up Argument Parser ---
    parser = argparse.ArgumentParser(description="Federated Learning Training Script for UP-Fall Dataset")
    
    parser.add_argument(
        '--model_name', 
        type=str, 
        required=True, 
        choices=['SensorModel', 'Img1Model', 'Img2Model', 'CSVIMG1Model', 'CSVIMG2Model', 'ResModel'],
        help='The name of the model to train. This determines which training function is called.'
    )
    parser.add_argument('--epochs', type=int, default=200, help='Number of global training rounds (epochs).')
    parser.add_argument('--total_clients', type=int, default=12, help='The total number of clients available for the simulation.')
    parser.add_argument('--num_clients_per_round', type=int, default=12, help='The number of clients to select for training in each round.')
    parser.add_argument('--max_acc', type=float, default=80.0, help='The accuracy threshold (in percent) for saving the best model.')
    
    args = parser.parse_args()

    total_clients = args.total_clients
    epoch_size = 64
    local_epoch_per_round = 3
    round_early_stop = 10

    # --- 2. Initialize and Load Data ---
    set_seed()
    
    # Hyperparameters are now loaded from args
    print(f"🚀 Starting training for model: {args.model_name}")
    print(f"Hyperparameters: Epochs={args.epochs}, Total Clients={args.total_clients}, Clients/Round={args.num_clients_per_round}\n")


    X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits, \
            Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits, \
            X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits, \
            Y_train_1_splits, Y_test_1_splits, Y_val_1_splits, \
            X_train_2_scaled_splits, X_test_2_scaled_splits, X_val_2_scaled_splits, \
            Y_train_2_splits, Y_test_2_splits, Y_val_2_splits = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    # Load data
    print("Loading and preprocessing data...")
    if args.model_name == 'ResModel':
        X_train_csv_scaled_splits, X_test_csv_scaled_splits, \
            Y_train_csv_splits, Y_test_csv_splits, \
            X_train_1_scaled_splits, X_test_1_scaled_splits, \
            Y_train_1_splits, Y_test_1_splits, \
            X_train_2_scaled_splits, X_test_2_scaled_splits, \
            Y_train_2_splits, Y_test_2_splits = loadClientsData()
        # ✅ ADD THESE LINES FOR DEBUGGING
        print(f"Total clients parameter: {total_clients}")
        print(f"Number of partitions created: {len(X_test_csv_scaled_splits)}")
        print(f"Keys in partition dictionary: {list(X_test_csv_scaled_splits.keys())}")
        print("Data loading complete.\n")
    else:
        X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled, \
            Y_train_csv, Y_test_csv, Y_val_csv, \
            X_train_1_scaled, X_test_1_scaled, X_val_1_scaled, \
            Y_train_1, Y_test_1, Y_val_1, \
            X_train_2_scaled, X_test_2_scaled, X_val_2_scaled, \
            Y_train_2, Y_test_2, Y_val_2 = loadData()
        
            # Split data for clients
        ratios = [1 / args.total_clients] * args.total_clients
        X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits, \
            Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits, \
            X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits, \
            Y_train_1_splits, Y_test_1_splits, Y_val_1_splits, \
            X_train_2_scaled_splits, X_test_2_scaled_splits, X_val_2_scaled_splits, \
            Y_train_2_splits, Y_test_2_splits, Y_val_2_splits = splitForClients(
                args.total_clients, ratios, X_train_csv_scaled, X_test_csv_scaled, X_val_csv_scaled,
                Y_train_csv, Y_test_csv, Y_val_csv,
                X_train_1_scaled, X_test_1_scaled, X_val_1_scaled,
                Y_train_1, Y_test_1, Y_val_1,
                X_train_2_scaled, X_test_2_scaled, X_val_2_scaled,
                Y_train_2, Y_test_2, Y_val_2
        )
        

    

    # --- 3. Conditionally Call the Training Function ---
    if args.model_name == 'SensorModel':
        trainValSensorModel(
            total_client=args.total_clients, 
            num_clients=args.num_clients_per_round, 
            epoch=args.epochs, 
            max_acc=args.max_acc,
            X_train_csv_scaled_splits=X_train_csv_scaled_splits, 
            X_test_csv_scaled_splits=X_test_csv_scaled_splits, 
            X_val_csv_scaled_splits=X_val_csv_scaled_splits,
            Y_train_csv_splits=Y_train_csv_splits, 
            Y_test_csv_splits=Y_test_csv_splits, 
            Y_val_csv_splits=Y_val_csv_splits
        )
    elif args.model_name == 'Img1Model':
        trainValImg1Model(
            total_client=args.total_clients, 
            num_clients=args.num_clients_per_round, 
            epoch=args.epochs, 
            max_acc=args.max_acc,
            X_train_1_scaled_splits=X_train_1_scaled_splits, 
            X_test_1_scaled_splits=X_test_1_scaled_splits, 
            X_val_1_scaled_splits=X_val_1_scaled_splits,
            Y_train_1_splits=Y_train_1_splits, 
            Y_test_1_splits=Y_test_1_splits, 
            Y_val_1_splits=Y_val_1_splits
        )
    elif args.model_name == 'Img2Model':
        trainValImg2Model(
            total_client=args.total_clients, 
            num_clients=args.num_clients_per_round, 
            epoch=args.epochs, 
            max_acc=args.max_acc,
            X_train_2_scaled_splits=X_train_2_scaled_splits, 
            X_test_2_scaled_splits=X_test_2_scaled_splits, 
            X_val_2_scaled_splits=X_val_2_scaled_splits,
            Y_train_2_splits=Y_train_2_splits, 
            Y_test_2_splits=Y_test_2_splits, 
            Y_val_2_splits=Y_val_2_splits
        )
    elif args.model_name == 'CSVIMG1Model':
        trainValModelCSVIMG1(
            total_client=args.total_clients, 
            num_clients=args.num_clients_per_round, 
            epoch=args.epochs, 
            max_acc=args.max_acc,
            X_train_csv_scaled_splits=X_train_csv_scaled_splits, 
            X_test_csv_scaled_splits=X_test_csv_scaled_splits, 
            X_val_csv_scaled_splits=X_val_csv_scaled_splits,
            X_train_1_scaled_splits=X_train_1_scaled_splits, 
            X_test_1_scaled_splits=X_test_1_scaled_splits, 
            X_val_1_scaled_splits=X_val_1_scaled_splits,
            Y_train_csv_splits=Y_train_csv_splits, 
            Y_test_csv_splits=Y_test_csv_splits, 
            Y_val_csv_splits=Y_val_csv_splits
        )
    elif args.model_name == 'CSVIMG2Model':
        trainValModelCSVIMG2(
            total_client=args.total_clients, 
            num_clients=args.num_clients_per_round, 
            epoch=args.epochs, 
            max_acc=args.max_acc,
            X_train_csv_scaled_splits=X_train_csv_scaled_splits, 
            X_test_csv_scaled_splits=X_test_csv_scaled_splits, 
            X_val_csv_scaled_splits=X_val_csv_scaled_splits,
            X_train_2_scaled_splits=X_train_2_scaled_splits, 
            X_test_2_scaled_splits=X_test_2_scaled_splits, 
            X_val_2_scaled_splits=X_val_2_scaled_splits,
            Y_train_csv_splits=Y_train_csv_splits, 
            Y_test_csv_splits=Y_test_csv_splits, 
            Y_val_csv_splits=Y_val_csv_splits
        )
    elif args.model_name == 'ResModel':
        trainValModelRes(
            total_client=args.total_clients, 
            num_clients=args.num_clients_per_round, 
            epoch=args.epochs, 
            max_acc=args.max_acc,
            epoch_size=epoch_size,
            local_epoch_per_round=local_epoch_per_round,
            round_early_stop=round_early_stop,
            X_train_csv_scaled_splits=X_train_csv_scaled_splits, 
            X_test_csv_scaled_splits=X_test_csv_scaled_splits, 
            X_val_csv_scaled_splits=X_val_csv_scaled_splits,
            X_train_1_scaled_splits=X_train_1_scaled_splits, 
            X_test_1_scaled_splits=X_test_1_scaled_splits, 
            X_val_1_scaled_splits=X_val_1_scaled_splits,
            X_train_2_scaled_splits=X_train_2_scaled_splits, 
            X_test_2_scaled_splits=X_test_2_scaled_splits, 
            X_val_2_scaled_splits=X_val_2_scaled_splits,
            Y_train_csv_splits=Y_train_csv_splits, 
            Y_test_csv_splits=Y_test_csv_splits, 
            Y_val_csv_splits=Y_val_csv_splits
        )
    else:
        print(f"Error: Model name '{args.model_name}' is not recognized or implemented.")


if __name__ == "__main__":
    main() # Run the main function when the script is executed