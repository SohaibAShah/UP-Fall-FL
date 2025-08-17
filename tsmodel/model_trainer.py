import torch
import torch.nn as nn
from tsmodel.models import SensorModel1, SensorModel2




def trainValSensorModel(total_clients, 
                        num_clients, 
                        epoch, 
                        max_acc,
                        model_name,
                        X_train_csv_scaled_splits,
                        X_val_csv_scaled_splits,
                        X_test_csv_scaled_splits,
                        Y_train_csv_splits,
                        Y_val_csv_splits,
                        Y_test_csv_splits
                        ):
    
    # Instantiate the model the total_client'th split used for server
    input_shapes = X_train_csv_scaled_splits[total_clients-1].shape[1]
    model_MLP = SensorModel1(input_shapes)
    model_MLP = model_MLP.double()
    model_MLP = model_MLP.cuda() if torch.cuda.is_available() else model_MLP