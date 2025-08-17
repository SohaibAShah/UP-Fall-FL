from turtle import pd
import torch, random, numpy as np
import torch.nn as nn, os, csv
from tsmodel.models import SensorModel1, SensorModel2
from tsmodel.utils import CustomDataset, display_result, scaled_data
from tsmodel.federated_learning import Server, Client
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

    # Instantiate the server with the model and evaluation dataset
    server = Server(model_MLP, 
                    [X_test_csv_scaled_splits[total_clients-1], 
                     Y_test_csv_splits[total_clients-1]],
                     num_clients
                    )
    
    # Initialize the list to hold client objects
    clients = []
    for c in range(num_clients):
        clients.append(Client(server.global_model, 
                              [X_train_csv_scaled_splits[c], Y_train_csv_splits[c]], 
                              [X_val_csv_scaled_splits[c], Y_val_csv_splits[c]], 
                              client_id=c))
        
        # Local training of each client
        clients_acc = {}
        clients_loss = {}
        for client in range(num_clients):
            epoch_acc = []
            epoch_loss = []
            clients_acc[client] = epoch_acc
            clients_loss[client] = epoch_loss

        for e in range(epoch):
            print(f"Epoch {e+1}/{epoch}")
            # Select random clients for this round 
            candidates = random.sample(clients, num_clients) 
            weight_accumulator = {}
            for name, params in server.global_model.state_dict().items():
                # Initialize the weight accumulator for each parameter
                weight_accumulator[name] = torch.zeros_like(params)

            # Accumulate weights from each client
            client_index = 0
            for c in candidates:
                # Local training for each client
                diff, acc_client, loss_client = c.local_train(server.global_model)
                # Append the results to the respective client's history
                clients_acc[client_index].append(acc_client)
                clients_loss[client_index].append(loss_client)
                client_index += 1
                # Accumulate the weight updates from this client
                for name, param in server.global_model.state_dict().items():
                    weight_accumulator[name].add_(diff[name])

            # Aggregate the weights from all clients
            server.model_aggregate(weight_accumulator)
            acc, loss = server.model_eval()
            clients_acc[num_clients].append(acc)
            clients_loss[num_clients].append(loss)
            print("Epoch %d, global_acc: %f, global_loss: %f\n" % (e, acc, loss))

            # Save the model if the accuracy is the best we've seen
            if acc > max_acc:
                max_acc = acc
                os.makedirs("./acc_lossFiles/", exist_ok=True) # Ensure directory exists
                torch.save(server.global_model.state_dict(), "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.pth".format(model_name, total_clients, num_clients, epoch, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))) # [37]
                print("save model")

        # test the model
        model = model_MLP
        # The original source tries to load a model based on the total `epoch` parameter.
        # This implies it attempts to load a model named after the full training duration,
        # rather than the specific epoch where `max_acc` was achieved.
        # If no model was saved (e.g., `max_acc` not reached), this `torch.load` will fail.
        # A more robust solution would be to store the path of the best saved model.
        try:
            model.load_state_dict(torch.load("./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.pth".format(model_name, total_clients, num_clients, epoch, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))) # [38]
        except FileNotFoundError:
            print("Model file not found. Continuing with the model's last state. Consider verifying the saving path and logic.")
            # If the file is not found, the model remains in its last trained state from the loop.

        # Set the model to evaluation mode
        model.eval()
        model = model.cuda() if torch.cuda.is_available() else model
        y_test, y_predict = [], []
        test_loader = DataLoader(CustomDataset(
            X_test_csv_scaled_splits[total_clients-1], 
            Y_test_csv_splits[total_clients-1]), 
            batch_size=32, 
            shuffle=False
            )
        for batch_id, batch in enumerate(test_loader):
            data = batch
            target = torch.squeeze(data[1]).float()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            outputs = model(data)
            y_test.extend(torch.argmax(target, dim=1).cpu().numpy())
            y_predict.extend(torch.argmax(outputs, dim=1).cpu().numpy())

        # Classification Report
        print(classification_report(y_test, y_predict, target_names=[str(i) for i in range(1, 13)], digits=4))
        print("max_acc:", max_acc)

        csv_file_name = "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.csv".format(
            model_name, total_clients, num_clients, epoch, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        
        # Save to CSV file
        with open(csv_file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['client', 'Epoch', 'Loss', 'Accuracy'])
            for i in range(num_clients + 1):
                losses = clients_loss[i]
                accuracies = clients_acc[i]
                for j in range(epoch):
                    writer.writerow([i, j + 1, losses[j], accuracies[j]])

        # Confusion Matrix
        plt.figure(dpi=150, figsize=(6, 4))
        classes = [f'A{i}' for i in range(1, 13)]
        mat = confusion_matrix(y_test, y_predict)
        df_cm = pd.DataFrame(mat, index=classes, columns=classes)
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.ylabel('True label', fontsize=15)
        plt.xlabel('Predicted label', fontsize=15)
        plt.savefig("./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.png".format(
            model_name, total_clients, num_clients, epoch, datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
        plt.show()