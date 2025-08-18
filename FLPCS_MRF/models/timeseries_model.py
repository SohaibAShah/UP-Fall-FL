from turtle import pd
import torch, random, numpy as np
import torch.nn as nn, os, csv
from utils import CustomDataset, display_result, scaled_data
from fl_simple import Server, Client
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm # 1. Import tqdm


class SensorModel(nn.Module):
    def __init__(self, input_shapes):
        super(SensorModel, self).__init__()
        self.fc1 = nn.Linear(input_shapes, 2000)
        self.bn1 = nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(2000, 600)
        self.bn2 = nn.BatchNorm1d(600)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(600, 12)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.softmax(self.fc3(x), dim=1)
        return x
    

model_name = 'SensorModel'
# Define the trainValSensorModel function
def trainValSensorModel(total_client,num_clients,epoch,max_acc,
                        X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits,
                        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits):
    # Instantiate the model and the total_client’th split used for server
    print("Model initiated for %d clients" % total_client)
    input_shapes = X_train_csv_scaled_splits[total_client-1].shape[1]
    model_MLP = SensorModel(input_shapes)
    model_MLP = model_MLP.double()
    model_MLP = model_MLP.cuda()

    # initialize server and clients
    print("\nServer and clients initialized")
    server = Server(model_MLP, [X_test_csv_scaled_splits[total_client-1],Y_test_csv_splits[total_client-1]], num_clients)
    clients = []

    for c in range(total_client):
        clients.append(Client(server.global_model, [X_train_csv_scaled_splits[c],Y_train_csv_splits[c]],
                              [X_val_csv_scaled_splits[c],Y_val_csv_splits[c]], id=c))

    # train
    clients_acc = {}
    clients_loss = {}
    server_acc_history = []
    server_loss_history = []
    # Define a single path for the best model to avoid filename mismatches
    best_model_path = ""
    model_saved = False

    for i in range(num_clients+1): # one more for server
        """epoch_acc = []
        epoch_loss = []
        clients_acc[i] = epoch_acc
        clients_loss[i] = epoch_loss"""

        # --- MODIFICATION 1: Correctly initialize storage for client metrics ---
        # Use a dictionary with the client's actual ID as the key
        clients_acc = {i: [] for i in range(total_client)}
        clients_loss = {i: [] for i in range(total_client)}
        
        

    for e in tqdm(range(epoch), desc="Training Progress"):
        candidates = random.sample(clients, num_clients)  # randomly select clients
        print("\nSelected clients: %s" % ([(c.client_id+1) for c in candidates]))
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)  # initialize weight_accumulator
        client_index= 0
        for c in candidates:
            print("Client %d training..." % (c.client_id+1))
            # --- MODIFICATION 2: Store and display accuracy using the correct client ID ---
            diff, acc_client, loss_client = c.local_train(server.global_model)
            
            # Store metrics using the actual client ID
            clients_acc[c.client_id].append(acc_client)
            clients_loss[c.client_id].append(loss_client)
            
            # Display the individual client's accuracy for this round
            print(f"  > Client {c.client_id+1} local validation acc: {acc_client:.2f}%")
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])  # update weight_accumulator
        server.model_aggregate(weight_accumulator)  # aggregate global model
        acc, loss = server.model_eval()
        
        # Store server/global model accuracy history
        server_acc_history.append(acc)
        server_loss_history.append(loss)
        
        """# 3. Calculate and print the elapsed time for the epoch
        epoch_end_time = time.time()
        elapsed_seconds = epoch_end_time - epoch_start_time
        #print(f"Global Acc: {acc:.2f}%, Global Loss: {loss:.4f}")
        #print(f"Epoch {e+1} completed in {elapsed_seconds:.2f} seconds.\n")"""
        if acc > max_acc:
            max_acc = acc
            # Define the path only when saving to capture the correct epoch and timestamp
            best_model_path = (f"./acc_lossFiles/{model_name}_totalClient_{total_client}_"
                             f"NumClient_{num_clients}_epoch_{e+1}_acc_{acc:.2f}.pth")
            torch.save(server.global_model.state_dict(), best_model_path)
            model_saved = True
            print("Accuracy improved. Saving model....")

        tqdm.write(f"Epoch {e+1} -> Global Acc: {acc:.2f}%, Global Loss: {loss:.4f}")
        

    # --- MODIFICATION 3: Add a summary of client performance after training ---
    print("\n--- Training Summary ---")
    for client_id, accs in clients_acc.items():
        if accs: # Check if the client was ever selected for training
            avg_acc = np.mean(accs)
            print(f"Client {client_id+1}: Average validation accuracy = {avg_acc:.2f}% (trained {len(accs)} times)")
    print("------------------------\n")

    # --- Test Phase ---
    if not model_saved:
        print("Warning: Model was never saved as accuracy did not exceed the initial threshold.")
        print("Skipping final evaluation from a loaded model.")
        return

    print(f"Loading best model from {best_model_path} for final evaluation...")
    model = SensorModel(input_shapes) # Re-initialize the model structure
    model = model.double().cuda()
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_test, y_predict = [], []

    test_loader = torch.utils.data.DataLoader(CustomDataset(X_val_csv_scaled_splits[total_client-1],Y_val_csv_splits[total_client-1]), batch_size=32)

    for batch_id, batch in enumerate(test_loader):
        data = batch[0]
        # target = torch.squeeze(batch[1]).int()
        # target = torch.tensor(target, dtype=torch.int64)
        target = torch.squeeze(batch[1]).float()

        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        y_test.extend(torch.argmax(target, dim=1).cpu().numpy())
        y_predict.extend(torch.argmax(output, dim=1).cpu().numpy())

    # classification_report
    print(classification_report(y_test, y_predict,
                                target_names=[f'A{i}' for i in range(1, 13)], digits=4))
    print('max_acc', max_acc)
    csv_file_name = "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.csv".format(model_name, total_client,
                                                                                         num_clients, epoch,
                                                                                         datetime.now().strftime(
                                                                                             '%Y-%m-%d-%H-%M-%S'))
    print("Saving client and server performance to CSV...")
    # Open the CSV file to write
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write a more descriptive header
        writer.writerow(['ClientID', 'Training_Round_Instance', 'Loss', 'Accuracy'])

        # Iterate through each client's recorded metrics
        for client_id, accuracies in clients_acc.items():
            losses = clients_loss[client_id]
            
            # This inner loop now iterates only over the actual number of times
            # this client was trained, preventing the IndexError.
            for i in range(len(accuracies)):
                # We record the client's ID, which instance of training this was for them (1st, 2nd, etc.),
                # and their corresponding loss and accuracy.
                writer.writerow([client_id + 1, i + 1, losses[i], accuracies[i]])

        # Optionally, add server performance to the same file for a complete record
        writer.writerow(['---', '---', '---', '---']) # Add a separator for clarity
        writer.writerow(['Global_Model', 'Epoch', 'Loss', 'Accuracy'])
        for i in range(len(server_acc_history)):
            writer.writerow(['global', i + 1, server_loss_history[i], server_acc_history[i]])

    print(f"Results saved to {csv_file_name}")
    # confusion matrix
    plt.figure(dpi=150, figsize=(6, 4))
    classes = [f'A{i}' for i in range(1, 13)]
    mat = confusion_matrix(y_test, y_predict)

    df_cm = pd.DataFrame(mat, index=classes, columns=classes)
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)
    plt.savefig(
        "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.png".format(model_name, total_client, num_clients,
                                                                             epoch,
                                                                             datetime.now().strftime(
                                                                                 '%Y-%m-%d-%H-%M-%S')))
    plt.show()