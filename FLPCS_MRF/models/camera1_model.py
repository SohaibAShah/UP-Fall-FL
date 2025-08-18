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
from tqdm import tqdm # 1. Import tqdm

class Img1Model(nn.Module):
    def __init__(self):
        super(Img1Model, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        # self.batch_norm = nn.BatchNorm2d(16)
        # self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        # self.fc1 = nn.Linear(16 * 15 * 15, 200)  # 假设输入是 32x32 的图像
        # self.dropout = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(200, 12)

        #V2======================
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(18)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(18 * (16) * 16, 100)  # 假设输入是 32x32 的图像
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 12)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.softmax(x, dim=1)
    

model_name = 'Img1Model'
def trainValImg1Model(total_client,num_clients,epoch,max_acc,
                        X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits,
                        Y_train_1_splits, Y_test_1_splits, Y_val_1_splits):
    # Instantiate the model    the total_client’th split used for server
    # input_shapes = X_train_csv_scaled_splits[total_client-1].shape[1]
    print("Model initiated for client:", total_client)
    model_MLP = Img1Model()
    model_MLP = model_MLP.double()
    model_MLP = model_MLP.cuda()

    # initialize server and clients
    print("\nServer and clients initialized")
  
    server = Server(model_MLP, [X_test_1_scaled_splits[total_client-1],Y_test_1_splits[total_client-1]], num_clients)
    clients = []

    for c in range(total_client):
        clients.append(Client(server.global_model, [X_train_1_scaled_splits[c],Y_train_1_splits[c]],
                              [X_val_1_scaled_splits[c],Y_val_1_splits[c]], id=c))

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
    model = model_MLP
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    model = model.cuda()

    y_test, y_predict = [], []

    test_loader = torch.utils.data.DataLoader(
        CustomDataset(X_val_1_scaled_splits[total_client - 1], Y_val_1_splits[total_client - 1]), batch_size=32)

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