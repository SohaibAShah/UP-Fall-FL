from turtle import pd
import torch, random, numpy as np
import torch.nn as nn, os, csv
from utils import CustomDataset, display_result, scaled_data, CustomDatasetIMG
from fl_combined import Server, Client
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bar

class ModelCSVIMG1(nn.Module):
    def __init__(self, num_csv_features):
        super(ModelCSVIMG1, self).__init__()

        # v2==========================================
        # 第一输入分支：处理CSV特征
        self.csv_fc_1 = nn.Linear(num_csv_features, 2000)
        self.csv_bn_1 = nn.BatchNorm1d(2000)
        self.csv_fc_2 = nn.Linear(2000, 600)
        self.csv_bn_2 = nn.BatchNorm1d(600)
        self.csv_fc_3 = nn.Linear(600, 100)
        self.csv_dropout = nn.Dropout(0.2)

        # 第二输入分支：处理第一张图像的2D卷积
        self.img1_conv_1 = nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
        self.img1_batch_norm = nn.BatchNorm2d(18)
        self.img1_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.img1_fc1 = nn.Linear(18 * (16) * 16, 100)
        self.img1_dropout = nn.Dropout(0.2)

        # 全连接层
        self.fc1 = nn.Linear(200, 600)
        self.fc2 = nn.Linear(600, 1200)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1200, 12)

    def forward(self, x_csv, x_img1):
        x_csv = F.relu(self.csv_bn_1(self.csv_fc_1(x_csv)))
        x_csv = F.relu(self.csv_bn_2(self.csv_fc_2(x_csv)))
        x_csv = F.relu(self.csv_fc_3(x_csv))
        x_csv = self.csv_dropout(x_csv)
        # x_csv = self.fc_csv_3(x_csv)

        x_img1 = x_img1.permute(0, 3, 1, 2)
        # 第二分支：第一张图像
        x_img1 = F.relu(self.img1_conv_1(x_img1))
        x_img1 = self.img1_batch_norm(x_img1)
        x_img1 = self.img1_pool(x_img1)
        x_img1 = x_img1.view(x_img1.size(0), -1)
        x_img1 = F.relu(self.img1_fc1(x_img1))
        x_img1 =self.img1_dropout(x_img1)

        # 连接三个分支
        x = torch.cat((x_csv, x_img1), dim=1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.softmax(self.fc3(x), dim=1)

        return x

model_name = 'ModelCSVIMG1'
def trainValModelCSVIMG1(total_client,num_clients,epoch,max_acc,
                        X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits,
                        X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits,
                        Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits):
    # Instantiate the model    the total_client’th split used for server
    # input_shapes = X_train_csv_scaled_splits[total_client-1].shape[1]
    model_MLP = ModelCSVIMG1(X_train_csv_scaled_splits[total_client-1].shape[1])
    model_MLP = model_MLP.double()
    model_MLP = model_MLP.cuda()

    # initialize server and clients
    server = Server(model_MLP, [X_test_csv_scaled_splits[total_client-1],X_test_1_scaled_splits[total_client-1],Y_test_csv_splits[total_client-1]], num_clients)
    clients = []

    for c in range(total_client):
        clients.append(Client(server.global_model, [X_train_csv_scaled_splits[c],X_train_1_scaled_splits[c],Y_train_csv_splits[c]],
                              [X_val_csv_scaled_splits[c],X_val_1_scaled_splits[c],Y_val_csv_splits[c]], id=c))

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

    test_loader = torch.utils.data.DataLoader(CustomDatasetIMG(X_val_csv_scaled_splits[total_client-1],X_val_1_scaled_splits[total_client-1],Y_val_csv_splits[total_client-1]), batch_size=32)

    for batch_id, batch in enumerate(test_loader):
        data1 = batch[0]
        data2 = batch[1]
        # target = torch.squeeze(batch[3]).int()
        # target = torch.tensor(target, dtype=torch.int64)
        target = torch.squeeze(batch[2]).float()

        if torch.cuda.is_available():
            data1 = data1.cuda()
            data2 = data2.cuda()
            target = target.cuda()

        output = model(data1,data2)
        y_test.extend(torch.argmax(target, dim=1).cpu().numpy())
        y_predict.extend(torch.argmax(output, dim=1).cpu().numpy())

    # classification_report
    print(classification_report(y_test, y_predict,
                                target_names=[f'A{i}' for i in range(1, 13)], digits=4))

    print('max_acc',max_acc)
    csv_file_name = "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.csv".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
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
    # 保存图像
    plt.savefig("./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.png".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    plt.show()