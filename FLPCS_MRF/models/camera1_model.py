from turtle import pd
import torch, random, numpy as np
import torch.nn as nn, os, csv
from utils import CustomDataset, display_result, scaled_data
from federated_learning import Server, Client
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
    for i in range(num_clients + 1):  # one more for server
        epoch_acc = []
        epoch_loss = []
        clients_acc[i] = epoch_acc
        clients_loss[i] = epoch_loss

    for e in tqdm(range(epoch), desc="Training Progress"):
        candidates = random.sample(clients, num_clients)  # randomly select clients
        print("selected clients: %s" % ([(c.client_id+1) for c in candidates]))
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)  # initialize weight_accumulator
        client_index= 0
        for c in candidates:
            print("Client %d training..." % (c.client_id+1))
            diff,acc_client,loss_client = c.local_train(server.global_model)  # train local model
            clients_acc[client_index].append(acc_client)
            clients_loss[client_index].append(loss_client)
            client_index += 1
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])  # update weight_accumulator
        server.model_aggregate(weight_accumulator)  # aggregate global model
        acc, loss = server.model_eval()
        clients_acc[num_clients].append(acc_client)
        clients_loss[num_clients].append(loss_client)
        
        """# 3. Calculate and print the elapsed time for the epoch
        epoch_end_time = time.time()
        elapsed_seconds = epoch_end_time - epoch_start_time
        #print(f"Global Acc: {acc:.2f}%, Global Loss: {loss:.4f}")
        #print(f"Epoch {e+1} completed in {elapsed_seconds:.2f} seconds.\n")"""
        if acc > max_acc:
            max_acc = acc
            torch.save(server.global_model.state_dict(), "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.pth".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
            print("Accuracy improved. Saving model...")
        tqdm.write(f"Epoch {e+1} -> Global Acc: {acc:.2f}%, Global Loss: {loss:.4f}")


    # test
    model = model_MLP
    model.load_state_dict(torch.load("./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.pth".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))))
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
    # 保存到 CSV 文件
    with open(csv_file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['client', 'Epoch', 'Loss', 'Accuracy'])
        for i in range(num_clients + 1):
            # 添加列名
            losses = clients_loss[i]
            accuracies = clients_acc[i]
            for j in range(epoch):
                writer.writerow([i, j + 1, losses[j], accuracies[j]])
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