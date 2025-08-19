from turtle import pd
import torch, random, numpy as np
import torch.nn as nn, os, csv
from utils import CustomDataset, display_result, scaled_data, CustomDatasetIMG, CustomDataseRest, set_seed, validate, train_one_epoch
from fl_Res import Server, Client
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # Import tqdm for progress bar
from models.Res_Model import ModelCSVIMG



model_name = 'ResModel'

def trainValModelRes(total_client,num_clients,epoch,max_acc,epoch_size,local_epoch_per_round,round_early_stop,
                     X_train_csv_scaled_splits, X_test_csv_scaled_splits, X_val_csv_scaled_splits,
                     X_train_1_scaled_splits, X_test_1_scaled_splits, X_val_1_scaled_splits,
                     X_train_2_scaled_splits, X_test_2_scaled_splits, X_val_2_scaled_splits,
                     Y_train_csv_splits, Y_test_csv_splits, Y_val_csv_splits):
    # Instantiate the model    the total_client’th split used for server
    # input_shapes = X_train_csv_scaled_splits[total_client-1].shape[1]
    # ✅ STEP 2: Add this print statement at the top of the function
    print(f"--- In trainValModelRes: Value of total_client RECEIVED = {total_client}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_MLP = ModelCSVIMG(X_train_csv_scaled_splits[0].shape[1],32,32)
    # model_MLP = model_MLP.double()
    model_MLP = model_MLP.to(device)

    # initialize server and clients
    server = Server(model_MLP,epoch_size, [X_test_csv_scaled_splits[total_client-1],X_test_1_scaled_splits[total_client-1],X_test_2_scaled_splits[total_client-1],Y_test_csv_splits[total_client-1]], num_clients)
    clients = []

    for client_index in range(total_client):
        clients.append(Client(epoch_size = epoch_size, local_epoch_per_round = local_epoch_per_round,
                              train_dataset = [X_train_csv_scaled_splits[client_index], X_train_1_scaled_splits[client_index], X_train_2_scaled_splits[client_index],
                               Y_train_csv_splits[client_index]],
                              val_dataset =  [X_test_csv_scaled_splits[client_index], X_test_1_scaled_splits[client_index], X_test_2_scaled_splits[client_index],
                               Y_test_csv_splits[client_index]], id = client_index))
    print("Total clients: %d, Selected clients: %d" % (total_client, num_clients))
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
            model, diff,val_acc,val_loss,min_loss,max_loss,losses,train_acc = c.local_train(server.global_model)  # train local model
            clients_acc[client_index].append(train_acc)
            clients_loss[client_index].append(losses)
            client_index += 1
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])  # update weight_accumulator
        server.model_aggregate(weight_accumulator)  # aggregate global model
        acc, loss = server.model_eval()
        clients_acc[num_clients].append(train_acc)
        clients_loss[num_clients].append(losses)
        
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
    total_loss = 0.0
    correct = 0
    dataset_size = 0
    for test_data_index in range(total_client):
        test_server_loader = torch.utils.data.DataLoader(
            CustomDataseRest(X_test_csv_scaled_splits[test_data_index], X_test_1_scaled_splits[test_data_index],
                          X_test_2_scaled_splits[test_data_index], Y_test_csv_splits[test_data_index]),
            batch_size=epoch_size)
        for batch_id, batch in enumerate(test_server_loader):
            data1 = batch[0]
            data2 = batch[1]
            data3 = batch[2]
            target = torch.squeeze(batch[3])
            # target = torch.tensor(target, dtype=torch.int64)
            # target = torch.squeeze(batch[3]).float()

            dataset_size += data1.size()[0]
            data1 = data1.to(device).float()
            data2 = data2.to(device).float()
            data3 = data3.to(device).float()
            target = target.to(device).float()

            output = model(data1, data2, data3)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.detach().max(1)[1]
            y_test.extend(torch.argmax(target, dim=1).cpu().numpy())
            y_predict.extend(torch.argmax(output, dim=1).cpu().numpy())
            correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    loss = total_loss / dataset_size

    print('server_test_acc', acc)
    print('server_test_loss', loss)

    # classification_report
    print(classification_report(y_test, y_predict,
                                target_names=[f'A{i}' for i in range(1, 13)], digits=4))

    print('max_acc',max_acc)
    csv_file_name = "./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.csv".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
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
    # 保存图像
    plt.savefig("./acc_lossFiles/{}_totalClient_{}_NumClient_{}_epoch_{}.png".format(model_name,total_client,num_clients,epoch,
                                                             datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
    plt.show()