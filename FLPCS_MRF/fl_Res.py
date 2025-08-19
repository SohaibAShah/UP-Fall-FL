# federated_learning.py 
# This file contains the core logic for the federated learning process, including the Server and Client classes, and the training and validation functions.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import csv
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from models.Res_Model import ModelCSVIMG
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from utils import CustomDataset, display_result, scaled_data, CustomDatasetIMG, CustomDataseRest, set_seed, validate, train_one_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Server
class Server(object):
    def __init__(self, model, epoch_size, eval_dataset, num_clients):
        
        self.global_model = model
        self.epoch_size = epoch_size
        self.num_clients = num_clients
        self.serverTestDataSet = CustomDataseRest(eval_dataset[0],eval_dataset[1],eval_dataset[2],eval_dataset[3])
        self.eval_loader = torch.utils.data.DataLoader(self.serverTestDataSet, batch_size=epoch_size)
	
    def model_aggregate(self, weight_accumulator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * (1/self.num_clients)   # average
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data1 = batch[0]
            data2 = batch[1]
            data3 = batch[2]
            # target = torch.squeeze(batch[1]).int()
            # target = torch.tensor(target, dtype=torch.int64)
            target = torch.squeeze(batch[3])
            # target = torch.tensor(target,dtype=float)

            dataset_size += data1.size()[0]

            data1 = data1.to(device).float()
            data2 = data2.to(device).float()
            data3 = data3.to(device).float()
            target = target.to(device).float()
            
            output = self.global_model(data1,data2,data3)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.detach().max(1)[1]
            correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

        acc = 100.0 *(float(correct) / float(dataset_size))
        loss = total_loss / dataset_size

        return acc, loss


# Client
class Client(object):
    def __init__(self, epoch_size, local_epoch_per_round, train_dataset,val_dataset, id = -1):
                self.epoch_size = epoch_size
                self.local_epoch_per_round = local_epoch_per_round
                self.client_id = id
                self.train_dataset = CustomDataseRest(train_dataset[0],train_dataset[1],train_dataset[2],train_dataset[3])
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=epoch_size,shuffle=True)
                self.eval_dataset = CustomDataseRest(val_dataset[0], val_dataset[1], val_dataset[2], val_dataset[3])
                self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=epoch_size,shuffle=False)
    
    def local_train(self, global_model):
        # for name, param in model.state_dict().items():
        #     self.local_model.state_dict()[name].copy_(param.clone())
        # # optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.0001)
        # optimizer = torch.optim.Adam(self.local_model.parameters(), lr=0.001)
        # self.local_model.train()
        # min_loss = -100000.00
        # max_loss = 100000.00
        # losses = []
        # accs = []
        # correct = 0
        # dataset_size = 0
        # for e in range(self.local_epoch_per_round):
        #     for batch_id, batch in enumerate(self.train_loader):
        #         data1 = batch[0]
        #         data2 = batch[1]
        #         data3 = batch[2]
        #         target = torch.squeeze(batch[3]).int()
        #         target = torch.tensor(target, dtype=torch.int64)
        #         # target = torch.squeeze(batch[3]).float()
        #         # target = torch.tensor(target, dtype=float)
        #         dataset_size += data1.size()[0]
        #         if torch.cuda.is_available():
        #             data1 = data1.cuda().float()
        #             data2 = data2.cuda().float()
        #             data3 = data3.cuda().float()
        #             target = target.cuda().float()
        #
        #         output = self.local_model(data1,data2,data3)
        #         loss = nn.functional.cross_entropy(output, target)
        #         pred = output.max(1)[1]
        #         correct += pred.eq(target.max(1)[1].view_as(pred)).sum().item()
        #
        #         if loss > max_loss:
        #             max_loss = loss
        #         if loss < min_loss:
        #             min_loss = loss
        #         losses.append(loss)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        # train_acc = 100.0 * (float(correct) / float(dataset_size))
        # # after epoch train, eval model and save client model acc,loss to csv
        # # self.local_model.eval()
        # # total_loss = 0.0
        # # correct = 0
        # # dataset_size = 0
        # # for batch_id, batch in enumerate(self.eval_loader):
        # #     data1 = batch[0]
        # #     data2 = batch[1]
        # #     data3 = batch[2]
        # #     # target = torch.squeeze(batch[1]).int()
        # #     # target = torch.tensor(target, dtype=torch.int64)
        # #     target = torch.squeeze(batch[3]).float()
        # #     # target = torch.tensor(target,dtype=float)
        # #
        # #     dataset_size += data1.size()[0]
        # #
        # #     if torch.cuda.is_available():
        # #         data1 = data1.cuda().double()
        # #         data2 = data2.cuda().double()
        # #         data3 = data3.cuda().double()
        # #         target = target.cuda()
        # #
        # #     output = self.local_model(data1,data2,data3)
        # #     total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
        # #
        # #     pred = output.detach().max(1)[1]
        # #     correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()
        # #
        # # test_acc = 100.0 * (float(correct) / float(dataset_size))
        # # loss = total_loss / dataset_size
        #
        # self.local_model.eval()
        # running_loss = 0.0
        # correct = 0
        # total = 0
        # with torch.no_grad():
        #     for batch_id, batch in enumerate(self.eval_loader):
        #         data1 = batch[0].cuda().float()
        #         data2 = batch[1].cuda().float()
        #         data3 = batch[2].cuda().float()
        #         target = torch.squeeze(batch[3]).int()
        #         target = torch.tensor(target, dtype=torch.int64).cuda().float()
        #
        #         # target = torch.squeeze(batch[3]).cuda().float()
        #
        #         output = model(data1, data2, data3)
        #         loss = nn.functional.cross_entropy(output, target)
        #
        #         # 统计
        #         running_loss += loss.item() * data1.size()[0]
        #         _, predicted = output.max(1)
        #         total += target.size(0)
        #         correct += predicted.eq(target.max(1)[1]).sum().item()
        # test_acc = 100.0 * correct / total

        print(self.train_dataset.features1.shape)
        model = ModelCSVIMG(self.train_dataset.features1.shape[1], 32, 32)
        model = model.to(device)

        for name, param in global_model.state_dict().items():
            model.state_dict()[name].copy_(param.clone())

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        num_epochs = self.local_epoch_per_round
        best_acc = 0.0

        min_loss = -100000.00
        max_loss = 100000.00
        losses = []
        for epoch in range(num_epochs):

            # 训练
            train_loss, train_acc = train_one_epoch(model, self.train_loader, criterion, optimizer)
            if train_loss > max_loss:
                max_loss = train_loss
            if train_loss < min_loss:
                min_loss = train_loss
            losses.append(train_loss)

            # 验证
            val_loss, val_acc = validate(model, self.eval_loader, criterion)

            # 保存最优模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")


            # 打印每个epoch的结果
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - ")

        print("client_{} 训练完成，最佳验证准确率: {:.2f}%".format(self.client_id, best_acc))

        diff = dict()
        for name, data in model.state_dict().items():
            diff[name] = (data - global_model.state_dict()[name])
        return model, diff,val_acc,val_loss,min_loss,max_loss,losses,train_acc



