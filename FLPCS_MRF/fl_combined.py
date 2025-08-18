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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from utils import CustomDataset, display_result, scaled_data, CustomDatasetIMG

# Server
class Server(object):
    def __init__(self, model, eval_dataset, num_clients):
        
        self.global_model = model
        self.num_clients = num_clients
        self.serverTestDataSet = CustomDatasetIMG(eval_dataset[0],eval_dataset[1],eval_dataset[2])
        self.eval_loader = torch.utils.data.DataLoader(self.serverTestDataSet, batch_size=32)
	
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
            # target = torch.squeeze(batch[1]).int()
            # target = torch.tensor(target, dtype=torch.int64)
            target = torch.squeeze(batch[2]).float()
            # target = torch.tensor(target,dtype=float)

            dataset_size += data1.size()[0]

            if torch.cuda.is_available():
                data1 = data1.cuda()
                data2 = data2.cuda()
                target = target.cuda()
            
            output = self.global_model(data1,data2)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.detach().max(1)[1]
            correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

        acc = 100.0 *(float(correct) / float(dataset_size))
        loss = total_loss / dataset_size

        return acc, loss


# Client
class Client(object):
    def __init__(self, model, train_dataset,val_dataset, id = -1):
                self.local_model = model
                self.client_id = id
                self.train_dataset = CustomDatasetIMG(train_dataset[0],train_dataset[1],train_dataset[2])
                self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32)
                self.eval_dataset = CustomDatasetIMG(val_dataset[0], val_dataset[1], val_dataset[2])
                self.eval_loader = torch.utils.data.DataLoader(self.eval_dataset, batch_size=32)

    def local_train(self, model):
        for name, param in model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.0001)
        self.local_model.train()
        for e in range(3):
            for batch_id, batch in enumerate(self.train_loader):
                data1 = batch[0]
                data2 = batch[1]
                # target = torch.squeeze(batch[1]).int()
                # target = torch.tensor(target, dtype=torch.int64)
                target = torch.squeeze(batch[2]).float()
                # target = torch.tensor(target, dtype=float)

                if torch.cuda.is_available():
                    data1 = data1.cuda()
                    data2 = data2.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data1,data2)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

        # after epoch train, eval model and save client model acc,loss to csv
        self.local_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data1 = batch[0]
            data2 = batch[1]
            # target = torch.squeeze(batch[1]).int()
            # target = torch.tensor(target, dtype=torch.int64)
            target = torch.squeeze(batch[2]).float()
            # target = torch.tensor(target,dtype=float)

            dataset_size += data1.size()[0]

            if torch.cuda.is_available():
                data1 = data1.cuda()
                data2 = data2.cuda()
                target = target.cuda()

            output = self.local_model(data1,data2)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()

            pred = output.detach().max(1)[1]
            correct += pred.eq(target.detach().max(1)[1].view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        loss = total_loss / dataset_size

        diff = dict()
        for name, data in self.local_model.state_dict().items():
            diff[name] = (data - model.state_dict()[name])
        return diff,acc,loss