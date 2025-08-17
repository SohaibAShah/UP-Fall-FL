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

from tsmodel.models import SensorModel1, SensorModel2
from tsmodel.utils import CustomDataset, display_result, scaled_data

class Server(object):
    def __init__(self, model, num_clients, eval_dataset):
        self.global_model = model
        self.num_clients = num_clients
        self.serverTestDataSet = CustomDataset(eval_dataset, eval_dataset[1])
        self.eval_loader = torch.utils.data.DataLoader(self.serverTestDataSet, batch_size=32, shuffle=False)
        
    def model_aggregate(self, weigh_accumator):
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weigh_accumator[name] * (1/self.num_clients)
            
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
            data = batch
            target = torch.squeeze(data[1]).float()
            dataset_size += data.size()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.global_model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            total_loss += loss.item()
            preds = output.detach().max(dim=1)[1]
            correct += preds.eq(target.detach().max(1)[1].view_as(preds)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        loss = total_loss / dataset_size
        print(f"Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")
        return acc, loss
    
class Client(object):
    def __init__(self, model, train_dataset, val_dataset, client_id = -1):
        self.local_model = model
        self.train_dataset = CustomDataset(train_dataset, train_dataset[1])
        self.val_dataset = CustomDataset(val_dataset, val_dataset[1])
        self.client_id = client_id
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=32, shuffle=False)   

    def local_train(self, model):
        for name, param in model.state_dict().items():
            if name in self.local_model.state_dict():
                self.local_model.state_dict()[name].copy_(param.clone())
        optimizer = optim.Adam(self.local_model.parameters(), lr=0.001)
        self.local_model.train()

        for epoch in range(3):
            for batch_id, batch in enumerate(self.train_loader):
                data = batch
                target = torch.squeeze(data[1]).float()
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                output = self.local_model(data)
                loss = F.cross_entropy(output, target, reduction='sum')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if batch_id % 10 == 0:
                    print(f"Client {self.client_id}, Epoch {epoch}, Batch {batch_id}, Loss: {loss.item()}")

        # After epoch train, eval model and save client model acc.loss to csv
        self.local_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.val_loader):
            data = batch
            target = torch.squeeze(data[1]).float()
            dataset_size += data.size()
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            output = self.local_model(data)
            loss = F.cross_entropy(output, target, reduction='sum')
            total_loss += loss.item()
            preds = output.detach().max(dim=1)[1]
            correct += preds.eq(target.detach().max(1)[1].view_as(preds)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(dataset_size))
        loss = total_loss / dataset_size
        print(f"Client {self.client_id}, Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")

        diff = dict()
        for name, param in self.local_model.state_dict().items():
            if name in model.state_dict():
                diff[name] = param - model.state_dict()[name]

        # Save to CSV
        result_df = pd.DataFrame({"Client ID": [self.client_id], "Accuracy": [acc], "Loss": [loss]})
        result_df.to_csv(f"client_{self.client_id}_results.csv", index=False)

        return diff, acc, loss