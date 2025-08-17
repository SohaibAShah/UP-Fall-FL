# models.py
# This file defines the neural network model used for sensor data.

import torch.nn as nn
import torch.nn.functional as F

class SensorModel1(nn.Module):
    def __init__(self, input_size):
        super(SensorModel1, self).__init__()
        self.fc1 = nn.Linear(input_size, 2000)
        self.bn1 = nn.BatchNorm1d(2000)
        self.fc2 = nn.Linear(2000, 600)
        self.bn2 = nn.BatchNorm1d(600)
        self.dropout = nn.Dropout(0.2)
        self.fc3 = nn.Linear(600, 12)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.fc3(x)
        return x

class SensorModel2(nn.Module):
    def __init__(self, input_shapes):
        super(SensorModel2, self).__init__()
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
