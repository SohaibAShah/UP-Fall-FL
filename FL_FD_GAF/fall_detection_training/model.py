import torch
from torch import nn

class MyModel(nn.Module):
    """CNN model for fall detection with 3 convolutional layers and 1 fully connected layer."""
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1)
        self.activation1 = nn.Softsign()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.bn1 = nn.BatchNorm2d(3)

        self.conv2 = nn.Conv2d(3, 6, kernel_size=3, stride=1)
        self.activation2 = nn.Softsign()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.bn2 = nn.BatchNorm2d(6)

        self.conv3 = nn.Conv2d(6, 12, kernel_size=3, stride=1)
        self.activation3 = nn.Softsign()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.bn3 = nn.BatchNorm2d(12)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2700, 11)  # 11 classes for activities 1-11
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.bn3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        
        return x