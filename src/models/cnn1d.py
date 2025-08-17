import torch.nn as nn

class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class AdvancedCNN1D(nn.Module):
    """
    A more robust 1D CNN with multiple convolutional blocks and batch normalization.
    """
    def __init__(self, in_channels=6, num_classes=2):
        super().__init__()
        
        # First convolutional block
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding='same'),
            nn.BatchNorm1d(64), # Stabilizes training
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Second convolutional block
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Third convolutional block
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # Adaptive pooling handles variable input sizes and gives a fixed output size
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # Dropout for regularization
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

