import torch.nn as nn

class SimpleMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for tabular or vector data.
    This model is suitable for single-row sensor readings.
    """
    def __init__(self, input_size=6, num_classes=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout helps prevent overfitting
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """Forward pass through the network."""
        return self.layers(x)
