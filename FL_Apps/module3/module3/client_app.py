"""Module3: A Flower / PyTorch app."""

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl
import numpy as np

from .models import GatedResidualFusionModel

class FallDetectionClient(fl.client.NumPyClient):
    def __init__(self, cid, data, device, num_features):
        self.cid = cid
        self.device = device
        self.model = GatedResidualFusionModel(num_features)
        self.data = data # (X_csv, X_img1, X_img2, y)
        # For SCAFFOLD
        self.control_variate = [torch.zeros_like(p) for p in self.model.parameters()]

    def get_parameters(self, config):
        return [p.cpu().detach().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.to(self.device); self.model.train()
        
        loader = DataLoader(TensorDataset(
            torch.from_numpy(self.data[0]).float(), torch.from_numpy(self.data[1]).float(),
            torch.from_numpy(self.data[2]).float(), torch.from_numpy(self.data[3]).long()
        ), batch_size=32, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        initial_global_params = [p.clone() for p in self.model.parameters()]

        for _ in range(config['local_epochs']):
            for x_csv, x_img1, x_img2, y in loader:
                # Modality Dropout
                if np.random.rand() < 0.3: x_img1.zero_(); x_img2.zero_()
                
                optimizer.zero_grad()
                outputs = self.model(x_csv.to(self.device), x_img1.to(self.device), x_img2.to(self.device))
                loss = criterion(outputs, y.to(self.device))

                if config.get('algorithm') == 'fedprox':
                    proximal_term = 0.0
                    for param, initial_param in zip(self.model.parameters(), initial_global_params):
                        proximal_term += torch.sum((param - initial_param) ** 2)
                    loss += (config.get('mu', 0.01) / 2) * proximal_term
                
                loss.backward()

                if config.get('algorithm') == 'scaffold':
                    server_cv = [torch.tensor(p, device=self.device) for p in config['server_cv']]
                    for param, scv, ccv in zip(self.model.parameters(), server_cv, self.control_variate):
                        param.grad += (scv - ccv).to(self.device)
                
                optimizer.step()

        # Handle SCAFFOLD return logic
        # (SCAFFOLD implementation is complex and omitted here for clarity, but this is where it goes)

        return self.get_parameters(config={}), len(self.data[3]), {}
