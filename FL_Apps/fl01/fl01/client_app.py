"""FL01: A Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import logging

from .task import CNN_Attention, load_data, get_num_features, set_weights, test, train, get_weights

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, cid, device, net, trainloader, valloader, control_variate=None):
        self.cid = cid
        self.device = device
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.control_variate = control_variate

    def fit(self, parameters, config):
        # --- TRACE ---
        logging.info(f"[Client {self.cid}] received parameters and starting local training (fit).")
        
        set_weights(self.net, parameters)
        
        initial_params = [torch.tensor(p, device=self.device) for p in parameters]
        
        train(self.net, self.trainloader, self.device, config, initial_params, self.control_variate)

        # --- TRACE ---
        logging.info(f"[Client {self.cid}] finished local training.")
        
        if config["algorithm"] == "scaffold":
            final_params = list(self.net.parameters())
            model_delta = [(p_final.cpu() - p_initial.cpu()).detach().numpy() for p_final, p_initial in zip(final_params, initial_params)]
            
            new_control_variate = []
            control_variate_delta = []
            coef = 1 / (config['local-epochs'] * len(self.trainloader) * config['learning-rate'])
            for ccv, scv, p_final, p_initial in zip(self.control_variate, config['server_cv'], final_params, initial_params):
                ccv_new = ccv - torch.tensor(scv).to(self.device) + coef * (p_initial - p_final)
                new_control_variate.append(ccv_new)
                control_variate_delta.append((ccv_new - ccv).cpu().detach().numpy())
            self.control_variate = new_control_variate
            return model_delta, len(self.trainloader.dataset), {"cv_delta": control_variate_delta}

        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # --- TRACE ---
        logging.info(f"[Client {self.cid}] received parameters for local evaluation (evaluate).")
        set_weights(self.net, parameters)
        loss, metrics = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), metrics

# Define client_fn
def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    # --- TRACE ---
    logging.info(f"[Client Factory] Creating client for partition-id: {partition_id}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    client_data = load_data(partition_id=partition_id, num_partitions=context.run_config["num_clients"])
    num_features = get_num_features()
    
    X_client, y_client = client_data
    trainloader = DataLoader(TensorDataset(torch.from_numpy(X_client).float(), torch.from_numpy(y_client).long()), batch_size=32, shuffle=True)
    
    net = CNN_Attention(num_features).to(device)
    
    return FlowerClient(partition_id, device, net, trainloader, trainloader).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)