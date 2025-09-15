import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

from .task import CNN_Attention, set_weights, test, train, get_weights

# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, cid, device, net, trainloader, valloader, control_variate=None):
        self.cid = cid; self.device = device; self.net = net
        self.trainloader = trainloader; self.valloader = valloader
        self.control_variate = control_variate

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        initial_params = [torch.tensor(p, device=self.device) for p in parameters]
        train(self.net, self.trainloader, self.device, config, initial_params, self.control_variate)

        if config["algorithm"] == "scaffold":
            final_params = list(self.net.parameters())
            model_delta = [(p_final.cpu() - p_initial.cpu()).detach().numpy() for p_final, p_initial in zip(final_params, initial_params)]
            new_control_variate, control_variate_delta = [], []
            coef = 1 / (config['local-epochs'] * len(self.trainloader) * config['learning-rate'])
            for ccv, scv, p_final, p_initial in zip(self.control_variate, config['server_cv'], final_params, initial_params):
                ccv_new = ccv - torch.tensor(scv).to(self.device) + coef * (p_initial - p_final)
                new_control_variate.append(ccv_new)
                control_variate_delta.append((ccv_new - ccv).cpu().detach().numpy())
            self.control_variate = new_control_variate
            return model_delta, len(self.trainloader.dataset), {"cv_delta": control_variate_delta}

        return get_weights(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, metrics = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), metrics

# Define client_fn
def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    partitions_dir = os.path.join(os.path.dirname(__file__), "..", "partitions")
    client_ids = sorted([int(fname.split('_')[-1].split('.')[0]) for fname in os.listdir(partitions_dir) if fname.startswith('X_client_')])
    cid_to_load = client_ids[partition_id]
    
    # **FIX:** Added weights_only=False to allow loading NumPy arrays
    X_client, y_client = torch.load(os.path.join(partitions_dir, f'client_{cid_to_load}.pt'), weights_only=False)
    
    X_client = np.transpose(X_client, (0, 2, 1))
    
    with open(os.path.join(partitions_dir, 'num_features.txt'), 'r') as f:
        num_features = int(f.read())
    
    trainloader = DataLoader(TensorDataset(torch.from_numpy(X_client).float(), torch.from_numpy(y_client).long()), batch_size=32, shuffle=True)
    
    net = CNN_Attention(num_features).to(device)
    
    return FlowerClient(partition_id, device, net, trainloader, trainloader).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
)