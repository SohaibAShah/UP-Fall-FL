import flwr as fl
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, TensorDataset

from .task import CNN_Attention, test, get_weights

print("[server_app.py] Module loaded.")

# Custom SCAFFOLD Strategy
class Scaffold(FedAvg):
    def __init__(self, **kwargs):
        print("[server_app.py] Scaffold strategy initialized.")

        super().__init__(**kwargs)
        self.server_control_variate = [np.zeros_like(p) for p in self.initial_parameters.tensors]
    def configure_fit(self, server_round, parameters, client_manager):
        config_list = super().configure_fit(server_round, parameters, client_manager)
        for _, config in config_list: config["server_cv"] = self.server_control_variate
        return config_list
    def aggregate_fit(self, server_round, results, failures):
        aggregated_deltas, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_deltas is not None:
            current_params = parameters_to_ndarrays(self.initial_parameters)
            new_params = [current + delta for current, delta in zip(current_params, parameters_to_ndarrays(aggregated_deltas))]
            self.initial_parameters = ndarrays_to_parameters(new_params)
        if results:
            avg_cv_delta = [np.mean([res.metrics["cv_delta"][i] for _, res in results], axis=0) for i in range(len(self.server_control_variate))]
            self.server_control_variate = [scv + cvd for scv, cvd in zip(self.server_control_variate, avg_cv_delta)]
        return self.initial_parameters, {}

# Server-side evaluation function
def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    print("[server_app.py] Server-side evaluate() called.")

    partitions_dir = os.path.join(os.path.dirname(__file__), "..", "partitions")
    with open(os.path.join(partitions_dir, 'num_features.txt'), 'r') as f:
        num_features = int(f.read())
    
    # **FIX:** Added weights_only=False to allow loading NumPy arrays
    X_test, y_test = torch.load(os.path.join(partitions_dir, 'test.pt'), weights_only=False)
    
    X_test = np.transpose(X_test, (0, 2, 1))
    testloader = DataLoader(TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long()), batch_size=256)
    
    net = CNN_Attention(num_features)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    
    loss, metrics = test(net, testloader, device)
    print(f"Server-side evaluation round {server_round}, f1_score: {metrics['f1_score']}")
    return loss, metrics

# Define server_fn
def server_fn(context: Context):
    print("[server_app.py] server_fn called.")
    # This server_fn does not need to load data itself.
    # The 'evaluate' function handles loading the test set when called by the strategy.
    num_rounds = context.run_config["num-server-rounds"]
    algorithm = context.run_config["algorithm"]
    num_clients = context.run_config["num-clients"]
    clients_per_round = context.run_config["clients-per-round"]
    
    partitions_dir = os.path.join(os.path.dirname(__file__), "..", "partitions")
    with open(os.path.join(partitions_dir, 'num_features.txt'), 'r') as f:
        num_features = int(f.read())
    
    net = CNN_Attention(num_features)
    parameters = ndarrays_to_parameters(get_weights(net))

    fit_config_fn = lambda sr: {
        "learning-rate": context.run_config["learning-rate"],
        "local-epochs": context.run_config["local-epochs"],
        "algorithm": algorithm,
        "mu": context.run_config.get("mu", 0.01),
    }

    strategy = FedAvg(
        fraction_fit=clients_per_round/num_clients, 
        min_available_clients=num_clients,
        initial_parameters=parameters, 
        evaluate_fn=evaluate, 
        on_fit_config_fn=fit_config_fn
        )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


print("[server_app.py] ServerApp ready.")
# Create ServerApp
app = ServerApp(server_fn=server_fn)