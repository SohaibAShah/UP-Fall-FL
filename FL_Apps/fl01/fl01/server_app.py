import flwr as fl
from flwr.common import Context, ndarrays_to_parameters, parameters_to_ndarrays, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import numpy as np
import torch
import logging

from .task import CNN_Attention, get_test_loader, get_num_features, test, get_weights, load_data

# Custom SCAFFOLD Strategy
class Scaffold(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.server_control_variate = [np.zeros_like(p) for p in self.initial_parameters.tensors]

    def configure_fit(self, server_round, parameters, client_manager):
        # --- TRACE ---
        logging.info(f"[Server Strategy] Configuring fit for round {server_round}...")
        config_list = super().configure_fit(server_round, parameters, client_manager)
        for _, config in config_list:
            config["server_cv"] = self.server_control_variate
        return config_list

    def aggregate_fit(self, server_round, results, failures):
        # --- TRACE ---
        logging.info(f"[Server Strategy] Aggregating {len(results)} results for round {server_round}...")
        aggregated_deltas, _ = super().aggregate_fit(server_round, results, failures)
        if aggregated_deltas is not None:
            current_params = parameters_to_ndarrays(self.initial_parameters)
            new_params = [current + delta for current, delta in zip(current_params, parameters_to_ndarrays(aggregated_deltas))]
            aggregated_parameters = ndarrays_to_parameters(new_params)
            self.initial_parameters = aggregated_parameters

        if results:
            avg_cv_delta = [np.mean([res.metrics["cv_delta"][i] for _, res in results], axis=0) for i in range(len(self.server_control_variate))]
            self.server_control_variate = [scv + cvd for scv, cvd in zip(self.server_control_variate, avg_cv_delta)]
        
        # --- TRACE ---
        logging.info(f"[Server Strategy] Aggregation complete for round {server_round}.")
        return aggregated_parameters, {}

# Server-side evaluation function
def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    # --- TRACE ---
    logging.info(f"[Server] Starting server-side evaluation for round {server_round}.")
    
    net = CNN_Attention(get_num_features())
    testloader = get_test_loader()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
    
    loss, metrics = test(net, testloader, device)
    
    # --- TRACE ---
    logging.info(f"[Server] Finished server-side evaluation for round {server_round}. F1-Score: {metrics['f1_score']:.4f}")
    return loss, metrics

# Define server_fn
def server_fn(context: Context):
    # --- TRACE ---
    logging.info("[Server Factory] Initializing server...")
    
    num_clients = context.run_config["num-clients"]
    load_data(partition_id=0, num_partitions=num_clients)

    num_rounds = context.run_config["num-server-rounds"]
    algorithm = context.run_config["algorithm"]
    clients_per_round = context.run_config["clients-per-round"]
    
    net = CNN_Attention(get_num_features())
    parameters = ndarrays_to_parameters(get_weights(net))

    fit_config_fn = lambda sr: {
        "learning-rate": context.run_config["learning-rate"],
        "local-epochs": context.run_config["local-epochs"],
        "algorithm": algorithm,
        "mu": context.run_config.get("mu", 0.01),
    }

    if algorithm == "scaffold":
        strategy = Scaffold(
            fraction_fit=clients_per_round/num_clients,
            min_available_clients=num_clients,
            initial_parameters=parameters,
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config_fn
        )
    else: # FedAvg or FedProx
        strategy = FedAvg(
            fraction_fit=clients_per_round/num_clients,
            min_available_clients=num_clients,
            initial_parameters=parameters,
            evaluate_fn=evaluate,
            on_fit_config_fn=fit_config_fn
        )

    config = ServerConfig(num_rounds=num_rounds)
    
    # --- TRACE ---
    logging.info(f"[Server Factory] Server initialized with strategy: {algorithm.upper()}")
    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)