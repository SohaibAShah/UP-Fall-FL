import flwr as fl
from flwr.common import Parameters

def get_strategy(algorithm_name: str, initial_params: Parameters, evaluate_fn, config: dict):
    """Returns the appropriate Flower strategy based on the algorithm name."""
    
    fit_config = {
        "learning_rate": config['learning_rate'],
        "local_epochs": config['local_epochs'],
        "algorithm": algorithm_name,
        "mu": config.get('mu', 0.01) # For FedProx
    }

    if algorithm_name in ["fedavg", "fedprox"]:
        strategy = fl.server.strategy.FedAvg(
            initial_parameters=initial_params,
            min_fit_clients=config['clients_per_round'],
            min_available_clients=config['num_clients'],
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=lambda sr: fit_config,
        )
    # elif algorithm_name == "scaffold":
        # The custom Scaffold strategy would be defined and returned here
        # pass 
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
    return strategy