"""Module3: A Flower / PyTorch app."""

from collections import OrderedDict

import flwr as fl
import torch
import logging
import pandas as pd
import matplotlib.pyplot as plt
import os, sys
import argparse

from module3.dataset import load_and_partition_data
from module3.models import GatedResidualFusionModel
from module3.client_app import FallDetectionClient
from module3.server_app import get_evaluate_fn
from module3.strategy import get_strategy
from flwr.common import Parameters
from flwr.server import ServerConfig

def setup_logging(log_dir):
    """Configures logging to save to a file and print to the console."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.setLevel(logging.INFO)

    # File handler to save logs
    file_handler = logging.FileHandler(os.path.join(log_dir, "fl_run_log.txt"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Stream handler to print to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

def main(args):
    # Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("output", exist_ok=True)
    setup_logging("output")

    # Load data
    client_partitions, test_data, num_features = load_and_partition_data(args.data_path)
    client_ids = list(client_partitions.keys())

    # --- Configuration ---
    config = {
        "learning_rate": 0.001, "local_epochs": 3,
        "fl_rounds": 20, "clients_per_round": 5, "num_clients": len(client_ids)
    }

    # --- Client Function ---
    def client_fn(cid: str) -> FallDetectionClient:
        return FallDetectionClient(cid, client_partitions[cid], DEVICE, num_features)

    # --- Initial Parameters ---
    initial_model = GatedResidualFusionModel(num_features)
    initial_params = [p.cpu().detach().numpy() for p in initial_model.parameters()]
    initial_parameters = fl.common.ndarrays_to_parameters(initial_params)

    # --- Evaluation Function ---
    evaluate_fn = get_evaluate_fn(test_data, DEVICE, num_features)
    
    # --- Run Experiment ---
    logging.info(f"--- Running Federated Learning Experiment: {args.algorithm.upper()} ---")
    strategy = get_strategy(args.algorithm, initial_parameters, evaluate_fn, config)
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(client_ids),
        config=fl.server.ServerConfig(num_rounds=config['fl_rounds']),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0.5} if DEVICE.type == "cuda" else {"num_cpus": 2}
    )

    # --- Save and Plot Results ---
    results_df = pd.DataFrame(history.metrics_distributed['f1_score_fall'], columns=['round', 'f1_score'])
    results_df.to_csv(f"output/{args.algorithm}_results.csv", index=False)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['round'], results_df['f1_score'], marker='o')
    plt.title(f'Global Model F1-Score vs. Rounds ({args.algorithm.upper()})')
    plt.xlabel('Communication Round'); plt.ylabel('F1-Score (Fall)')
    plt.grid(True); plt.tight_layout()
    plt.savefig(f"output/{args.algorithm}_convergence.png")
    
    logging.info(f"✅ Simulation for {args.algorithm.upper()} complete. Results saved to 'output' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fall Detection FL Experiments")
    parser.add_argument("--algorithm", type=str, default="fedavg", choices=["fedavg", "fedprox", "scaffold"], help="FL algorithm to run")
    parser.add_argument("--data_path", type=str, default="/home/syed/PhD/UP_Fall_Dataset/Sensor + Image/", help="Path to the dataset directory")
    args = parser.parse_args()
    main(args)
