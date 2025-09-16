"""fed-fall: A Flower / pytorch_msg_api app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdam
from flwr.common import Metrics

from typing import Callable, Dict, List, Optional, Tuple, Union, Any
from flwr.common import NDArrays
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score
import wandb
from .task import Net, load_data, get_num_features

# ===================================================================
# 1. SERVER-SIDE EVALUATION FUNCTION
# ===================================================================
def get_evaluate_fn(test_data_path: str, num_features: int):
    """Return an evaluation function for server-side evaluation."""
    # Load the centralized test set once
    X_csv, X_img1, X_img2, y = torch.load(test_data_path, weights_only=False)
    testloader = DataLoader(TensorDataset(
        torch.from_numpy(X_csv).float(), torch.from_numpy(X_img1).float(),
        torch.from_numpy(X_img2).float(), torch.from_numpy(y).long()
    ), batch_size=64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, parameters: NDArrays, config: Dict[str, str]):
        model = Net(num_csv_features=num_features)
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.to(device)

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        all_labels, all_predicted = [], []
        with torch.no_grad():
            for x_csv, x_img1, x_img2, labels in testloader:
                x_csv, x_img1, x_img2, labels = x_csv.to(device), x_img1.to(device), x_img2.to(device), labels.to(device)
                outputs = model(x_csv, x_img1, x_img2)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = loss / len(testloader) if len(testloader) > 0 else 0.0
        f1 = f1_score(all_labels, all_predicted, pos_label=0, zero_division=0)
        precision = precision_score(all_labels, all_predicted, pos_label=0, zero_division=0)
        recall = recall_score(all_labels, all_predicted, pos_label=0, zero_division=0)
        
        return avg_loss, {"accuracy": accuracy, "f1_score": f1, "precision": precision, "recall": recall}
    return evaluate

# ===================================================================
# 2. ALL-IN-ONE CUSTOM STRATEGY FOR LOGGING & SERVER-SIDE EVALUATION
# ===================================================================
class CustomStrategy(FedAdam):
    def __init__(self, *args, **kwargs):
        self.server_eval_fn = kwargs.pop("evaluate_fn", None)
        super().__init__(*args, **kwargs)

    def evaluate(self, server_round: int, parameters: NDArrays):
        if self.server_eval_fn:
            loss, metrics = self.server_eval_fn(server_round, parameters, {})
            print(f"Round {server_round} SERVER-SIDE evaluation: loss {loss}, metrics {metrics}")
            wandb_metrics = {f"server_{k}": v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=server_round)
        return super().evaluate(server_round, parameters)

# ===================================================================
# 3. SERVER APP
# ===================================================================
app = ServerApp()
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    config = context.run_config
    num_rounds = config["num-server-rounds"]
    
    wandb.init(project="federated-fall-detection", name=f"Final-Run-Rounds-{num_rounds}", config=config)

    NUM_FEATURES = get_num_features()
    global_model = Net(num_csv_features=NUM_FEATURES)
    arrays = ArrayRecord(global_model.state_dict())
    
    test_data_path = "/home/syed/PhD/UP-Fall-FL/FL_Apps/fed-fall/fed_fall/UP_Fall_partitions/test.pt"
    server_evaluate_fn = get_evaluate_fn(test_data_path, NUM_FEATURES)

    # Instantiate our new all-in-one custom strategy
    strategy = CustomStrategy(
        fraction_train=config["fraction-train"],
        fraction_evaluate=1.0,
        eta=config.get("eta", 0.001),
        beta_1=config.get("beta_1", 0.9),
        beta_2=config.get("beta_2", 0.999),
        evaluate_fn=server_evaluate_fn,  # Pass the function to our custom strategy
    )

    result = strategy.start(grid=grid, initial_arrays=arrays, train_config=ConfigRecord({"lr": config["lr"]}), num_rounds=num_rounds)
    
    wandb.finish()

    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")