"""fed-fall: A Flower / pytorch_msg_api app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg, FedAdam

from .task import Net

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]

    # This number should match what's in your 'partitions/num_features.txt'
    NUM_FEATURES = 10 # <-- IMPORTANT: Set this value manually

    # Load global model
    global_model = Net(num_features=NUM_FEATURES)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    #strategy = FedAvg(fraction_train=fraction_train)

    # Initialize FedAdam strategy instead of FedAvg
    # The parameters eta (server-side learning rate) and beta_1/beta_2 are the standard Adam hyperparameters
    strategy = FedAdam(
        fraction_train=fraction_train,
        eta=0.001,
        beta_1=0.9,
        beta_2=0.999,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
