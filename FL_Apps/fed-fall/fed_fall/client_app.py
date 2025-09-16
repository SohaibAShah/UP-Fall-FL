"""fed-fall: A Flower / pytorch_msg_api app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from .task import Net, load_data, ConvLSTMNet
from .task import test as test_fn
from .task import train as train_fn

# Flower ClientApp
app = ClientApp()

if torch.cuda.is_available():
    print(f"[client_app.py] Using GPU: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
else:
    print("[client_app.py] Using CPU")
    device = torch.device("cpu")

@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

       # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, num_features = load_data(partition_id=partition_id)


    # Load the model and initialize it with the received weights
    model = ConvLSTMNet(num_features=num_features)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    model.to(device)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

       # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader, num_features = load_data(partition_id=partition_id)

     # Load the model and initialize it with the received weights
    model = ConvLSTMNet(num_features=num_features)
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Call the evaluation function
    eval_loss, eval_acc, advanced_metrics = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
        "F1-score": advanced_metrics["f1_score"],
        "Precision": advanced_metrics["precision"],
        "Recall": advanced_metrics["recall"]
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
