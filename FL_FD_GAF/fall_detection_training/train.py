import random
import torch
from server import Server
from client import Client

def train_federated_model(server, clients, num_epochs, num_clients, max_acc=80.0):
    """Train the federated learning model.
    
    Args:
        server (Server): Server instance with global model.
        clients (list): List of Client instances.
        num_epochs (int): Number of federated learning rounds.
        num_clients (int): Number of clients to select per round.
        max_acc (float): Accuracy threshold for saving the best model.
    
    Returns:
        float: Best accuracy achieved.
    """
    best_acc = max_acc
    for e in range(num_epochs):
        candidates = random.sample(clients, num_clients)  # Randomly select clients
        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)  # Initialize accumulator
        
        for c in candidates:
            diff = c.local_train(server.global_model)  # Train local model
            for name, params in server.global_model.state_dict().items():
                weight_accumulator[name].add_(diff[name])  # Update accumulator
        
        server.model_aggregate(weight_accumulator)  # Aggregate global model
        acc, loss = server.model_eval()
        print(f"Epoch {e}, global_acc: {acc:.4f}, global_loss: {loss:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(server.global_model.state_dict(), 'model.pth')
            print("Saved model with accuracy:", acc)
    
    return best_acc