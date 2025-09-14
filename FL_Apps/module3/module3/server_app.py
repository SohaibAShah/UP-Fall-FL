"""Module3: A Flower / PyTorch app."""

from collections import OrderedDict
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import flwr as fl
from flwr.common import NDArrays, Metrics
import numpy as np

from .models import GatedResidualFusionModel

def get_evaluate_fn(test_data, device, num_features):
    """Return an evaluation function for server-side evaluation."""
    def evaluate(server_round: int, parameters: NDArrays, config) -> tuple[float, Metrics]:
        model = GatedResidualFusionModel(num_features)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)})
        model.load_state_dict(state_dict)
        model.to(device); model.eval()
        
        test_loader = DataLoader(TensorDataset(
            torch.from_numpy(test_data[0]).float(), torch.from_numpy(test_data[1]).float(),
            torch.from_numpy(test_data[2]).float(), torch.from_numpy(test_data[3]).long()
        ), batch_size=128)
        
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x_csv, x_img1, x_img2, y in test_loader:
                outputs = model(x_csv.to(device), x_img1.to(device), x_img2.to(device))
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(y.numpy())
        
        report = classification_report(all_labels, all_preds, target_names=['Fall', 'No Fall'], output_dict=True, zero_division=0)
        
        metrics = {
            "accuracy": report['accuracy'],
            "f1_score_fall": report['Fall']['f1-score'],
            "precision_fall": report['Fall']['precision'],
            "recall_fall": report['Fall']['recall'],
        }
        # Loss is not calculated here as it's not the primary metric for comparison
        return 0.0, metrics
    return evaluate
