import torch
from torch import nn

class Server(object):
    """Server for federated learning, handling model aggregation and evaluation.
    
    Args:
        model (nn.Module): Global model.
        eval_dataset (dict): Test dataset for evaluation.
        num_clients (int): Number of clients to select per round.
    """
    def __init__(self, model, eval_dataset, num_clients):
        self.global_model = model
        self.num_clients = num_clients
        self.eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=32)
	
    def model_aggregate(self, weight_accumulator):
        """Aggregate client model updates into the global model.
        
        Args:
            weight_accumulator (dict): Accumulated weight differences from clients.
        """
        for name, data in self.global_model.state_dict().items():
            update_per_layer = weight_accumulator[name] * (1 / self.num_clients)  # Average
            if data.type() != update_per_layer.type():
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    def model_eval(self):
        """Evaluate the global model on the test dataset.
        
        Returns:
            tuple: (accuracy, loss) as percentages and average loss.
        """
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data = batch[0]
            target = torch.squeeze(batch[1]).int()
            target = torch.tensor(target, dtype=torch.int64)

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            
            output = self.global_model(data)
            total_loss += nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            dataset_size += data.size()[0]

        acc = 100.0 * (float(correct) / float(dataset_size))
        loss = total_loss / dataset_size
        return acc, loss