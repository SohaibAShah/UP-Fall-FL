import torch
from torch import nn

class Client(object):
    """Client for federated learning, handling local model training.
    
    Args:
        model (nn.Module): Local model copy.
        train_dataset (dict): Training dataset for the client.
        id (int): Client ID.
    """
    def __init__(self, model, train_dataset, id=-1):
        self.local_model = model
        self.client_id = id
        self.train_dataset = train_dataset
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset[id], batch_size=32)

    def local_train(self, global_model):
        """Train the local model and compute weight differences.
        
        Args:
            global_model (nn.Module): Global model to compute differences against.
        
        Returns:
            dict: Weight differences between local and global models.
        """
        for name, param in global_model.state_dict().items():
            self.local_model.state_dict()[name].copy_(param.clone())
        
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.001, momentum=0.0001)
        self.local_model.train()
        
        for e in range(3):  # Local epochs
            for batch_id, batch in enumerate(self.train_loader):
                data = batch[0]
                target = torch.squeeze(batch[1]).int()
                target = torch.tensor(target, dtype=torch.int64)

                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = self.local_model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
            
        diff = {}
        for name, data in self.local_model.state_dict().items():
            diff[name] = data - global_model.state_dict()[name]
        
        return diff