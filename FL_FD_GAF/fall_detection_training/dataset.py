from torch.utils.data import Dataset

class MyDataset(Dataset):
    """Custom Dataset for loading preprocessed fall detection data.
    
    Args:
        input (list): List of input data (e.g., fused GAF and camera images).
        label (list): List of corresponding labels.
        transform (callable, optional): Optional transform to apply to samples.
    """
    def __init__(self, input, label, transform=None):
        self.input = input
        self.label = label
        self.transform = transform

    def __len__(self):
        return len(self.input)
    
    def __getitem__(self, index):
        sample = self.input[index], self.label[index]
        if self.transform:
            sample = self.transform(sample)
        return sample