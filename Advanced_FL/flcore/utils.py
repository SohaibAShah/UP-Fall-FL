import torch, random, numpy as np

def to_device(batch, device):
    if len(batch)==3:
        imu, rgb, y = batch
        imu = None if imu is None else imu.to(device)
        rgb = None if rgb is None else rgb.to(device)
        y = y.to(device)
        return imu, rgb, y
    else:
        imu, rgb = batch
        imu = None if imu is None else imu.to(device)
        rgb = None if rgb is None else rgb.to(device)
        return imu, rgb

def one_hot(y, num_classes=2):
    return torch.nn.functional.one_hot(y, num_classes=num_classes).float()

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)