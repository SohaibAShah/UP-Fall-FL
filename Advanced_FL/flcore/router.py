import torch
class IMUGate:
    def __init__(self, thresh=0.6):
        self.thresh = thresh
    def skip(self, imu=None, rgb=None):
        if imu is None: return False
        var = torch.var(imu, dim=(1,2)).mean().item()
        return var < self.thresh