import torch
import torch.nn as nn
import torch.nn.functional as F

class IMUEncoder(nn.Module):
    def __init__(self, in_ch=6, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, hid, 5, padding=2), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):  # B x C x T
        return self.net(x).squeeze(-1)

class RGBEncoder(nn.Module):
    def __init__(self, in_dim=34, hid=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, hid), nn.ReLU()
        )
    def forward(self, x):  # B x F
        return self.net(x)

class ResidualFusion(nn.Module):
    def __init__(self, hid=128, gated=True):
        super().__init__()
        self.proj = nn.Linear(hid, hid)
        self.gated = gated
        if gated:
            self.gate = nn.Sequential(
                nn.Linear(2*hid, hid), nn.ReLU(),
                nn.Linear(hid, 1)
            )
    def forward(self, h_imu, h_rgb):
        add = self.proj(h_rgb)
        if self.gated:
            g = torch.sigmoid(self.gate(torch.cat([h_imu, h_rgb], dim=-1)))
            add = add * g
        return h_imu + add

class Adapter(nn.Module):
    def __init__(self, hid=128, bottleneck=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hid, bottleneck), nn.ReLU(),
            nn.Linear(bottleneck, hid)
        )
    def forward(self, h):
        return h + self.net(h)

class ELMHead(nn.Module):
    def __init__(self, in_dim=128, num_classes=2, lam=1e-2, device='cpu'):
        super().__init__()
        self.in_dim = in_dim
        self.num_classes = num_classes
        self.lam = lam
        self.register_buffer('P', torch.eye(in_dim, device=device) / lam)
        self.beta = nn.Parameter(torch.zeros(in_dim, num_classes, device=device), requires_grad=False)

    @torch.no_grad()
    def online_update(self, h, y_onehot):
        for i in range(h.size(0)):
            hi = h[i:i+1].T
            Pi = self.P
            denom = (1 + (hi.T @ Pi @ hi)).item()
            k = (Pi @ hi) / denom
            self.P = Pi - k @ (hi.T @ Pi)
            err = (y_onehot[i:i+1].T - self.beta.T @ hi).T
            self.beta += (hi @ err)

    def forward(self, h):
        return h @ self.beta

class MultimodalFD(nn.Module):
    def __init__(self, imu_ch=6, rgb_dim=34, hid=128, fusion='residual_gate', adapters=False, num_classes=2, device='cpu'):
        super().__init__()
        self.has_rgb = bool(rgb_dim and rgb_dim > 0)
        self.imu = IMUEncoder(imu_ch, hid)
        if self.has_rgb:
            self.rgb = RGBEncoder(rgb_dim, hid)
            if fusion == 'residual':
                self.fuse = ResidualFusion(hid, gated=False)
            elif fusion == 'residual_gate':
                self.fuse = ResidualFusion(hid, gated=True)
            else:
                self.fuse = None
        self.fusion = fusion
        self.adapters = adapters
        if adapters:
            self.adapter = Adapter(hid)
        self.head = nn.Linear(hid, num_classes)

    def forward(self, imu_x, rgb_x=None, fusion_mask=None):
        h_imu = self.imu(imu_x)
        if self.has_rgb and (rgb_x is not None) and (fusion_mask is None or fusion_mask.get('rgb',1)==1):
            h_rgb = self.rgb(rgb_x)
            if self.fusion in ['residual', 'residual_gate']:
                h = self.fuse(h_imu, h_rgb)
            elif self.fusion == 'early':
                h = torch.cat([h_imu, h_rgb], dim=-1)
                h = F.relu(nn.Linear(h.size(-1), h_imu.size(-1)).to(h.device)(h))
            elif self.fusion == 'late':
                z_imu = self.head(h_imu if not self.adapters else self.adapter(h_imu))
                z_rgb = self.head(h_rgb if not self.adapters else self.adapter(h_rgb))
                return (z_imu + z_rgb) / 2
        else:
            h = h_imu
        if self.adapters:
            h = self.adapter(h)
        return self.head(h)