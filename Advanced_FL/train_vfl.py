import yaml, argparse, torch, numpy as np
from flcore.datasets import load_upfall
from flcore.models import IMUEncoder, RGBEncoder
from flcore.utils import set_seed

class SplitHead(torch.nn.Module):
    def __init__(self, hid=64, num_classes=2):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2*hid, hid), torch.nn.ReLU(),
            torch.nn.Linear(hid, num_classes)
        )
    def forward(self, hA, hB): return self.net(torch.cat([hA, hB], dim=-1))

def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg['experiment']['seed'])
    device = cfg['experiment']['device'] if torch.cuda.is_available() else 'cpu'
    X_imu, X_rgb, y, subj = load_upfall(args.data_dir, modalities=('imu','rgb'))
    # normalize IMU
    mu = X_imu.mean(dim=(0,2), keepdim=True); std = X_imu.std(dim=(0,2), keepdim=True)+1e-6
    X_imu = (X_imu - mu) / std

    A = IMUEncoder(in_ch=X_imu.size(1), hid=cfg['vfl']['split_hidden']).to(device)
    B = RGBEncoder(in_dim=X_rgb.size(1), hid=cfg['vfl']['split_hidden']).to(device)
    H = SplitHead(hid=cfg['vfl']['split_hidden'], num_classes=2).to(device)
    optA = torch.optim.Adam(A.parameters(), lr=1e-3)
    optB = torch.optim.Adam(B.parameters(), lr=1e-3)
    optH = torch.optim.Adam(H.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    idx = np.arange(len(y)); np.random.shuffle(idx)
    X_imu = X_imu[idx]; X_rgb = X_rgb[idx]; y = y[idx]
    for epoch in range(10):
        for i in range(0, len(y), 64):
            imu = X_imu[i:i+64].to(device)
            rgb = X_rgb[i:i+64].to(device)
            yy  = y[i:i+64].to(device)
            hA = A(imu)     # Party A
            hB = B(rgb)     # Party B
            logits = H(hA, hB)  # aggregator
            loss = loss_fn(logits, yy)
            optA.zero_grad(); optB.zero_grad(); optH.zero_grad()
            loss.backward(); optA.step(); optB.step(); optH.step()
        print(f"[VFL] epoch {epoch}, loss={loss.item():.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--data_dir', type=str, default='data/upfall_proc')
    args = ap.parse_args()
    main(args)