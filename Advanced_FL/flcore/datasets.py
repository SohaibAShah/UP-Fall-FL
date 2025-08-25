import os, numpy as np, torch, random
from torch.utils.data import Dataset

class IMURGBDataset(Dataset):
    def __init__(self, imu, rgb, y, modal_completeness=1.0, device='cpu'):
        self.imu = imu    # tensor B x C x T
        self.rgb = rgb    # tensor B x F or None
        self.y = y.long()
        self.modal_completeness = float(modal_completeness)
        self.feature_diversity = float(torch.var(imu).item())
        self.device = device
    def __len__(self): return self.y.size(0)
    def __getitem__(self, i):
        return self.imu[i], (None if self.rgb is None else self.rgb[i]), self.y[i]

def load_synthetic(n=6000, T=200, C=6, F=34, fall_ratio=0.2, subjects=30):
    X_imu, X_rgb, y, subj = [], [], [], []
    for s in range(subjects):
        bias = np.random.randn(C) * 0.1
        for i in range(n//subjects):
            is_fall = (random.random()<fall_ratio)
            imu = np.random.randn(C, T)*0.2 + bias[:,None]
            if is_fall:
                t0 = np.random.randint(T//4, 3*T//4)
                imu[:, t0:t0+5] += np.random.randn(C,1)*3.0
            rgb = np.random.randn(F) * (1.0 if is_fall else 0.5) + (0.5 if is_fall else 0.0)
            X_imu.append(imu); X_rgb.append(rgb); y.append(int(is_fall)); subj.append(s)
    X_imu = torch.tensor(np.stack(X_imu), dtype=torch.float32)
    X_rgb = torch.tensor(np.stack(X_rgb), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.long)
    subj = np.array(subj)
    return X_imu, X_rgb, y, subj

def subject_wise_split(subj, train_frac=0.7, val_frac=0.1, seed=1337):
    rng = np.random.default_rng(seed)
    uniq = np.unique(subj)
    rng.shuffle(uniq)
    n_tr = int(len(uniq)*train_frac); n_val = int(len(uniq)*val_frac)
    tr_subj = set(uniq[:n_tr]); val_subj=set(uniq[n_tr:n_tr+n_val]); te_subj=set(uniq[n_tr+n_val:])
    def mask(S): return np.array([s in S for s in subj])
    return mask(tr_subj), mask(val_subj), mask(te_subj)

def load_upfall(data_dir, modalities=('imu','rgb')):
    imu = torch.tensor(np.load(os.path.join(data_dir, 'imu.npy')), dtype=torch.float32)
    y   = torch.tensor(np.load(os.path.join(data_dir, 'y.npy')), dtype=torch.long)
    subj= np.load(os.path.join(data_dir, 'subj.npy'))
    rgb = None
    if 'rgb' in modalities and os.path.exists(os.path.join(data_dir, 'rgb.npy')):
        rgb = torch.tensor(np.load(os.path.join(data_dir, 'rgb.npy')), dtype=torch.float32)
    return imu, rgb, y, subj