# %% [markdown]
# # FL-Fall Worked Example
# - Synthetic sanity check
# - UP-FALL preprocessing
# - HFL training with multimodal residual fusion, DP, personalization, Pareto selection, robust agg
# - Ablations and metrics

# %%
import os, yaml, torch, numpy as np
from flcore.utils import set_seed
from flcore.datasets import load_synthetic, load_upfall
from flcore.models import MultimodalFD
from flcore.selection import pareto_front
from flcore.server import aggregate
from flcore.client import FLClient

# %% [markdown]
# ## 1) Synthetic sanity check

# %%
set_seed(1337)
X_imu, X_rgb, y, subj = load_synthetic()
mu = X_imu.mean(dim=(0,2), keepdim=True); std = X_imu.std(dim=(0,2), keepdim=True)+1e-6
X_imu = (X_imu - mu) / std

# %%
cfg = yaml.safe_load(open('config.yaml'))
cfg['experiment']['rounds'] = 5
cfg['experiment']['clients_per_round'] = 5
cfg['experiment']['total_clients'] = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# %%
# Make clients
from flcore.datasets import IMURGBDataset
def make_clients_quick():
    N_clients = cfg['experiment']['total_clients']
    subjects = np.unique(subj)
    assign = {s: i % N_clients for i, s in enumerate(subjects)}
    client_idxs = {i: [] for i in range(N_clients)}
    for i in range(len(y)):
        client_idxs[assign[subj[i]]].append(i)
    clients = []
    for cid in range(N_clients):
        idxs = np.array(client_idxs[cid], dtype=int)
        m = np.random.rand(len(idxs)) < 0.8
        tr = idxs[m]; va = idxs[~m]
        ds_train = IMURGBDataset(X_imu[tr], X_rgb[tr], y[tr])
        ds_val   = IMURGBDataset(X_imu[va], X_rgb[va], y[va])
        def model_fn():
            return MultimodalFD(
                imu_ch=cfg['model']['imu_channels'],
                rgb_dim=cfg['model']['rgb_feat_dim'],
                hid=cfg['model']['hidden_dim'],
                fusion=cfg['model']['fusion'],
                adapters=cfg['model']['adapters'],
                num_classes=2, device=device)
        clients.append(FLClient(cid, model_fn, {'train':ds_train,'val':ds_val}, cfg, device))
    return clients

clients = make_clients_quick()
global_model = MultimodalFD(imu_ch=6, rgb_dim=34, hid=128, fusion='residual_gate', adapters=True).to(device)
global_state = global_model.state_dict()

# %%
for rnd in range(cfg['experiment']['rounds']):
    sel = pareto_front(clients)
    sel = np.random.choice(sel, size=min(cfg['experiment']['clients_per_round'], len(sel)), replace=False)
    updates = []
    for c in sel:
        updates.append(c.local_train(global_state, epochs=cfg['experiment']['local_epochs']))
    global_state = aggregate(updates, global_state, method='trimmed_mean', trim_frac=0.1, max_norm=5.0)
    m = $$c.validate(global_state)['f1'$$ for c in np.random.choice(clients, size=min(5, len(clients)), replace=False)]
    print(f"Round {rnd} mean F1={np.mean(m):.3f}")

# %% [markdown]
# ## 2) UP-FALL preprocessing (run once in a terminal)
# ```bash
# python data_prep/upfall_prep.py --raw_root /path/to/UP-FALL --out_dir data/upfall_proc \
#   --imu_hz 100 --window_sec 2.0 --stride_sec 1.0
# ```

# %% [markdown]
# ## 3) HFL training on UP-FALL
# Run:
# ```bash
# python train_hfl.py --config config.yaml --data_dir data/upfall_proc
# ```

# %% [markdown]
# ## 4) Ablations
# - Change fusion: early/late/residual/residual_gate
# - Toggle adapters or ELM
# - Toggle DP (`privacy.enable_dp`)
# - Switch aggregator: fedavg/median/trimmed_mean
# - Comm compression toggles in config.yaml