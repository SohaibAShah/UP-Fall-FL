import argparse, yaml, numpy as np, torch, copy, random
from flcore.datasets import load_synthetic, load_upfall, IMURGBDataset, subject_wise_split
from flcore.models import MultimodalFD
from flcore.client import FLClient
from flcore.server import aggregate, apply_comm_policies, undo_comm_policies
from flcore.selection import pareto_front
from flcore.utils import set_seed

def make_clients(cfg, device, X_imu, X_rgb, y, subj):
    N_clients = cfg['experiment']['total_clients']
    subjects = np.unique(subj)
    rng = np.random.default_rng(cfg['experiment']['seed'])
    rng.shuffle(subjects)
    assign = {s: i % N_clients for i, s in enumerate(subjects)}
    client_idxs = {i: [] for i in range(N_clients)}
    for i in range(len(y)):
        client_idxs[assign[subj[i]]].append(i)

    clients = []
    for cid in range(N_clients):
        idxs = np.array(client_idxs[cid], dtype=int)
        if len(idxs) < 10: continue
        m = np.random.rand(len(idxs)) < 0.8
        tr = idxs[m]; va = idxs[~m]
        ds_train = IMURGBDataset(X_imu[tr], (None if X_rgb is None else X_rgb[tr]), y[tr], modal_completeness=float(X_rgb is not None), device=device)
        ds_val   = IMURGBDataset(X_imu[va], (None if X_rgb is None else X_rgb[va]), y[va], modal_completeness=float(X_rgb is not None), device=device)
        dataset = {'train': ds_train, 'val': ds_val}

        def model_fn():
            return MultimodalFD(
                imu_ch=cfg['model']['imu_channels'],
                rgb_dim=cfg['model']['rgb_feat_dim'],
                hid=cfg['model']['hidden_dim'],
                fusion=cfg['model']['fusion'],
                adapters=cfg['model']['adapters'],
                num_classes=2, device=device)
        clients.append(FLClient(cid, model_fn, dataset, cfg, device))
    return clients

def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg['experiment']['seed'])
    device = cfg['experiment']['device'] if torch.cuda.is_available() else 'cpu'

    if args.synthetic:
        X_imu, X_rgb, y, subj = load_synthetic()
    else:
        X_imu, X_rgb, y, subj = load_upfall(args.data_dir, modalities=cfg['data']['modalities'])
        mu = X_imu.mean(dim=(0,2), keepdim=True); std = X_imu.std(dim=(0,2), keepdim=True)+1e-6
        X_imu = (X_imu - mu) / std

    clients = make_clients(cfg, device, X_imu, X_rgb, y, subj)

    model = MultimodalFD(
        imu_ch=cfg['model']['imu_channels'],
        rgb_dim=cfg['model']['rgb_feat_dim'],
        hid=cfg['model']['hidden_dim'],
        fusion=cfg['model']['fusion'],
        adapters=cfg['model']['adapters'],
        num_classes=2, device=device).to(device)
    global_state = model.state_dict()

    for rnd in range(cfg['experiment']['rounds']):
        cand = pareto_front(clients) if cfg['selection']['enable_pareto'] else clients
        selected = random.sample(cand, k=min(cfg['experiment']['clients_per_round'], len(cand)))

        updates, f1s = [], []
        for c in selected:
            update = c.local_train(copy.deepcopy(global_state), epochs=cfg['experiment']['local_epochs'])
            update = apply_comm_policies(update, cfg)
            update = undo_comm_policies(update)
            updates.append(update)
            f1s.append(c.validate(global_state)['f1'])

        global_state = aggregate(
            updates, global_state,
            method=cfg['robust']['aggregator'],
            trim_frac=cfg['robust']['trim_frac'],
            max_norm=cfg['robust']['max_update_norm'])

        # quick validation on a small random set
        val_metrics = $$c.validate(global_state) for c in random.sample(clients, k=min(10, len(clients)))$$
        mean_f1 = np.mean([m['f1'] for m in val_metrics])
        print(f"[Round {rnd:03d}] mean F1={mean_f1:.3f} | selected avg F1={np.mean(f1s):.3f} | clients={len(selected)}")

    torch.save(global_state, f"{cfg['experiment']['name']}_final.pt")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config.yaml')
    ap.add_argument('--synthetic', type=int, default=0)
    ap.add_argument('--data_dir', type=str, default='data/upfall_proc')
    args = ap.parse_args()
    main(args)