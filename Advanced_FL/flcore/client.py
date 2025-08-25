import torch, time, random
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from .utils import to_device, one_hot
from .router import IMUGate
from .models import MultimodalFD, ELMHead

class FLClient:
    def __init__(self, cid, model_fn, dataset, cfg, device):
        self.cid = cid
        self.cfg = cfg
        self.device = device
        self.dataset = dataset
        bs = cfg['experiment']['batch_size']
        self.train_loader = DataLoader(dataset['train'], batch_size=bs, shuffle=True, drop_last=True)
        self.unlab_loader = DataLoader(dataset.get('unlabeled', []), batch_size=bs, shuffle=False)
        self.val_loader = DataLoader(dataset['val'], batch_size=bs, shuffle=False)
        self.model = model_fn().to(device)
        self.elm = ELMHead(in_dim=cfg['model']['hidden_dim'], num_classes=2, device=device) if cfg['model'].get('elm_head', False) else None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg['experiment']['lr'])
        self.privacy_engine = None
        if cfg['privacy'].get('enable_dp', False):
            self.privacy_engine = PrivacyEngine()
            self.model, self.optimizer, self.train_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=cfg['privacy']['noise_multiplier'],
                max_grad_norm=cfg['privacy']['clip_norm'],
            )
        self.gate = IMUGate(thresh=cfg['model']['gate_threshold'])
        # selection metrics
        self.last_loss = 1.0; self.last_seen = time.time()
        self.modal_completeness = getattr(dataset['train'], 'modal_completeness', 1.0)
        self.size = len(dataset['train']); self.feature_diversity = getattr(dataset['train'], 'feature_diversity', 1.0)

    def _fusion_mask(self):
        p = self.cfg['model']['modality_dropout_p']
        mask = {'rgb': 1}
        if random.random() < p:
            mask['rgb'] = 0
        return mask

    def local_train(self, global_state, epochs=None):
        self.model.load_state_dict(global_state, strict=False)
        self.model.train()
        epochs = epochs or self.cfg['experiment']['local_epochs']
        loss_fn = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        for _ in range(epochs):
            for batch in self.train_loader:
                imu, rgb, y = to_device(batch, self.device)
                mask = self._fusion_mask()
                if rgb is not None and self.gate.skip(imu=imu):
                    rgb = None
                self.optimizer.zero_grad()
                logits = self.model(imu, rgb, fusion_mask=mask)
                loss = loss_fn(logits, y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        self.last_loss = total_loss / max(1, len(self.train_loader) * epochs)
        self.last_seen = time.time()

        # Semi-supervised pseudo-labeling
        if self.cfg['semi_supervised']['enable'] and len(self.unlab_loader) > 0:
            self.model.eval()
            with torch.no_grad():
                for batch in self.unlab_loader:
                    imu, rgb = to_device(batch, self.device)
                    p = torch.softmax(self.model(imu, rgb), dim=-1)
                    conf, pl = p.max(dim=-1)
                    sel = conf > self.cfg['semi_supervised']['pl_threshold']
                    if sel.sum() > 0:
                        self.model.train()
                        self.optimizer.zero_grad()
                        loss = loss_fn(self.model(imu[sel], None if rgb is None else rgb[sel]), pl[sel])
                        loss.backward(); self.optimizer.step()

        # ELM personalization
        if self.elm is not None:
            self.model.eval()
            feats, labels = [], []
            with torch.no_grad():
                for i, batch in enumerate(self.train_loader):
                    imu, rgb, y = to_device(batch, self.device)
                    h = self.model.imu(imu)
                    feats.append(h); labels.append(y)
                    if (i+1)*self.cfg['experiment']['batch_size'] > 1024: break
            if feats:
                H = torch.cat(feats, dim=0)
                Y = one_hot(torch.cat(labels, dim=0), num_classes=2).to(H.device)
                self.elm.online_update(H, Y)

        update = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        return update

    def validate(self, global_state=None):
        if global_state is not None:
            self.model.load_state_dict(global_state, strict=False)
        self.model.eval()
        correct = total = tp = tn = fp = fn = 0
        with torch.no_grad():
            for batch in self.val_loader:
                imu, rgb, y = to_device(batch, self.device)
                logits = self.model(imu, rgb)
                if self.elm is not None:
                    h = self.model.imu(imu)
                    logits = logits + self.elm(h)
                pred = logits.argmax(dim=-1)
                correct += (pred==y).sum().item(); total += y.numel()
                tp += ((pred==1)&(y==1)).sum().item(); tn += ((pred==0)&(y==0)).sum().item()
                fp += ((pred==1)&(y==0)).sum().item(); fn += ((pred==0)&(y==1)).sum().item()
        acc = correct/total if total else 0.0
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        f1 = (2*tp) / (2*tp + fp + fn + 1e-8)
        return {'accuracy': acc, 'f1': f1, 'sensitivity': sens, 'specificity': spec}