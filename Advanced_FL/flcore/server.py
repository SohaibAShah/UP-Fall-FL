import torch
from .compression import quantize_state, dequantize_state, topk_state, important_mask_state

def _state_to_vec(state): return torch.cat([p.flatten() for _, p in state.items()])
def _vec_to_state(vec, ref_state):
    out, idx = {}, 0
    for k, p in ref_state.items():
        n = p.numel()
        out[k] = vec[idx:idx+n].view_as(p).clone()
        idx += n
    return out

def aggregate(updates, base_state, method='fedavg', trim_frac=0.1, max_norm=None):
    keys = list(base_state.keys())
    stacked = torch.stack([_state_to_vec(u) for u in updates], dim=0)
    if max_norm is not None:
        norms = torch.norm(stacked - _state_to_vec(base_state), dim=1)
        keep = norms <= max_norm
        if keep.sum() > 0:
            stacked = stacked[keep]
    if method == 'fedavg':
        agg_vec = stacked.mean(dim=0)
    elif method == 'median':
        agg_vec = stacked.median(dim=0).values
    elif method == 'trimmed_mean':
        N = stacked.size(0); k = int(trim_frac * N)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        take = sorted_vals[k: N-k if N-2*k>0 else N]
        agg_vec = take.mean(dim=0)
    else:
        raise ValueError(method)
    return _vec_to_state(agg_vec, base_state)

def apply_comm_policies(update, cfg, heavy_prefix='rgb'):
    out = {k: v.clone() for k, v in update.items()}
    if cfg['comm'].get('important_weights', False):
        out = important_mask_state(out, ratio=cfg['comm']['important_ratio'], prefix=heavy_prefix)
    if cfg['comm'].get('topk_frac', 0) > 0:
        out = topk_state(out, frac=cfg['comm']['topk_frac'])
    if cfg['comm'].get('quantize_8bit', False):
        out = quantize_state(out)
    return out

def undo_comm_policies(update):
    if isinstance(list(update.values())⸨[0]⸩, dict) and 'q' in list(update.values())⸨[0]⸩:
        return dequantize_state(update)
    return update