import torch, copy

def quantize_state(state):
    qstate = {}
    for k, v in state.items():
        v = v.float()
        mn, mx = v.min(), v.max()
        scale = (mx - mn) / 255.0 + 1e-8
        q = torch.clamp(((v - mn) / scale).round(), 0, 255).to(torch.uint8)
        qstate[k] = {'q': q, 'mn': mn, 'scale': scale}
    return qstate

def dequantize_state(qstate):
    state = {}
    for k, d in qstate.items():
        state[k] = (d['q'].float() * d['scale'] + d['mn']).to(torch.float32)
    return state

def topk_state(state, frac=0.2):
    out = {}
    for k, v in state.items():
        flat = v.flatten()
        knum = max(1, int(frac * flat.numel()))
        vals, idx = torch.topk(flat.abs(), knum, sorted=False)
        sparse = torch.zeros_like(flat); sparse[idx] = flat[idx]
        out[k] = sparse.view_as(v)
    return out

def important_mask_state(state, ratio=0.4, prefix='rgb'):
    out = {}
    for k, v in state.items():
        if prefix in k:
            flat = v.flatten()
            knum = max(1, int(ratio * flat.numel()))
            vals, idx = torch.topk(flat.abs(), knum, sorted=False)
            pruned = torch.zeros_like(flat); pruned[idx] = flat[idx]
            out[k] = pruned.view_as(v)
        else:
            out[k] = v
    return out