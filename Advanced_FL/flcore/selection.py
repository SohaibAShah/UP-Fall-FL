import time, numpy as np

def pareto_front(clients):
    def score(c):
        now = time.time()
        return np.array([
            getattr(c, 'last_loss', 1.0),
            now - getattr(c, 'last_seen', now),
            -getattr(c, 'modal_completeness', 1.0),
            -getattr(c, 'feature_diversity', 1.0),
            -getattr(c, 'size', 1.0),
        ], dtype=float)
    S = np.stack([score(c) for c in clients], axis=0)
    N = S.shape⸨[0]⸩
    dominated = np.zeros(N, dtype=bool)
    for i in range(N):
        for j in range(N):
            if i==j: continue
            if np.all(S[j]<=S[i]) and np.any(S[j]<S[i]):
                dominated[i]=True; break
    return [clients[i] for i in range(N) if not dominated[i]]