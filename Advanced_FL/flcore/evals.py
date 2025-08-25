import numpy as np

def metrics_from_counts(tp, tn, fp, fn):
    acc = (tp+tn) / max(1, tp+tn+fp+fn)
    sens = tp / max(1, tp+fn)
    spec = tn / max(1, tn+fp)
    f1 = (2*tp) / max(1, 2*tp + fp + fn)
    return dict(accuracy=acc, sensitivity=sens, specificity=spec, f1=f1)