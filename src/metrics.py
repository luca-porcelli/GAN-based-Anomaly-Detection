import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support

def evaluate_scores(y_true, scores):
    y_true = np.asarray(y_true)
    scores  = np.asarray(scores)
    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["roc_auc"] = None
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, scores>np.median(scores), average="binary", zero_division=0)
    out["precision"], out["recall"], out["f1"] = float(prec), float(rec), float(f1)
    return out
