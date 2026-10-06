"""SPLC-style lexicographic preference model, as an alternative to
bradley_terry.py's single linear-weight fit. Instead of combining all
features into one weighted sum, this compares candidates one feature at a
time in a learned priority order: the highest-priority feature decides
unless the two candidates are within an epsilon tie-zone on it, in which
case the next feature in the order breaks the tie (same spirit as SPLC's
collision > goal-progress > risk-exposure hierarchy, ported to our
turn_angle/clearance style features instead of their safety-specific
ones -- safety itself is already handled separately by
src/preference/safety_filter.py's hard filter, upstream of this).

Fit from pairwise data only (diff_features, labels), no architecture
assumption beyond "one feature dominates unless tied":
  - priority order: features ranked by how often their sign alone agrees
    with the recorded winner, across comparisons where it isn't near-tied
    (most consistently decisive feature first).
  - direction: whether winning means a lower or higher value on each
    feature (sign of its mean signed difference on comparisons it decided).
  - epsilon per feature: a fraction of that feature's its own
    diff-magnitude scale in the training data (the tie-zone width).
"""
import numpy as np


def fit_lexicographic(diff_features, labels, tie_frac=0.25):
    """diff_features: (N, D) feat_a - feat_b per comparison. labels: (N,)
    1.0 if a won, 0.0 if b won. Returns dict: order (list of feature
    indices, most consistently decisive first), direction (+1 if winning
    means higher value on that feature, -1 if lower), epsilon (per
    feature, the tie-zone half-width)."""
    X = np.asarray(diff_features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    D = X.shape[1]
    signed = X * (2 * y[:, None] - 1)  # positive = this feature's diff favored the winner

    epsilon = tie_frac * np.std(X, axis=0)
    epsilon = np.where(epsilon < 1e-9, 1e-9, epsilon)

    agreement = []
    direction = np.zeros(D)
    for k in range(D):
        decisive = np.abs(X[:, k]) > epsilon[k]
        if decisive.sum() == 0:
            agreement.append(0.0)
            direction[k] = 1.0
            continue
        agree_rate = float((signed[decisive, k] > 0).mean())
        agreement.append(max(agree_rate, 1 - agree_rate))
        direction[k] = 1.0 if signed[decisive, k].mean() >= 0 else -1.0

    order = list(np.argsort(agreement)[::-1])
    return {"order": order, "direction": direction, "epsilon": epsilon}


def predict_lex(model, feat_a, feat_b):
    """Returns 1.0 if a wins, 0.0 if b wins, 0.5 if every tier tied
    (falls back to the largest-margin tier)."""
    diff = (np.asarray(feat_a) - np.asarray(feat_b)) * model["direction"]
    for k in model["order"]:
        if abs(diff[k]) > model["epsilon"][k]:
            return 1.0 if diff[k] > 0 else 0.0
    k = int(np.argmax(np.abs(diff)))
    return 1.0 if diff[k] >= 0 else 0.0


def best_lex(model, feats):
    """feats: (K, D). Iteratively narrows the candidate set tier by tier:
    at each tier, keeps only candidates within epsilon of that tier's best
    (direction-aware) value among current survivors; the final survivor
    (ties broken by the last tier's raw value) is the pick."""
    feats = np.asarray(feats, dtype=np.float64)
    signed = feats * model["direction"][None, :]
    survivors = np.arange(len(feats))
    for k in model["order"]:
        vals = signed[survivors, k]
        best_val = vals.max()
        keep = survivors[vals >= best_val - model["epsilon"][k]]
        if len(keep) >= 1:
            survivors = keep
        if len(survivors) == 1:
            break
    last_k = model["order"][-1]
    return int(survivors[np.argmax(signed[survivors, last_k])])
