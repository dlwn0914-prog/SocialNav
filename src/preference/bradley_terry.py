"""Few-shot preference-weight fitting via a linear Bradley-Terry model.

Given a small number of pairwise comparisons "trajectory A preferred over
trajectory B", fit a weight vector w such that

    P(A preferred over B) = sigmoid(w . (features(A) - features(B)))

This is the same model DPO/RLHF-style preference learning uses, just without a
neural reward model -- linear-in-features so it converges from very few (N<=10)
comparisons and needs no gradient descent through the navigation backbone.
"""
import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fit_preference_weights(diff_features, labels, l2=1.0, lr=0.1, num_steps=500, w_init=None, normalize=True):
    """diff_features: (N, D) array, features(A) - features(B) per comparison.
    labels: (N,) array of 1.0 if A was preferred, 0.0 if B was preferred.
    l2: L2 regularization strength (important for few-shot stability -- with
        N<D comparisons the unregularized MLE is not identified).
    normalize: rescale each feature dimension by its std across `diff_features`
        before fitting, so features with very different natural scales (e.g.
        degrees vs. meters) get comparable weights. The returned `w` is in the
        *original* (unnormalized) feature space, so `score()`/`select_best()`
        keep using raw `extract_features()` output unchanged.
    Returns: w, (D,) array.
    """
    diff_features = np.asarray(diff_features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n, d = diff_features.shape

    if normalize:
        scale = np.std(diff_features, axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        x = diff_features / scale
    else:
        scale = np.ones(d)
        x = diff_features

    w = np.zeros(d) if w_init is None else np.array(w_init, dtype=np.float64) * scale

    for _ in range(num_steps):
        logits = x @ w
        p = _sigmoid(logits)
        grad = x.T @ (labels - p) / n - l2 * w / n
        w = w + lr * grad

    return w / scale


def predict_preference_prob(w, feat_a, feat_b):
    """P(A preferred over B) under the fitted weights."""
    return float(_sigmoid(np.dot(w, np.asarray(feat_a) - np.asarray(feat_b))))


def score(w, feats):
    """Scalar preference score of a single candidate's feature vector."""
    return float(np.dot(w, feats))
