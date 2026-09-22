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
        degrees vs. meters) get comparable weights.

    Returns: (w, scale), both (D,) arrays. `w` is in *normalized* space (it
    was fit against `diff_features / scale`) and is returned as-is -- NOT
    divided back by `scale` into the original feature space. An earlier
    version did that final `w / scale` un-normalization, which blows up
    whenever some feature's comparison-diff std is small: dividing by a small
    number inflates that dimension's raw-space weight arbitrarily, regardless
    of how many/which features are used (seen repeatedly across
    utils/crowd_scenario_demo.py, utils/preference_integration_demo.py and
    utils/crowd_citywalker_integration_demo.py -- reducing feature count just
    moved the blowup to a *different* dimension each time, confirming the
    normalize/un-normalize step itself was the bug, not feature selection).

    Callers must score/compare using the *same* `scale`: divide any raw
    feature vector by `scale` before dotting it with `w` -- see score()'s and
    predict_preference_prob()'s `scale` argument, and
    src/preference/select_and_refine.py's `scale` passthrough. A hand-specified
    ground-truth `w_true` (not fit by this function) stays in raw feature
    space and should be used with `scale=None` (the default).
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

    return w, scale


def predict_preference_prob(w, feat_a, feat_b, scale=None):
    """P(A preferred over B) under the fitted weights. Pass `scale` (as
    returned by fit_preference_weights) when `w` is in normalized space;
    leave it None (default) for a hand-specified raw-space `w_true`."""
    diff = np.asarray(feat_a, dtype=np.float64) - np.asarray(feat_b, dtype=np.float64)
    if scale is not None:
        diff = diff / np.asarray(scale, dtype=np.float64)
    return float(_sigmoid(np.dot(w, diff)))


def score(w, feats, scale=None):
    """Scalar preference score of a single candidate's feature vector. Pass
    `scale` when `w` is in normalized space; leave it None (default) for a
    hand-specified raw-space `w_true`."""
    feats = np.asarray(feats, dtype=np.float64)
    if scale is not None:
        feats = feats / np.asarray(scale, dtype=np.float64)
    return float(np.dot(w, feats))
