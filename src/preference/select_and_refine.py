"""Select the best candidate trajectory under a fitted preference scorer, and
optionally nudge it a little further toward higher preference score
(numeric-gradient ascent on the feature score) -- the "select, then slightly
adjust" behavior the advisor described, and what the turn-category sanity check
(utils/turn_multimodal_check.py) showed plain best-of-N alone under-delivers on.
"""
import numpy as np

from src.preference.features import extract_features
from src.preference.bradley_terry import score as bt_score


def select_best(w, candidates, target):
    """candidates: list of (T, 2) waypoint arrays. Returns (best_index, best_score, all_scores)."""
    scores = [bt_score(w, extract_features(c, target)) for c in candidates]
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx], scores


def refine_towards_preference(waypoints, target, w, num_steps=3, lr=0.05, eps=1e-3):
    """Numeric-gradient ascent of `w . features(waypoints)` w.r.t. the waypoints
    themselves, a few small steps. This is a cheap stand-in for classifier-guidance
    during the flow ODE (guiding the *sampling process*); here we guide the already
    -sampled discrete waypoints directly, which needs no access to the flow model.
    """
    wp = np.array(waypoints, dtype=np.float64)
    for _ in range(num_steps):
        grad = np.zeros_like(wp)
        base_score = bt_score(w, extract_features(wp, target))
        for i in range(wp.shape[0]):
            for j in range(wp.shape[1]):
                perturbed = wp.copy()
                perturbed[i, j] += eps
                s = bt_score(w, extract_features(perturbed, target))
                grad[i, j] = (s - base_score) / eps
        norm = np.linalg.norm(grad)
        if norm > 1e-8:
            wp = wp + lr * grad / norm
    return wp


def select_and_refine(w, candidates, target, num_refine_steps=3, refine_lr=0.05):
    best_idx, best_score, scores = select_best(w, candidates, target)
    refined = refine_towards_preference(candidates[best_idx], target, w, num_steps=num_refine_steps, lr=refine_lr)
    return refined, best_idx, scores
