#!/usr/bin/env python
"""Does SPLC-style lexicographic preference fitting (src/preference/lexicographic.py)
actually predict held-out pairwise comparisons better than the existing
single linear-weight Bradley-Terry fit (src/preference/bradley_terry.py),
on our real two decisive features (total_turn_angle_deg, clearance)?

Tested under BOTH assumptions about what the "true" user actually is, not
just the one that would favor either model:
  - linear_truth: true preference is a linear combination of the two
    features (what bradley_terry.py assumes)
  - lex_truth: true preference genuinely checks turn_angle first and only
    falls back to clearance when turn_angle is within some tolerance (what
    SPLC's hierarchy assumes)

For each: generate a synthetic candidate pool (curvy paths + a synthetic
crowd placed in the path's own reach, same spirit as
crowd_citywalker_integration_demo.py), fit both models from N=10 random
training comparisons, then measure prediction accuracy on 50 held-out
random pairs from the SAME pool -- a direct test of which model
generalizes better to comparisons it wasn't trained on.

Usage:
    PYTHONPATH=. python utils/lexicographic_vs_linear.py --repeats 30
"""
import argparse

import numpy as np

from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES
from src.preference.bradley_terry import fit_preference_weights, predict_preference_prob
from src.preference.lexicographic import fit_lexicographic, predict_lex
from utils.preference_poc import make_synthetic_candidates

TURN_IDX = FEATURE_NAMES.index("total_turn_angle_deg")
FEATURE_LABELS = ["total_turn_angle_deg", "clearance"]


def build_pool(rng, num_candidates=20):
    target = np.array([0.0, 5.0])
    candidates = make_synthetic_candidates(target, num_candidates=num_candidates, rng=rng)
    all_wp = np.concatenate([np.asarray(c) for c in candidates], axis=0)
    reach = float(np.mean(np.linalg.norm(all_wp, axis=1)))
    centroid = all_wp.mean(axis=0)
    centroid_dir = centroid / (np.linalg.norm(centroid) + 1e-9)
    normal = np.array([-centroid_dir[1], centroid_dir[0]])
    num_agents = 3
    block_point = centroid_dir * reach * rng.uniform(0.5, 1.0)
    agents = block_point[None, :] + rng.normal(scale=0.3 * reach, size=(num_agents, 1)) * normal[None, :]

    feats = []
    for c in candidates:
        base = extract_features(c, target)
        clearance = agent_avoidance_feature(c, agents)
        clearance = min(clearance, reach)  # cap inf when no agents are ever close
        feats.append([base[TURN_IDX], clearance])
    return np.array(feats, dtype=np.float64)


def true_label_linear(feat_a, feat_b, w_true, rng):
    p = predict_preference_prob(w_true, feat_a, feat_b)
    return 1.0 if rng.random() < p else 0.0


def true_label_lex(feat_a, feat_b, direction, eps, rng, noise=0.05):
    diff = (feat_a - feat_b) * direction
    if abs(diff[0]) > eps[0]:
        winner = 1.0 if diff[0] > 0 else 0.0
    else:
        winner = 1.0 if diff[1] > 0 else 0.0
    return winner if rng.random() > noise else 1.0 - winner


def run_once(rng, truth_kind, n_train=10, n_test=50):
    feats_raw = build_pool(rng)
    scale = feats_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    feats_norm = feats_raw / scale
    K = len(feats_raw)

    if truth_kind == "linear_truth":
        w_true = np.array([-1.0, 1.0])  # prefers less turning AND more clearance
        label_fn = lambda i, j: true_label_linear(feats_raw[i], feats_raw[j], w_true, rng)
    else:
        direction = np.array([-1.0, 1.0])
        eps_true = np.array([0.15 * feats_raw[:, 0].std(), 0.0])
        label_fn = lambda i, j: true_label_lex(feats_raw[i], feats_raw[j], direction, eps_true, rng)

    diffs, labels = [], []
    for _ in range(n_train):
        i, j = rng.choice(K, size=2, replace=False)
        w = label_fn(i, j)
        diff = feats_norm[i] - feats_norm[j] if w == 1.0 else feats_norm[j] - feats_norm[i]
        diffs.append(diff)
        labels.append(1.0)
    diffs = np.array(diffs)
    labels = np.array(labels)

    w_hat, _ = fit_preference_weights(diffs, labels)
    lex_model = fit_lexicographic(diffs, labels)

    n_correct_linear, n_correct_lex = 0, 0
    for _ in range(n_test):
        i, j = rng.choice(K, size=2, replace=False)
        true_w = label_fn(i, j)
        p_linear = predict_preference_prob(w_hat, feats_norm[i], feats_norm[j])
        pred_linear = 1.0 if p_linear > 0.5 else 0.0
        pred_lex = predict_lex(lex_model, feats_norm[i], feats_norm[j])
        n_correct_linear += int(pred_linear == true_w)
        n_correct_lex += int(pred_lex == true_w)

    return n_correct_linear / n_test, n_correct_lex / n_test


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    for truth_kind in ("linear_truth", "lex_truth"):
        acc_linear, acc_lex = [], []
        for rep in range(args.repeats):
            rng = np.random.default_rng(5000 + rep)
            a, b = run_once(rng, truth_kind)
            acc_linear.append(a)
            acc_lex.append(b)
        print(f"=== true user is {truth_kind} ({args.repeats} repeats) ===")
        print(f"  linear Bradley-Terry fit : held-out accuracy {np.mean(acc_linear):.3f} (+-{np.std(acc_linear):.3f})")
        print(f"  lexicographic fit        : held-out accuracy {np.mean(acc_lex):.3f} (+-{np.std(acc_lex):.3f})")


if __name__ == "__main__":
    main()
