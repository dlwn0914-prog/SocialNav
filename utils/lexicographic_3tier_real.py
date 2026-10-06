#!/usr/bin/env python
"""3-tier (total_turn_angle_deg, clearance, passing_side) lexicographic vs
single linear Bradley-Terry fit, on REAL Crowd-FM candidates from the 3
scenarios where passing_side's sign actually flips within the pool
(dense_6, sparse_1, spread_3 -- see src/preference/lexicographic.py's
2-tier version and this session's scenario sweep). The other 9/12
scenarios structurally can't test this: people positioned off to one
side make every candidate curve the "same side" by construction, so
passing_side never has two real options to choose between there.

IMPORTANT CAVEAT, by design, not an oversight: these same 3 scenarios are
exactly the ones where src/preference/safety_filter.py's hard filter
leaves 0 safe candidates (fallback only) -- people central enough for
passing_side to vary are also close enough to violate PRIEST's safety
ellipse for every candidate. This test therefore deliberately skips the
safety filter and uses the raw 20-candidate pool, to isolate "is
passing_side a useful tier at all" from "does the full safety-filtered
pipeline ever reach a scene where it matters" -- those are two separate
questions, and this script only answers the first. Report both findings
together, not just the accuracy numbers.
"""
import argparse
import json

import numpy as np

from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES
from src.preference.bradley_terry import fit_preference_weights, predict_preference_prob
from src.preference.lexicographic import fit_lexicographic, predict_lex

TURN_IDX = FEATURE_NAMES.index("total_turn_angle_deg")
SCENARIOS = ["dense_6", "sparse_1", "spread_3"]
FEATURE_LABELS = ["total_turn_angle_deg", "clearance", "passing_side"]


def passing_side(wp, agents):
    full = np.vstack([[0, 0], wp])
    sides = []
    for agent in agents:
        dists = np.linalg.norm(full - agent[None, :], axis=1)
        k = int(np.argmin(dists))
        k0, k1 = max(k - 1, 0), min(k + 1, len(full) - 1)
        tangent = full[k1] - full[k0]
        if np.linalg.norm(tangent) < 1e-8:
            continue
        tangent = tangent / np.linalg.norm(tangent)
        normal = np.array([-tangent[1], tangent[0]])
        sides.append(np.dot(agent - full[k], normal))
    return float(np.mean(sides)) if sides else 0.0


def load_scenario_feats(name):
    d = json.load(open(f"/home/nuri2/crowd_fm_ws/scenario_{name}.json"))
    cands = [np.array(c, dtype=np.float64) for c in d["candidates"]]
    goal = np.array(d["goal"], dtype=np.float64)
    people = np.array(d["people"], dtype=np.float64)
    people = people[np.linalg.norm(people, axis=1) < 500]

    reach = float(np.mean(np.linalg.norm(np.concatenate(cands, axis=0), axis=1)))
    feats = []
    for c in cands:
        base = extract_features(c, goal)
        clearance = min(agent_avoidance_feature(c, people), reach)
        side = passing_side(c, people)
        feats.append([base[TURN_IDX], clearance, side])
    return np.array(feats, dtype=np.float64)


def true_label_linear(feat_a, feat_b, w_true, rng):
    p = predict_preference_prob(w_true, feat_a, feat_b)
    return 1.0 if rng.random() < p else 0.0


def true_label_lex(feat_a, feat_b, direction, eps, rng, noise=0.05):
    diff = (feat_a - feat_b) * direction
    winner = 0.5
    for k in range(len(diff)):
        if abs(diff[k]) > eps[k]:
            winner = 1.0 if diff[k] > 0 else 0.0
            break
    if winner == 0.5:
        k = int(np.argmax(np.abs(diff)))
        winner = 1.0 if diff[k] >= 0 else 0.0
    return winner if rng.random() > noise else 1.0 - winner


def run_once(rng, feats_raw, truth_kind, n_train=10, n_test=50):
    scale = feats_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    feats_norm = feats_raw / scale
    K = len(feats_raw)

    if truth_kind == "linear_truth":
        w_true = np.array([-1.0, 1.0, 0.8])  # less turning, more clearance, prefers + side
        label_fn = lambda i, j: true_label_linear(feats_raw[i], feats_raw[j], w_true, rng)
    else:
        direction = np.array([-1.0, 1.0, 1.0])
        eps_true = np.array([0.15 * feats_raw[:, 0].std(), 0.15 * feats_raw[:, 1].std(), 0.0])
        label_fn = lambda i, j: true_label_lex(feats_raw[i], feats_raw[j], direction, eps_true, rng)

    diffs, labels = [], []
    for _ in range(n_train):
        i, j = rng.choice(K, size=2, replace=False)
        w = label_fn(i, j)
        diff = feats_norm[i] - feats_norm[j] if w == 1.0 else feats_norm[j] - feats_norm[i]
        diffs.append(diff)
        labels.append(1.0)
    diffs, labels = np.array(diffs), np.array(labels)

    w_hat, _ = fit_preference_weights(diffs, labels)
    lex_model = fit_lexicographic(diffs, labels)

    n_correct_linear, n_correct_lex = 0, 0
    for _ in range(n_test):
        i, j = rng.choice(K, size=2, replace=False)
        true_w = label_fn(i, j)
        pred_linear = 1.0 if predict_preference_prob(w_hat, feats_norm[i], feats_norm[j]) > 0.5 else 0.0
        pred_lex = predict_lex(lex_model, feats_norm[i], feats_norm[j])
        n_correct_linear += int(pred_linear == true_w)
        n_correct_lex += int(pred_lex == true_w)
    return n_correct_linear / n_test, n_correct_lex / n_test, lex_model["order"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=10)
    args = p.parse_args()

    print("feature ranges per scenario (checking passing_side actually varies):")
    pooled = {}
    for name in SCENARIOS:
        feats = load_scenario_feats(name)
        pooled[name] = feats
        print(f"  {name}: " + ", ".join(
            f"{FEATURE_LABELS[k]}=[{feats[:,k].min():.3f},{feats[:,k].max():.3f}]" for k in range(3)))

    for truth_kind in ("linear_truth", "lex_truth"):
        print(f"\n=== true user is {truth_kind} ===")
        all_lin, all_lex = [], []
        for name in SCENARIOS:
            feats = pooled[name]
            lins, lexs, orders = [], [], []
            for rep in range(args.repeats):
                rng = np.random.default_rng(7000 + rep)
                a, b, order = run_once(rng, feats, truth_kind)
                lins.append(a); lexs.append(b); orders.append(tuple(order))
            all_lin.extend(lins); all_lex.extend(lexs)
            from collections import Counter
            print(f"  {name}: linear={np.mean(lins):.3f} lex={np.mean(lexs):.3f}  "
                  f"fitted_order_counts={Counter(orders)}")
        print(f"  POOLED across 3 scenarios: linear={np.mean(all_lin):.3f} lex={np.mean(all_lex):.3f}")


if __name__ == "__main__":
    main()
