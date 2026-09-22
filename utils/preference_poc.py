#!/usr/bin/env python
"""Proof-of-concept: does the few-shot Bradley-Terry preference scorer
(src/preference/) actually recover a synthetic user's preference from a
handful of pairwise comparisons, and does selecting/refining by it pick
trajectories that synthetic user would actually prefer?

This is deliberately decoupled from the SocialNav VLM/flow-matching model --
it validates the new preference-fitting machinery in isolation, on synthetic
candidate trajectories, before it gets wired into the (expensive) full
generation pipeline. Matches the advisor's "verify the idea with something
simple first" guidance.

Usage:
    PYTHONPATH=. python utils/preference_poc.py
"""
import numpy as np

from src.preference.features import extract_features, FEATURE_NAMES, NUM_FEATURES
from src.preference.bradley_terry import fit_preference_weights, predict_preference_prob
from src.preference.select_and_refine import select_best, refine_towards_preference

RNG = np.random.default_rng(0)

# Two synthetic user "personalities" with clearly different priorities, expressed
# as ground-truth weights over the same feature axes (see features.py FEATURE_NAMES).
PROFILES = {
    "smooth_and_direct": {
        # likes short, direct, low-turn, low-deviation, on-target paths
        "path_length": -1.0,
        "directness": 2.0,
        "total_turn_angle_deg": -1.5,
        "final_heading_error_deg": -1.0,
        "lateral_deviation": -1.5,
        "endpoint_dist_to_target": -2.0,
        "target_alignment_cos": 1.5,
    },
    "wide_berth": {
        # tolerates detours / lateral deviation, cares less about directness,
        # still wants to end up near the target
        "path_length": -0.2,
        "directness": -0.3,
        "total_turn_angle_deg": 0.2,
        "final_heading_error_deg": -0.5,
        "lateral_deviation": 1.5,
        "endpoint_dist_to_target": -2.0,
        "target_alignment_cos": 0.5,
    },
}


def profile_to_vector(profile):
    return np.array([profile[name] for name in FEATURE_NAMES], dtype=np.float64)


def make_synthetic_candidates(target, num_candidates=20, num_steps=5, rng=RNG):
    """Random curvy paths toward-ish a target, some direct, some detouring,
    to stand in for what a multi-modal flow policy would sample."""
    candidates = []
    target = np.asarray(target, dtype=np.float64)
    for _ in range(num_candidates):
        directness_bias = rng.uniform(0.2, 1.0)
        turn_noise = rng.uniform(0.0, 25.0)
        step_dir = target / (np.linalg.norm(target) + 1e-6)
        wp = []
        pos = np.zeros(2)
        for t in range(1, num_steps + 1):
            target_step = (target * t / num_steps) - pos
            noise_angle = np.radians(rng.normal(0, turn_noise))
            rot = np.array([[np.cos(noise_angle), -np.sin(noise_angle)], [np.sin(noise_angle), np.cos(noise_angle)]])
            step = directness_bias * (rot @ target_step) + (1 - directness_bias) * step_dir * (
                np.linalg.norm(target) / num_steps
            )
            pos = pos + step
            wp.append(pos.copy())
        candidates.append(np.array(wp))
    return candidates


def sample_pairwise_comparisons(w_true, candidates, target, n_comparisons, rng=RNG):
    feats = [extract_features(c, target) for c in candidates]
    diff_feats, labels = [], []
    for _ in range(n_comparisons):
        i, j = rng.choice(len(candidates), size=2, replace=False)
        p_i_over_j = predict_preference_prob(w_true, feats[i], feats[j])
        label = 1.0 if rng.random() < p_i_over_j else 0.0
        if label == 1.0:
            diff_feats.append(feats[i] - feats[j])
        else:
            diff_feats.append(feats[j] - feats[i])
        labels.append(1.0)  # diff_feats is always (preferred - other)
    return np.array(diff_feats), np.array(labels)


def run_profile(name, profile, target=(0.0, 5.0), n_shot=10, n_holdout_candidates=20):
    w_true = profile_to_vector(profile)
    train_candidates = make_synthetic_candidates(target, num_candidates=12)
    diff_feats, labels = sample_pairwise_comparisons(w_true, train_candidates, target, n_shot)

    w_hat, scale = fit_preference_weights(diff_feats, labels, l2=1.0, lr=0.2, num_steps=800)

    # w_hat is in normalized space (see bradley_terry.py); compare against
    # w_true in that same space (w_true * scale) rather than raw, otherwise
    # this cosine similarity is comparing two different unit systems.
    w_true_normalized = w_true * scale
    cos_sim = float(
        np.dot(w_true_normalized, w_hat) / (np.linalg.norm(w_true_normalized) * np.linalg.norm(w_hat) + 1e-8)
    )

    holdout = make_synthetic_candidates(target, num_candidates=n_holdout_candidates)
    true_best_idx, _, true_scores = select_best(w_true, holdout, target)
    hat_best_idx, _, hat_scores = select_best(w_hat, holdout, target, scale=scale)
    top1_agreement = int(true_best_idx == hat_best_idx)
    true_rank_of_hat_pick = int(np.sum(np.array(true_scores) > true_scores[hat_best_idx]))

    refined = refine_towards_preference(
        holdout[hat_best_idx], target, w_hat, num_steps=5, lr=0.1, scale=scale, candidates=holdout,
    )
    score_before = np.dot(w_hat, extract_features(holdout[hat_best_idx], target) / scale)
    score_after = np.dot(w_hat, extract_features(refined, target) / scale)
    true_score_before = np.dot(w_true, extract_features(holdout[hat_best_idx], target))
    true_score_after = np.dot(w_true, extract_features(refined, target))

    print(f"\n=== profile: {name} (n_shot={n_shot}) ===")
    print("w_true:", dict(zip(FEATURE_NAMES, np.round(w_true, 2))))
    print("w_hat :", dict(zip(FEATURE_NAMES, np.round(w_hat, 2))))
    print(f"cosine(w_true, w_hat) = {cos_sim:.3f}")
    print(f"top-1 agreement on {n_holdout_candidates} held-out candidates: {top1_agreement}")
    print(f"true-preference rank of the fitted-scorer's pick (0 = actually the best): {true_rank_of_hat_pick}")
    print(f"refine step (own scorer): score {score_before:.3f} -> {score_after:.3f}")
    print(f"refine step (true profile's own judgement of the same move): {true_score_before:.3f} -> {true_score_after:.3f}")

    return {
        "cos_sim": cos_sim,
        "top1_agreement": top1_agreement,
        "true_rank_of_hat_pick": true_rank_of_hat_pick,
    }


def main():
    print(f"feature axes: {FEATURE_NAMES}")
    all_results = {}
    for n_shot in (5, 10, 20):
        for name, profile in PROFILES.items():
            key = f"{name}_n{n_shot}"
            all_results[key] = run_profile(name, profile, n_shot=n_shot)

    print("\n=== summary across n_shot in {5,10,20} ===")
    for n_shot in (5, 10, 20):
        cos_sims = [all_results[f"{name}_n{n_shot}"]["cos_sim"] for name in PROFILES]
        agree = [all_results[f"{name}_n{n_shot}"]["top1_agreement"] for name in PROFILES]
        print(f"n_shot={n_shot}: mean cosine(w_true,w_hat)={np.mean(cos_sims):.3f}  mean top1_agreement={np.mean(agree):.2f}")


if __name__ == "__main__":
    main()
