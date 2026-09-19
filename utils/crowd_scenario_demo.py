#!/usr/bin/env python
"""End-to-end demo: "6 people blocking a corridor" preference-learning pipeline,
entirely offline (no live robot navigation needed -- sidesteps the nav2
controller instability found tonight).

1. A realistic crowd layout (6 people clustered mid-corridor), sized to match
   the actual map_empty world (30m x 23m, confirmed from
   arena4_ws/src/arena/simulation-setup/worlds/map_empty/map/map.yaml).
2. Three candidate paths from src/preference/crowd_path_generator.py:
   direct (through the crowd), moderate_detour, wide_detour.
3. Three synthetic user profiles with different true preferences:
   crowd_avoider -> wide_detour, path_shortener -> direct, balanced -> moderate_detour.
4. Bradley-Terry pairwise comparisons generated per profile (N=10 each, with
   replacement over the 3 candidate pairs), weights fit with
   src/preference/bradley_terry.py, and checked against the true ranking.

Usage:
    PYTHONPATH=/home/nuri2/SocialNav python3 utils/crowd_scenario_demo.py
"""
import numpy as np

from src.preference.crowd_path_generator import generate_candidate_paths, to_ego_frame
from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES
from src.preference.bradley_terry import fit_preference_weights, predict_preference_prob

RNG = np.random.default_rng(0)

# Map is 30m x 23m (map_empty/map/map.yaml). Robot crosses the room along a
# corridor; 6 people are clustered in the middle of that corridor.
START = np.array([2.0, 11.5])
GOAL = np.array([28.0, 11.5])
AGENTS = np.array(
    [
        [14.0, 10.2],
        [14.5, 11.0],
        [15.2, 11.8],
        [13.8, 12.3],
        [15.5, 10.6],
        [14.2, 12.9],
    ]
)

FULL_FEATURE_NAMES = FEATURE_NAMES + ["min_dist_to_agents"]

PROFILES = {
    "crowd_avoider": {  # cares about staying away from people, tolerates a longer path
        "path_length": -0.1,
        "directness": 0.0,
        "total_turn_angle_deg": -0.05,
        "final_heading_error_deg": -0.2,
        "lateral_deviation": 0.0,
        "endpoint_dist_to_target": -0.3,
        "target_alignment_cos": 0.3,
        "min_dist_to_agents": 2.0,
    },
    "path_shortener": {  # cares about getting there fast, indifferent to crowd distance
        "path_length": -1.5,
        "directness": 1.5,
        "total_turn_angle_deg": -0.1,
        "final_heading_error_deg": -0.2,
        "lateral_deviation": -0.1,
        "endpoint_dist_to_target": -0.3,
        "target_alignment_cos": 0.3,
        "min_dist_to_agents": 0.05,
    },
    "balanced": {  # moderate weight on both
        "path_length": -0.5,
        "directness": 0.5,
        "total_turn_angle_deg": -0.1,
        "final_heading_error_deg": -0.2,
        "lateral_deviation": -0.05,
        "endpoint_dist_to_target": -0.3,
        "target_alignment_cos": 0.3,
        "min_dist_to_agents": 0.6,
    },
}

EXPECTED_BEST = {
    "crowd_avoider": "wide_detour",
    "path_shortener": "direct",
    # NOTE: with this path generator's convex clearance-vs-length tradeoff,
    # "moderate_detour" is never a linear-utility optimum among these 3
    # candidates (going wider is *more* length-efficient at buying clearance
    # here, not less) -- so any profile lands on an endpoint. The real test
    # is fitted == true, not fitted == a specific candidate name.
    "balanced": None,
}


def profile_to_vector(profile):
    return np.array([profile[n] for n in FULL_FEATURE_NAMES], dtype=np.float64)


def full_features(waypoints_ego, target_ego, agents_ego):
    base = extract_features(waypoints_ego, target_ego)
    dist = agent_avoidance_feature(waypoints_ego, agents_ego)
    return np.concatenate([base, [dist]])


def main():
    candidates_world = generate_candidate_paths(START, GOAL, AGENTS, num_steps=5)
    names = list(candidates_world.keys())

    agents_ego = to_ego_frame(AGENTS, START)
    target_ego = to_ego_frame(GOAL[None, :], START)[0]

    feats = {}
    for name, wp_world in candidates_world.items():
        wp_ego = to_ego_frame(wp_world, START)
        feats[name] = full_features(wp_ego, target_ego, agents_ego)
        print(f"{name}: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(FULL_FEATURE_NAMES, feats[name])))

    print()
    for profile_name, profile in PROFILES.items():
        w_true = profile_to_vector(profile)

        # true ranking under this profile
        true_scores = {n: float(np.dot(w_true, feats[n])) for n in names}
        true_best = max(true_scores, key=true_scores.get)

        # N=10 pairwise comparisons among the 3 candidates, Bradley-Terry sampled
        diff_feats, labels = [], []
        for _ in range(10):
            a, b = RNG.choice(names, size=2, replace=False)
            p_a_over_b = predict_preference_prob(w_true, feats[a], feats[b])
            if RNG.random() < p_a_over_b:
                diff_feats.append(feats[a] - feats[b])
            else:
                diff_feats.append(feats[b] - feats[a])
            labels.append(1.0)

        w_hat = fit_preference_weights(np.array(diff_feats), np.array(labels), l2=1.0, lr=0.2, num_steps=800)
        hat_scores = {n: float(np.dot(w_hat, feats[n])) for n in names}
        hat_best = max(hat_scores, key=hat_scores.get)

        expected = EXPECTED_BEST[profile_name]
        fit_matches_truth = "OK" if hat_best == true_best else "FIT MISMATCH"
        expected_note = "" if expected is None else (" (matches expectation)" if true_best == expected else " (NOTE: true preference differs from naive expectation, see comment above)")
        print(f"=== {profile_name} ===")
        print(f"  true-profile best: {true_best}{expected_note} | fitted-scorer best: {hat_best} -> {fit_matches_truth}")
        print(f"  true_scores:   " + ", ".join(f"{n}={v:.2f}" for n, v in true_scores.items()))
        print(f"  fitted_scores: " + ", ".join(f"{n}={v:.2f}" for n, v in hat_scores.items()))
        print()


if __name__ == "__main__":
    main()
