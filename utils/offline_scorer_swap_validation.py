#!/usr/bin/env python
"""Small offline validation, before touching Crowd-FM's actual model code:
does swapping its generic pretrained TrajectoryScorer's pick for a
Bradley-Terry scorer learned from N<=10 active-query feedback (restricted
to safety-filtered candidates) actually change which candidate gets
selected -- and does the new pick move toward what the simulated
personalized user actually wants?

Pipeline under test, all offline, using REAL Crowd-FM candidates/scores
already extracted (crowd_fm_ws/extract_real_candidates.py) -- no changes
to Crowd-FM's own code yet:

  1. "original" pick: argmax of Crowd-FM's own pretrained scorer_scores
     (the score the real TrajectoryScorer assigned), restricted to the
     safety-filtered safe set (src/preference/safety_filter.py) -- this
     is what ros_interface.py's plan() would pick today.
  2. "personalized" pick: ActiveQueryStrategy (src/preference/active_query.py)
     asks up to N<=10 pairwise questions (answered by a simulated
     CROWD_AVOIDER_PROFILE "true" user, same as earlier scripts this
     session) over the SAME safe set, fits a Bradley-Terry theta_hat, and
     picks argmax(feats @ theta_hat) -- this is what the proposed
     CFM -> safety filter -> N<=10 personalized scorer pipeline would pick.
  3. "true best": argmax of the simulated true preference over the safe
     set -- the ground truth the personalized pick is trying to recover.

Reports, per scenario: whether original != personalized (swap actually
changes the decision) and whether personalized == true_best (the learned
scorer recovers what the simulated user actually wants), plus each pick's
raw (turn_angle, target_alignment, clearance) features so the difference
is concrete, not just an index.

Usage:
    PYTHONPATH=. python utils/offline_scorer_swap_validation.py \
        --scenarios offside_3 offside_4_right offside_5_wide
"""
import argparse
import json

import numpy as np

from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import ActiveQueryStrategy
from src.preference.safety_filter import filter_safe_candidates

REDUCED_FEATURE_NAMES = ["total_turn_angle_deg", "target_alignment_cos", "min_dist_to_agents"]
_BASE_IDX = {n: FEATURE_NAMES.index(n) for n in REDUCED_FEATURE_NAMES if n in FEATURE_NAMES}

CROWD_AVOIDER_PROFILE = {
    "total_turn_angle_deg": -0.05,
    "target_alignment_cos": 0.3,
    "min_dist_to_agents": 2.0,
}


def candidate_reach_scale(candidates):
    all_wp = np.concatenate([np.asarray(c, dtype=np.float64) for c in candidates], axis=0)
    return float(np.mean(np.linalg.norm(all_wp, axis=1)))


def reduced_features(waypoints, target, agents, reach):
    base = extract_features(waypoints, target)
    dist = agent_avoidance_feature(waypoints, agents)
    return np.array(
        [
            base[_BASE_IDX["total_turn_angle_deg"]],
            base[_BASE_IDX["target_alignment_cos"]],
            dist / max(reach, 1e-6),
        ],
        dtype=np.float64,
    )


def run_one(name, path, n_shot, rng_seed):
    with open(path) as f:
        data = json.load(f)

    candidates = [np.array(c, dtype=np.float64) for c in data["candidates"]]
    goal = np.array(data["goal"], dtype=np.float64)
    people = np.array(data["people"], dtype=np.float64)
    static_obs = np.array(data["static_obstacles"], dtype=np.float64)
    people = people[np.linalg.norm(people, axis=1) < 500]
    pretrained_scores = np.array(data["scorer_scores"], dtype=np.float64)

    filt = filter_safe_candidates(candidates, static_obs=static_obs, dynamic_obs=people)
    safe_idx = filt["safe_indices"]
    print(f"\n=== {name} === ({len(safe_idx)} safe candidates"
          f"{' [fallback]' if filt['used_fallback'] else ''})")

    if len(safe_idx) < 2:
        print("fewer than 2 safe candidates -- can't compare original vs personalized pick")
        return None

    safe_candidates = [candidates[i] for i in safe_idx]
    reach = candidate_reach_scale(safe_candidates)
    feats_raw = np.array([reduced_features(c, goal, people, reach) for c in safe_candidates])
    safe_pretrained_scores = pretrained_scores[safe_idx]

    original_local = int(np.argmax(safe_pretrained_scores))
    original_global = safe_idx[original_local]

    w_true = np.array([CROWD_AVOIDER_PROFILE[n] for n in REDUCED_FEATURE_NAMES])
    scale = feats_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    feats_norm = feats_raw / scale

    rng = np.random.default_rng(rng_seed)
    strat = ActiveQueryStrategy(rng=rng)
    n_queryable = len(safe_candidates)
    budget = min(n_shot, n_queryable - 1)
    for _ in range(budget):
        if strat.done():
            break
        i, j = strat.select(feats_norm)
        p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
        winner = i if rng.random() < p_i else j
        strat.update(feats_norm, i, j, winner)

    personalized_local = strat.best(feats_norm)
    personalized_global = safe_idx[personalized_local]

    true_local = int(np.argmax(feats_raw @ w_true))
    true_global = safe_idx[true_local]

    def fmt(idx_local):
        f = feats_raw[idx_local]
        return {"turn_angle_deg": round(float(f[0]), 2),
                "target_alignment_cos": round(float(f[1]), 3),
                "clearance_ratio": round(float(f[2]), 3)}

    result = {
        "scenario": name,
        "n_safe": len(safe_idx),
        "queries_used": budget,
        "original_pick": {"global_idx": int(original_global), **fmt(original_local),
                           "pretrained_score": round(float(safe_pretrained_scores[original_local]), 3)},
        "personalized_pick": {"global_idx": int(personalized_global), **fmt(personalized_local)},
        "true_best": {"global_idx": int(true_global), **fmt(true_local)},
        "swap_changed_decision": bool(original_global != personalized_global),
        "personalized_matches_true_best": bool(personalized_global == true_global),
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", default="/home/nuri2/crowd_fm_ws")
    p.add_argument("--scenarios", nargs="+",
                    default=["offside_3", "offside_4_right", "offside_5_wide", "offside_2_left", "offside_3_left"])
    p.add_argument("--n-shot", type=int, default=10)
    p.add_argument("--output", default="offline_scorer_swap_validation.json")
    args = p.parse_args()

    results = []
    for name in args.scenarios:
        path = f"{args.scenario_dir}/scenario_{name}.json"
        r = run_one(name, path, args.n_shot, rng_seed=hash(name) % (2**31))
        if r is not None:
            results.append(r)

    n_changed = sum(r["swap_changed_decision"] for r in results)
    n_matches_true = sum(r["personalized_matches_true_best"] for r in results)
    print(f"\n=== summary over {len(results)} testable scenarios ===")
    print(f"swap changed the decision in {n_changed}/{len(results)} scenarios")
    print(f"personalized pick matched the simulated true-best in {n_matches_true}/{len(results)} scenarios")

    with open(args.output, "w") as f:
        json.dump({"results": results, "n_changed": n_changed, "n_matches_true": n_matches_true}, f, indent=2)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
