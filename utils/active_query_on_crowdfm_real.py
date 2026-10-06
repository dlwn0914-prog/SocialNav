#!/usr/bin/env python
"""Does active query selection (UserAlign) actually help on REAL Crowd-FM
candidates, not just preference_poc.py's synthetic ones?

utils/active_vs_random_query.py validated RandomQueryStrategy vs
ActiveQueryStrategy (src/preference/active_query.py) only on synthetic
made-up candidate pools. This script runs the same comparison on the K=20
trajectory candidates Crowd-FM's OWN pipeline actually produced for a real
crowd scenario: generate_trajectories_for_timestep (flow model) ->
TrajectoryScorer -> PriestPlanner.run_optimization (collision/smoothness
refinement) -- extracted via crowd_fm_ws/extract_real_candidates.py (run
inside the crowd_fm_dev Docker container, GPU) and dumped to
crowd_fm_ws/real_candidates.json.

Same reduced 3-feature space and CROWD_AVOIDER_PROFILE "true" preference as
utils/crowd_citywalker_integration_demo.py (total_turn_angle_deg,
target_alignment_cos, min_dist_to_agents/reach), and the same
scale-normalization fix active_vs_random_query.py already validated
(fit in per-dimension-std-normalized space, compare against w_true rescaled
into that space) -- both root-caused from the same bradley_terry.py bug this
session's earlier work fixed.

Usage:
    PYTHONPATH=. python utils/active_query_on_crowdfm_real.py \
        --candidates-json /home/nuri2/crowd_fm_ws/real_candidates.json --repeats 30
"""
import argparse
import json

import numpy as np

from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import RandomQueryStrategy, ActiveQueryStrategy

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


def run_strategy(strategy, feats_raw, feats_norm, w_true, w_true_scaled_norm, true_best_idx, max_queries, rng):
    cos_track, top1_track = [], []
    used = 0
    for _ in range(max_queries):
        if strategy.done():
            break
        i, j = strategy.select(feats_norm)
        p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
        winner = i if rng.random() < p_i else j
        strategy.update(feats_norm, i, j, winner)
        used += 1

        theta = strategy.theta_hat
        if theta is not None and np.linalg.norm(theta) > 1e-8:
            cos = float(np.dot(w_true_scaled_norm, theta) / (np.linalg.norm(w_true_scaled_norm) * np.linalg.norm(theta)))
        else:
            cos = 0.0
        cos_track.append(cos)
        top1_track.append(int(strategy.best(feats_norm) == true_best_idx))
    while len(cos_track) < max_queries:
        cos_track.append(cos_track[-1] if cos_track else 0.0)
        top1_track.append(top1_track[-1] if top1_track else 0)
    return cos_track, top1_track, used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates-json", default="/home/nuri2/crowd_fm_ws/real_candidates.json")
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--max-queries", type=int, default=10)
    p.add_argument("--output", default="active_query_on_crowdfm_real.json")
    args = p.parse_args()

    with open(args.candidates_json) as f:
        data = json.load(f)

    candidates = [np.array(c, dtype=np.float64) for c in data["candidates"]]
    goal = np.array(data["goal"], dtype=np.float64)
    people = np.array(data["people"], dtype=np.float64)
    reach = candidate_reach_scale(candidates)

    print(f"loaded {len(candidates)} real Crowd-FM candidates (flow+scorer+PRIEST-refined)")
    print(f"goal={goal}, {len(people)} people, candidate_reach_scale={reach:.3f}")

    feats_raw = np.array([reduced_features(c, goal, people, reach) for c in candidates])
    print("feature stats (raw):")
    for k, name in enumerate(REDUCED_FEATURE_NAMES):
        print(f"  {name}: min={feats_raw[:,k].min():.3f} max={feats_raw[:,k].max():.3f} std={feats_raw[:,k].std():.3f}")

    w_true = np.array([CROWD_AVOIDER_PROFILE[n] for n in REDUCED_FEATURE_NAMES])
    scale = feats_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    feats_norm = feats_raw / scale
    w_true_scaled = w_true * scale
    w_true_scaled_norm = w_true_scaled / (np.linalg.norm(w_true_scaled) + 1e-8)

    true_scores = feats_raw @ w_true
    true_best_idx = int(np.argmax(true_scores))
    print(f"true_best_idx (crowd-avoider profile) = {true_best_idx}, "
          f"Crowd-FM's own PRIEST-selected idx = {data.get('priest_best_idx')}")

    results = {"random": {"cos": [], "top1": [], "queries_used": []},
               "active": {"cos": [], "top1": [], "queries_used": []}}

    for rep in range(args.repeats):
        rng_r = np.random.default_rng(2000 + rep)
        rand_strat = RandomQueryStrategy(rng=rng_r)
        cos_r, top1_r, used_r = run_strategy(
            rand_strat, feats_raw, feats_norm, w_true, w_true_scaled_norm, true_best_idx, args.max_queries, rng_r
        )
        results["random"]["cos"].append(cos_r)
        results["random"]["top1"].append(top1_r)
        results["random"]["queries_used"].append(used_r)

        rng_a = np.random.default_rng(3000 + rep)
        act_strat = ActiveQueryStrategy(rng=rng_a)
        cos_a, top1_a, used_a = run_strategy(
            act_strat, feats_raw, feats_norm, w_true, w_true_scaled_norm, true_best_idx, args.max_queries, rng_a
        )
        results["active"]["cos"].append(cos_a)
        results["active"]["top1"].append(top1_a)
        results["active"]["queries_used"].append(used_a)

    def summarize(key):
        cos_arr = np.array(results[key]["cos"])
        top1_arr = np.array(results[key]["top1"])
        used_arr = np.array(results[key]["queries_used"])
        return {
            "mean_cos_per_query": cos_arr.mean(axis=0).tolist(),
            "mean_top1_per_query": top1_arr.mean(axis=0).tolist(),
            "mean_queries_used": float(used_arr.mean()),
            "final_mean_cos": float(cos_arr[:, -1].mean()),
            "final_mean_top1": float(top1_arr[:, -1].mean()),
        }

    summary = {"random": summarize("random"), "active": summarize("active")}
    print(json.dumps(summary, indent=2))

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "raw": results, "true_best_idx": true_best_idx,
                    "priest_best_idx": data.get("priest_best_idx")}, f, indent=2)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
