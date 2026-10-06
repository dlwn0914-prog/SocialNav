#!/usr/bin/env python
"""End-to-end test of the proposed research pipeline (CFM candidate
generation -> hard safety filtering -> N<=10 A/B personalized preference
scorer) across several real Crowd-FM scenarios, not just the single one
utils/active_query_on_crowdfm_real.py tested.

Each scenario (crowd_fm_ws/scenario_<name>.json) was produced by actually
running Crowd-FM's flow model + scorer + PRIEST refinement
(crowd_fm_ws/extract_real_candidates.py) for a different crowd layout:
  sparse_1   - 1 person, dead center
  spread_3   - 3 people, wide spacing across the corridor
  cluster_4  - 4 people clustered close together (same as the earlier
               single-scenario test)
  dense_6    - 6 people, clustered even tighter
  offside_3  - 3 people all pushed to one side, leaving a clear lane

For each scenario:
  1. src/preference/safety_filter.py filters the K real refined candidates
     down to the safe subset (or the single fallback candidate if the
     scene is too crowded for any candidate to clear PRIEST's own
     collision ellipses everywhere).
  2. If >=2 safe candidates survive, RandomQueryStrategy vs
     ActiveQueryStrategy (src/preference/active_query.py) are run on that
     SAFE subset only (not the raw candidate pool) -- this is what "safety
     filtering before the N<=10 preference step" actually means in
     practice, not just a conceptual pipeline order.
  3. If <2 candidates survive (crowded-scene fallback), there is nothing
     to run a pairwise preference query on -- this is reported as its own
     outcome, not skipped silently.

Usage:
    PYTHONPATH=. python utils/multi_scenario_pipeline_test.py \
        --scenario-dir /home/nuri2/crowd_fm_ws --repeats 20
"""
import argparse
import glob
import json
import os

import numpy as np

from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import RandomQueryStrategy, ActiveQueryStrategy
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


def run_one_scenario(name, path, repeats, max_queries):
    with open(path) as f:
        data = json.load(f)

    candidates = [np.array(c, dtype=np.float64) for c in data["candidates"]]
    goal = np.array(data["goal"], dtype=np.float64)
    people = np.array(data["people"], dtype=np.float64)
    static_obs = np.array(data["static_obstacles"], dtype=np.float64)
    people = people[np.linalg.norm(people, axis=1) < 500]

    filt = filter_safe_candidates(candidates, static_obs=static_obs, dynamic_obs=people)
    safe_idx = filt["safe_indices"]
    n_candidates = len(candidates)
    n_safe = 0 if filt["used_fallback"] else len(safe_idx)

    report = {
        "scenario": name,
        "n_people": int(len(people)),
        "n_candidates": n_candidates,
        "n_safe": n_safe,
        "used_fallback": filt["used_fallback"],
        "margins_min": float(min(filt["margins"])),
        "margins_max": float(max(filt["margins"])),
    }

    if len(safe_idx) < 2:
        report["query_test"] = "skipped (fewer than 2 candidates survive safety filtering -- nothing to compare)"
        return report

    safe_candidates = [candidates[i] for i in safe_idx]
    reach = candidate_reach_scale(safe_candidates)
    feats_raw = np.array([reduced_features(c, goal, people, reach) for c in safe_candidates])

    w_true = np.array([CROWD_AVOIDER_PROFILE[n] for n in REDUCED_FEATURE_NAMES])
    scale = feats_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    feats_norm = feats_raw / scale
    w_true_scaled = w_true * scale
    w_true_scaled_norm = w_true_scaled / (np.linalg.norm(w_true_scaled) + 1e-8)
    true_best_idx = int(np.argmax(feats_raw @ w_true))

    n_queryable = len(safe_candidates)
    eff_max_queries = min(max_queries, n_queryable - 1) if n_queryable >= 2 else 0

    results = {"random": {"cos": [], "top1": []}, "active": {"cos": [], "top1": []}}
    for rep in range(repeats):
        rng_r = np.random.default_rng(2000 + rep)
        rand_strat = RandomQueryStrategy(rng=rng_r)
        cos_r, top1_r, _ = run_strategy(
            rand_strat, feats_raw, feats_norm, w_true, w_true_scaled_norm, true_best_idx, eff_max_queries, rng_r
        )
        results["random"]["cos"].append(cos_r)
        results["random"]["top1"].append(top1_r)

        rng_a = np.random.default_rng(3000 + rep)
        act_strat = ActiveQueryStrategy(rng=rng_a)
        cos_a, top1_a, _ = run_strategy(
            act_strat, feats_raw, feats_norm, w_true, w_true_scaled_norm, true_best_idx, eff_max_queries, rng_a
        )
        results["active"]["cos"].append(cos_a)
        results["active"]["top1"].append(top1_a)

    def final_mean(key, metric):
        arr = np.array(results[key][metric])
        return float(arr[:, -1].mean()) if arr.shape[1] > 0 else float("nan")

    report["query_test"] = {
        "n_queryable_safe_candidates": n_queryable,
        "queries_used": eff_max_queries,
        "random_final_cos": final_mean("random", "cos"),
        "random_final_top1": final_mean("random", "top1"),
        "active_final_cos": final_mean("active", "cos"),
        "active_final_top1": final_mean("active", "top1"),
    }
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", default="/home/nuri2/crowd_fm_ws")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--max-queries", type=int, default=10)
    p.add_argument("--output", default="multi_scenario_pipeline_test.json")
    args = p.parse_args()

    paths = sorted(glob.glob(os.path.join(args.scenario_dir, "scenario_*.json")))
    reports = []
    for path in paths:
        name = os.path.basename(path).replace("scenario_", "").replace(".json", "")
        print(f"\n=== {name} ===")
        rep = run_one_scenario(name, path, args.repeats, args.max_queries)
        print(json.dumps(rep, indent=2))
        reports.append(rep)

    queryable = [r for r in reports if isinstance(r.get("query_test"), dict)]
    if queryable:
        agg = {
            "n_scenarios_queryable": len(queryable),
            "n_scenarios_fallback_only": len(reports) - len(queryable),
            "mean_random_final_cos": float(np.mean([r["query_test"]["random_final_cos"] for r in queryable])),
            "mean_active_final_cos": float(np.mean([r["query_test"]["active_final_cos"] for r in queryable])),
            "mean_random_final_top1": float(np.mean([r["query_test"]["random_final_top1"] for r in queryable])),
            "mean_active_final_top1": float(np.mean([r["query_test"]["active_final_top1"] for r in queryable])),
        }
        print("\n=== aggregate across scenarios ===")
        print(json.dumps(agg, indent=2))
    else:
        agg = None
        print("\nNo scenario had >=2 safe candidates to compare -- no aggregate query-test stats.")

    with open(args.output, "w") as f:
        json.dump({"scenarios": reports, "aggregate": agg}, f, indent=2)
    print(f"\nsaved to {args.output}")


if __name__ == "__main__":
    main()
