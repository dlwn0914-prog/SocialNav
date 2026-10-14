#!/usr/bin/env python
"""Does FisherOptimalStrategy's win over random (utils/dpb_estimation_regret.py,
synthetic candidates) hold up on REAL Crowd-FM candidates?

Reuses the 12 real scenarios already extracted this session
(crowd_fm_ws/scenario_*.json -- flow model + scorer + PRIEST, see
scripts/personalized_validation/extract_real_candidates.py) and
src/preference/safety_filter.py's hard filter, so no Docker/GPU needed
here -- the expensive part (generating real candidates) was already done.

For each scenario with >=2 safety-filtered candidates: run Random,
ActiveQueryStrategy, DPBStrategy(gamma=0.8), and FisherOptimalStrategy,
tracking cumulative ESTIMATION regret (1 - cosine(theta_hat, true
CROWD_AVOIDER_PROFILE) per round, summed) -- the metric this session
settled on as matching our actual research goal (full preference-vector
recovery), not the paper's best-arm-style regret.

Usage:
    PYTHONPATH=. python utils/fisher_optimal_on_real_candidates.py --repeats 20
"""
import argparse
import glob
import json
import os

import numpy as np

from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES
from src.preference.bradley_terry import predict_preference_prob
from src.preference.safety_filter import filter_safe_candidates
from src.preference.active_query import ActiveQueryStrategy, RandomQueryStrategy
from src.preference.dpb import DPBStrategy
from src.preference.fisher_optimal import FisherOptimalStrategy

REDUCED_FEATURE_NAMES = ["total_turn_angle_deg", "target_alignment_cos", "min_dist_to_agents"]
_BASE_IDX = {n: FEATURE_NAMES.index(n) for n in REDUCED_FEATURE_NAMES if n in FEATURE_NAMES}
CROWD_AVOIDER_PROFILE = {"total_turn_angle_deg": -0.05, "target_alignment_cos": 0.3, "min_dist_to_agents": 2.0}


def candidate_reach_scale(candidates):
    all_wp = np.concatenate([np.asarray(c, dtype=np.float64) for c in candidates], axis=0)
    return float(np.mean(np.linalg.norm(all_wp, axis=1)))


def reduced_features(waypoints, target, agents, reach):
    base = extract_features(waypoints, target)
    dist = agent_avoidance_feature(waypoints, agents)
    return np.array([
        base[_BASE_IDX["total_turn_angle_deg"]],
        base[_BASE_IDX["target_alignment_cos"]],
        min(dist, reach) if np.isfinite(dist) else reach,
    ], dtype=np.float64)


def cos_to(theta_hat, w_true_norm):
    if theta_hat is None or np.linalg.norm(theta_hat) < 1e-8:
        return 0.0
    return float(np.dot(w_true_norm, theta_hat) / (np.linalg.norm(w_true_norm) * np.linalg.norm(theta_hat)))


def load_scenario_feats(path):
    d = json.load(open(path))
    cands = [np.array(c, dtype=np.float64) for c in d["candidates"]]
    goal = np.array(d["goal"], dtype=np.float64)
    people = np.array(d["people"], dtype=np.float64)
    people = people[np.linalg.norm(people, axis=1) < 500]
    static_obs = np.array(d.get("static_obstacles", []), dtype=np.float64)
    if static_obs.size:
        static_obs = static_obs[np.linalg.norm(static_obs, axis=1) > 1e-5]

    filt = filter_safe_candidates(cands, static_obs=static_obs, dynamic_obs=people)
    if filt["used_fallback"] or len(filt["safe_indices"]) < 2:
        return None

    safe = [cands[i] for i in filt["safe_indices"]]
    reach = candidate_reach_scale(safe)
    feats_raw = np.array([reduced_features(c, goal, people, reach) for c in safe])
    return feats_raw


def run_session(strategy, feats_raw, feats_norm, w_true, w_true_scaled_norm, rng, n_queries):
    cum, running = [], 0.0
    for _ in range(n_queries):
        i, j = strategy.select(feats_norm)
        p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
        winner = i if rng.random() < p_i else j
        strategy.update(feats_norm, i, j, winner)
        running += 1.0 - cos_to(strategy.theta_hat, w_true_scaled_norm)
        cum.append(running)
    return cum


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario-dir", default="/home/nuri2/crowd_fm_ws")
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()

    methods = {
        "Random": lambda rng: RandomQueryStrategy(rng=rng),
        "ActiveQuery": lambda rng: ActiveQueryStrategy(rng=rng),
        "DPB(g=0.8)": lambda rng: DPBStrategy(gamma=0.8, discount_unit="round", rng=rng),
        "Fisher-optimal": lambda rng: FisherOptimalStrategy(rng=rng),
    }

    paths = sorted(glob.glob(os.path.join(args.scenario_dir, "scenario_*.json")))
    all_finals = {name: [] for name in methods}
    per_scenario = {}
    n_testable = 0

    for path in paths:
        name_sc = os.path.basename(path).replace("scenario_", "").replace(".json", "")
        feats_raw = load_scenario_feats(path)
        if feats_raw is None:
            print(f"{name_sc}: skipped (fallback or <2 safe candidates)")
            continue
        n_testable += 1
        K = len(feats_raw)
        n_queries = min(10, K - 1)
        print(f"{name_sc}: {K} safe candidates, {n_queries} queries")

        scale = feats_raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        feats_norm = feats_raw / scale
        w_true = np.array([CROWD_AVOIDER_PROFILE[n] for n in REDUCED_FEATURE_NAMES])
        w_true_scaled_norm = (w_true * scale) / (np.linalg.norm(w_true * scale) + 1e-8)

        scenario_finals = {name: [] for name in methods}
        for rep in range(args.repeats):
            for name, make_strat in methods.items():
                rng = np.random.default_rng(1000 * rep + hash((name_sc, name)) % 1000)
                strat = make_strat(rng)
                cum = run_session(strat, feats_raw, feats_norm, w_true, w_true_scaled_norm, rng, n_queries)
                all_finals[name].append(cum[-1])
                scenario_finals[name].append(cum[-1])
        per_scenario[name_sc] = {"K": K, "n_queries": n_queries,
                                  "means": {name: np.mean(v) for name, v in scenario_finals.items()}}

    print(f"\n=== per-scenario final cumulative estimation regret (mean over {args.repeats} repeats) ===")
    header = f"{'scenario':>16} {'K':>4} {'nq':>4}" + "".join(f"{name:>16}" for name in methods)
    print(header)
    for name_sc, info in per_scenario.items():
        row = f"{name_sc:>16} {info['K']:>4} {info['n_queries']:>4}"
        for name in methods:
            row += f"{info['means'][name]:>16.3f}"
        print(row)

    print(f"\n=== final cumulative estimation regret, pooled across {n_testable} testable scenarios "
          f"x {args.repeats} repeats (lower = better) ===")
    for name in methods:
        vals = all_finals[name]
        print(f"  {name}: mean={np.mean(vals):.4f} (+-{np.std(vals):.4f}), n={len(vals)}")


if __name__ == "__main__":
    main()
