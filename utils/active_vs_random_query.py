#!/usr/bin/env python
"""Does UserAlign's active query selection actually beat picking pairs at
random, for the same query budget? Every other script in this repo
(crowd_scenario_demo.py, preference_integration_demo.py,
crowd_citywalker_integration_demo.py, real_feedback_test.py) samples
comparison pairs uniformly at random -- this compares that directly against
src/preference/active_query.py's ActiveQueryStrategy (ported from UserAlign,
NeurIPS 2025) on the same synthetic candidate pools utils/preference_poc.py
already validated the fitting/selection machinery on.

For each of N repeats x each synthetic profile: generate a holdout candidate
pool once, then run both strategies for up to --max-queries rounds (same
simulated human, same true w), tracking cosine(w_true, theta_hat) and
top-1 agreement (does the strategy's current `best()` match the pool's true
best candidate) after every query. ActiveQueryStrategy can also stop early
(done()) once no candidate could plausibly beat the current best -- that
query-count savings is reported too.

Usage:
    PYTHONPATH=. python utils/active_vs_random_query.py --repeats 10 --max-queries 10
"""
import argparse
import json

import numpy as np

from src.preference.features import extract_features, FEATURE_NAMES
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import RandomQueryStrategy, ActiveQueryStrategy
from utils.preference_poc import PROFILES, profile_to_vector, make_synthetic_candidates


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--max-queries", type=int, default=10)
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--output", default="active_vs_random_query.json")
    p.add_argument("--plot", default="/home/nuri2/Desktop/active_vs_random_query.png")
    return p.parse_args()


def run_strategy(strategy, feats_raw, feats_norm, w_true, w_true_scaled_norm, true_best_idx, max_queries, rng):
    """feats_raw: used only to simulate the (true, unnormalized-w_true-driven)
    human's answer -- the actual preference model being simulated is defined
    in raw feature space, same as every other script.
    feats_norm: what the strategy itself sees/fits on -- per-dimension-std
    normalized, so fit_logit_mle's norm constraint applies to comparable
    scales instead of being dominated by whichever raw feature (e.g.
    total_turn_angle_deg, 0-180) happens to have the largest magnitude."""
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
    # pad to max_queries with the last value (for early-stopped active runs)
    while len(cos_track) < max_queries:
        cos_track.append(cos_track[-1] if cos_track else 0.0)
        top1_track.append(top1_track[-1] if top1_track else 0)
    return cos_track, top1_track, used


def main():
    args = parse_args()
    results = {"random": {"cos": [], "top1": [], "queries_used": []},
               "active": {"cos": [], "top1": [], "queries_used": []}}

    for rep in range(args.repeats):
        rng_data = np.random.default_rng(1000 + rep)
        for profile_name, profile in PROFILES.items():
            w_true = profile_to_vector(profile)
            target = (0.0, 5.0)
            candidates = make_synthetic_candidates(target, num_candidates=args.pool_size, rng=rng_data)
            feats_raw = np.array([extract_features(c, target) for c in candidates])
            true_scores = feats_raw @ w_true
            true_best_idx = int(np.argmax(true_scores))

            # Normalize features by per-dimension std (same fix bradley_terry.py
            # already applies) so fit_logit_mle's norm-constrained theta isn't
            # dominated by whichever raw feature has the largest scale. Compare
            # fitted theta against w_true rescaled into that same space.
            scale = feats_raw.std(axis=0)
            scale = np.where(scale < 1e-6, 1.0, scale)
            feats_norm = feats_raw / scale
            w_true_scaled = w_true * scale
            w_true_scaled_norm = w_true_scaled / (np.linalg.norm(w_true_scaled) + 1e-8)

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

        print(f">>> repeat {rep + 1}/{args.repeats} done")

    def summarize(key):
        cos_arr = np.array(results[key]["cos"])  # (runs, max_queries)
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
        json.dump({"summary": summary, "raw": results}, f, indent=2)
    print(f"saved to {args.output}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(1, args.max_queries + 1)
    axes[0].plot(x, summary["random"]["mean_cos_per_query"], "o-", label="random pair (baseline, matches every other script)")
    axes[0].plot(x, summary["active"]["mean_cos_per_query"], "s-", label="active query (UserAlign)")
    axes[0].set_xlabel("query round")
    axes[0].set_ylabel("cosine(w_true, theta_hat)")
    axes[0].set_title("Fit quality vs query count")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, summary["random"]["mean_top1_per_query"], "o-", label="random pair")
    axes[1].plot(x, summary["active"]["mean_top1_per_query"], "s-", label="active query (UserAlign)")
    axes[1].axvline(summary["active"]["mean_queries_used"], color="gray", linestyle="--",
                     label=f"active mean stop: {summary['active']['mean_queries_used']:.1f} queries")
    axes[1].set_xlabel("query round")
    axes[1].set_ylabel("top-1 agreement (best() == true best)")
    axes[1].set_title("Selection accuracy vs query count")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
