#!/usr/bin/env python
"""How much does N (few-shot feedback count) actually help
utils/crowd_citywalker_integration_demo.py's fit? A single run at N=10 gave
cosine(w_true, w_hat) anywhere from 0.19 to 0.92 depending on random draw and
model-sampling noise (flow-matching has no fixed seed), so a single run at any
one N is not evidence of a trend -- this sweeps N in {5,10,20,30}, repeating
each R times (default 5) on the *same* candidate pool (the expensive GPU step,
generated once) so only the pairwise-comparison sampling varies, and reports
mean +/- std per N.

Usage:
    PYTHONPATH=. python utils/nshot_sweep.py \
        --model-path model --data-path data/citywalker_test.jsonl \
        --limit 40 --repeats 5 --n-shots 5,10,20,30
"""
import argparse
import json

import numpy as np
from tqdm import tqdm

from utils.citywalker import SocialNavModel, TEST_CATEGORIES
from utils.turn_multimodal_check import infer_multi, max_angle_and_hit
from src.preference.features import agent_avoidance_feature
from src.preference.bradley_terry import fit_preference_weights, predict_preference_prob
import utils.crowd_citywalker_integration_demo as ccd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--flow-steps", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--category", default="crowd", choices=TEST_CATEGORIES)
    p.add_argument("--num-agents", type=int, default=6)
    p.add_argument("--refine-steps", type=int, default=8)
    p.add_argument("--refine-bound-scale", type=float, default=2.0)
    p.add_argument("--n-shots", default="5,10,20,30", help="Comma-separated N values to sweep.")
    p.add_argument("--repeats", type=int, default=5, help="Independent comparison-draws per N.")
    p.add_argument("--output", default="nshot_sweep.json")
    p.add_argument("--plot", default="/home/nuri2/Desktop/nshot_sweep.png")
    return p.parse_args()


def build_candidate_pool(args):
    print(">>> loading model:", args.model_path)
    model = SocialNavModel(args.model_path, device=args.device, flow_steps=args.flow_steps)

    print(f">>> loading '{args.category}'-category samples:", args.data_path)
    items = ccd.load_category_samples(args.data_path, args.category)
    print(f">>> {len(items)} '{args.category}' samples with path_distance >= 1m")
    items = items[: args.limit]

    crowd_rng = np.random.default_rng(1)
    per_item = []
    for item in tqdm(items, desc="sampling candidates (once, reused for every N/repeat)"):
        msg1 = item["messages"][1]
        gt = np.asarray(msg1["gt_waypoints"], dtype=np.float32)
        step_scale = float(msg1["step_scale"])
        input_waypoints = np.asarray(msg1["input_waypoints"], dtype=np.float32)
        target = input_waypoints[5]
        try:
            wp_pred = infer_multi(model, item, args.num_samples)
        except Exception as e:
            print(f"[WARN] failed: {e}")
            continue
        reach = ccd.candidate_reach_scale(wp_pred)
        spread = ccd.candidate_spread(wp_pred)
        agents = ccd.synthetic_crowd(wp_pred, args.num_agents, crowd_rng)
        feats = [ccd.reduced_features(c, target, agents, reach) for c in wp_pred]
        per_item.append(
            {
                "candidates": wp_pred, "gt": gt, "step_scale": step_scale, "target": target,
                "agents": agents, "reach": reach, "spread": spread, "feats": feats,
            }
        )
    return per_item


def fit_and_eval(per_item, w_true, n_shot, rng, args):
    diff_feats, labels = [], []
    for _ in range(n_shot):
        rec = per_item[rng.integers(len(per_item))]
        i, j = rng.choice(len(rec["feats"]), size=2, replace=False)
        p_i_over_j = predict_preference_prob(w_true, rec["feats"][i], rec["feats"][j])
        if rng.random() < p_i_over_j:
            diff_feats.append(rec["feats"][i] - rec["feats"][j])
        else:
            diff_feats.append(rec["feats"][j] - rec["feats"][i])
        labels.append(1.0)
    w_hat, scale = fit_preference_weights(np.array(diff_feats), np.array(labels), l2=1.0, lr=0.2, num_steps=800)

    w_true_norm = w_true * scale
    cos_sim = float(
        np.dot(w_true_norm, w_hat) / (np.linalg.norm(w_true_norm) * np.linalg.norm(w_hat) + 1e-8)
    )

    single_angles, single_clears, refined_angles, refined_clears, oracle_clears = [], [], [], [], []
    for rec in per_item:
        candidates, gt, step_scale, target, agents, reach, spread = (
            rec["candidates"], rec["gt"], rec["step_scale"], rec["target"], rec["agents"], rec["reach"], rec["spread"]
        )
        cand_list = list(candidates)
        single_angles.append(max_angle_and_hit(candidates[0], gt, step_scale))
        single_clears.append(agent_avoidance_feature(candidates[0], agents))
        oracle_clears.append(float(max(agent_avoidance_feature(c, agents) for c in cand_list)))

        best_idx, _ = ccd.local_select_best(w_hat, cand_list, target, agents, reach, scale=scale)
        selected = candidates[best_idx]
        max_step = args.refine_bound_scale * spread
        refined = ccd.local_refine(
            selected, target, w_hat, agents, reach,
            num_steps=args.refine_steps, scale=scale, max_step_per_waypoint=max_step,
        )
        refined_angles.append(max_angle_and_hit(refined, gt, step_scale))
        refined_clears.append(agent_avoidance_feature(refined, agents))

    return {
        "cos_sim": cos_sim,
        "mean_single_angle": float(np.mean(single_angles)),
        "mean_single_clearance": float(np.mean(single_clears)),
        "mean_oracle_clearance": float(np.mean(oracle_clears)),
        "mean_refined_angle": float(np.mean(refined_angles)),
        "mean_refined_clearance": float(np.mean(refined_clears)),
    }


def main():
    args = parse_args()
    n_shots = [int(x) for x in args.n_shots.split(",")]

    per_item = build_candidate_pool(args)
    w_true = ccd.w_true_vector()
    print(f">>> candidate pool: {len(per_item)} items, sweeping N={n_shots} x {args.repeats} repeats")

    sweep = {}
    for n_shot in n_shots:
        runs = []
        for rep in range(args.repeats):
            rng = np.random.default_rng(1000 * n_shot + rep)
            runs.append(fit_and_eval(per_item, w_true, n_shot, rng, args))
        sweep[n_shot] = runs
        cos_vals = [r["cos_sim"] for r in runs]
        print(f"N={n_shot:3d}: cosine mean={np.mean(cos_vals):.3f} std={np.std(cos_vals):.3f} "
              f"(range {min(cos_vals):.2f}-{max(cos_vals):.2f})")

    with open(args.output, "w") as f:
        json.dump({"n_shots": n_shots, "repeats": args.repeats, "sweep": sweep}, f, indent=2)
    print(f"saved to {args.output}")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cos_means = [np.mean([r["cos_sim"] for r in sweep[n]]) for n in n_shots]
    cos_stds = [np.std([r["cos_sim"] for r in sweep[n]]) for n in n_shots]
    axes[0].errorbar(n_shots, cos_means, yerr=cos_stds, marker="o", capsize=4)
    axes[0].set_xlabel("N (pairwise feedback count)")
    axes[0].set_ylabel("cosine(w_true, w_hat)")
    axes[0].set_title("Preference-fit quality vs N")
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(-0.1, 1.05)

    single_c = [np.mean([r["mean_single_clearance"] for r in sweep[n]]) for n in n_shots]
    refined_c = [np.mean([r["mean_refined_clearance"] for r in sweep[n]]) for n in n_shots]
    oracle_c = [np.mean([r["mean_oracle_clearance"] for r in sweep[n]]) for n in n_shots]
    axes[1].plot(n_shots, single_c, "o-", label="single-sample")
    axes[1].plot(n_shots, refined_c, "s-", label="scorer-selected+refined")
    axes[1].plot(n_shots, oracle_c, "k--", label="oracle-best-of-10")
    axes[1].set_xlabel("N (pairwise feedback count)")
    axes[1].set_ylabel("mean clearance")
    axes[1].set_title("Crowd clearance vs N")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.plot, dpi=150)
    print(f"plot saved to {args.plot}")


if __name__ == "__main__":
    main()
