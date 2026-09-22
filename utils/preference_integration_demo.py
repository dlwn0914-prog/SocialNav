#!/usr/bin/env python
"""Wire the few-shot Bradley-Terry preference scorer (src/preference/) into the
real SocialNav flow-matching Action Expert's multi-sample output, on real
turn-category CityWalker data -- the integration step after utils/preference_poc.py
validated the fitting/selection machinery on synthetic trajectories.

Profile tested: "target_tracking" -- prefers candidates whose final heading and
endpoint closely match the given <input_target>. This is the natural corrective
preference for the failure mode utils/turn_multimodal_check.py diagnosed (the
released checkpoint largely ignores the target direction on turn-category
samples), so it doubles as a mechanism check: does plugging the scorer into the
real model's own candidates recover some of that gap? It is deliberately aligned
with the eval metric, so a positive result validates the *mechanism*, not
"arbitrary preference following" (that needs a preference orthogonal to target
-tracking, left for follow-up once real/synthetic preference data beyond this
sanity profile exists).

Usage:
    PYTHONPATH=. python utils/preference_integration_demo.py \
        --model-path model --data-path data/citywalker_test.jsonl \
        --num-samples 10 --limit 40 --n-shot 10
"""
import argparse
import json

import numpy as np
import torch
from tqdm import tqdm

from utils.citywalker import SocialNavModel
from utils.turn_multimodal_check import load_turn_samples, infer_multi, max_angle_and_hit
from src.preference.features import extract_features, FEATURE_NAMES
from src.preference.bradley_terry import fit_preference_weights, predict_preference_prob
from src.preference.select_and_refine import select_best, refine_towards_preference

RNG = np.random.default_rng(0)

TARGET_TRACKING_PROFILE = {
    "path_length": -0.1,
    "directness": 0.3,
    "total_turn_angle_deg": -0.1,
    "final_heading_error_deg": -1.0,
    "lateral_deviation": -0.2,
    "endpoint_dist_to_target": -0.5,
    "target_alignment_cos": 2.0,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--flow-steps", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--n-shot", type=int, default=10)
    p.add_argument("--refine-steps", type=int, default=3)
    p.add_argument("--output", default="preference_integration_demo.json")
    return p.parse_args()


def w_true_vector():
    return np.array([TARGET_TRACKING_PROFILE[n] for n in FEATURE_NAMES], dtype=np.float64)


def main():
    args = parse_args()
    print(">>> loading model:", args.model_path)
    model = SocialNavModel(args.model_path, device=args.device, flow_steps=args.flow_steps)

    print(">>> loading turn-category samples:", args.data_path)
    items = load_turn_samples(args.data_path)
    items = items[: args.limit]
    print(f">>> {len(items)} turn samples")

    # --- Generate candidates for every item once (shared for fitting + eval) ---
    per_item = []
    for item in tqdm(items, desc="sampling candidates"):
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
        feats = [extract_features(c, target) for c in wp_pred]
        per_item.append(
            {"item": item, "candidates": wp_pred, "gt": gt, "step_scale": step_scale, "target": target, "feats": feats}
        )

    # --- Fit w_hat from N pairwise comparisons on the model's own real ---
    # --- candidates (not synthetic trajectories), each drawn *within one ---
    # --- randomly chosen item* (two of its own candidates) rather than from a
    # --- pool across items: path_length (and, less directly, the other
    # --- features it correlates with) scales with each CityWalker scene's own
    # --- absolute coordinate size, so pooling across items before sampling
    # --- pairs let cross-item scale variance dominate the fit instead of real
    # --- within-item ranking signal (see utils/crowd_citywalker_integration_demo.py,
    # --- where this was root-caused after w_hat repeatedly blew up on
    # --- whichever feature had the smallest within-comparison variance). A
    # --- real user also only ever judges "A vs B for this navigation
    # --- instance", never across two unrelated scenes.
    w_true = w_true_vector()
    print(f">>> {len(per_item)} items x {args.num_samples} real candidates each")

    diff_feats, labels = [], []
    for _ in range(args.n_shot):
        rec = per_item[RNG.integers(len(per_item))]
        i, j = RNG.choice(len(rec["feats"]), size=2, replace=False)
        p_i_over_j = predict_preference_prob(w_true, rec["feats"][i], rec["feats"][j])
        if RNG.random() < p_i_over_j:
            diff_feats.append(rec["feats"][i] - rec["feats"][j])
        else:
            diff_feats.append(rec["feats"][j] - rec["feats"][i])
        labels.append(1.0)
    w_hat, scale = fit_preference_weights(np.array(diff_feats), np.array(labels), l2=1.0, lr=0.2, num_steps=800)
    print("w_true:", dict(zip(FEATURE_NAMES, np.round(w_true, 2))))
    print("w_hat (normalized space):", dict(zip(FEATURE_NAMES, np.round(w_hat, 2))))

    # --- Evaluate on each item's own real candidates: naive vs scorer-selected+refined ---
    results = []
    for rec in per_item:
        candidates = rec["candidates"]
        gt, step_scale, target = rec["gt"], rec["step_scale"], rec["target"]

        single_angle = max_angle_and_hit(candidates[0], gt, step_scale)
        oracle_best_angle = float(min(max_angle_and_hit(c, gt, step_scale) for c in candidates))

        best_idx, _, _ = select_best(w_hat, list(candidates), target, scale=scale)
        selected = candidates[best_idx]
        selected_angle = max_angle_and_hit(selected, gt, step_scale)

        refined = refine_towards_preference(
            selected, target, w_hat, num_steps=args.refine_steps, scale=scale, candidates=list(candidates),
        )
        refined_angle = max_angle_and_hit(refined, gt, step_scale)

        results.append(
            {
                "single_angle": single_angle,
                "oracle_best_angle": oracle_best_angle,
                "scorer_selected_angle": selected_angle,
                "scorer_selected_refined_angle": refined_angle,
            }
        )

    def mean(key):
        return float(np.mean([r[key] for r in results]))

    summary = {
        "n_items": len(results),
        "n_shot_comparisons": args.n_shot,
        "num_samples_per_item": args.num_samples,
        "mean_single_sample_angle": mean("single_angle"),
        "mean_oracle_best_of_n_angle": mean("oracle_best_angle"),
        "mean_scorer_selected_angle": mean("scorer_selected_angle"),
        "mean_scorer_selected_refined_angle": mean("scorer_selected_refined_angle"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(args.output, "w") as f:
        json.dump(
            {"summary": summary, "w_true": w_true.tolist(), "w_hat": w_hat.tolist(), "scale": scale.tolist(), "per_item": results},
            f,
            indent=2,
        )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
