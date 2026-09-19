#!/usr/bin/env python
"""Sanity-check whether SocialNav's flow-matching action expert produces diverse
candidate trajectories for `turn`-category CityWalker samples, and whether any
candidate among several actually tracks the given target direction.

Motivation: a full single-sample (`num_samples=1`) CityWalker eval on the
reconstructed benchmark showed turn-category mean_angle (34.2 deg) far worse than
other categories (16-19 deg), with predicted trajectories barely correlated with
`<input_target>` (sign match ~53%, corr ~-0.065) despite the target itself being
strongly correlated with the true path (sign match ~96%, corr ~0.70). This script
checks whether that is a mode-collapse of the underlying generative distribution
(no candidate tracks the target) or a single-sample selection problem (some
candidates do track it, just not the one usually drawn) -- the latter would support
a preference/selection-conditioned sampling approach on top of the existing policy.

Usage:
    PYTHONPATH=. python utils/turn_multimodal_check.py \
        --model-path model --data-path data/citywalker_test.jsonl \
        --num-samples 10 --output turn_multimodal_check.json
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from utils.citywalker import SocialNavModel, TEST_CATEGORIES


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-path", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--flow-steps", type=int, default=5)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--limit", type=int, default=None, help="Cap number of turn samples checked.")
    p.add_argument("--output", default="turn_multimodal_check.json")
    return p.parse_args()


def load_turn_samples(data_path):
    turn_idx = TEST_CATEGORIES.index("turn")
    items = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            msg1 = obj["messages"][1]
            cats = msg1.get("categories", [0] * len(TEST_CATEGORIES))
            if int(round(cats[turn_idx])) != 1:
                continue
            gt = np.asarray(msg1["gt_waypoints"], dtype=np.float32)
            step_scale = float(msg1["step_scale"])
            gt_abs = gt * step_scale
            path_distance_m = (
                float(np.linalg.norm(gt_abs[-1] - gt_abs[0])) if gt_abs.shape[0] >= 2 else float(np.linalg.norm(gt_abs[-1]))
            )
            if path_distance_m < 1.0:
                continue
            items.append(obj)
    return items


def max_angle_and_hit(pred_rel, gt_rel, step_scale):
    pred_abs = pred_rel * step_scale
    gt_abs = gt_rel * step_scale
    pred_t = torch.from_numpy(pred_abs).view(-1, 2)
    gt_t = torch.from_numpy(gt_abs).view(-1, 2)
    cos_sim = F.cosine_similarity(pred_t, gt_t, dim=1).clamp(-1.0, 1.0)
    angles = torch.acos(cos_sim) * 180.0 / torch.pi
    return float(angles.max().item())


@torch.no_grad()
def infer_multi(model, item, num_samples):
    messages = item["messages"]
    user_content = messages[0]["content"]
    images = [Image.open(p).convert("RGB") for p in item["images"]]

    content = [{"type": "image", "image": img} for img in images]
    content.append({"type": "text", "text": user_content})
    messages_for_model = [{"role": "user", "content": content}]

    text = model.processor.apply_chat_template(messages_for_model, tokenize=False, add_generation_prompt=True) + "<|im_end|>"
    inputs = model.processor(text=text, images=images, padding=True, return_tensors="pt").to(model.device)

    input_waypoints = torch.tensor(messages[1]["input_waypoints"], dtype=torch.float32).unsqueeze(0).to(model.device)
    inputs["input_waypoints"] = input_waypoints

    outputs = model.model(
        **inputs,
        train=False,
        train_branch="fm",
        num_samples=num_samples,
        special_token2id=model.special_token2id,
    )
    wp_pred = outputs[0].detach().cpu().float().numpy()  # (num_samples, action_chunk, 2)
    return wp_pred


def main():
    args = parse_args()
    print(">>> loading model:", args.model_path)
    model = SocialNavModel(args.model_path, device=args.device, flow_steps=args.flow_steps)

    print(">>> loading turn-category samples:", args.data_path)
    items = load_turn_samples(args.data_path)
    print(f">>> {len(items)} turn samples with path_distance >= 1m")
    if args.limit is not None:
        items = items[: args.limit]
        print(f">>> capped to {len(items)}")

    results = []
    for item in tqdm(items, desc="multi-sample inference"):
        msg1 = item["messages"][1]
        gt = np.asarray(msg1["gt_waypoints"], dtype=np.float32)
        step_scale = float(msg1["step_scale"])
        input_waypoints = np.asarray(msg1["input_waypoints"], dtype=np.float32)
        target = input_waypoints[5]  # 6th entry = <input_target>
        gt_target_sign_match = int(np.sign(target[0]) == np.sign(gt[-1, 0]))

        try:
            wp_pred = infer_multi(model, item, args.num_samples)
        except Exception as e:
            print(f"[WARN] failed on {item.get('meta')}: {e}")
            continue

        angles = [max_angle_and_hit(wp_pred[i], gt, step_scale) for i in range(wp_pred.shape[0])]
        final_x = wp_pred[:, -1, 0]
        sign_match = (np.sign(final_x) == np.sign(target[0])).astype(int)

        results.append(
            {
                "meta": item.get("meta"),
                "single_sample_angle": angles[0],
                "best_of_n_angle": float(np.min(angles)),
                "mean_angle_over_n": float(np.mean(angles)),
                "final_x_std": float(np.std(final_x)),
                "sign_match_rate": float(np.mean(sign_match)),
                "any_hit_sign": int(np.any(sign_match)),
                "any_hit_lt30deg": int(np.any(np.asarray(angles) < 30.0)),
                "gt_target_sign_match": gt_target_sign_match,
            }
        )

    n = len(results)
    single_angles = [r["single_sample_angle"] for r in results]
    best_angles = [r["best_of_n_angle"] for r in results]
    stds = [r["final_x_std"] for r in results]
    any_hit_sign_rate = np.mean([r["any_hit_sign"] for r in results])
    any_hit_lt30_rate = np.mean([r["any_hit_lt30deg"] for r in results])
    mean_sign_match_rate = np.mean([r["sign_match_rate"] for r in results])

    summary = {
        "n_turn_samples": n,
        "num_samples_per_item": args.num_samples,
        "mean_single_sample_angle": float(np.mean(single_angles)),
        "mean_best_of_n_angle": float(np.mean(best_angles)),
        "mean_final_x_std_across_candidates": float(np.mean(stds)),
        "mean_sign_match_rate_per_item": float(mean_sign_match_rate),
        "fraction_items_with_any_candidate_matching_target_sign": float(any_hit_sign_rate),
        "fraction_items_with_any_candidate_lt30deg": float(any_hit_lt30_rate),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_item": results}, f, indent=2, ensure_ascii=False)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
