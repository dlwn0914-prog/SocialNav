#!/usr/bin/env python
"""Merge the two previously-separate validation tracks into one end-to-end
experiment: real SocialNav flow-matching candidates (utils/preference_integration_demo.py's
track) scored for crowd-avoidance (src/preference/features.py's agent_avoidance_feature,
so far only exercised on rule-based synthetic paths in utils/crowd_scenario_demo.py).

Why not "real images + real crowd coordinates" directly: CityWalker has no real
crowd-position labels (its 'crowd'/'person_close_by' categories are binary tags,
not coordinates -- see the category breakdown run this session), and the Gazebo
Jackal model has no camera sensor wired up (checked
arena4_ws/src/arena/simulation-setup/entities/robots/jackal/urdf/jackal.gazebo --
only imu/gpu_lidar/navsat sensors are defined, no camera). Neither side alone can
produce that combination without real engineering work (adding + rebuilding a
Gazebo camera sensor).

This script uses the honest middle ground instead: real CityWalker images (from
scenes the dataset itself flagged as having people nearby) feed the real model to
get real flow-matching candidates, and a synthetic crowd is placed in that scene's
ego-frame coordinate space purely for min_dist_to_agents scoring. The model never
sees or reacts to this synthetic crowd (it isn't in the pixels) -- this validates
whether preference-guided selection/refinement can steer *real* model candidates
toward higher clearance from *some* crowd, not whether the model perceives crowds
from images. A fully faithful version needs an actual camera sensor in Gazebo.

Feature set (3, not the full 8 in src/preference/features.py): total_turn_angle_deg,
target_alignment_cos, min_dist_to_agents. Two earlier attempts at this script fit
all 8 features and got increasingly unstable w_hat (wrong-signed, |weight| in the
hundreds) once tested against 40 *different* real CityWalker scenes instead of one
fixed synthetic scenario -- N=10 comparisons cannot reliably identify 8 free
parameters when several of them (path_length, lateral_deviation,
endpoint_dist_to_target) are collinear measures of "how much did you deviate" AND
scale with each scene's own absolute coordinate size. Dropping to 3 well-separated,
naturally scale-free features (angles/cosine, plus min_dist_to_agents rescaled by
candidate_reach_scale(), see below) is a more honest match to what N=10 examples
can actually identify; select_and_refine.py's shared select_best/refine_towards_preference
assume the *full* 8-dim feature space (via src/preference/features.py's
extract_features), so this script reimplements a local select+refine over the
reduced 3-dim space rather than changing that already-validated shared module.

Crowd placement scale (candidate_reach_scale, not |target|): CityWalker's 5-step
task predicts only the *next* few steps, not a full path to the eventual goal --
ground truth itself typically covers under half of |target|'s distance (confirmed
visually: /home/nuri2/Desktop/crowd_citywalker_demo.png). An earlier version placed
the synthetic crowd relative to |target|, which put it well outside the region any
candidate could reach, making "avoids the crowd" untestable. Rescaling to each
item's own candidate_reach_scale() (mean waypoint distance from origin, pooled
across all 10 real candidates) fixed that -- but exposed a real, unresolved
tradeoff rather than a bug: with the crowd now genuinely in the way,
total_turn_angle_deg (the *true* user's dominant preference, weight -3.68 in
normalized space) got almost entirely crowded out by min_dist_to_agents in the
fitted w_hat (weight 1.37 vs true 0.08) -- cosine(w_true, w_hat) dropped from 0.92
(crowd out of reach, no real conflict to resolve) to 0.19 (crowd in reach, real
conflict). refine_towards_preference's unconstrained backtracking line search then
chased that skewed clearance weight past what any of the 10 original candidates
achieved (mean clearance 0.76 vs oracle-best-of-10's 0.26) while destroying angle
accuracy (19.3 -> 65.0 deg mean). This is the most honest result this pipeline has
produced: when clearance and target-tracking *genuinely* compete, N=10 comparisons
+ unconstrained refinement does not yet resolve the tradeoff sensibly. Left
unfixed as a documented finding rather than patched further -- candidate next
steps: bound refine's total displacement to the original candidate spread (so it
can only interpolate/nudge within observed diversity, not extrapolate arbitrarily
far), and/or increase n_shot so the fit can separate the two competing features.

Usage:
    PYTHONPATH=. python utils/crowd_citywalker_integration_demo.py \
        --model-path model --data-path data/citywalker_test.jsonl \
        --num-samples 10 --limit 40 --n-shot 10
"""
import argparse
import json

import numpy as np
from tqdm import tqdm

from utils.citywalker import SocialNavModel, TEST_CATEGORIES
from utils.turn_multimodal_check import infer_multi, max_angle_and_hit
from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES
from src.preference.bradley_terry import fit_preference_weights, predict_preference_prob, score as bt_score

RNG = np.random.default_rng(0)

REDUCED_FEATURE_NAMES = ["total_turn_angle_deg", "target_alignment_cos", "min_dist_to_agents"]
_BASE_IDX = {n: FEATURE_NAMES.index(n) for n in REDUCED_FEATURE_NAMES if n in FEATURE_NAMES}

# Same relative weights as utils/crowd_scenario_demo.py's crowd_avoider on
# these two dimensions, reused for continuity across experiments (same "true"
# user, now tested against real model candidates instead of rule-based
# synthetic paths); final_heading_error_deg/path_length/etc. dropped, see
# module docstring.
CROWD_AVOIDER_PROFILE = {
    "total_turn_angle_deg": -0.05,
    "target_alignment_cos": 0.3,
    "min_dist_to_agents": 2.0,
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
    p.add_argument("--category", default="crowd", choices=TEST_CATEGORIES)
    p.add_argument("--num-agents", type=int, default=6)
    p.add_argument("--refine-steps", type=int, default=8)
    p.add_argument("--output", default="crowd_citywalker_integration_demo.json")
    return p.parse_args()


def load_category_samples(data_path, category):
    cat_idx = TEST_CATEGORIES.index(category)
    items = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            msg1 = obj["messages"][1]
            cats = msg1.get("categories", [0] * len(TEST_CATEGORIES))
            if int(round(cats[cat_idx])) != 1:
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


def candidate_reach_scale(candidates):
    """Mean distance-from-origin of every waypoint across all of this item's
    real candidates -- "how far do these 5-step predictions actually reach",
    as opposed to |target| (the eventual-goal direction CityWalker's 5-step
    task is *not* trying to reach in one shot -- ground truth itself only
    covers a fraction of the distance to target; see module docstring).
    Placing/scoring the synthetic crowd relative to |target| put agents well
    outside the region any candidate could possibly pass through (visually
    confirmed: /home/nuri2/Desktop/crowd_citywalker_demo.png's crowd sat near
    the target star while every candidate stayed within a third of that
    distance), making "avoids the crowd" untestable for that sample. This is
    the scale everything crowd-related should use instead."""
    all_wp = np.concatenate([np.asarray(c, dtype=np.float64) for c in candidates], axis=0)
    return float(np.mean(np.linalg.norm(all_wp, axis=1)))


def synthetic_crowd(candidates, num_agents, rng):
    """Place num_agents agents near a random point along the direction the
    candidates actually travel (their pooled centroid direction), at a
    fraction of candidate_reach_scale(candidates) -- i.e. *inside* the region
    the model's own 5-step predictions can plausibly reach, not near the
    (out-of-reach) target. The model never sees these (not in the image);
    this is purely for scoring min_dist_to_agents (see module docstring)."""
    all_wp = np.concatenate([np.asarray(c, dtype=np.float64) for c in candidates], axis=0)
    reach = float(np.mean(np.linalg.norm(all_wp, axis=1)))
    if reach < 1e-6:
        return np.zeros((num_agents, 2))
    centroid = all_wp.mean(axis=0)
    centroid_norm = float(np.linalg.norm(centroid))
    direction = centroid / centroid_norm if centroid_norm > 1e-6 else np.array([1.0, 0.0])
    normal = np.array([-direction[1], direction[0]])
    frac = rng.uniform(0.5, 1.1)
    block_point = direction * reach * frac
    spread = 0.3 * reach
    offsets = rng.normal(scale=spread, size=(num_agents, 1)) * normal[None, :]
    offsets = offsets + rng.normal(scale=spread * 0.4, size=(num_agents, 1)) * direction[None, :]
    return block_point[None, :] + offsets


def reduced_features(waypoints, target, agents, reach):
    """3-dim reduced feature vector, in REDUCED_FEATURE_NAMES order.
    min_dist_to_agents is rescaled by this item's own candidate_reach_scale
    (not |target|, see synthetic_crowd's docstring -- keeping the same
    divisor used to *place* the crowd is what makes "clearance as a fraction
    of how far the candidates go" a consistent, comparable ratio across
    items) into a scale-free ratio, so a fitted preference generalizes across
    CityWalker scenes with very different absolute coordinate scale (this
    dataset's per-scene scale spread was directly measured: single_clearance
    ranged ~0.1-24 in raw units across just 40 items). total_turn_angle_deg
    and target_alignment_cos are already scale-free (degrees / cosine
    similarity), so they pass through unscaled.
    """
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


def w_true_vector():
    return np.array([CROWD_AVOIDER_PROFILE[n] for n in REDUCED_FEATURE_NAMES], dtype=np.float64)


def local_select_best(w, candidates, target, agents, reach, scale=None):
    scores = [bt_score(w, reduced_features(c, target, agents, reach), scale=scale) for c in candidates]
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx]


def local_refine(waypoints, target, w, agents, reach, num_steps=8, eps=1e-3, scale=None):
    """Same backtracking-line-search design as src/preference/select_and_refine.py's
    refine_towards_preference (only accept a step if it actually improves the
    score, halving on failure) -- reimplemented locally over reduced_features()
    instead of the shared module's fixed 8-dim feature space. See module
    docstring for why. Pass `scale` (from fit_preference_weights) when `w` is
    a fitted weight vector in normalized space."""
    wp = np.array(waypoints, dtype=np.float64)
    target_scale = max(float(np.linalg.norm(target)), 1e-6)
    best_wp = wp
    best_score = bt_score(w, reduced_features(best_wp, target, agents, reach), scale=scale)
    step = 0.02 * target_scale
    for _ in range(num_steps):
        grad = np.zeros_like(best_wp)
        for i in range(best_wp.shape[0]):
            for j in range(best_wp.shape[1]):
                perturbed = best_wp.copy()
                perturbed[i, j] += eps
                s = bt_score(w, reduced_features(perturbed, target, agents, reach), scale=scale)
                grad[i, j] = (s - best_score) / eps
        norm = np.linalg.norm(grad)
        if norm < 1e-8:
            break
        direction = grad / norm
        improved = False
        cur_step = step
        for _ in range(6):
            candidate = best_wp + cur_step * direction
            cand_score = bt_score(w, reduced_features(candidate, target, agents, reach), scale=scale)
            if cand_score > best_score:
                best_wp, best_score = candidate, cand_score
                step = cur_step
                improved = True
                break
            cur_step *= 0.5
        if not improved:
            break
    return best_wp


def main():
    args = parse_args()
    print(">>> loading model:", args.model_path)
    model = SocialNavModel(args.model_path, device=args.device, flow_steps=args.flow_steps)

    print(f">>> loading '{args.category}'-category samples:", args.data_path)
    items = load_category_samples(args.data_path, args.category)
    print(f">>> {len(items)} '{args.category}' samples with path_distance >= 1m")
    items = items[: args.limit]

    crowd_rng = np.random.default_rng(1)
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
        reach = candidate_reach_scale(wp_pred)
        agents = synthetic_crowd(wp_pred, args.num_agents, crowd_rng)
        feats = [reduced_features(c, target, agents, reach) for c in wp_pred]
        per_item.append(
            {
                "candidates": wp_pred,
                "gt": gt,
                "step_scale": step_scale,
                "target": target,
                "agents": agents,
                "reach": reach,
                "feats": feats,
                "images": item["images"],
                "meta": item.get("meta"),
            }
        )

    w_true = w_true_vector()
    print(f">>> {len(per_item)} items x {args.num_samples} real candidates each")

    # Draw each of the N=10 comparisons *within one randomly chosen item* (two
    # of its own candidates), not pooled across items -- a real user judges
    # "A vs B for this navigation instance", never across two unrelated scenes.
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
    print("w_true:", dict(zip(REDUCED_FEATURE_NAMES, np.round(w_true, 3))))
    print("w_hat (normalized space):", dict(zip(REDUCED_FEATURE_NAMES, np.round(w_hat, 3))))
    print("scale :", dict(zip(REDUCED_FEATURE_NAMES, np.round(scale, 3))))

    results = []
    for rec in per_item:
        candidates, gt, step_scale, target, agents, reach = (
            rec["candidates"], rec["gt"], rec["step_scale"], rec["target"], rec["agents"], rec["reach"]
        )
        cand_list = list(candidates)

        single_angle = max_angle_and_hit(candidates[0], gt, step_scale)
        single_clearance = agent_avoidance_feature(candidates[0], agents)
        oracle_best_clearance = float(max(agent_avoidance_feature(c, agents) for c in cand_list))

        best_idx, _ = local_select_best(w_hat, cand_list, target, agents, reach, scale=scale)
        selected = candidates[best_idx]
        selected_angle = max_angle_and_hit(selected, gt, step_scale)
        selected_clearance = agent_avoidance_feature(selected, agents)

        refined = local_refine(selected, target, w_hat, agents, reach, num_steps=args.refine_steps, scale=scale)
        refined_angle = max_angle_and_hit(refined, gt, step_scale)
        refined_clearance = agent_avoidance_feature(refined, agents)

        results.append(
            {
                "single_angle": single_angle,
                "single_clearance": single_clearance,
                "oracle_best_clearance": oracle_best_clearance,
                "scorer_selected_angle": selected_angle,
                "scorer_selected_clearance": selected_clearance,
                "scorer_selected_refined_angle": refined_angle,
                "scorer_selected_refined_clearance": refined_clearance,
                "geometry": {
                    "image": rec["images"][-1],
                    "meta": rec["meta"],
                    "target": target.tolist(),
                    "gt": gt.tolist(),
                    "agents": agents.tolist(),
                    "single": candidates[0].tolist(),
                    "all_candidates": [c.tolist() for c in cand_list],
                    "selected": selected.tolist(),
                    "refined": refined.tolist(),
                },
            }
        )

    def mean(key):
        return float(np.mean([r[key] for r in results]))

    summary = {
        "n_items": len(results),
        "category": args.category,
        "n_shot_comparisons": args.n_shot,
        "num_samples_per_item": args.num_samples,
        "num_agents": args.num_agents,
        "mean_single_sample_angle": mean("single_angle"),
        "mean_single_sample_clearance": mean("single_clearance"),
        "mean_oracle_best_of_n_clearance": mean("oracle_best_clearance"),
        "mean_scorer_selected_angle": mean("scorer_selected_angle"),
        "mean_scorer_selected_clearance": mean("scorer_selected_clearance"),
        "mean_scorer_selected_refined_angle": mean("scorer_selected_refined_angle"),
        "mean_scorer_selected_refined_clearance": mean("scorer_selected_refined_clearance"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    with open(args.output, "w") as f:
        json.dump(
            {"summary": summary, "w_true": w_true.tolist(), "w_hat": w_hat.tolist(), "scale": scale.tolist(), "per_item": results},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()
