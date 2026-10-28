#!/usr/bin/env python
"""How well can ANY linear preference over a feature set pick the candidate
closest to the path a real pedestrian walked? Run before tuning query
selection: if the best possible weights cannot pick it, more or smarter A/B
questions cannot either -- the feature set is the bottleneck.

Data: Grand Central scenes (crowd_fm_ws/scenarios_v2/gc) + the replaced
pedestrian's real 3 s path (utils/gc_ego_future.py). Per scene: safe
candidates (constant-velocity people), "answer" = candidate with the lowest
ADE to the real path, features z-normalized within the scene (as the
benchmark does).

Two bounds per feature set, by searching weight directions on the unit
sphere (--n-dirs samples; exact enough for d <= 8):
  personal   best direction chosen PER SCENE (person) -- the ceiling for any
             personalized linear method, however it asks
  population ONE direction for all scenes -- what a single non-personalized
             preference achieves
Reported: top-1 rate and normalized regret (ADE(pick) - ADE_min) / (ADE_max -
ADE_min). personal - population = the most personalization could add.
The population direction is fit on the very scenes it is scored on (in-sample,
optimistic); "population cv" instead picks it on the other folds' scenes and
scores the held-out ones -- the fair non-personalized baseline.

Feature sets are cumulative, in the order the next ones would be added.

Usage:
    PYTHONPATH=. .venv/bin/python utils/feature_upper_bound.py --candidates raw_candidates
"""
import argparse
import glob
import json
import os

import numpy as np

from src.preference.features import extract_features, social_features, FEATURE_NAMES
from src.preference.safety_filter import filter_safe_candidates
from utils.fisher_optimal_on_real_candidates import candidate_reach_scale
from src.preference.features import agent_avoidance_feature

BASE = ["total_turn_angle_deg", "target_alignment_cos"]
FEATURE_SETS = [
    ("base3 (current)", BASE + ["min_dist_static"]),
    ("base2 + min_dist_moving", BASE + ["min_dist_moving"]),
    ("+ passing_side", BASE + ["min_dist_moving", "passing_side"]),
    ("+ intrusion_time", BASE + ["min_dist_moving", "passing_side", "intrusion_time_s"]),
    ("+ progress", BASE + ["min_dist_moving", "passing_side", "intrusion_time_s", "progress_m"]),
]


def all_features(c, d, people, vel, reach):
    base = extract_features(c, np.array(d["goal"]))
    soc = social_features(c, people, vel, d["traj_duration"])
    dist = agent_avoidance_feature(c, people)
    f = {n: base[FEATURE_NAMES.index(n)] for n in BASE}
    f["min_dist_static"] = min(dist, reach) if np.isfinite(dist) else reach
    f.update(dict(zip(["min_dist_moving", "passing_side", "intrusion_time_s", "progress_m"], soc)))
    return f


def load_scenes(scene_dir, futures, key):
    scenes = []
    for path in sorted(glob.glob(os.path.join(scene_dir, "scenario_gc_*.json"))):
        name = os.path.basename(path)[len("scenario_"):-len(".json")]
        if futures.get(name) is None:
            continue
        d = json.load(open(path))
        people = np.array(d["people"], dtype=np.float64).reshape(-1, 2)
        vel = np.array(d["people_vel"], dtype=np.float64).reshape(-1, 2)
        static = np.array(d["static_obstacles"], dtype=np.float64)
        cands = np.array(d[key], dtype=np.float64)
        f = filter_safe_candidates(list(cands), static_obs=static, dynamic_obs=people,
                                   dynamic_vel=vel, duration=d["traj_duration"])
        if f["used_fallback"] or len(f["safe_indices"]) < 2:
            continue
        safe = cands[f["safe_indices"]]
        reach = candidate_reach_scale(list(safe))
        ade = np.linalg.norm(safe - np.array(futures[name])[None], axis=-1).mean(axis=1)
        scenes.append({"name": name, "feats": [all_features(c, d, people, vel, reach) for c in safe], "ade": ade})
    return scenes


def matrix(scene, names):
    X = np.array([[f[n] for n in names] for f in scene["feats"]])
    sd = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(sd < 1e-9, 1.0, sd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", default="/home/nuri2/crowd_fm_ws/scenarios_v2/gc")
    p.add_argument("--ego-future", default="/home/nuri2/crowd_fm_ws/scenarios_gc/ego_future.json")
    p.add_argument("--candidates", choices=["candidates", "raw_candidates"], default="raw_candidates")
    p.add_argument("--n-dirs", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    scenes = load_scenes(args.scene_dir, json.load(open(args.ego_future)), args.candidates)
    print(f"{len(scenes)} usable scenes ({args.candidates}, moving-people safety filter)")
    rand_top1 = np.mean([1 / len(s["ade"]) for s in scenes])
    print(f"random pick: top-1 {rand_top1:.3f}\n")
    folds = rng.permutation(len(scenes)) % args.cv_folds
    print(f"{'feature set':>26} {'d':>2} {'personal top1':>14} {'regret':>7} {'population top1':>16} {'regret':>7}"
          f" {'pop-cv top1':>12} {'regret':>7}")
    report = {}
    for label, names in FEATURE_SETS:
        dirs = rng.standard_normal((args.n_dirs, len(names)))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        per_regret, per_top1 = [], []
        regs, hits = [], []
        for s in scenes:
            ade = s["ade"]
            picks = np.argmax(matrix(s, names) @ dirs.T, axis=0)  # (n_dirs,)
            reg = (ade[picks] - ade.min()) / max(ade.max() - ade.min(), 1e-9)
            hit = (picks == np.argmin(ade)).astype(float)
            per_regret.append(reg.min())
            per_top1.append(hit.max())
            regs.append(reg)
            hits.append(hit)
        regs, hits = np.array(regs), np.array(hits)  # (n_scenes, n_dirs)
        pop_regret, pop_top1 = regs.sum(0), hits.sum(0)
        b = int(np.argmin(pop_regret))
        cv_reg, cv_hit = np.zeros(len(scenes)), np.zeros(len(scenes))
        for k in range(args.cv_folds):
            bk = int(np.argmin(regs[folds != k].sum(0)))
            cv_reg[folds == k], cv_hit[folds == k] = regs[folds == k, bk], hits[folds == k, bk]
        row = {"d": len(names), "features": names,
               "personal_top1": float(np.mean(per_top1)), "personal_regret": float(np.mean(per_regret)),
               "population_top1": float(pop_top1[b] / len(scenes)), "population_regret": float(pop_regret[b] / len(scenes)),
               "population_direction": dirs[b].tolist(),
               "population_cv_top1": float(cv_hit.mean()), "population_cv_regret": float(cv_reg.mean())}
        report[label] = row
        print(f"{label:>26} {len(names):>2} {row['personal_top1']:>14.3f} {row['personal_regret']:>7.3f}"
              f" {row['population_top1']:>16.3f} {row['population_regret']:>7.3f}"
              f" {row['population_cv_top1']:>12.3f} {row['population_cv_regret']:>7.3f}")
    out = args.out or f"results/feature_upper_bound_{args.candidates}.json"
    with open(out, "w") as f:
        json.dump({"args": vars(args), "n_scenes": len(scenes), "random_top1": rand_top1, "sets": report}, f, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
