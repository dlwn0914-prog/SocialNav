#!/usr/bin/env python
"""Are there >= 2 SAFE and meaningfully DIFFERENT candidates to ask a user
about? Compares Crowd-FM's raw flow samples ("raw_candidates") against the
PRIEST-refined ones ("candidates"), under the old frozen-people safety check
and the constant-velocity one (src/preference/safety_filter.py).

Needs scenario files from the updated extractor (raw candidates + people
velocities saved): crowd_fm_ws/scenarios_v2/{hand,more,gc}.

Per (scene set, candidate type, filter):
  usable      scenes with >= 2 safe candidates (no fallback)
  safe        mean number of safe candidates per scene (of 20)
  gap         mean over usable scenes of the mean pairwise max pointwise
              distance between safe candidates [m]
  lateral     mean over usable scenes of the spread (max - min) of the safe
              candidates' lateral endpoint y [m]

Usage:
    PYTHONPATH=. .venv/bin/python utils/candidate_diversity_report.py
"""
import argparse
import glob
import json
import os

import numpy as np

from src.preference.safety_filter import filter_safe_candidates


def scene_stats(path, key, moving):
    d = json.load(open(path))
    cands = np.array(d[key], dtype=np.float64)
    people = np.array(d["people"], dtype=np.float64).reshape(-1, 2)
    static = np.array(d.get("static_obstacles", []), dtype=np.float64)
    if static.size:
        static = static[np.linalg.norm(static, axis=1) > 1e-5]
    vel = np.array(d["people_vel"], dtype=np.float64).reshape(-1, 2) if moving else None
    f = filter_safe_candidates(list(cands), static_obs=static, dynamic_obs=people,
                               dynamic_vel=vel, duration=d["traj_duration"] if moving else None)
    safe = [] if f["used_fallback"] else f["safe_indices"]
    if len(safe) < 2:
        return len(safe), None, None
    C = cands[safe]
    gaps = np.linalg.norm(C[:, None] - C[None], axis=-1).max(-1)[np.triu_indices(len(C), 1)]
    return len(safe), float(gaps.mean()), float(np.ptp(C[:, -1, 1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/nuri2/crowd_fm_ws/scenarios_v2")
    p.add_argument("--out", default="results/candidate_diversity.json")
    args = p.parse_args()

    report = {}
    print(f"{'set':>5} {'candidates':>15} {'people':>7} {'usable':>9} {'safe':>6} {'gap[m]':>7} {'lateral[m]':>11}")
    for sub in ["hand", "more", "gc"]:
        paths = sorted(glob.glob(os.path.join(args.root, sub, "scenario_*.json")))
        for key in ["raw_candidates", "candidates"]:
            for moving in [False, True]:
                rows = [scene_stats(pth, key, moving) for pth in paths]
                usable = [r for r in rows if r[1] is not None]
                entry = {
                    "n_scenes": len(rows),
                    "usable": len(usable),
                    "mean_safe": float(np.mean([r[0] for r in rows])),
                    "mean_gap": float(np.mean([r[1] for r in usable])) if usable else None,
                    "mean_lateral": float(np.mean([r[2] for r in usable])) if usable else None,
                }
                report[f"{sub}/{key}/{'moving' if moving else 'frozen'}"] = entry
                gap = f"{entry['mean_gap']:.2f}" if usable else "-"
                lat = f"{entry['mean_lateral']:.2f}" if usable else "-"
                print(f"{sub:>5} {key:>15} {'moving' if moving else 'frozen':>7} {len(usable):>4}/{len(rows):<4}"
                      f" {entry['mean_safe']:>6.1f} {gap:>7} {lat:>11}")
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
