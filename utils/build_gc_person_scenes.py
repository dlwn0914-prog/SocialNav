#!/usr/bin/env python
"""Per-PERSON Grand Central scenes for the "learn a preference, use it later"
setting: several scenes from the same pedestrian's walk, so queries answered
in the first scenes can be scored on later ones the person was never asked
about.

Differences from utils/build_gc_scenes.py (one scene per person):
  - goal = where the ego actually is --goal-horizon seconds later (robot
    frame), not a fixed point 8 m straight ahead. The destination is then
    given to the planner, so answers taken from the real path reflect how
    the person walks, not where they were going.
  - per person, up to --scenes-per-person frames at least --min-gap seconds
    apart (in walk order), each with 2..10 other people in Crowd-FM's
    window and the full answer path (--duration s) and goal available.
  - the replaced pedestrian's real path over --duration s (the gc-ego
    answer) is written next to the scenes in ego_future.json, keyed by
    scene name, on Crowd-FM's 30-point grid (as utils/gc_ego_future.py).

Scene names: gcp_<person:03d>_<k>, k = 0.. in walk order; the split into
query scenes and held-out scenes is left to the benchmark.

Usage:
    python utils/build_gc_person_scenes.py --people 150 \
        --out-dir /home/nuri2/crowd_fm_ws/scenarios_gcp/scenes
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from utils.build_gc_scenes import FPS, load_tracks, to_robot_frame

STEP = 20  # frames between annotations (0.8 s)


def interp_path(track, f0, t):
    """track: {frame: (x, y)}; positions at f0 + t seconds, None if the
    annotations stop or skip a step before the last t."""
    fr = [f0]
    while (fr[-1] - f0) / FPS < t[-1]:
        if fr[-1] + STEP not in track:
            return None
        fr.append(fr[-1] + STEP)
    th = (np.array(fr) - f0) / FPS
    xy = np.array([track[f] for f in fr])
    return np.stack([np.interp(t, th, xy[:, 0]), np.interp(t, th, xy[:, 1])], axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", default="/home/nuri2/datasets/grandcentral/raw/opentraj/datasets/GC/Annotation")
    p.add_argument("--out-dir", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/scenes")
    p.add_argument("--people", type=int, default=150)
    p.add_argument("--scenes-per-person", type=int, default=6)
    p.add_argument("--min-gap", type=float, default=5.0, help="seconds between a person's scenes")
    p.add_argument("--goal-horizon", type=float, default=6.4, help="goal = ego position this many seconds ahead")
    p.add_argument("--min-goal-dist", type=float, default=3.0, help="skip frames where the ego barely moves on")
    p.add_argument("--min-ego-speed", type=float, default=0.5)
    p.add_argument("--duration", type=float, default=3.0, help="answer path length [s] (Crowd-FM horizon)")
    p.add_argument("--n-points", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    by_frame = load_tracks(args.annot_dir)
    tracks = defaultdict(dict)  # pid -> {frame: (x, y)}
    for f, peds in by_frame.items():
        for pid, x, y, *_ in peds:
            tracks[pid][f] = (x, y)

    t_ans = np.linspace(0.0, args.duration, args.n_points)
    gap = int(round(args.min_gap * FPS))
    cands = defaultdict(list)  # pid -> [(frame, goal_robot, future_robot, people)]
    for f in sorted(by_frame):
        peds = by_frame[f]
        for ego in peds:
            pid, ex, ey, evx, evy = ego
            if np.hypot(evx, evy) < args.min_ego_speed or (cands[pid] and f - cands[pid][-1][0] < gap):
                continue
            people = to_robot_frame(ego, [q for q in peds if q[0] != pid])
            if not 2 <= len(people) <= 10:
                continue
            fut = interp_path(tracks[pid], f, t_ans)
            goal = interp_path(tracks[pid], f, np.array([0.0, args.goal_horizon]))
            if fut is None or goal is None:
                continue
            th = np.arctan2(evy, evx)
            R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])  # world -> robot
            g = R @ (goal[1] - (ex, ey))
            if np.linalg.norm(g) < args.min_goal_dist:
                continue
            cands[pid].append((f, g, (fut - (ex, ey)) @ R.T, people))

    eligible = sorted(pid for pid, c in cands.items() if len(c) >= args.scenes_per_person)
    rng = np.random.default_rng(args.seed)
    chosen = sorted(rng.choice(eligible, size=min(args.people, len(eligible)), replace=False))
    print(f"{len(eligible)} people with >= {args.scenes_per_person} scenes; using {len(chosen)}")

    ring = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    static = (20.0 * np.stack([np.cos(ring), np.sin(ring)], axis=1)).tolist()
    os.makedirs(args.out_dir, exist_ok=True)
    futures, goal_dist, goal_ang = {}, [], []
    for n, pid in enumerate(chosen):
        for k, (f, g, fut, people) in enumerate(cands[pid][:args.scenes_per_person]):
            name = f"gcp_{n:03d}_{k}"
            with open(os.path.join(args.out_dir, f"{name}.json"), "w") as fh:
                json.dump({"source": {"dataset": "GrandCentral/OpenTraj", "frame": int(f), "ego_ped_id": int(pid),
                                      "person": n, "order": k, "goal_horizon_s": args.goal_horizon},
                           "goal": [float(g[0]), float(g[1])],
                           "people": [[float(v) for v in r[1:]] for r in people],  # x, y, vx, vy
                           "static_obstacles": static}, fh)
            futures[name] = fut.tolist()
            goal_dist.append(np.linalg.norm(g))
            goal_ang.append(np.degrees(np.arctan2(g[1], g[0])))
    with open(os.path.join(os.path.dirname(args.out_dir.rstrip("/")), "ego_future.json"), "w") as fh:
        json.dump(futures, fh)
    ga = np.abs(goal_ang)
    print(f"{len(futures)} scenes; goal distance median {np.median(goal_dist):.1f} m "
          f"[{np.percentile(goal_dist, 5):.1f}, {np.percentile(goal_dist, 95):.1f}], "
          f"|goal angle| median {np.median(ga):.0f} deg, >20 deg in {np.mean(ga > 20):.0%}")


if __name__ == "__main__":
    main()
