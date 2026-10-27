#!/usr/bin/env python
"""Build Crowd-FM input scenes from REAL pedestrian layouts in the Grand
Central Station dataset (Zhou et al. CVPR 2011 video; per-pedestrian
annotations by Yi et al. CVPR 2015, as packaged in OpenTraj, ACCV 2020),
replacing the hand-placed people of crowd_fm_ws/run_scenarios.sh.

Coordinates: each Annotation/<id>.txt holds (a, b, frame) triplets in image
pixels every 20 frames (0.8 s at 25 fps). Converted exactly as OpenTraj's
loader_gcs.py does: world = H . [b, a, 1] (dehomogenized) * 0.8, with
OpenTraj's manually estimated homography (from the concourse's published
84 m x 37 m size) -- approximate, so absolute distances carry some error.
Sanity check: median walking speed comes out at 1.12 m/s.

Scene construction ("replace one pedestrian with the robot"): pick an ego
pedestrian walking >= --min-ego-speed at frame f; the robot starts at its
position facing its walking direction, goal 8 m ahead (same goal as the
hand-built scenes). Every other pedestrian annotated at frame f is moved
into that robot frame with its finite-difference velocity; those inside
Crowd-FM's reach window (x in [0.3, 4.0], |y| <= 2.5 -- candidates end at
~3 m) are kept, nearest NUM_DYNAMIC_OBSTACLES at most. No static obstacles
are annotated, so static_obstacles is a ring 20 m away (outside the window).

Scenes are stratified by people-in-window count (2-3 / 4-6 / 7-10).

Usage:
    python utils/build_gc_scenes.py --per-bin 15 \
        --out-dir /home/nuri2/crowd_fm_ws/scenarios_gc/scenes
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

H = np.array([[4.97412897e-02, -4.24730883e-02, 7.25543911e01],
              [1.45017874e-01, -3.35678711e-03, 7.97920970e00],
              [1.36068797e-03, -4.98339188e-05, 1.00000000e00]])
FPS = 25.0
NUM_DYNAMIC_OBSTACLES = 10
WINDOW_X = (0.3, 4.0)
WINDOW_Y = 2.5
GOAL_DIST = 8.0
BINS = {"2-3": (2, 3), "4-6": (4, 6), "7-10": (7, 10)}


def load_tracks(annot_dir):
    """Returns {frame: [(ped_id, x, y, vx, vy), ...]} in world metres."""
    by_frame = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(annot_dir, "*.txt"))):
        a = np.array(open(path).read().split(), dtype=np.float64).reshape(-1, 3)
        p = np.c_[a[:, 1], a[:, 0], np.ones(len(a))] @ H.T
        w = p[:, :2] / p[:, 2:] * 0.8
        frames = a[:, 2].astype(int)
        pid = int(os.path.basename(path)[:-4])
        for k in range(len(a)):
            # central/one-sided difference, only across consecutive annotations (gap of 20 frames)
            nb = [m for m in (k - 1, k + 1) if 0 <= m < len(a) and abs(frames[m] - frames[k]) == 20]
            if not nb:
                continue
            lo, hi = min(nb + [k]), max(nb + [k])
            v = (w[hi] - w[lo]) / ((frames[hi] - frames[lo]) / FPS)
            by_frame[frames[k]].append((pid, w[k, 0], w[k, 1], v[0], v[1]))
    return by_frame


def to_robot_frame(ego, others):
    _, ex, ey, evx, evy = ego
    th = np.arctan2(evy, evx)
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])  # world -> robot
    people = []
    for pid, x, y, vx, vy in others:
        pos = R @ np.array([x - ex, y - ey])
        vel = R @ np.array([vx, vy])
        if WINDOW_X[0] <= pos[0] <= WINDOW_X[1] and abs(pos[1]) <= WINDOW_Y:
            people.append([pid, pos[0], pos[1], vel[0], vel[1]])
    people.sort(key=lambda r: np.hypot(r[1], r[2]))
    return people[:NUM_DYNAMIC_OBSTACLES]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", default="/home/nuri2/datasets/grandcentral/raw/opentraj/datasets/GC/Annotation")
    p.add_argument("--out-dir", default="/home/nuri2/crowd_fm_ws/scenarios_gc/scenes")
    p.add_argument("--per-bin", type=int, default=15)
    p.add_argument("--min-ego-speed", type=float, default=0.5)
    p.add_argument("--min-frame-gap", type=int, default=500, help="frames between chosen scenes (20 s), for diversity")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    by_frame = load_tracks(args.annot_dir)
    frames = np.array(sorted(by_frame))
    os.makedirs(args.out_dir, exist_ok=True)

    chosen = {b: [] for b in BINS}
    used_frames = []
    for f in rng.permutation(frames):
        if all(len(v) >= args.per_bin for v in chosen.values()):
            break
        if any(abs(f - u) < args.min_frame_gap for u in used_frames):
            continue
        peds = by_frame[f]
        movers = [q for q in peds if np.hypot(q[3], q[4]) >= args.min_ego_speed]
        for ego in rng.permutation(len(movers)):
            people = to_robot_frame(movers[ego], [q for q in peds if q[0] != movers[ego][0]])
            b = next((b for b, (lo, hi) in BINS.items() if lo <= len(people) <= hi), None)
            if b is None or len(chosen[b]) >= args.per_bin:
                continue
            chosen[b].append((int(f), movers[ego][0], people))
            used_frames.append(f)
            break

    ring = np.linspace(0, 2 * np.pi, 500, endpoint=False)
    static = (20.0 * np.stack([np.cos(ring), np.sin(ring)], axis=1)).tolist()
    for b, scenes in chosen.items():
        for k, (f, ego_id, people) in enumerate(scenes):
            name = f"gc_{b.replace('-', 'to')}_{k:02d}"
            with open(os.path.join(args.out_dir, f"{name}.json"), "w") as fh:
                json.dump({"source": {"dataset": "GrandCentral/OpenTraj", "frame": f, "ego_ped_id": ego_id},
                           "goal": [GOAL_DIST, 0.0],
                           "people": [[float(v) for v in r[1:]] for r in people],  # x, y, vx, vy
                           "static_obstacles": static}, fh)
        print(f"bin {b}: {len(scenes)} scenes")


if __name__ == "__main__":
    main()
