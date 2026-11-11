#!/usr/bin/env python
"""ETH/UCY pedestrian paths in the same format as
utils/build_gc_traj_dataset.py, to check whether the walking-behaviour
codebook finds situation-dependent (social) behaviour on denser, finer data
than Grand Central.

Input: the Social-GAN / Trajectron++ text files (frame, ped id, x, y in
metres; one annotation every 0.4 s), one per scene: biwi_eth, biwi_hotel,
students001, students003, crowds_zara01, crowds_zara02 (the "test" file of
each leave-one-out split is the whole scene). Downloaded from
StanfordASL/Trajectron-plus-plus experiments/pedestrians/raw.

Paths, past and goal are taken every 2nd annotation (0.8 s) so the codebook
input matches Grand Central exactly; velocity uses the 0.4 s neighbours.
Split by person (ids made unique per scene); --test-scene instead puts one
whole scene in test.

Usage:
    PYTHONPATH=. .venv/bin/python utils/build_ethucy_traj_dataset.py \
        --out /home/nuri2/datasets/ethucy/traj_codebook.npz
"""
import argparse
import os
from collections import defaultdict

import numpy as np

SCENES = ["biwi_eth", "biwi_hotel", "students001", "students003", "crowds_zara01", "crowds_zara02"]
DT = 0.4  # seconds between annotations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="/home/nuri2/datasets/ethucy/raw")
    p.add_argument("--out", default="/home/nuri2/datasets/ethucy/traj_codebook.npz")
    p.add_argument("--horizon", type=int, default=6, help="future points, every 0.8 s")
    p.add_argument("--goal-horizon", type=float, default=6.4)
    p.add_argument("--min-speed", type=float, default=0.5)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--max-people", type=int, default=12)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--test-scene", default=None, choices=SCENES)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cols = {k: [] for k in ["pid", "frame", "future", "past", "speed", "goal", "people", "n_people", "scene", "dest", "dest_t"]}
    g_steps = int(round(args.goal_horizon / DT))
    for si, scene in enumerate(SCENES):
        a = np.loadtxt(os.path.join(args.raw_dir, f"{scene}.txt"))
        frames = np.unique(a[:, 0])
        step = int(np.median(np.diff(frames)))  # frame numbers per 0.4 s annotation
        track = defaultdict(dict)
        for f, q, x, y in a:
            track[int(q)][int(f)] = (x, y)
        last = {q: max(tr) for q, tr in track.items()}
        by_frame = defaultdict(list)  # frame -> [(pid, x, y, vx, vy)]
        for q, tr in track.items():
            for f, (x, y) in tr.items():
                if f - step in tr and f + step in tr:
                    v = (np.array(tr[f + step]) - tr[f - step]) / (2 * DT)
                    by_frame[f].append((q, x, y, v[0], v[1]))
        for f, peds in by_frame.items():
            for q, ex, ey, evx, evy in peds:
                tr = track[q]
                sp = np.hypot(evx, evy)
                need = [f + k * step for k in range(1, max(2 * args.horizon, g_steps) + 1)] + \
                       [f - k * step for k in range(1, 5)]
                if sp < args.min_speed or any(n not in tr for n in need):
                    continue
                th = np.arctan2(evy, evx)
                R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])  # world -> robot
                rel = lambda n: R @ (np.array(tr[f + n * step]) - (ex, ey))
                others = []
                for o, x, y, vx, vy in peds:
                    if o == q:
                        continue
                    pos = R @ np.array([x - ex, y - ey])
                    if np.linalg.norm(pos) <= args.radius:
                        others.append([*pos, *(R @ np.array([vx, vy]))])
                others.sort(key=lambda r: np.hypot(r[0], r[1]))
                others = others[:args.max_people]
                ppl = np.zeros((args.max_people, 4))
                if others:
                    ppl[:len(others)] = others
                cols["pid"].append(si * 100000 + q)
                cols["frame"].append(f)
                cols["future"].append([rel(2 * k) for k in range(1, args.horizon + 1)])
                cols["past"].append([rel(-4), rel(-2)])
                cols["speed"].append(sp)
                cols["goal"].append(rel(g_steps))
                cols["people"].append(ppl)
                cols["n_people"].append(len(others))
                cols["scene"].append(si)
                cols["dest"].append(R @ (np.array(tr[last[q]]) - (ex, ey)))  # track end, as build_gc_traj_dataset.py
                cols["dest_t"].append((last[q] - f) / step * DT)
    data = {k: np.asarray(v, dtype=np.float32 if k not in ("pid", "frame", "n_people", "scene") else np.int64)
            for k, v in cols.items()}

    rng = np.random.default_rng(args.seed)
    pids = np.array(sorted(set(data["pid"].tolist())))
    if args.test_scene is not None:
        test = set(q for q in pids if q // 100000 == SCENES.index(args.test_scene))
        pids = np.array([q for q in pids if q not in test])
    rng.shuffle(pids)
    n_val = int(len(pids) * args.val_frac)
    n_test = 0 if args.test_scene is not None else int(len(pids) * args.test_frac)
    split_of = {int(q): 1 for q in pids[:n_val]}
    split_of.update({int(q): 2 for q in pids[n_val:n_val + n_test]})
    if args.test_scene is not None:
        split_of.update({int(q): 2 for q in test})
    data["split"] = np.array([split_of.get(int(q), 0) for q in data["pid"]], dtype=np.int64)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **data, horizon_s=np.float32(args.horizon * 2 * DT))
    print(f"{len(data['pid'])} samples from {len(set(data['pid'].tolist()))} people "
          f"(train {np.sum(data['split'] == 0)}, val {np.sum(data['split'] == 1)}, test {np.sum(data['split'] == 2)})")
    for si, s in enumerate(SCENES):
        m = data["scene"] == si
        print(f"  {s:>14}: {m.sum():6d} samples, people within {args.radius} m median {np.median(data['n_people'][m]):.0f}")
    print(f"speed median {np.median(data['speed']):.2f} m/s; end of path median x {np.median(data['future'][:, -1, 0]):.2f} m, "
          f"|y| median {np.median(np.abs(data['future'][:, -1, 1])):.2f} m")


if __name__ == "__main__":
    main()
