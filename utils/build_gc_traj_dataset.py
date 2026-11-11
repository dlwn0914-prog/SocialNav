#!/usr/bin/env python
"""Grand Central pedestrian paths + their situation, for learning a codebook
of walking behaviours (VQ-VAE) and which behaviour people use in which
situation.

One sample per annotated (person, frame) where the person walks >= --min-speed:
  future   (H, 2)  positions at 0.8, 1.6, ... H*0.8 s, robot frame (x = current
                   heading from velocity, y = left), straight from the 0.8 s
                   annotations -- no interpolation, so samples need H
                   consecutive annotations
  past     (2, 2)  positions 1.6 and 0.8 s ago, same frame
  speed    ()      current speed [m/s]
  goal     (2,)    position --goal-horizon s ahead (as build_gc_person_scenes.py),
                   the destination a robot would be given
  dest     (2,)    the track's last position, same frame, and dest_t (s) how far
                   ahead it is: a destination that does not carry the next few
                   seconds' detour the way the 6.4 s goal does (when dest_t is
                   large)
  people   (N, 4)  the N nearest others within --radius: x, y, vx, vy, robot
                   frame; zero rows past n_people
Split by person id (no person in two splits). The 150 people of
build_gc_person_scenes.py (same seed and eligibility) go to test, so the
codebook never sees the people the transfer benchmark uses.

Usage:
    PYTHONPATH=. .venv/bin/python utils/build_gc_traj_dataset.py \
        --out /home/nuri2/datasets/grandcentral/traj_codebook.npz
"""
import argparse
import json
import glob
import os

import numpy as np

from utils.build_gc_scenes import FPS, load_tracks

STEP = 20  # frames between annotations (0.8 s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", default="/home/nuri2/datasets/grandcentral/raw/opentraj/datasets/GC/Annotation")
    p.add_argument("--out", default="/home/nuri2/datasets/grandcentral/traj_codebook.npz")
    p.add_argument("--horizon", type=int, default=6, help="future annotations (x 0.8 s)")
    p.add_argument("--goal-horizon", type=float, default=6.4)
    p.add_argument("--min-speed", type=float, default=0.5)
    p.add_argument("--radius", type=float, default=6.0)
    p.add_argument("--max-people", type=int, default=12)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--gcp-scene-dir", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/scenes")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    by_frame = load_tracks(args.annot_dir)
    track = {}  # pid -> {frame: (x, y, vx, vy)}
    for f, peds in by_frame.items():
        for pid, x, y, vx, vy in peds:
            track.setdefault(pid, {})[f] = (x, y, vx, vy)
    last = {pid: max(tr) for pid, tr in track.items()}
    g_steps = int(round(args.goal_horizon * FPS / STEP))
    need = max(args.horizon, g_steps)

    cols = {k: [] for k in ["pid", "frame", "future", "past", "speed", "goal", "people", "n_people", "dest", "dest_t"]}
    for f in sorted(by_frame):
        peds = by_frame[f]
        for pid, ex, ey, evx, evy in peds:
            sp = np.hypot(evx, evy)
            tr = track[pid]
            if sp < args.min_speed or any(f + k * STEP not in tr for k in range(1, need + 1)) \
                    or any(f - k * STEP not in tr for k in (1, 2)):
                continue
            th = np.arctan2(evy, evx)
            R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])  # world -> robot
            rel = lambda k: R @ (np.array(tr[f + k * STEP][:2]) - (ex, ey))
            others = []
            for q, x, y, vx, vy in peds:
                if q == pid:
                    continue
                pos = R @ np.array([x - ex, y - ey])
                if np.linalg.norm(pos) <= args.radius:
                    others.append([*pos, *(R @ np.array([vx, vy]))])
            others.sort(key=lambda r: np.hypot(r[0], r[1]))
            others = others[:args.max_people]
            ppl = np.zeros((args.max_people, 4))
            if others:
                ppl[:len(others)] = others
            cols["pid"].append(pid)
            cols["frame"].append(f)
            cols["future"].append([rel(k) for k in range(1, args.horizon + 1)])
            cols["past"].append([rel(-2), rel(-1)])
            cols["speed"].append(sp)
            cols["goal"].append(rel(g_steps))
            cols["people"].append(ppl)
            cols["n_people"].append(len(others))
            cols["dest"].append(R @ (np.array(tr[last[pid]][:2]) - (ex, ey)))
            cols["dest_t"].append((last[pid] - f) / FPS)
    data = {k: np.asarray(v, dtype=np.float32 if k not in ("pid", "frame", "n_people") else np.int64)
            for k, v in cols.items()}

    held = set()
    for path in glob.glob(os.path.join(args.gcp_scene_dir, "gcp_*.json")):
        held.add(int(json.load(open(path))["source"]["ego_ped_id"]))
    pids = np.array(sorted(set(data["pid"].tolist()) - held))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(pids)
    n_val, n_test = int(len(pids) * args.val_frac), int(len(pids) * args.test_frac)
    split_of = {int(q): 1 for q in pids[:n_val]}
    split_of.update({int(q): 2 for q in pids[n_val:n_val + n_test]})
    split_of.update({q: 2 for q in held})
    data["split"] = np.array([split_of.get(int(q), 0) for q in data["pid"]], dtype=np.int64)  # 0 train 1 val 2 test

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **data, horizon_s=np.float32(args.horizon * STEP / FPS))
    n = len(data["pid"])
    print(f"{n} samples from {len(set(data['pid'].tolist()))} people "
          f"(train {np.sum(data['split'] == 0)}, val {np.sum(data['split'] == 1)}, test {np.sum(data['split'] == 2)}; "
          f"{len(held)} transfer-benchmark people in test)")
    print(f"speed median {np.median(data['speed']):.2f} m/s; people within {args.radius} m median "
          f"{np.median(data['n_people']):.0f}; end of path median x {np.median(data['future'][:, -1, 0]):.2f} m, "
          f"|y| median {np.median(np.abs(data['future'][:, -1, 1])):.2f} m")


if __name__ == "__main__":
    main()
