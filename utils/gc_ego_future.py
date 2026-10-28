#!/usr/bin/env python
"""Where did the pedestrian the robot replaced ACTUALLY walk? For each Grand
Central scene from utils/build_gc_scenes.py, the ego pedestrian's real path
over the next traj_duration seconds, in the scene's robot frame and on the
same time grid as Crowd-FM's candidates (30 waypoints over 3.0 s).

Used as a data-derived simulated user (utils/unified_query_benchmark.py
--user gc-ego): a person answers "A or B?" by which candidate stays closer
to the path they really took. Their preference is then whatever their
walking revealed -- not a sampled linear theta* -- and differs per person.

Annotations come every 20 frames (0.8 s), so 3 s is 4-5 samples, linearly
interpolated. A scene is marked unavailable when the ego's annotations stop
(or skip a 20-frame step) before frame + traj_duration.

Usage:
    python utils/gc_ego_future.py
"""
import argparse
import glob
import json
import os

import numpy as np

from utils.build_gc_scenes import FPS, read_track, velocity_at


def ego_future(annot_dir, frame, ego_id, duration, n_points):
    frames, w = read_track(os.path.join(annot_dir, f"{ego_id:06d}.txt"))
    k = int(np.flatnonzero(frames == frame)[0])
    v = velocity_at(frames, w, k)
    th = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])  # world -> robot, as in build_gc_scenes
    m = k
    while m + 1 < len(frames) and frames[m + 1] - frames[m] == 20 and (frames[m] - frame) / FPS < duration:
        m += 1
    t_have = (frames[k:m + 1] - frame) / FPS
    if t_have[-1] < duration:
        return None
    t = np.linspace(0.0, duration, n_points)
    pos = np.stack([np.interp(t, t_have, w[k:m + 1, 0]), np.interp(t, t_have, w[k:m + 1, 1])], axis=1)
    return ((pos - w[k]) @ R.T).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--annot-dir", default="/home/nuri2/datasets/grandcentral/raw/opentraj/datasets/GC/Annotation")
    p.add_argument("--scene-dir", default="/home/nuri2/crowd_fm_ws/scenarios_gc/scenes")
    p.add_argument("--out", default="/home/nuri2/crowd_fm_ws/scenarios_gc/ego_future.json")
    p.add_argument("--duration", type=float, default=3.0)
    p.add_argument("--n-points", type=int, default=30)
    args = p.parse_args()

    out = {}
    for path in sorted(glob.glob(os.path.join(args.scene_dir, "gc_*.json"))):
        src = json.load(open(path))["source"]
        fut = ego_future(args.annot_dir, src["frame"], src["ego_ped_id"], args.duration, args.n_points)
        out[os.path.basename(path)[:-5]] = fut
    with open(args.out, "w") as f:
        json.dump(out, f)
    ok = {k: v for k, v in out.items() if v is not None}
    ends = np.array([v[-1] for v in ok.values()])
    print(f"{len(ok)}/{len(out)} scenes have a full {args.duration:.1f} s future; "
          f"endpoint x mean {ends[:, 0].mean():.2f} m, |y| mean {np.abs(ends[:, 1]).mean():.2f} m (max {np.abs(ends[:, 1]).max():.2f})")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
