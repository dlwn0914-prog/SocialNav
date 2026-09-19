#!/usr/bin/env python
"""Build a citywalker_test.jsonl for utils/citywalker.py from the raw teleportation
data (data/obs/traj_nav_XX + data/pose_label/pose_traj_XX.txt).

This is a faithful port of the official CityWalker `TeleopDataset`
(ai4ce/CityWalker, data/teleop_dataset.py) sample-construction logic, run under the
exact hyperparameters in that repo's config/finetune.yaml:
  context_size=5, len_traj_pred=5, cord_embedding=input_target,
  search_window=50, arrived_threshold=5, arrived_prob=0.3,
  num_train=8, num_val=1, num_test=9  ->  test split = pose_traj_10..18 (9 sequences)

Two details carried over verbatim from the official loader (easy to get wrong by
guessing, so re-derived directly from its source instead):
  - History waypoints (<input_pos1>..<input_pos5>) are built from GPS-derived local
    ENU positions (lat/lon -> meters via an equirectangular approx, translate+rotate
    so the 2nd-to-last history point faces -y), NOT from the odometry `poses` array.
  - The target waypoint (<input_target>) and `gt_waypoints` (future ground truth) are
    built from the odometry `poses` array (tx,ty,tz,rx,ry,rz) via a full 4x4
    current-pose-relative transform, NOT from GPS.
  - Both are divided by a per-sequence `step_scale` (mean consecutive-pose step
    length) before being written out, matching what utils/citywalker.py expects
    (it multiplies back by `step_scale` to recover absolute-meter waypoints).

KNOWN LIMITATION: the literal instruction text surrounding the placeholder tokens
(<input_pos1>..<input_pos5>, <input_target>) that SocialNav's own training/benchmark
jsonls used is not recoverable from any available source (not in the SocialNav repo,
its released model cards, or its arXiv paper). SocialNav's architecture injects
waypoint embeddings at the placeholder token ids rather than parsing the surrounding
wording, so PROMPT_TEMPLATE below is a best-effort stand-in that preserves the
required tokens/order; absolute metric values may therefore not exactly reproduce the
paper's reported numbers even though the CityWalker-side data construction here is
faithful to the official preprocessing.

The `arrive` field involves a random target-frame selection in the official loader
(np.random.rand()/random.randint, even in test mode); a fixed seed is used below for
reproducibility since the original seed (if any) used to freeze SocialNav's benchmark
jsonl is not recoverable either.
"""
import argparse
import json
import os
import random

import numpy as np
from scipy.spatial.transform import Rotation as Rot

TEST_CATEGORIES = ["crowd", "person_close_by", "turn", "action_target_mismatch", "crossing", "other"]

CONTEXT_SIZE = 5
WP_LENGTH = 5
SEARCH_WINDOW = 50
ARRIVED_THRESHOLD = 5
ARRIVED_PROB = 0.3
NUM_TRAIN, NUM_VAL, NUM_TEST = 8, 1, 9
SEED = 42

PROMPT_TEMPLATE = (
    f"You are a robot navigating towards a goal. You are given {CONTEXT_SIZE} history observation "
    "images (oldest to most recent) and their positions relative to the current pose: "
    + ", ".join(f"<input_pos{i}>" for i in range(1, CONTEXT_SIZE + 1))
    + ". The goal position relative to the current pose is <input_target>. "
    f"Predict the next {WP_LENGTH} waypoints towards the goal."
)


def latlon_to_local(lat, lon, lat0, lon0):
    r_earth = 6378137.0
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)
    lat0_rad, lon0_rad = np.radians(lat0), np.radians(lon0)
    dlat = lat_rad - lat0_rad
    dlon = lon_rad - lon0_rad
    x = dlon * np.cos((lat_rad + lat0_rad) / 2) * r_earth
    y = dlat * r_earth
    return x, y


def pose_to_matrix(pose):
    tx, ty, tz, rx, ry, rz = pose
    m = np.eye(4)
    m[:3, :3] = Rot.from_rotvec([rx, ry, rz]).as_matrix()
    m[:3, 3] = [tx, ty, tz]
    return m


def poses_to_matrices(poses):
    mats = np.tile(np.eye(4), (poses.shape[0], 1, 1))
    mats[:, :3, :3] = Rot.from_rotvec(poses[:, 3:6]).as_matrix()
    mats[:, :3, 3] = poses[:, :3]
    return mats


def transform_poses(poses, current_pose):
    cur_inv = np.linalg.inv(pose_to_matrix(current_pose))
    mats = poses_to_matrices(poses)
    transformed = np.matmul(cur_inv[np.newaxis, :, :], mats)
    pos = transformed[:, :3, 3].copy()
    pos[:, [0, 1]] = pos[:, [1, 0]]
    pos[:, 1] *= -1
    return pos


def transform_pose(pose, current_pose):
    cur_inv = np.linalg.inv(pose_to_matrix(current_pose))
    t = np.matmul(cur_inv, pose_to_matrix(pose))
    pos = t[:3, 3].copy()
    pos[[0, 1]] = pos[[1, 0]]
    pos[1] *= -1
    return pos


def transform_input(input_positions):
    current_position = input_positions[-1]
    translated = input_positions - current_position
    second_last = translated[-2]
    angle = -np.pi / 2 - np.arctan2(second_last[1], second_last[0])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return translated[:, :2] @ rot.T


def select_target_index(future_positions):
    arrived = np.random.rand() < ARRIVED_PROB
    max_idx = future_positions.shape[0] - 1
    if arrived:
        target_idx = random.randint(WP_LENGTH, min(WP_LENGTH + ARRIVED_THRESHOLD, max_idx))
    else:
        target_idx = random.randint(WP_LENGTH + ARRIVED_THRESHOLD, max_idx)
    return target_idx, arrived


def load_sequence(pose_path, image_root_dir):
    seq_idx = "".join(filter(str.isdigit, os.path.basename(pose_path)))
    image_folder = os.path.join(image_root_dir, f"traj_nav_{seq_idx}")
    if not os.path.isdir(image_folder):
        raise FileNotFoundError(image_folder)

    with open(pose_path) as f:
        lines = f.readlines()

    gps_positions, poses, images, categories = [], [], [], []
    ref_lat = ref_lon = ref_alt = None

    for i in range(0, len(lines), 3):
        gps_tokens = lines[i].strip().split(",")
        lat, lon, alt = float(gps_tokens[1]), float(gps_tokens[2]), float(gps_tokens[4])
        if ref_lat is None:
            ref_lat, ref_lon, ref_alt = lat, lon, alt
        x, y = latlon_to_local(lat, lon, ref_lat, ref_lon)
        gps_positions.append([x, y, alt - ref_alt])

        pose_tokens = lines[i + 1].strip().split(",")
        tx, ty, tz, rx, ry, rz = (float(v) for v in pose_tokens[1:7])
        poses.append([tx, ty, tz, rx, ry, rz])
        images.append(f"forward_{int(pose_tokens[7]):04d}.jpg")

        categories.append([int(v) for v in lines[i + 2].strip().split(",")])

    gps_positions = np.array(gps_positions, dtype=np.float64)
    poses = np.array(poses, dtype=np.float64)
    categories = np.array(categories, dtype=np.int32)

    step_scale = float(np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1).mean())
    step_scale = max(step_scale, 1e-2)

    return seq_idx, image_folder, gps_positions, poses, images, categories, step_scale


def build_sequence_samples(seq_idx, image_folder, gps_positions, poses, images, categories, step_scale):
    n = poses.shape[0]
    usable = n - CONTEXT_SIZE - max(ARRIVED_THRESHOLD * 2, WP_LENGTH)
    samples = []

    for pose_start in range(0, max(usable, 0)):
        future_waypoints = poses[pose_start + CONTEXT_SIZE : pose_start + CONTEXT_SIZE + SEARCH_WINDOW]
        if future_waypoints.shape[0] == 0:
            continue
        target_idx, arrived = select_target_index(future_waypoints)

        input_poses = poses[pose_start : pose_start + CONTEXT_SIZE]
        waypoint_start = pose_start + CONTEXT_SIZE
        waypoint_end = waypoint_start + WP_LENGTH
        gt_waypoint_poses = poses[waypoint_start:waypoint_end]
        if gt_waypoint_poses.shape[0] < WP_LENGTH:
            continue

        current_pose = input_poses[-1]
        gt_waypoints = transform_poses(gt_waypoint_poses, current_pose)[:, :2]

        target_pose_idx = pose_start + CONTEXT_SIZE + target_idx
        if target_pose_idx >= n:
            continue
        target_pose = poses[target_pose_idx]
        target_transformed = transform_pose(target_pose, current_pose)[:2]

        input_gps_positions = gps_positions[pose_start : pose_start + CONTEXT_SIZE]
        rotated_input = transform_input(input_gps_positions)
        input_positions = np.concatenate([rotated_input, target_transformed[np.newaxis, :]], axis=0)

        gt_waypoints_scaled = gt_waypoints / step_scale
        input_positions_scaled = input_positions / step_scale

        cat_idx = pose_start + CONTEXT_SIZE - 1
        cats = categories[cat_idx].tolist()

        image_paths = [
            os.path.join(image_folder, name) for name in images[pose_start : pose_start + CONTEXT_SIZE]
        ]

        samples.append(
            {
                "images": image_paths,
                "messages": [
                    {"role": "user", "content": PROMPT_TEMPLATE},
                    {
                        "role": "assistant",
                        "gt_waypoints": gt_waypoints_scaled.tolist(),
                        "input_waypoints": input_positions_scaled.tolist(),
                        "step_scale": step_scale,
                        "arrive": [int(arrived)],
                        "categories": cats,
                    },
                ],
                "meta": {"seq": seq_idx, "pose_start": pose_start},
            }
        )
    return samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "pose_label"))
    p.add_argument("--image-root-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "obs"))
    p.add_argument("--output", default=os.path.join(os.path.dirname(__file__), "..", "data", "citywalker_test.jsonl"))
    p.add_argument("--limit-per-seq", type=int, default=None, help="Cap samples per sequence (debug).")
    args = p.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)

    pose_files = [f"pose_traj_{i:02d}.txt" for i in range(1, NUM_TRAIN + NUM_VAL + NUM_TEST + 1)]
    test_files = pose_files[NUM_TRAIN + NUM_VAL : NUM_TRAIN + NUM_VAL + NUM_TEST]
    print(f"Test split sequences: {test_files}")

    total = 0
    with open(args.output, "w") as fout:
        for fname in test_files:
            pose_path = os.path.join(args.pose_dir, fname)
            seq_idx, image_folder, gps_positions, poses, images, categories, step_scale = load_sequence(
                pose_path, args.image_root_dir
            )
            samples = build_sequence_samples(
                seq_idx, image_folder, gps_positions, poses, images, categories, step_scale
            )
            if args.limit_per_seq is not None:
                samples = samples[: args.limit_per_seq]
            print(f"seq {seq_idx}: {len(samples)} samples, step_scale={step_scale:.4f}")
            for s in samples:
                fout.write(json.dumps(s) + "\n")
                total += 1

    print(f"Wrote {total} samples to {args.output}")


if __name__ == "__main__":
    main()
