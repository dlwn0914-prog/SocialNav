#!/usr/bin/env python
"""Validate src/preference/safety_filter.py against the real Crowd-FM
candidates extracted in crowd_fm_ws/extract_real_candidates.py: (1) the
normal scenario as generated (check how many of the K real candidates pass
the hard safety filter), and (2) an artificially crowded version of the
same scene (inject obstacles directly on top of every candidate's path) to
confirm the fallback path actually triggers and returns exactly one
least-unsafe candidate instead of an empty set.

Usage:
    PYTHONPATH=. python utils/test_safety_filter.py \
        --candidates-json /home/nuri2/crowd_fm_ws/real_candidates.json
"""
import argparse
import json

import numpy as np

from src.preference.safety_filter import filter_safe_candidates, STATIC_RADIUS, DYNAMIC_RADIUS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidates-json", default="/home/nuri2/crowd_fm_ws/real_candidates.json")
    args = p.parse_args()

    with open(args.candidates_json) as f:
        data = json.load(f)

    candidates = [np.array(c, dtype=np.float64) for c in data["candidates"]]
    people = np.array(data["people"], dtype=np.float64)
    static_obs = np.array(data["static_obstacles"], dtype=np.float64)
    # Crowd-FM pads unused dynamic-obstacle slots with x=y=1000 ("no obstacle");
    # drop those before passing real obstacle positions into the filter.
    people = people[np.linalg.norm(people, axis=1) < 500]

    print(f"loaded {len(candidates)} real Crowd-FM candidates, "
          f"{len(people)} real people, {len(static_obs)} static obstacle points")

    print("\n=== scenario as generated (normal) ===")
    result = filter_safe_candidates(candidates, static_obs=static_obs, dynamic_obs=people)
    print(f"safe_indices: {result['safe_indices']}")
    print(f"used_fallback: {result['used_fallback']}")
    print(f"margins (min clearance - required radius, >=0 is safe): "
          f"min={min(result['margins']):.3f} max={max(result['margins']):.3f}")
    n_safe = len(result["safe_indices"]) if not result["used_fallback"] else 0
    print(f"-> {n_safe}/{len(candidates)} candidates pass the hard safety filter")

    print("\n=== artificially crowded scenario (force every candidate unsafe) ===")
    # Place a dense ring of people directly along the path every real
    # candidate actually travels (their pooled centroid direction), close
    # enough that no candidate can avoid violating dynamic_radius anywhere.
    all_wp = np.concatenate(candidates, axis=0)
    centroid_dir = all_wp.mean(axis=0)
    centroid_dir = centroid_dir / (np.linalg.norm(centroid_dir) + 1e-9)
    crowd_positions = np.stack([centroid_dir * t for t in np.linspace(0.3, 2.8, 12)], axis=0)
    crowded_people = np.concatenate([people, crowd_positions], axis=0)

    result_crowded = filter_safe_candidates(candidates, static_obs=static_obs, dynamic_obs=crowded_people)
    print(f"safe_indices: {result_crowded['safe_indices']}")
    print(f"used_fallback: {result_crowded['used_fallback']}")
    print(f"margins: min={min(result_crowded['margins']):.3f} max={max(result_crowded['margins']):.3f}")

    assert result_crowded["used_fallback"], "expected every candidate to be forced unsafe"
    assert len(result_crowded["safe_indices"]) == 1, "fallback must return exactly one candidate, never empty"
    fallback_idx = result_crowded["safe_indices"][0]
    best_margin_idx = int(np.argmax(result_crowded["margins"]))
    assert fallback_idx == best_margin_idx, "fallback must pick the least-unsafe (max margin) candidate"
    print(f"-> PASS: fallback correctly returned 1 candidate (idx={fallback_idx}, "
          f"its margin {result_crowded['margins'][fallback_idx]:.3f} is the best/least-negative of all {len(candidates)})")


if __name__ == "__main__":
    main()
