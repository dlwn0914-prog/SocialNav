#!/usr/bin/env python
"""Pilot v3: a 20-question subset of pilot v2 (results/codebook/pilot_v2/
questions.json) with a strict train / test split in presentation order.

11/18 feedback: theta must be fit only on answers given BEFORE any test
question is shown, and 60 questions are too many. v3 has
  train   10 questions, always shown first (shuffled within the block):
          side 2 (GC, slow walker), front 1, yield 2 (GC, slow walker),
          detour 1, human 1, group 3 (queue, conversation, walking party)
  test     8 questions, shown after the whole train block:
          side, front, yield (slow walker), detour, human 1 each,
          group 3 (queue, conversation, walking party)
  repeat   2 train questions asked again with the sides swapped (in the
          second block), for answer consistency
Each slot takes, among the v2 questions of that axis / scene / split, the one
that adds most information about theta given the slots filled so far
(greedy D-optimal: max log det(I + sum dx dx^T), dx = option 0 - option 1,
normalized features). Test questions come only from v2 test questions, so
no situation is shared between the blocks.

Usage:
    PYTHONPATH=. .venv/bin/python utils/build_pilot_v3.py
"""
import argparse
import json
import os

import numpy as np

TRAIN = [("side", "gc"), ("side", "slow"), ("front", "gc"), ("yield", "gc"), ("yield", "slow"),
         ("detour", "gc"), ("human", "gc"), ("group", "queue"), ("group", "talk"), ("group", "pair")]
TEST = [("side", "gc"), ("front", "gc"), ("yield", "slow"), ("detour", "gc"), ("human", "gc"),
        ("group", "queue"), ("group", "talk"), ("group", "pair")]


def dx(q):
    x = np.asarray(q["x"])
    return x[0] - x[1]


def greedy(slots, pool, D):
    """One question per (axis, scene) slot, each maximizing the log det gain."""
    A, out = np.eye(D), []
    for axis, scene in slots:
        cand = [q for q in pool if q["axis"] == axis and q["scene"] == scene and q not in out]
        assert cand, (axis, scene)
        best = max(cand, key=lambda q: np.linalg.slogdet(A + np.outer(dx(q), dx(q)))[1])
        out.append(best)
        A += np.outer(dx(best), dx(best))
    return out, np.linalg.slogdet(A)[1]


def swapped(q):
    r = dict(q, id=q["id"] + "r", split="repeat", repeat_of=q["id"])
    for k in ("codes", "paths", "raw", "x", "min_dist"):
        r[k] = q[k][::-1]
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2", default="results/codebook/pilot_v2/questions.json")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", default="results/codebook/pilot_v3/questions.json")
    args = p.parse_args()

    V2 = json.load(open(args.v2))
    D = len(V2["features"])
    qs = [q for q in V2["questions"] if q["split"] != "repeat"]
    train, ld_train = greedy(TRAIN, [q for q in qs if q["split"] == "fit"], D)
    test, ld_test = greedy(TEST, [q for q in qs if q["split"] == "test"], D)
    # repeat the GC train questions whose two options differ most (clearest answers)
    rep = sorted([q for q in train if q["scene"] == "gc"], key=lambda q: -np.linalg.norm(dx(q)))[:args.repeats]
    out = ([dict(q, split="train", block=0) for q in train] + [dict(q, split="test", block=1) for q in test]
           + [dict(swapped(q), split="repeat", block=1) for q in rep])
    for q in out:
        q.pop("situation", None)
    meta = {k: V2[k] for k in V2 if k != "questions"}
    meta["args"] = {"v3": vars(args), "v2": V2["args"]}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(dict(meta, questions=out), open(args.out, "w"))
    print(f"{len(out)} questions: train {len(train)} (log det {ld_train:.2f}), test {len(test)} (log det {ld_test:.2f}), "
          f"repeat {len(rep)} -> {args.out}")
    for q in out:
        print(f"  {q['split']:6s} {q['id']:10s} {q['axis']:6s} {q['scene']}")


if __name__ == "__main__":
    main()
