#!/usr/bin/env python
"""Why does a person's learned preference not transfer to their later scenes?
Two checks on the same scenes, features and split as person_transfer_benchmark.py.

1. Person signal (gc-ego answers): fit the best direction on the person's OWN
   query scenes vs on ANOTHER person's query scenes, both scored on the
   person's held-out scenes. own ~= other means the real paths carry no
   person-specific preference that these features can see.
2. Synthetic users with a fixed theta* per person (u = c * X theta*, c chosen
   so the median per-scene u range matches gc-ego): the same pipeline should
   transfer when the preference really is fixed. If it does not, the
   benchmark itself is broken.

Usage:
    PYTHONPATH=. .venv/bin/python utils/person_signal_check.py --features draft5
"""
import argparse
import json

import numpy as np

from utils.person_transfer_benchmark import (METHODS, boot, load_people, regret, run_session,
                                             scene_regrets)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/v3")
    p.add_argument("--ego-future", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/ego_future.json")
    p.add_argument("--features", default="draft5")
    p.add_argument("--n-train", type=int, default=3)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--beta", type=float, default=17.0)
    p.add_argument("--n-dirs", type=int, default=20000)
    p.add_argument("--n-others", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    people = load_people(args.scene_dir, json.load(open(args.ego_future)), args.features, args.beta)
    split = {}
    for pid, sc in people.items():
        tr = [sc[k] for k in sorted(sc) if k < args.n_train]
        te = [sc[k] for k in sorted(sc) if k >= args.n_train]
        if tr and te:
            split[pid] = (tr, te)
    pids = sorted(split)
    pool = np.vstack([s["raw"] for pid in pids for s in split[pid][0]])
    mu, sd = pool.mean(axis=0), pool.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    for pid in pids:
        for s in split[pid][0] + split[pid][1]:
            s["X"] = (s["raw"] - mu) / sd
    dirs = rng.standard_normal((args.n_dirs, len(mu)))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    out = {}

    # 1. own vs other person's query scenes, gc-ego answers
    tr_reg = {pid: np.mean([scene_regrets(s, dirs) for s in split[pid][0]], axis=0) for pid in pids}
    own, other, gap = [], [], []
    for pid in pids:
        te_reg = np.mean([scene_regrets(s, dirs) for s in split[pid][1]], axis=0)
        o = te_reg[int(np.argmin(tr_reg[pid]))]
        others = rng.choice([q for q in pids if q != pid], size=args.n_others, replace=False)
        x = np.mean([te_reg[int(np.argmin(tr_reg[q]))] for q in others])
        own.append(o); other.append(x); gap.append(o - x)
    out["gc_ego"] = {"own": boot(own, rng), "other": boot(other, rng), "own_minus_other": boot(gap, rng)}
    print(f"=== 1. gc-ego, {args.features}, {len(pids)} people: best direction fitted on query scenes, "
          f"scored on own held-out scenes ===")
    for k, v in out["gc_ego"].items():
        print(f"  {k:>16}  {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")

    # 2. synthetic fixed theta* per person on the same scenes
    c = np.median([np.ptp(s["u"]) for pid in pids for s in split[pid][0] + split[pid][1]]) / \
        np.median([np.ptp(s["X"] @ (d / np.linalg.norm(d))) for pid in pids
                   for s in split[pid][0] for d in rng.standard_normal((5, len(mu)))])
    for pid in pids:
        th = rng.standard_normal(len(mu))
        th /= np.linalg.norm(th)
        for s in split[pid][0] + split[pid][1]:
            s["u"] = c * s["X"] @ th
    ref = {k: [] for k in ["random", "shared", "personal-fit", "personal-best"]}
    allreg = {pid: np.mean([scene_regrets(s, dirs) for s in split[pid][0] + split[pid][1]], axis=0) for pid in pids}
    b = int(np.argmin(np.mean([allreg[q] for q in pids], axis=0)))  # in-sample shared: optimistic for shared
    for pid in pids:
        tr, te = split[pid]
        te_reg = np.mean([scene_regrets(s, dirs) for s in te], axis=0)
        ref["random"].append(np.mean([np.mean((s["u"].max() - s["u"]) / max(np.ptp(s["u"]), 1e-9)) for s in te]))
        ref["shared"].append(te_reg[b])
        ref["personal-fit"].append(te_reg[int(np.argmin(np.mean([scene_regrets(s, dirs) for s in tr], axis=0)))])
        ref["personal-best"].append(te_reg.min())
    meth = {}
    for m in ["Random", "ActiveQuery", "EUBO"]:
        r = []
        for pid in pids:
            tr, te = split[pid]
            rr = []
            for rep in range(args.repeats):
                srng = np.random.default_rng([args.seed, pid, rep])
                theta, _ = run_session(METHODS[m](srng), tr, args.n_queries, srng)
                pick = (lambda X: int(np.argmax(X @ theta))) if theta is not None else (lambda X: int(srng.integers(len(X))))
                rr.append(np.mean([regret(s["u"], pick(s["X"])) for s in te]))
            r.append(np.mean(rr))
        meth[m] = r
    out["synthetic"] = {k: boot(v, rng) for k, v in {**ref, **meth}.items()}
    out["synthetic"]["EUBO-ActiveQuery"] = boot(np.array(meth["EUBO"]) - np.array(meth["ActiveQuery"]), rng)
    print(f"=== 2. synthetic fixed theta* per person (u scale {c:.2f}), held-out regret ===")
    for k, v in out["synthetic"].items():
        print(f"  {k:>16}  {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
