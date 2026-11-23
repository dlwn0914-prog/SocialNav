#!/usr/bin/env python
"""Learn a person's preference from A/B queries in some scenes, then use it
in scenes they were never asked about -- the "use it on later drives"
setting, on Grand Central people with several scenes each
(utils/build_gc_person_scenes.py: real-future goals, v3 candidates).

Per person: scenes with order < --n-train are query scenes, the rest are
held out. A session spends --n-queries A/B queries over the query scenes
(cycled; EUBO-anyscene picks the scene each time), updating ONE posterior (strategies keep winner - loser feature
diffs, so answers from different scenes accumulate). The final theta_hat
then picks argmax theta_hat . phi in every held-out scene with no queries.

Answers: gc-ego, shape answer -- u = -beta * shape_distance(candidate, the
path the person actually walked); P(A) = sigmoid(u_A - u_B). Regret per
scene is normalized by that scene's u range.

Features are normalized with ONE mean/std for all scenes (taken from the
query scenes' candidates of all people, no answers used), so a weight means
the same thing in every scene -- unlike the per-scene z-score of
unified_query_benchmark.py.

Reference points on the held-out scenes:
  random        expected regret of a uniformly random candidate
  shared (cv)   one weight direction for everyone, chosen on the other
                folds' people (all their scenes, noise-free answers) --
                the non-personalized baseline
  personal-fit  best direction for this person on their QUERY scenes
                (noise-free, as if all pairs were answered) -- how far a
                personal linear preference transfers at all
  personal-best best direction on the held-out scenes themselves (ceiling)
CIs: 95% bootstrap over people (sessions of one person are not independent).

Usage:
    PYTHONPATH=. .venv/bin/python utils/person_transfer_benchmark.py --features draft5 \
        --out results/person_transfer/v3_draft5.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

from src.preference.active_query import ActiveQueryStrategy, RandomQueryStrategy
from src.preference.maxinp import MaxInPStrategy
from src.preference.bayesian_query import EUBOStrategy, InfoGainStrategy
from utils.fisher_optimal_on_real_candidates import load_scenario_feats
from utils.unified_query_benchmark import shape_distance

class EUBOAnySceneStrategy(EUBOStrategy):
    """EUBO that also chooses WHICH query scene to ask in, instead of cycling
    through them. A question still compares two candidates of one scene (a
    cross-scene pair could not be asked on a real drive, nor answered from
    one real path). Scenes are compared by the gain of asking there,
    max_pair EUBO - max_k E[u_k], since raw EUBO values carry each scene's
    own utility level; one set of posterior samples serves all scenes."""

    def select_scene(self, scenes):
        mean, cov = self._posterior(scenes[0].shape[1])
        th = self.rng.multivariate_normal(mean, cov, size=self.n_samples).T  # (d, S)
        best = None
        for k, feats in enumerate(scenes):
            utils = np.asarray(feats, dtype=np.float64) @ th  # (K, S)
            eubo = np.maximum(utils[:, None, :], utils[None, :, :]).mean(axis=2)
            np.fill_diagonal(eubo, -np.inf)
            i, j = np.unravel_index(np.argmax(eubo), eubo.shape)
            gain = eubo[i, j] - utils.mean(axis=1).max()
            if best is None or gain > best[0]:
                best = (gain, k, int(i), int(j))
        return best[1:]


METHODS = {
    "Random": lambda rng: RandomQueryStrategy(rng=rng),
    "ActiveQuery": lambda rng: ActiveQueryStrategy(rng=rng),
    "MaxInP": lambda rng: MaxInPStrategy(rng=rng),
    "EUBO": lambda rng: EUBOStrategy(rng=rng),
    "InfoGain": lambda rng: InfoGainStrategy(rng=rng),
    "EUBO-anyscene": lambda rng: EUBOAnySceneStrategy(rng=rng),
}


def load_people(scene_dir, futures, feature_set, beta):
    people = defaultdict(dict)  # person -> {order: scene}
    for path in sorted(glob.glob(os.path.join(scene_dir, "scenario_gcp_*.json"))):
        name = os.path.basename(path)[len("scenario_"):-len(".json")]
        person, order = map(int, re.match(r"gcp_(\d+)_(\d+)", name).groups())
        loaded = load_scenario_feats(path, "candidates", moving_people=True, return_candidates=True,
                                     feature_set=feature_set)
        if loaded is None:
            continue
        feats, cands = loaded
        u = -beta * shape_distance(cands, np.array(futures[name]))
        people[person][order] = {"name": name, "raw": feats, "u": u}
    return people


def regret(u, pick):
    return float((u.max() - u[pick]) / max(u.max() - u.min(), 1e-9))


def picks(X, dirs):
    return np.argmax(X @ dirs.T, axis=0)  # (n_dirs,)


def scene_regrets(s, dirs):
    u = s["u"]
    return (u.max() - u[picks(s["X"], dirs)]) / max(u.max() - u.min(), 1e-9)


def run_session(strategy, train, n_queries, rng):
    """Query scenes are cycled (query q in scene q mod n), except for
    strategies with select_scene, which pick the scene themselves."""
    stopped = False
    per_scene = np.zeros(len(train), dtype=int)  # queries asked in each query scene
    for q in range(n_queries):
        if stopped:
            break
        if hasattr(strategy, "select_scene"):
            k, i, j = strategy.select_scene([t["X"] for t in train])
        else:
            k = q % len(train)
            i, j = strategy.select(train[k]["X"])
        s = train[k]
        if strategy.done():
            stopped = True
            break
        p_i = 1.0 / (1.0 + np.exp(-(s["u"][i] - s["u"][j])))
        strategy.update(s["X"], i, j, i if rng.random() < p_i else j)
        per_scene[k] += 1
    return strategy.theta_hat, per_scene


def boot(x, rng, n=2000):
    x = np.asarray(x)
    b = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/v3")
    p.add_argument("--ego-future", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/ego_future.json")
    p.add_argument("--features", choices=["base3", "social6", "fixed6", "draft5"], default="draft5")
    p.add_argument("--n-train", type=int, default=3, help="scenes with order < n-train are query scenes")
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--beta", type=float, default=17.0)
    p.add_argument("--n-dirs", type=int, default=20000)
    p.add_argument("--cv-folds", type=int, default=5)
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
    print(f"{len(pids)} people with >= 1 usable query scene and >= 1 held-out scene "
          f"({sum(len(t) for t, _ in split.values())} query / {sum(len(e) for _, e in split.values())} held-out scenes)")

    # one normalization for every scene, from query-scene candidates only
    pool = np.vstack([s["raw"] for pid in pids for s in split[pid][0]])
    mu, sd = pool.mean(axis=0), pool.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    for pid in pids:
        for s in split[pid][0] + split[pid][1]:
            s["X"] = (s["raw"] - mu) / sd

    d = len(mu)
    dirs = rng.standard_normal((args.n_dirs, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    ref = {k: {} for k in ["random", "shared (cv)", "personal-fit", "personal-best"]}
    per_person_all = {pid: np.mean([scene_regrets(s, dirs) for s in split[pid][0] + split[pid][1]], axis=0)
                      for pid in pids}
    folds = {pid: f for pid, f in zip(pids, rng.permutation(len(pids)) % args.cv_folds)}
    for k in range(args.cv_folds):
        others = [pid for pid in pids if folds[pid] != k]
        b = int(np.argmin(np.mean([per_person_all[q] for q in others], axis=0)))
        for pid in pids:
            if folds[pid] == k:
                ref["shared (cv)"][pid] = float(np.mean([regret(s["u"], int(np.argmax(s["X"] @ dirs[b])))
                                                         for s in split[pid][1]]))
    for pid in pids:
        tr, te = split[pid]
        te_reg = np.mean([scene_regrets(s, dirs) for s in te], axis=0)
        ref["personal-fit"][pid] = float(te_reg[int(np.argmin(np.mean([scene_regrets(s, dirs) for s in tr], axis=0)))])
        ref["personal-best"][pid] = float(te_reg.min())
        ref["random"][pid] = float(np.mean([np.mean((s["u"].max() - s["u"]) / max(np.ptp(s["u"]), 1e-9)) for s in te]))

    res = {m: {} for m in METHODS}
    in_sample = {m: {} for m in METHODS}
    top_share = {m: {} for m in METHODS}  # share of a session's queries in its most-asked query scene
    for m, make in METHODS.items():
        for pid in pids:
            tr, te = split[pid]
            r_te, r_tr, share = [], [], []
            for rep in range(args.repeats):
                srng = np.random.default_rng([args.seed, pid, rep])
                strat = make(srng)
                theta, per_scene = run_session(strat, tr, args.n_queries, srng)
                share.append(per_scene.max() / max(per_scene.sum(), 1))
                pick = (lambda X: int(np.argmax(X @ theta))) if theta is not None else (lambda X: int(srng.integers(len(X))))
                r_te.append(np.mean([regret(s["u"], pick(s["X"])) for s in te]))
                r_tr.append(np.mean([regret(s["u"], pick(s["X"])) for s in tr]))
            res[m][pid], in_sample[m][pid] = float(np.mean(r_te)), float(np.mean(r_tr))
            top_share[m][pid] = float(np.mean(share))
        print(f"  {m} done")

    brng = np.random.default_rng(1)
    rows = {}
    print(f"\n=== held-out scenes, {args.features}, {args.n_queries} queries over {args.n_train} query scenes, "
          f"{len(pids)} people x {args.repeats} ===")
    print(f"{'':>14} {'regret held-out [95% CI]':>28} {'vs shared (cv) [95% CI]':>26} {'regret query scenes':>20}")
    base = np.array([ref["shared (cv)"][q] for q in pids])
    for name, vals, ins in ([(k, ref[k], None) for k in ref] + [(m, res[m], in_sample[m]) for m in METHODS]):
        x = np.array([vals[q] for q in pids])
        mean, lo, hi = boot(x, brng)
        dm, dlo, dhi = boot(x - base, brng)
        rows[name] = {"held_out": [mean, lo, hi], "vs_shared": [dm, dlo, dhi],
                      "query_scenes": float(np.mean([ins[q] for q in pids])) if ins else None}
        tail = f"{rows[name]['query_scenes']:>20.3f}" if ins else f"{'':>20}"
        print(f"{name:>14} {mean:>8.3f} [{lo:.3f}, {hi:.3f}]       {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]    {tail}")
    print("\nquery concentration (mean share of queries in the most-asked query scene; "
          f"even split = {1 / args.n_train:.2f}):")
    for m in METHODS:
        print(f"  {m:>14} {np.mean([top_share[m][q] for q in pids]):.2f}")
    eu, aq = np.array([res["EUBO"][q] for q in pids]), np.array([res["ActiveQuery"][q] for q in pids])
    dm, dlo, dhi = boot(eu - aq, brng)
    print(f"\nEUBO - ActiveQuery on held-out scenes: {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]")

    out = args.out or f"results/person_transfer/v3_{args.features}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "n_people": len(pids), "norm_mean": mu.tolist(), "norm_std": sd.tolist(),
                   "summary": rows, "per_person": {"ref": ref, "methods": res, "query_scenes": in_sample, "top_scene_share": top_share}}, f, indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
