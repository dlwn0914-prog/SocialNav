#!/usr/bin/env python
"""One shared protocol for every query-selection strategy, replacing the
per-script setups used so far (each earlier script had its own user model,
metric, K, and repeat count -- which is why ActiveQuery's top-1 rate ranged
from 60% to 97.5% across scripts and no number was comparable).

Protocol (identical for every method):
  - Pools: "real" = the 12 cached Crowd-FM scenarios (crowd_fm_ws/scenario_*.json),
    hard-safety-filtered, 3 reduced features (same as
    utils/fisher_optimal_on_real_candidates.py); "synthetic" = K=20
    make_synthetic_candidates pools, 7 features. Features are std-normalized
    per pool, and the true user lives in that normalized space.
  - True users: theta* sampled uniformly on the sphere with ||theta*|| =
    --theta-norm (controls how noisy the simulated answers are), a fresh
    one per session -- not one hand-picked profile, so results don't hinge
    on a single preference direction.
  - Feedback: Bradley-Terry, P(i > j) = sigmoid(theta* . (phi_i - phi_j)).
  - Budget: N=10 queries. If a strategy early-stops (ActiveQuery.done()),
    its recommendation is frozen for the remaining rounds and the number
    of queries actually used is recorded.
  - Metrics after every query t: normalized simple regret of the
    recommended candidate, (u*_max - u*(rec)) / (u*_max - u*_min);
    top-1 match; cosine(theta_hat, theta*).
  - Pairing: within a session every method sees the same pool and theta*,
    so method differences are reported as paired differences vs ActiveQuery
    with 95% bootstrap CIs.
  - --estimator: "native" lets each strategy recommend from its own
    estimator (norm-constrained MLE for ActiveQuery/Random/Fisher/DPB,
    Laplace MAP for MaxInP/TS/EUBO). "mle" / "map" refit ONE shared
    estimator on each method's own answered queries for the recommendation
    and cosine, so selection-rule effects are separated from estimator
    effects (queries are still chosen by each strategy's own rule).

Usage:
    PYTHONPATH=. .venv/bin/python utils/unified_query_benchmark.py --track real --repeats 30
    PYTHONPATH=. .venv/bin/python utils/unified_query_benchmark.py --track synthetic --sessions 360
"""
import argparse
import glob
import json
import os
import time

import numpy as np

from src.preference.features import extract_features
from src.preference.active_query import ActiveQueryStrategy, RandomQueryStrategy, fit_logit_mle
from src.preference.dpb import DPBStrategy
from src.preference.fisher_optimal import FisherOptimalStrategy
from src.preference.maxinp import MaxInPStrategy
from src.preference.bayesian_query import TopTwoTSStrategy, EUBOStrategy, laplace_posterior
from utils.fisher_optimal_on_real_candidates import load_scenario_feats
from utils.preference_poc import make_synthetic_candidates

METHODS = {
    "Random": lambda rng: RandomQueryStrategy(rng=rng),
    "ActiveQuery": lambda rng: ActiveQueryStrategy(rng=rng),
    "DPB(g=0.8)": lambda rng: DPBStrategy(gamma=0.8, discount_unit="round", rng=rng),
    "DPB(g=1.0)": lambda rng: DPBStrategy(gamma=1.0, discount_unit="round", rng=rng),
    "Fisher-optimal": lambda rng: FisherOptimalStrategy(rng=rng),
    "MaxInP": lambda rng: MaxInPStrategy(rng=rng),
    "TopTwo-TS": lambda rng: TopTwoTSStrategy(rng=rng),
    "EUBO": lambda rng: EUBOStrategy(rng=rng),
}
REFERENCE = "ActiveQuery"


def normalize(feats_raw):
    scale = feats_raw.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    return (feats_raw - feats_raw.mean(axis=0)) / scale


def sample_theta(dim, norm, rng):
    v = rng.standard_normal(dim)
    return norm * v / np.linalg.norm(v)


def cosine(a, b):
    if a is None or np.linalg.norm(a) < 1e-8:
        return 0.0
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def shared_estimate(history, dim, estimator):
    """history: list of winner - loser feature diffs."""
    if not history:
        return None
    if estimator == "mle":
        return fit_logit_mle(np.array(history), np.ones(len(history)))
    return laplace_posterior(np.array(history), dim)[0]


def run_session(strategy, feats, theta_star, rng, n_queries, estimator="native"):
    u = feats @ theta_star
    u_best, u_range = u.max(), max(u.max() - u.min(), 1e-8)
    true_best = int(np.argmax(u))
    regret, top1, cos = [], [], []
    used, stopped, history, theta = 0, False, [], None
    for _ in range(n_queries):
        if not stopped:
            i, j = strategy.select(feats)
            if strategy.done():
                stopped = True
            else:
                p_i = 1.0 / (1.0 + np.exp(-(u[i] - u[j])))
                winner = i if rng.random() < p_i else j
                strategy.update(feats, i, j, winner)
                history.append(feats[winner] - feats[j if winner == i else i])
                used += 1
                theta = strategy.theta_hat if estimator == "native" else shared_estimate(history, feats.shape[1], estimator)
        rec = int(np.argmax(feats @ theta)) if theta is not None else strategy.best(feats)
        regret.append(float((u_best - u[rec]) / u_range))
        top1.append(float(rec == true_best))
        cos.append(cosine(theta, theta_star))
    return {"regret": regret, "top1": top1, "cos": cos, "queries_used": used}


def build_sessions(args, rng):
    """Yields (pool_name, feats_norm) once per session."""
    if args.track == "real":
        pools = []
        for path in sorted(glob.glob(os.path.join(args.scenario_dir, "scenario_*.json"))):
            feats_raw = load_scenario_feats(path, args.candidates, args.moving_people)
            if feats_raw is None:
                print(f"{os.path.basename(path)}: skipped (fallback or <2 safe candidates)")
                continue
            pools.append((os.path.basename(path)[len("scenario_"):-len(".json")], normalize(feats_raw)))
        for _ in range(args.repeats):
            for name, feats in pools:
                yield name, feats
    else:
        target = np.array([0.0, 5.0])
        for s in range(args.sessions):
            cands = make_synthetic_candidates(target, num_candidates=args.k, rng=rng)
            yield f"syn_{s}", normalize(np.array([extract_features(c, target) for c in cands]))


def bootstrap_ci(x, rng, n_boot=2000):
    x = np.asarray(x, dtype=np.float64)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--track", choices=["real", "synthetic"], default="real")
    p.add_argument("--scenario-dir", default="/home/nuri2/crowd_fm_ws")
    p.add_argument("--repeats", type=int, default=30, help="real track: theta* draws per scenario")
    p.add_argument("--sessions", type=int, default=360, help="synthetic track: number of pools")
    p.add_argument("--k", type=int, default=20, help="synthetic track: candidates per pool")
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--theta-norm", type=float, default=2.0)
    p.add_argument("--estimator", choices=["native", "mle", "map"], default="native")
    p.add_argument("--candidates", choices=["candidates", "raw_candidates"], default="candidates",
                   help="real track: PRIEST-refined or pre-refinement flow candidates")
    p.add_argument("--moving-people", action="store_true",
                   help="real track: safety filter predicts people at constant velocity")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    raw = {m: [] for m in METHODS}
    pools_seen = []
    t0 = time.time()
    for s, (pool_name, feats) in enumerate(build_sessions(args, rng)):
        theta_star = sample_theta(feats.shape[1], args.theta_norm, rng)
        session_seed = int(rng.integers(2**31))
        for m, make in METHODS.items():
            # same seed per method -> same feedback-noise stream where queries coincide
            srng = np.random.default_rng(session_seed)
            raw[m].append(run_session(make(srng), feats, theta_star, srng, args.n_queries, args.estimator))
        pools_seen.append({"pool": pool_name, "K": int(feats.shape[0]), "dim": int(feats.shape[1])})
        if (s + 1) % 20 == 0:
            print(f"  {s + 1} sessions done ({time.time() - t0:.0f}s)", flush=True)

    n = len(pools_seen)
    brng = np.random.default_rng(args.seed + 1)
    summary = {}
    print(f"\n=== {args.track} track: {n} sessions, N={args.n_queries}, ||theta*||={args.theta_norm}, estimator={args.estimator} ===")
    print(f"{'method':>16} {'regret@N [95% CI]':>26} {'top1@N':>8} {'cos@N':>7} {'d_regret vs ' + REFERENCE + ' [95% CI]':>36} {'q_used':>7}")
    ref_final = np.array([r["regret"][-1] for r in raw[REFERENCE]])
    for m in METHODS:
        reg = np.array([r["regret"] for r in raw[m]])
        top1 = np.array([r["top1"] for r in raw[m]])
        cos = np.array([r["cos"] for r in raw[m]])
        final = reg[:, -1]
        r_mean, r_lo, r_hi = bootstrap_ci(final, brng)
        d_mean, d_lo, d_hi = bootstrap_ci(final - ref_final, brng)
        summary[m] = {
            "regret_per_query": reg.mean(axis=0).tolist(),
            "top1_per_query": top1.mean(axis=0).tolist(),
            "cos_per_query": cos.mean(axis=0).tolist(),
            "final_regret": [r_mean, r_lo, r_hi],
            "final_regret_diff_vs_ref": [d_mean, d_lo, d_hi],
            "final_top1": float(top1[:, -1].mean()),
            "final_cos": float(cos[:, -1].mean()),
            "mean_queries_used": float(np.mean([r["queries_used"] for r in raw[m]])),
        }
        print(f"{m:>16} {r_mean:>8.3f} [{r_lo:.3f},{r_hi:.3f}]   {top1[:, -1].mean():>8.3f} {cos[:, -1].mean():>7.3f}"
              f" {d_mean:>+14.3f} [{d_lo:+.3f},{d_hi:+.3f}]   {summary[m]['mean_queries_used']:>7.2f}")

    print("\nmean normalized simple regret per query (t=1..N):")
    for m in METHODS:
        print(f"  {m:>16}: " + " ".join(f"{v:.3f}" for v in summary[m]["regret_per_query"]))
    print("\nmean top-1 per query:")
    for m in METHODS:
        print(f"  {m:>16}: " + " ".join(f"{v:.2f}" for v in summary[m]["top1_per_query"]))

    out = args.out or f"unified_query_benchmark_{args.track}.json"
    with open(out, "w") as f:
        json.dump({"args": vars(args), "summary": summary, "sessions": pools_seen, "raw": raw}, f)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
