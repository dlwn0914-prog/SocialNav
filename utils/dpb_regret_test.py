#!/usr/bin/env python
"""Actual cumulative regret (DPB paper's Eq. 6), not the cosine-similarity
proxy every earlier script in this session used. This is what "2단계 —
Regret 테스트" literally asked for.

R_T := sum_{t=1}^T [ mu(|X_t*^T theta*|) - mu(|X_t^T theta*|) ]

where mu is the logistic function, X_t is the query (feature-difference
vector of the pair compared) the strategy actually picked at round t, and
X_t* is the single most-informative possible query in the candidate pool
under the true preference theta* (mu(|X^T theta*|) is the same for every
round in a stationary session, since theta* and the pool don't change --
so X_t* is fixed once per session; a strategy's cumulative regret is small
exactly when it quickly converges to repeatedly asking that most-decisive
pair, OR any pair that is nearly as informative).

Compares DPBStrategy(gamma=0.8, the swept default), ActiveQueryStrategy
(UserAlign-style, the method being replaced), and RandomQueryStrategy
(uniform baseline) under a stationary true user.

Usage:
    PYTHONPATH=. python utils/dpb_regret_test.py --repeats 30
"""
import argparse

import numpy as np

from src.preference.features import extract_features
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import ActiveQueryStrategy, RandomQueryStrategy
from src.preference.dpb import DPBStrategy
from utils.preference_poc import PROFILES, profile_to_vector, make_synthetic_candidates

PROFILE_NAMES = list(PROFILES.keys())


def mu(x):
    return 1.0 / (1.0 + np.exp(-x))


def build_pool(rng):
    target = (0.0, 5.0)
    candidates = make_synthetic_candidates(target, num_candidates=20, rng=rng)
    return np.array([extract_features(c, target) for c in candidates])


def best_possible_mu(feats_norm, w_true_scaled):
    K = len(feats_norm)
    best = -np.inf
    for i in range(K):
        for j in range(i + 1, K):
            X = feats_norm[i] - feats_norm[j]
            val = mu(abs(float(X @ w_true_scaled)))
            if val > best:
                best = val
    return best


def run_session(strategy, feats_raw, feats_norm, w_true, w_true_scaled, best_mu, rng, n_queries=10):
    regret_per_round = []
    for _ in range(n_queries):
        i, j = strategy.select(feats_norm)
        X = feats_norm[i] - feats_norm[j]
        this_mu = mu(abs(float(X @ w_true_scaled)))
        regret_per_round.append(best_mu - this_mu)

        p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
        winner = i if rng.random() < p_i else j
        strategy.update(feats_norm, i, j, winner)
    return np.cumsum(regret_per_round)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--n-queries", type=int, default=10)
    args = p.parse_args()

    methods = {
        "Random": lambda rng: RandomQueryStrategy(rng=rng),
        "ActiveQuery (UserAlign)": lambda rng: ActiveQueryStrategy(rng=rng),
        "DPB (gamma=0.8)": lambda rng: DPBStrategy(gamma=0.8, discount_unit="round", rng=rng),
    }
    all_cum_regret = {name: [] for name in methods}

    for rep in range(args.repeats):
        rng_data = np.random.default_rng(9000 + rep)
        feats_raw = build_pool(rng_data)
        scale = feats_raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        feats_norm = feats_raw / scale
        w_true = profile_to_vector(PROFILES[PROFILE_NAMES[0]])
        w_true_scaled = w_true * scale
        best_mu = best_possible_mu(feats_norm, w_true_scaled)

        for name, make_strat in methods.items():
            rng_s = np.random.default_rng(1000 + rep)
            strat = make_strat(rng_s)
            cum = run_session(strat, feats_raw, feats_norm, w_true, w_true_scaled, best_mu, rng_s, args.n_queries)
            all_cum_regret[name].append(cum)

    print(f"cumulative regret R_t (DPB paper's Eq. 6), mean over {args.repeats} repeats, stationary user\n")
    header = f"{'query':>6}" + "".join(f"{name:>26}" for name in methods)
    print(header)
    for t in range(args.n_queries):
        row = f"{t+1:>6}"
        for name in methods:
            mean_t = np.mean([c[t] for c in all_cum_regret[name]])
            row += f"{mean_t:>26.4f}"
        print(row)

    print("\nfinal cumulative regret (lower = better):")
    for name in methods:
        finals = [c[-1] for c in all_cum_regret[name]]
        print(f"  {name}: {np.mean(finals):.4f} (+-{np.std(finals):.4f})")


if __name__ == "__main__":
    main()
