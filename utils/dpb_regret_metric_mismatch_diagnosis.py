#!/usr/bin/env python
"""Why does ActiveQueryStrategy win on the DPB paper's actual cumulative
regret metric (utils/dpb_regret_test.py: 0.0001 vs DPB's 0.0117), when
every earlier test this session found DPB recovers the true preference
VECTOR (cosine similarity) better?

Because the paper's regret (Eq. 6) only rewards repeatedly asking about
the single most-decisive pair under the true preference -- a best-arm
identification objective -- not full-vector estimation. This script
verifies that mechanically: for each method's 10 queries in a session, it
builds the (un-discounted) design matrix V = sum_t X_t X_t^T from the
actually-asked comparisons and reports its minimum eigenvalue (0 means at
least one direction in the 7-dim feature space was never probed at all,
so theta_hat is arbitrary along it) and how many distinct pairs were
asked.

Usage:
    PYTHONPATH=. python utils/dpb_regret_metric_mismatch_diagnosis.py --repeats 30
"""
import argparse

import numpy as np

from src.preference.features import extract_features
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import ActiveQueryStrategy
from src.preference.dpb import DPBStrategy
from utils.preference_poc import PROFILES, profile_to_vector, make_synthetic_candidates

PROFILE_NAMES = list(PROFILES.keys())


def build_pool(rng):
    target = (0.0, 5.0)
    candidates = make_synthetic_candidates(target, num_candidates=20, rng=rng)
    return np.array([extract_features(c, target) for c in candidates])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    methods = {
        "ActiveQuery": lambda rng: ActiveQueryStrategy(rng=rng),
        "DPB(g=0.8)": lambda rng: DPBStrategy(gamma=0.8, discount_unit="round", rng=rng),
    }
    min_eig = {name: [] for name in methods}
    n_unique_pairs = {name: [] for name in methods}

    for rep in range(args.repeats):
        rng_data = np.random.default_rng(9000 + rep)
        feats_raw = build_pool(rng_data)
        scale = feats_raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        feats_norm = feats_raw / scale
        w_true = profile_to_vector(PROFILES[PROFILE_NAMES[0]])

        for name, make_strat in methods.items():
            rng_s = np.random.default_rng(1000 + rep)
            strat = make_strat(rng_s)
            Xs, pairs_seen = [], set()
            for _ in range(10):
                i, j = strat.select(feats_norm)
                pairs_seen.add(tuple(sorted((i, j))))
                Xs.append(feats_norm[i] - feats_norm[j])
                p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
                winner = i if rng_s.random() < p_i else j
                strat.update(feats_norm, i, j, winner)
            V = sum(np.outer(X, X) for X in Xs)
            min_eig[name].append(np.linalg.eigvalsh(V).min())
            n_unique_pairs[name].append(len(pairs_seen))

    print(f"design-matrix coverage over 10 queries, {args.repeats} repeats, stationary user\n")
    for name in methods:
        print(f"  {name}: mean min-eigenvalue(V) = {np.mean(min_eig[name]):.4f}, "
              f"mean distinct pairs asked (of 10) = {np.mean(n_unique_pairs[name]):.2f}")
    print("\nmin-eigenvalue ~ 0 means at least one feature direction was never probed at all --")
    print("theta_hat is arbitrary along it, which is exactly why low-regret (best-arm-focused)")
    print("query selection can still produce poor cosine (full-vector) recovery.")


if __name__ == "__main__":
    main()
