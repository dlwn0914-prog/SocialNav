#!/usr/bin/env python
"""Estimation regret: redefines "regret" to match our actual research goal
(learn the user's full preference vector, reusable across future unseen
candidate sets) instead of the DPB paper's best-arm-identification regret
(utils/dpb_regret_test.py, which favored ActiveQueryStrategy -- see
utils/dpb_regret_metric_mismatch_diagnosis.py for why that metric doesn't
match our goal).

estimation_regret_t := 1 - cosine(theta_hat_t, theta*)   (bounded [0, 2], 0 = perfect)
cumulative_estimation_regret_T := sum_{t=1}^T estimation_regret_t

A method whose per-round estimation regret shrinks toward 0 as it collects
more comparisons gives sub-linear cumulative growth (the curve flattens);
a method stuck with a non-shrinking estimation error keeps accumulating
regret at a roughly constant rate (linear growth, never flattens).

Usage:
    PYTHONPATH=. python utils/dpb_estimation_regret.py --repeats 30
"""
import argparse

import numpy as np

from src.preference.features import extract_features
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import ActiveQueryStrategy, RandomQueryStrategy
from src.preference.dpb import DPBStrategy
from src.preference.doptimal import DOptimalStrategy
from src.preference.fisher_optimal import FisherOptimalStrategy
from utils.preference_poc import PROFILES, profile_to_vector, make_synthetic_candidates

PROFILE_NAMES = list(PROFILES.keys())


def build_pool(rng):
    target = (0.0, 5.0)
    candidates = make_synthetic_candidates(target, num_candidates=20, rng=rng)
    return np.array([extract_features(c, target) for c in candidates])


def cos_to(theta_hat, w_true_norm):
    if theta_hat is None or np.linalg.norm(theta_hat) < 1e-8:
        return 0.0
    return float(np.dot(w_true_norm, theta_hat) / (np.linalg.norm(w_true_norm) * np.linalg.norm(theta_hat)))


def run_session(strategy, feats_raw, feats_norm, w_true, w_true_scaled_norm, rng, n_queries=10):
    est_regret = []
    for _ in range(n_queries):
        i, j = strategy.select(feats_norm)
        p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
        winner = i if rng.random() < p_i else j
        strategy.update(feats_norm, i, j, winner)
        est_regret.append(1.0 - cos_to(strategy.theta_hat, w_true_scaled_norm))
    return np.cumsum(est_regret)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--n-queries", type=int, default=10)
    args = p.parse_args()

    methods = {
        "Random": lambda rng: RandomQueryStrategy(rng=rng),
        "ActiveQuery (UserAlign)": lambda rng: ActiveQueryStrategy(rng=rng),
        "DPB (gamma=0.8)": lambda rng: DPBStrategy(gamma=0.8, discount_unit="round", rng=rng),
        "D-optimal": lambda rng: DOptimalStrategy(rng=rng),
        "Fisher-optimal": lambda rng: FisherOptimalStrategy(rng=rng),
    }
    all_cum = {name: [] for name in methods}

    for rep in range(args.repeats):
        rng_data = np.random.default_rng(9000 + rep)
        feats_raw = build_pool(rng_data)
        scale = feats_raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        feats_norm = feats_raw / scale
        w_true = profile_to_vector(PROFILES[PROFILE_NAMES[0]])
        w_true_scaled_norm = (w_true * scale) / (np.linalg.norm(w_true * scale) + 1e-8)

        for name, make_strat in methods.items():
            rng_s = np.random.default_rng(1000 + rep)
            strat = make_strat(rng_s)
            cum = run_session(strat, feats_raw, feats_norm, w_true, w_true_scaled_norm, rng_s, args.n_queries)
            all_cum[name].append(cum)

    print(f"cumulative ESTIMATION regret (1-cosine per round), mean over {args.repeats} repeats, stationary user\n")
    header = f"{'query':>6}" + "".join(f"{name:>26}" for name in methods)
    print(header)
    for t in range(args.n_queries):
        row = f"{t+1:>6}"
        for name in methods:
            mean_t = np.mean([c[t] for c in all_cum[name]])
            row += f"{mean_t:>26.4f}"
        print(row)

    print("\nper-round regret increment (last 3 rounds, shrinking = sub-linear / flattening):")
    for name in methods:
        cum_arr = np.array(all_cum[name])
        incr = np.diff(cum_arr.mean(axis=0))
        print(f"  {name}: {np.round(incr[-3:], 4).tolist()}")

    print("\nfinal cumulative estimation regret (lower = better):")
    for name in methods:
        finals = [c[-1] for c in all_cum[name]]
        print(f"  {name}: {np.mean(finals):.4f} (+-{np.std(finals):.4f})")


if __name__ == "__main__":
    main()
