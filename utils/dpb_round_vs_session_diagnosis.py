#!/usr/bin/env python
"""Why does DPB with discount_unit="round" recover/perform better than
discount_unit="session", even with NO drift (pure stationary preference)?

Hypothesis: discounting doesn't just affect the FIT (penalized MLE weights),
it also shrinks the UCB design matrix V_t used by SELECT -- with
discount_unit="round", V_t's effective sample size stays smaller throughout
a single session (early-in-session comparisons are already aged by the
time of later queries), so the exploration bonus alpha*||X||_{V^-1} stays
larger for longer and SELECT keeps exploring diverse pairs. With
discount_unit="session", nothing decays within a session, so V_t
accumulates at full weight and SELECT converges to exploiting
(challenging near the current best) faster -- fewer genuinely
informative/diverse comparisons by the end of the session.

This script measures, for one stationary 10-query session, round vs
session vs no discount: (1) the UCB exploration-bonus term's magnitude at
each query (does it shrink faster for session-discount?), (2) how many
DISTINCT candidate indices get touched across the 10 queries (a cheap
diversity proxy), and (3) final cosine, to connect the mechanism to the
earlier result.

Usage:
    PYTHONPATH=. python utils/dpb_round_vs_session_diagnosis.py --repeats 30
"""
import argparse

import numpy as np

from src.preference.features import extract_features
from src.preference.bradley_terry import predict_preference_prob
from src.preference.dpb import DPBStrategy
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


def exploration_bonus(strategy, feats, i, j):
    """Re-derive just the alpha*||X||_{V^-1} term for the pair SELECT just chose."""
    dim = feats.shape[1]
    V = strategy._V(dim)
    V_inv = np.linalg.inv(V)
    X = feats[i] - feats[j]
    alpha = strategy.alpha_scale * np.sqrt(dim * np.log(1 + strategy.t)) + 1.0
    return alpha * np.sqrt(max(X @ V_inv @ X, 0.0))


def run_session(strategy, feats_raw, feats_norm, scale, w_true, rng):
    w_true_scaled_norm = (w_true * scale) / (np.linalg.norm(w_true * scale) + 1e-8)
    touched = set()
    bonus_track = []
    for _ in range(10):
        i, j = strategy.select(feats_norm)
        bonus_track.append(exploration_bonus(strategy, feats_norm, i, j))
        touched.add(i)
        touched.add(j)
        p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
        winner = i if rng.random() < p_i else j
        strategy.update(feats_norm, i, j, winner)
    final_cos = cos_to(strategy.theta_hat, w_true_scaled_norm)
    return bonus_track, len(touched), final_cos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    configs = {
        "no discount (g=1.0, round)": dict(gamma=1.0, discount_unit="round"),
        "round-discount (g=0.7)": dict(gamma=0.7, discount_unit="round"),
        "session-discount (g=0.7)": dict(gamma=0.7, discount_unit="session"),
    }

    all_bonus = {name: [] for name in configs}
    all_touched = {name: [] for name in configs}
    all_cos = {name: [] for name in configs}

    for rep in range(args.repeats):
        rng_data = np.random.default_rng(9000 + rep)
        feats_raw = build_pool(rng_data)
        scale = feats_raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        feats_norm = feats_raw / scale
        w_true = profile_to_vector(PROFILES[PROFILE_NAMES[0]])

        for name, kwargs in configs.items():
            rng_s = np.random.default_rng(1000 + rep)
            strat = DPBStrategy(rng=rng_s, **kwargs)
            bonus_track, n_touched, final_cos = run_session(strat, feats_raw, feats_norm, scale, w_true, rng_s)
            all_bonus[name].append(bonus_track)
            all_touched[name].append(n_touched)
            all_cos[name].append(final_cos)

    print(f"=== exploration bonus (alpha*||X||_V^-1) per query, mean over {args.repeats} stationary sessions ===")
    header = f"{'query':>6}" + "".join(f"{name:>28}" for name in configs)
    print(header)
    for t in range(10):
        row = f"{t+1:>6}"
        for name in configs:
            row += f"{np.mean([b[t] for b in all_bonus[name]]):>28.4f}"
        print(row)

    print(f"\n=== distinct candidate indices touched across 10 queries (out of 20 total) ===")
    for name in configs:
        print(f"  {name}: mean={np.mean(all_touched[name]):.2f}")

    print(f"\n=== final cosine (sanity check vs earlier result) ===")
    for name in configs:
        print(f"  {name}: mean={np.mean(all_cos[name]):.3f}")


if __name__ == "__main__":
    main()
