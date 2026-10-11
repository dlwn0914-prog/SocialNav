#!/usr/bin/env python
"""Sweep DPB's discount factor gamma (round-based, the variant chosen over
session-based -- see utils/dpb_round_vs_session_diagnosis.py and
utils/dpb_cross_session.py) to find a reasonable default, instead of the
arbitrary gamma=0.7 used in every prior test this session.

Two conditions, same as utils/dpb_cross_session.py:
  - stationary: single 10-query session, no drift at all
  - cross-session drift: 3 sessions of 10 queries, true preference switches
    between session 1 and session 2 (not mid-session)

For each gamma, reports final cosine in both conditions -- the right
default is the one that does well in the drifting case without giving up
much in the stationary case (gamma=1.0 is the "no discount" edge case).

Usage:
    PYTHONPATH=. python utils/dpb_gamma_sweep.py --repeats 30
"""
import argparse

import numpy as np

from src.preference.features import extract_features
from src.preference.bradley_terry import predict_preference_prob
from src.preference.dpb import DPBStrategy
from utils.preference_poc import PROFILES, profile_to_vector, make_synthetic_candidates

PROFILE_NAMES = list(PROFILES.keys())
GAMMAS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]


def build_pool(rng):
    target = (0.0, 5.0)
    candidates = make_synthetic_candidates(target, num_candidates=20, rng=rng)
    return np.array([extract_features(c, target) for c in candidates])


def cos_to(theta_hat, w_true_norm):
    if theta_hat is None or np.linalg.norm(theta_hat) < 1e-8:
        return 0.0
    return float(np.dot(w_true_norm, theta_hat) / (np.linalg.norm(w_true_norm) * np.linalg.norm(theta_hat)))


def run_stationary(gamma, feats_raw, feats_norm, scale, rng):
    w_true = profile_to_vector(PROFILES[PROFILE_NAMES[0]])
    w_true_scaled_norm = (w_true * scale) / (np.linalg.norm(w_true * scale) + 1e-8)
    strat = DPBStrategy(gamma=gamma, discount_unit="round", rng=rng)
    for _ in range(10):
        i, j = strat.select(feats_norm)
        p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
        winner = i if rng.random() < p_i else j
        strat.update(feats_norm, i, j, winner)
    return cos_to(strat.theta_hat, w_true_scaled_norm)


def run_drifting(gamma, feats_raw, feats_norm, scale, rng):
    w_a = profile_to_vector(PROFILES[PROFILE_NAMES[0]])
    w_b = profile_to_vector(PROFILES[PROFILE_NAMES[1]])
    strat = DPBStrategy(gamma=gamma, discount_unit="round", rng=rng)
    for sess in range(3):
        w_true = w_b if sess >= 2 else w_a
        for _ in range(10):
            i, j = strat.select(feats_norm)
            p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
            winner = i if rng.random() < p_i else j
            strat.update(feats_norm, i, j, winner)
    w_true_scaled_norm = (w_b * scale) / (np.linalg.norm(w_b * scale) + 1e-8)
    return cos_to(strat.theta_hat, w_true_scaled_norm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    stat_results = {g: [] for g in GAMMAS}
    drift_results = {g: [] for g in GAMMAS}

    for rep in range(args.repeats):
        rng_data = np.random.default_rng(9000 + rep)
        feats_raw = build_pool(rng_data)
        scale = feats_raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        feats_norm = feats_raw / scale

        for g in GAMMAS:
            rng_s = np.random.default_rng(1000 + rep)
            stat_results[g].append(run_stationary(g, feats_raw, feats_norm, scale, rng_s))

            rng_d = np.random.default_rng(2000 + rep)
            drift_results[g].append(run_drifting(g, feats_raw, feats_norm, scale, rng_d))

    print(f"gamma sweep, {args.repeats} repeats (round-based discount)\n")
    print(f"{'gamma':>8} {'stationary':>12} {'cross-session drift':>22}")
    for g in GAMMAS:
        print(f"{g:>8.2f} {np.mean(stat_results[g]):>12.3f} {np.mean(drift_results[g]):>22.3f}")


if __name__ == "__main__":
    main()
