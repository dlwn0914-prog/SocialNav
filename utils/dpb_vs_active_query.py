#!/usr/bin/env python
"""Does DPB's discount factor (src/preference/dpb.py, ported from Kim et al.
2024) actually help over the existing ActiveQueryStrategy (UserAlign-style,
no discounting) -- and does it help ONLY when it should (a user whose true
preference drifts mid-session), without hurting when the user is stationary?

Two conditions, both simulated on the same synthetic candidate pools
utils/preference_poc.py already validates the fitting machinery on:
  - stationary: one true w_true for the whole session (10 queries)
  - drifting: w_true = profile A for queries 1-5, then SWITCHES to profile
    B (opposite priorities) for queries 6-10 -- the exact scenario DPB's
    discounting is meant for for.

Tracks cosine(current true w, theta_hat) after every query, so a drift at
query 6 shows up as a cosine drop that either recovers quickly (adapts) or
stays low (stuck on the stale estimate).

Usage:
    PYTHONPATH=. python utils/dpb_vs_active_query.py --repeats 20
"""
import argparse

import numpy as np

from src.preference.features import extract_features, FEATURE_NAMES
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import ActiveQueryStrategy
from src.preference.dpb import DPBStrategy, fit_discounted_mle
from utils.preference_poc import PROFILES, profile_to_vector, make_synthetic_candidates

PROFILE_NAMES = list(PROFILES.keys())  # ["smooth_and_direct", "wide_berth"]


class ActiveQueryL2Strategy(ActiveQueryStrategy):
    """Ablation: UserAlign's confidence-region challenger SELECT (unchanged,
    inherited from ActiveQueryStrategy) but DPB's penalized fit (L2 term,
    gamma=1.0 so every past comparison is weighted equally -- same fit as
    DPBStrategy(gamma=1.0) would produce). Isolates whether DPB's
    stationary-case edge over ActiveQueryStrategy comes from its UCB-style
    SELECT rule or from this L2 penalty in the FIT."""

    def update(self, feats, i, j, winner):
        diff = feats[i] - feats[j] if winner == i else feats[j] - feats[i]
        self.diff_feats.append(diff)
        self.labels.append(1.0)
        weights = np.ones(len(self.diff_feats))
        try:
            self.theta_hat = fit_discounted_mle(
                np.array(self.diff_feats), np.array(self.labels), weights, norm_bound=self.norm_bound
            )
        except RuntimeError:
            pass


def build_pool(rng):
    target = (0.0, 5.0)
    candidates = make_synthetic_candidates(target, num_candidates=20, rng=rng)
    feats_raw = np.array([extract_features(c, target) for c in candidates])
    return feats_raw


def cos_to(theta_hat, w_true_norm):
    if theta_hat is None or np.linalg.norm(theta_hat) < 1e-8:
        return 0.0
    return float(np.dot(w_true_norm, theta_hat) / (np.linalg.norm(w_true_norm) * np.linalg.norm(theta_hat)))


def run_session(strategy, feats_raw, feats_norm, scale, condition, rng, n_queries=10, switch_at=5):
    w_a = profile_to_vector(PROFILES[PROFILE_NAMES[0]])
    w_b = profile_to_vector(PROFILES[PROFILE_NAMES[1]])

    cos_track = []
    for t in range(n_queries):
        if condition == "drifting" and t >= switch_at:
            w_true = w_b
        else:
            w_true = w_a
        w_true_scaled_norm = (w_true * scale) / (np.linalg.norm(w_true * scale) + 1e-8)

        i, j = strategy.select(feats_norm)
        p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
        winner = i if rng.random() < p_i else j
        strategy.update(feats_norm, i, j, winner)

        cos_track.append(cos_to(strategy.theta_hat, w_true_scaled_norm))
    return cos_track


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=10)
    args = p.parse_args()

    for condition in ("stationary", "drifting"):
        tracks_active, tracks_active_l2, tracks_dpb_stat, tracks_dpb_disc = [], [], [], []
        for rep in range(args.repeats):
            rng_data = np.random.default_rng(9000 + rep)
            feats_raw = build_pool(rng_data)
            scale = feats_raw.std(axis=0)
            scale = np.where(scale < 1e-6, 1.0, scale)
            feats_norm = feats_raw / scale

            rng_a = np.random.default_rng(1000 + rep)
            tracks_active.append(run_session(
                ActiveQueryStrategy(rng=rng_a), feats_raw, feats_norm, scale, condition, rng_a, args.n_queries))

            rng_a2 = np.random.default_rng(1000 + rep)
            tracks_active_l2.append(run_session(
                ActiveQueryL2Strategy(rng=rng_a2), feats_raw, feats_norm, scale, condition, rng_a2, args.n_queries))

            rng_b = np.random.default_rng(2000 + rep)
            tracks_dpb_stat.append(run_session(
                DPBStrategy(gamma=1.0, rng=rng_b), feats_raw, feats_norm, scale, condition, rng_b, args.n_queries))

            rng_c = np.random.default_rng(3000 + rep)
            tracks_dpb_disc.append(run_session(
                DPBStrategy(gamma=0.7, rng=rng_c), feats_raw, feats_norm, scale, condition, rng_c, args.n_queries))

        mean_active = np.mean(tracks_active, axis=0)
        mean_active_l2 = np.mean(tracks_active_l2, axis=0)
        mean_dpb_stat = np.mean(tracks_dpb_stat, axis=0)
        mean_dpb_disc = np.mean(tracks_dpb_disc, axis=0)

        print(f"\n=== {condition} true user ({args.repeats} repeats) ===")
        print(f"{'query':>6} {'ActiveQuery':>12} {'+L2 only':>10} {'DPB(g=1.0)':>12} {'DPB(g=0.7)':>12}")
        for t in range(args.n_queries):
            print(f"{t+1:>6} {mean_active[t]:>12.3f} {mean_active_l2[t]:>10.3f} {mean_dpb_stat[t]:>12.3f} {mean_dpb_disc[t]:>12.3f}")


if __name__ == "__main__":
    main()
