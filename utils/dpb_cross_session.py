#!/usr/bin/env python
"""Re-tests DPB's discount factor under a more realistic drift model than
utils/dpb_vs_active_query.py's mid-session switch: preference is CONSTANT
within each N=10 sitting (a user doesn't flip-flop answering 10 questions
in a row) and only changes BETWEEN separate sessions (different day /
crowd density / mood) -- 3 sessions of 10 queries each, true preference =
profile A for sessions 0-1, switches to profile B for session 2.

Compares:
  - ActiveQueryStrategy: no discount at all, accumulates everything (baseline)
  - DPB, discount_unit="round" (gamma=0.7 per query, as in the earlier
    mid-session test): ages every individual comparison, including ones
    from EARLIER IN THE SAME SESSION -- may discard useful within-session
    signal unnecessarily.
  - DPB, discount_unit="session" (gamma=0.7 per SESSION): every comparison
    in the current session is weighted equally; only whole past sessions
    get discounted as a block -- the more realistic model for this drift
    pattern.

Usage:
    PYTHONPATH=. python utils/dpb_cross_session.py --repeats 20
"""
import argparse

import numpy as np

from src.preference.features import extract_features
from src.preference.bradley_terry import predict_preference_prob
from src.preference.active_query import ActiveQueryStrategy
from src.preference.dpb import DPBStrategy
from utils.preference_poc import PROFILES, profile_to_vector, make_synthetic_candidates

PROFILE_NAMES = list(PROFILES.keys())
N_SESSIONS = 3
QUERIES_PER_SESSION = 10
SWITCH_AT_SESSION = 2  # sessions 0,1 = profile A; session 2 = profile B


def build_pool(rng):
    target = (0.0, 5.0)
    candidates = make_synthetic_candidates(target, num_candidates=20, rng=rng)
    return np.array([extract_features(c, target) for c in candidates])


def cos_to(theta_hat, w_true_norm):
    if theta_hat is None or np.linalg.norm(theta_hat) < 1e-8:
        return 0.0
    return float(np.dot(w_true_norm, theta_hat) / (np.linalg.norm(w_true_norm) * np.linalg.norm(theta_hat)))


def run_multi_session(strategy, feats_raw, feats_norm, scale, rng, is_dpb_session_mode):
    w_a = profile_to_vector(PROFILES[PROFILE_NAMES[0]])
    w_b = profile_to_vector(PROFILES[PROFILE_NAMES[1]])
    end_of_session_cos = []

    for sess in range(N_SESSIONS):
        w_true = w_b if sess >= SWITCH_AT_SESSION else w_a
        w_true_scaled_norm = (w_true * scale) / (np.linalg.norm(w_true * scale) + 1e-8)
        if is_dpb_session_mode and sess > 0:
            strategy.new_session()
        for _ in range(QUERIES_PER_SESSION):
            i, j = strategy.select(feats_norm)
            p_i = predict_preference_prob(w_true, feats_raw[i], feats_raw[j])
            winner = i if rng.random() < p_i else j
            strategy.update(feats_norm, i, j, winner)
        end_of_session_cos.append(cos_to(strategy.theta_hat, w_true_scaled_norm))
    return end_of_session_cos


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=20)
    args = p.parse_args()

    results = {"ActiveQuery (no discount)": [], "DPB round-discount (g=0.7)": [], "DPB session-discount (g=0.7)": []}

    for rep in range(args.repeats):
        rng_data = np.random.default_rng(9000 + rep)
        feats_raw = build_pool(rng_data)
        scale = feats_raw.std(axis=0)
        scale = np.where(scale < 1e-6, 1.0, scale)
        feats_norm = feats_raw / scale

        rng_a = np.random.default_rng(1000 + rep)
        results["ActiveQuery (no discount)"].append(
            run_multi_session(ActiveQueryStrategy(rng=rng_a), feats_raw, feats_norm, scale, rng_a, False))

        rng_b = np.random.default_rng(2000 + rep)
        results["DPB round-discount (g=0.7)"].append(
            run_multi_session(DPBStrategy(gamma=0.7, discount_unit="round", rng=rng_b),
                               feats_raw, feats_norm, scale, rng_b, False))

        rng_c = np.random.default_rng(3000 + rep)
        results["DPB session-discount (g=0.7)"].append(
            run_multi_session(DPBStrategy(gamma=0.7, discount_unit="session", rng=rng_c),
                               feats_raw, feats_norm, scale, rng_c, True))

    print(f"cosine(theta_hat, true-preference-of-that-session) at END of each session, {args.repeats} repeats")
    print(f"sessions 0-1 = profile A, session 2 = profile B (switch happens BETWEEN sessions, not mid-session)\n")
    header = f"{'method':>32}" + "".join(f"{'session '+str(s):>12}" for s in range(N_SESSIONS))
    print(header)
    for name, runs in results.items():
        means = np.mean(runs, axis=0)
        row = f"{name:>32}" + "".join(f"{m:>12.3f}" for m in means)
        print(row)


if __name__ == "__main__":
    main()
