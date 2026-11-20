#!/usr/bin/env python
"""How much does cutting the pilot from 60 (v2) to 20 (v3) questions cost?
Synthetic participants, as in utils/style_preference_benchmark.py: theta* ~
N(0, I) (gauss) or Dirichlet(0.3) magnitudes with random signs (sparse),
unit norm, over the pilot's 6 normalized features; each answers
P(A) = sigmoid(c (u_A - u_B)), c = --answer-scale.

Question sets (train = what theta is fit on, Laplace Bradley-Terry, prior N(0, I)):
  v2           42 fit questions
  v3           10 train questions (utils/build_pilot_v3.py, greedy D-optimal)
  v2-random10  10 fit questions drawn at random per participant (what the
               D-optimal choice adds over an arbitrary 10)
Scored on the 14 v2 test questions, which no set trains on:
  agreement   share of test questions where theta_hat prefers the option
              theta* prefers (noise-free)
  answer acc  expected share of the participant's noisy answers theta_hat
              predicts (what the pilot measures)
  cos         cosine(theta_hat, theta*)
and the spread of the MEASURED test accuracy per participant on the set's own
test block (v2: 14 questions, v3: 8) -- the pilot's measurement noise.
Means with a 95% bootstrap over participants.

Usage:
    PYTHONPATH=. .venv/bin/python utils/pilot_size_sim.py --out results/codebook/pilot_v3/size_sim.json
"""
import argparse
import json

import numpy as np

from src.preference.bayesian_query import laplace_posterior
from utils.person_transfer_benchmark import boot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--v2", default="results/codebook/pilot_v2/questions.json")
    p.add_argument("--v3", default="results/codebook/pilot_v3/questions.json")
    p.add_argument("--n-users", type=int, default=2000)
    p.add_argument("--answer-scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/codebook/pilot_v3/size_sim.json")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    V2, V3 = json.load(open(args.v2)), json.load(open(args.v3))
    dx = lambda qs: np.array([np.subtract(*np.asarray(q["x"])) for q in qs])  # option 0 - option 1
    v2_fit = dx([q for q in V2["questions"] if q["split"] == "fit"])
    held = dx([q for q in V2["questions"] if q["split"] == "test"])
    v3_train = dx([q for q in V3["questions"] if q["split"] == "train"])
    v3_test = dx([q for q in V3["questions"] if q["split"] == "test"])
    D = held.shape[1]
    print(f"v2 fit {len(v2_fit)}, v3 train {len(v3_train)}, held-out (v2 test) {len(held)}, v3 test {len(v3_test)}")

    res = {}
    for users in ("gauss", "sparse"):
        th = rng.standard_normal((args.n_users, D)) if users == "gauss" else \
            rng.dirichlet(0.3 * np.ones(D), size=args.n_users) * rng.choice([-1, 1], size=(args.n_users, D))
        th /= np.linalg.norm(th, axis=1, keepdims=True)
        sets = {"v2": lambda: v2_fit, "v3": lambda: v3_train,
                "v2-random10": lambda: v2_fit[rng.choice(len(v2_fit), len(v3_train), replace=False)]}
        own_test = {"v2": held, "v3": v3_test}
        out = {}
        for name, get in sets.items():
            m = []
            for t in th:
                F = get()
                u = args.answer_scale * (F @ t)
                win = rng.random(len(F)) < 1 / (1 + np.exp(-u))  # option 0 picked
                th_hat = laplace_posterior(np.where(win[:, None], F, -F), D)[0]
                du, dv = held @ t, held @ th_hat
                right = np.sign(dv) == np.sign(du)
                pt = 1 / (1 + np.exp(-args.answer_scale * np.abs(du)))
                row = [right.mean(), np.where(right, pt, 1 - pt).mean(),
                       th_hat @ t / max(np.linalg.norm(th_hat), 1e-12)]
                if name in own_test:  # one noisy test block, as the pilot would score it
                    T = own_test[name]
                    pa = 1 / (1 + np.exp(-args.answer_scale * (T @ t)))
                    a = rng.random(len(T)) < pa
                    row.append(np.mean((T @ th_hat > 0) == a))
                m.append(row)
            m = np.array(m)
            out[name] = {"agreement": boot(m[:, 0], rng), "answer_acc": boot(m[:, 1], rng), "cos": boot(m[:, 2], rng)}
            if m.shape[1] > 3:
                out[name]["measured_test_acc_mean"] = float(m[:, 3].mean())
                out[name]["measured_test_acc_sd"] = float(m[:, 3].std())
        res[users] = out
        print(f"=== {users} users ({args.n_users}), answer scale {args.answer_scale:g}, scored on the {len(held)} v2 test questions ===")
        for name, r in out.items():
            extra = (f"   measured test acc {r['measured_test_acc_mean']:.3f} +- {r['measured_test_acc_sd']:.3f} (sd over participants)"
                     if "measured_test_acc_sd" in r else "")
            print(f"  {name:12s} agreement {r['agreement'][0]:.3f} [{r['agreement'][1]:.3f},{r['agreement'][2]:.3f}]  "
                  f"answer acc {r['answer_acc'][0]:.3f}  cos {r['cos'][0]:.3f}" + extra)
    json.dump({"args": vars(args), "results": res}, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
