#!/usr/bin/env python
"""Personal preference over the walking-behaviour codebook, used to pick the
robot's path at test time.

Pipeline per situation (a real pedestrian state from held-out people):
  candidates  the K codes of the heading-frame codebook
              (utils/train_traj_vqvae.py), each as a path = constant velocity
              at the current speed + the code's decoded deviation
  data prior  log p(code | situation) from utils/train_code_prior.py (self +
              people summary + track-end destination): what people do here
  features    phi(code, situation), d = 6:
                log_prior   the data prior
                clearance   min distance to anyone over the path (people at
                            constant velocity), capped at 3 m
                progress    how much closer the path's end is to the
                            destination [m]
                intrusion   seconds-weighted shortfall below 1.2 m, summed
                            over people
                effort      mean |second difference| of the path [m]
                pass_left   side the path passes the nearest person ahead
                            (tanh of the lateral offset at closest approach
                            / 0.5 m; + = robot on their left, 0 if no one)
              normalized with one mean/std over all candidates of the query
              situations (no answers used)
  selection   argmax_k theta . phi(k) -- the test-time choice among sampled
              paths; theta = (1, 0, ..., 0) picks the data prior's favourite

Users: synthetic, theta* drawn per user (log_prior, clearance, progress >= 0,
intrusion, effort <= 0, pass_left either sign), unit norm, --users orthant
(similar users) or sparse (each mainly cares about one or two features); answers
P(A) = sigmoid(c (u_A - u_B)) with u = theta* . phi and c = --answer-scale.
A session asks --n-queries A/B questions over --n-query-situations
situations (cycled; EUBO-anyscene chooses), then theta_hat picks the code in
--n-test-situations held-out interactive situations with no questions.
Interactive (--interaction): someone ahead within --max-ahead (95% of
situations on both datasets), or a predicted conflict (62% on GC). Query
situations come from the interactive pool, its complement or all situations
(--query-pool); everything else is drawn the same way, so the runs pair up.

Reported (held-out situations, 95% bootstrap over users):
  paired       per-user differences of regret between methods (--pairs) at
               every query count, with the share of users where A wins
  regret       (u_max - u_pick) / (u_max - u_min) per situation
  near-miss    share of picked paths that come within 0.5 m of someone
  ADE to real  how far the picked path is from what the pedestrian really did
References: random code, human prior (argmax log p), population mean theta*
(one direction for everyone), oracle theta* (regret 0).

Usage:
    PYTHONPATH=. .venv/bin/python utils/code_preference_benchmark.py \
        --prior results/codebook/goalfix/gc_dest_s1.pt --out results/codebook/code_pref_gc.json
"""
import argparse
import json

import numpy as np
import torch

from src.preference.traj_vqvae import TrajVQVAE
from utils.person_transfer_benchmark import METHODS, boot
from utils.train_code_prior import load_prior, situation, social_summary

FEATURES = ["log_prior", "clearance", "progress", "intrusion", "effort", "pass_left"]
SIGNS = np.array([1, 1, 1, -1, -1, 0])  # sign constraint of theta* (0 = free)


def code_features(paths, ppl, n_ppl, dest, logp, dt=0.8):
    """paths (B, K, H, 2), ppl (B, N, 4), dest (B, 2), logp (B, K) -> (B, K, 6) raw features,
    plus the min distance (B, K) for the near-miss rate."""
    B, K, H, _ = paths.shape
    t = dt * np.arange(1, H + 1)
    others = ppl[:, None, :, :2] + t[None, :, None, None] * ppl[:, None, :, 2:]  # (B, H, N, 2)
    present = np.arange(ppl.shape[1])[None] < n_ppl[:, None]  # (B, N)
    dist = np.linalg.norm(paths[:, :, :, None] - others[:, None], axis=-1)  # (B, K, H, N)
    dist = np.where(present[:, None, None], dist, np.inf)
    dmin = dist.min(axis=(2, 3))
    clearance = np.minimum(dmin, 3.0)
    progress = np.linalg.norm(dest, axis=1)[:, None] - np.linalg.norm(dest[:, None] - paths[:, :, -1], axis=-1)
    intrusion = (np.maximum(0.0, 1.2 - np.where(np.isfinite(dist), dist, 9.0)) * dt).sum(axis=(2, 3))
    full = np.concatenate([np.zeros((B, K, 1, 2)), paths], axis=2)
    effort = np.linalg.norm(np.diff(full, n=2, axis=2), axis=-1).mean(axis=2)
    ahead = present & (ppl[..., 0] > 0)
    dn = np.where(ahead, np.linalg.norm(ppl[..., :2], axis=-1), np.inf)
    j = dn.argmin(1)
    r = np.arange(B)
    has = np.isfinite(dn[r, j])
    near = others[r, :, j]  # (B, H, 2) nearest person ahead over time
    gap = paths - near[:, None]  # (B, K, H, 2)
    tca = np.linalg.norm(gap, axis=-1).argmin(axis=2)  # (B, K)
    lat = np.take_along_axis(gap[..., 1], tca[..., None], axis=2)[..., 0]
    pass_left = np.where(has[:, None], np.tanh(lat / 0.5), 0.0)
    return np.stack([logp, clearance, progress, intrusion, effort, pass_left], axis=-1), dmin


def sample_theta(rng, n, kind="orthant"):
    """orthant: |N(0, I)| with the sign constraints (users similar: all trade the
    same things off the same way); sparse: Dirichlet(0.3) magnitudes, so each
    user mainly cares about one or two features (users differ)."""
    th = rng.standard_normal((n, len(FEATURES))) if kind == "orthant" else \
        rng.dirichlet(0.3 * np.ones(len(FEATURES)), size=n) * rng.choice([-1, 1], size=(n, len(FEATURES)))
    th = np.where(SIGNS != 0, np.abs(th) * np.where(SIGNS == 0, 1, SIGNS), th)
    return th / np.linalg.norm(th, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default="results/codebook/goalfix/gc_dest_s1.pt")
    p.add_argument("--n-users", type=int, default=200)
    p.add_argument("--n-query-situations", type=int, default=10)
    p.add_argument("--n-test-situations", type=int, default=20)
    p.add_argument("--n-queries", type=int, default=20)
    p.add_argument("--checkpoints", default="0,2,5,10,20")
    p.add_argument("--pairs", default="EUBO-ActiveQuery,EUBO-Random,EUBO-anyscene-EUBO",
                   help="paired differences A-B (same users, same situations), reported at every checkpoint")
    p.add_argument("--users", choices=["orthant", "sparse"], default="orthant")
    p.add_argument("--answer-scale", type=float, default=5.0)
    p.add_argument("--max-ahead", type=float, default=5.0,
                   help="situations need someone ahead within this distance (otherwise social features are moot)")
    p.add_argument("--interaction", choices=["ahead", "conflict"], default="ahead",
                   help="what makes a situation interactive: someone ahead within --max-ahead, or a predicted "
                        "conflict (constant velocity: someone passes within 1 m in the next 8 s)")
    p.add_argument("--query-pool", choices=["inter", "nointer", "all"], default="inter",
                   help="where the query situations come from; held-out situations, users and the feature "
                        "normalization stay the same (interactive), so runs differing only here are paired")
    p.add_argument("--methods", default="Random,ActiveQuery,EUBO,EUBO-anyscene")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    checkpoints = [int(c) for c in args.checkpoints.split(",")]

    net, pa = load_prior(args.prior)
    ck = torch.load(pa["ckpt"], weights_only=False)
    assert ck["args"].get("reference", "heading") == "heading", "candidates are built in the heading frame"
    d = dict(np.load(pa["data"]))
    vq = TrajVQVAE(horizon=d["future"].shape[1], num_codes=ck["args"]["codes"], latent=ck["args"]["latent"])
    vq.load_state_dict(ck["state_dict"])
    vq.eval()
    with torch.no_grad():
        code_dev = vq.decode_codes().numpy()  # (K, H, 2)

    # base: held-out (test) people with a far destination, as in the prior's training
    base = np.where((d["split"] == 2) & (d["dest_t"] >= pa["min_dest_t"]))[0]
    pos = d["people"][base][..., :2]
    present = np.arange(pos.shape[1])[None] < d["n_people"][base][:, None]
    if args.interaction == "ahead":
        inter = (present & (pos[..., 0] > 0) & (np.linalg.norm(pos, axis=-1) < args.max_ahead)).any(1)
    else:
        inter = social_summary(d["people"][base], present, d["speed"][base])[:, 4] < 8.0
    _, x = situation(d, "heading", True, pa["people_input"], pa["goal"])
    with torch.no_grad():
        logp = torch.log_softmax(net(torch.tensor((x[base] - net.mu) / net.sd)), dim=1).numpy()
    K, H = code_dev.shape[:2]
    t = 0.8 * np.arange(1, H + 1)
    cv = np.stack([d["speed"][base][:, None] * t, np.zeros((len(base), H))], axis=-1)  # (B, H, 2)
    paths = cv[:, None] + code_dev[None]  # (B, K, H, 2)
    raw, dmin = code_features(paths, d["people"][base], d["n_people"][base], d["dest"][base], logp)
    real = d["future"][base]
    ade_real = np.linalg.norm(paths - real[:, None], axis=-1).mean(axis=2)  # (B, K)
    pool = np.where(inter)[0]
    print(f"{len(base)} situations (test people, dest >= {pa['min_dest_t']:g} s), {len(pool)} interactive "
          f"({args.interaction}), {K} codes; queries from {args.query_pool}; prior {args.prior}")

    thetas = sample_theta(rng, args.n_users, args.users)
    pop_theta = thetas.mean(0)
    users, norm_q = [], []
    qrng = np.random.default_rng(args.seed + 7)  # query situations outside the interactive pool
    for u in range(args.n_users):
        idx = rng.choice(len(pool), args.n_query_situations + args.n_test_situations, replace=False)
        q_int, te = pool[idx[:args.n_query_situations]], pool[idx[args.n_query_situations:]]
        norm_q.append(q_int)
        if args.query_pool == "inter":
            qs = q_int
        else:
            cand = np.setdiff1d(np.where(~inter)[0] if args.query_pool == "nointer" else np.arange(len(base)), te)
            qs = qrng.choice(cand, args.n_query_situations, replace=False)
        users.append((qs, te))
    q_all = np.concatenate(norm_q)
    mu = raw[q_all].reshape(-1, raw.shape[-1]).mean(0)
    sd = raw[q_all].reshape(-1, raw.shape[-1]).std(0) + 1e-9
    X = (raw - mu) / sd

    def evaluate(theta_hat, theta, test):
        out = []
        for s in test:
            u = X[s] @ theta
            k = int(rng.integers(K)) if theta_hat is None else int(np.argmax(X[s] @ theta_hat))
            out.append(((u.max() - u[k]) / max(np.ptp(u), 1e-9), dmin[s, k] < 0.5, ade_real[s, k]))
        return np.mean(out, axis=0)

    rows = {}
    def add(name, vals):  # vals (n_users, 3)
        vals = np.asarray(vals)
        rows[name] = {m: boot(vals[:, i], rng) for i, m in enumerate(["regret", "near_miss", "ade_real"])}

    add("random code", [np.mean([evaluate(None, th, te) for _ in range(5)], axis=0) for th, (_, te) in zip(thetas, users)])
    e0 = np.eye(len(FEATURES))[0]
    add("human prior (argmax log p)", [evaluate(e0, th, te) for th, (_, te) in zip(thetas, users)])
    add("population mean theta*", [evaluate(pop_theta, th, te) for th, (_, te) in zip(thetas, users)])
    add("oracle theta*", [evaluate(th, th, te) for th, (_, te) in zip(thetas, users)])
    curves, per_user = {}, {}
    for m in args.methods.split(","):
        per = {c: [] for c in checkpoints}
        for ui, (th, (qs, te)) in enumerate(zip(thetas, users)):
            srng = np.random.default_rng(args.seed * 100003 + ui)
            strat = METHODS[m](srng)
            theta_hat = None
            for q in range(max(checkpoints) + 1):
                if q in per:
                    per[q].append(evaluate(theta_hat, th, te))
                if q == max(checkpoints):
                    break
                if hasattr(strat, "select_scene"):
                    k, i, j = strat.select_scene([X[s] for s in qs])
                else:
                    k = q % len(qs)
                    i, j = strat.select(X[qs[k]])
                if strat.done():
                    continue
                u = args.answer_scale * (X[qs[k]] @ th)
                strat.update(X[qs[k]], i, j, i if srng.random() < 1 / (1 + np.exp(-(u[i] - u[j]))) else j)
                theta_hat = strat.theta_hat
        curves[m] = {c: boot(np.asarray(v)[:, 0], rng) for c, v in per.items()}
        per_user[m] = {c: np.asarray(v)[:, 0] for c, v in per.items()}
        add(f"{m} ({max(checkpoints)} queries)", per[max(checkpoints)])
        print(f"  {m}: regret after " + ", ".join(f"{c}: {curves[m][c][0]:.3f}" for c in checkpoints))

    print(f"=== held-out situations, {args.n_users} synthetic users ({args.users}), answer scale {args.answer_scale:g} ===")
    print(f"  {'':>32} {'regret':>22} {'near-miss <0.5 m':>22} {'ADE to real [m]':>22}")
    for name, r in rows.items():
        print(f"  {name:>32} " + " ".join(f"{r[m][0]:6.3f} [{r[m][1]:.3f},{r[m][2]:.3f}]" for m in r))
    paired = {}
    for pair in args.pairs.split(","):
        a, b = next((a, pair[len(a) + 1:]) for a in per_user if pair.startswith(a + "-") and pair[len(a) + 1:] in per_user)
        paired[f"{a} - {b}"] = {c: boot(per_user[a][c] - per_user[b][c], rng) + (float(np.mean(per_user[a][c] < per_user[b][c])),)
                                for c in checkpoints if c > 0}
    print("  paired regret differences (mean [95% CI], share of users where A is better):")
    for k, v in paired.items():
        print(f"    {k:>28}: " + "; ".join(f"{c} q {d[0]:+.4f} [{d[1]:+.4f},{d[2]:+.4f}] {d[3]:.2f}" for c, d in v.items()))
    res = {"args": vars(args), "paired": {k: {str(c): d for c, d in v.items()} for k, v in paired.items()},
           "per_user": {m: {str(c): v.tolist() for c, v in pu.items()} for m, pu in per_user.items()}, "n_pool": int(len(pool)), "n_base": int(len(base)), "features": FEATURES, "rows": rows,
           "curves": {m: {str(c): v for c, v in cu.items()} for m, cu in curves.items()},
           "feature_mean": mu.tolist(), "feature_std": sd.tolist()}
    out = args.out or args.prior.replace(".pt", "_codepref.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
