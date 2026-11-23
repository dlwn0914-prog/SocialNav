#!/usr/bin/env python
"""Style preference over the walking-behaviour codebook: safety is a filter
on the candidates, the preference only chooses among safe ones.

Situations: held-out (test) people with a far destination, as in the prior's
training, and a predicted conflict (constant velocity: someone passes within
1 m in the next 8 s) -- where a walking style shows at all. --query-pool asks
the questions in conflict situations (inter), in the others (nointer) or in
any (all); test situations, users and the feature normalization stay those of
the inter run, so the three runs pair up per user.

Candidates: the K codes of the heading-frame codebook (constant velocity +
decoded deviation) that are SAFE -- min distance to anyone >= --clearance
(people at constant velocity) and progress to the destination >= --min-progress
x the best code's. Situations with fewer than --min-safe safe codes are
dropped. --mirror adds each code reflected left-right (32 candidates): the
learned codes mostly go the way people did, so without it both sides of the
conflict person are on offer in only ~22% of situations (35% before the
safety filter); a mirrored code's prior is the original code's prior in the
mirrored situation, and each set gets half the mass.

Style features phi (d = 5), none of them has a right sign:
  side       which side the path passes the conflict person (the one with the
             earliest predicted conflict; without one, the one passing
             closest; nobody around: 0): tanh(lateral offset at the closest
             approach / 0.5 m), + = robot on their left
  front      ahead of or behind them: tanh(offset along their walking
             direction at the closest approach / 0.5 m), + = cross in front;
             0 if they stand (< 0.2 m/s)
  yield      mean speed over the path / current speed - 1, < 0 = slow down
  detour     how far the path strays from the straight line to the
             destination (max over its points) [m]
  human      log p(code | situation), the data prior (how typical for people)
normalized with one mean/std over the safe candidates of all query
situations.

Users: synthetic, theta* ~ N(0, I) (--users gauss) or Dirichlet(0.3)
magnitudes with random signs (--users sparse), unit norm; answers
P(A) = sigmoid(c (u_A - u_B)), c = --answer-scale. These users differ by
construction -- whether real people do is for the pilot to show; what this
measures is how fast each query method finds a style preference.

Reported on held-out situations (95% bootstrap over users):
  agreement  share of safe-candidate pairs where theta_hat orders the pair as
             theta* does (0.5 = chance)
  answer acc expected share of the user's (noisy) A/B answers that theta_hat
             predicts -- the number a pilot can measure; the oracle row is
             its ceiling under the answer noise
  regret     (u_max - u_pick) / (u_max - u_min) among safe candidates
References: no preference (random safe code), human prior (argmax log p),
oracle theta*. Paired differences EUBO - AQ, EUBO - Random per query count.

Robustness (users the estimator's model does not describe): --lapse e makes
each answer a coin flip with probability e; --nonlinear w adds pairwise
feature interactions to the true utility, u = theta* . phi + w q(phi),
q(phi) = sum_{a<b} W_ab phi_a phi_b with a random unit-norm W per user, so
no linear theta reaches zero regret (the oracle theta* row is then > 0).
--answer-scale changes the answer noise.

DPO baselines (src/preference/dpo.py; --methods ...,DPO,DPO-EUBO): theta from
the DPO loss with the human prior as the reference policy (theta_ref =
sd_human e_human under this run's normalization), on random pairs (DPO) or
on the pairs EUBO picks from the Laplace posterior of the same answers
(DPO-EUBO, same questions as EUBO, different estimator).

Usage:
    PYTHONPATH=. .venv/bin/python utils/style_preference_benchmark.py \
        --prior results/codebook/goalfix/gc_dest_s1.pt --out results/codebook/style/gc_gauss.json
"""
import argparse
import json

import numpy as np
import torch

from src.preference.traj_vqvae import TrajVQVAE
from utils.code_preference_benchmark import code_features
from src.preference.dpo import DPOStrategy
from utils.person_transfer_benchmark import METHODS, boot
from utils.train_code_prior import load_prior, situation

FEATURES = ["side", "front", "yield", "detour", "human"]


def style_features(paths, ppl, n_ppl, speed, dest, logp, dt=0.8):
    """paths (B, K, H, 2) -> (B, K, 5) raw style features."""
    B, K, H, _ = paths.shape
    t = dt * np.arange(1, H + 1)
    present = np.arange(ppl.shape[1])[None] < n_ppl[:, None]
    rel_v = ppl[..., 2:] - np.stack([speed, np.zeros_like(speed)], axis=-1)[:, None]
    vv = (rel_v ** 2).sum(-1) + 1e-6
    t_ca = np.clip(-(ppl[..., :2] * rel_v).sum(-1) / vv, 0, 8)
    d_ca = np.linalg.norm(ppl[..., :2] + t_ca[..., None] * rel_v, axis=-1)
    j = np.where(present & (d_ca < 1.0), t_ca, np.inf).argmin(1)  # conflict person: earliest predicted conflict
    none = ~(present & (d_ca < 1.0)).any(1)
    j[none] = np.where(present, d_ca, np.inf)[none].argmin(1)
    r = np.arange(B)
    alone = ~present.any(1)
    pj = ppl[r, j]  # (B, 4)
    traj = pj[:, None, :2] + t[None, :, None] * pj[:, None, 2:]  # (B, H, 2)
    gap = paths - traj[:, None]  # (B, K, H, 2) robot relative to them
    k_ca = np.linalg.norm(gap, axis=-1).argmin(axis=2)
    g = np.take_along_axis(gap, k_ca[..., None, None].repeat(2, -1), axis=2)[:, :, 0]  # (B, K, 2)
    sp = np.linalg.norm(pj[:, 2:], axis=-1)
    vhat = pj[:, 2:] / np.maximum(sp, 1e-6)[:, None]
    side = np.where(alone[:, None], 0.0, np.tanh(g[..., 1] / 0.5))
    front = np.where((sp[:, None] > 0.2) & ~alone[:, None], np.tanh((g * vhat[:, None]).sum(-1) / 0.5), 0.0)
    full = np.concatenate([np.zeros((B, K, 1, 2)), paths], axis=2)
    length = np.linalg.norm(np.diff(full, axis=2), axis=-1).sum(-1)
    yld = length / np.maximum(speed[:, None] * t[-1], 1e-6) - 1
    u = dest / np.maximum(np.linalg.norm(dest, axis=1, keepdims=True), 1e-6)
    detour = np.abs(paths[..., 0] * u[:, None, None, 1] - paths[..., 1] * u[:, None, None, 0]).max(-1)
    return np.stack([side, front, yld, detour, logp], axis=-1)


def mirror_situation(x):
    """Reflect a (speed, past 2x2, dest, social_summary 8) situation row left-right."""
    m = x.copy()
    for c in (2, 4, 6, 8, 10):  # past y's, dest y, nearest person y and vy
        m[:, c] = -m[:, c]
    m[:, 12], m[:, 13], m[:, 14] = x[:, 13], x[:, 12], -x[:, 14]  # left/right counts swap
    return m


def sample_theta(rng, n, kind):
    th = rng.standard_normal((n, len(FEATURES))) if kind == "gauss" else \
        rng.dirichlet(0.3 * np.ones(len(FEATURES)), size=n) * rng.choice([-1, 1], size=(n, len(FEATURES)))
    return th / np.linalg.norm(th, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default="results/codebook/goalfix/gc_dest_s1.pt")
    p.add_argument("--clearance", type=float, default=0.5)
    p.add_argument("--min-progress", type=float, default=0.5)
    p.add_argument("--min-safe", type=int, default=4)
    p.add_argument("--mirror", action="store_true", help="add the left-right reflected codes as candidates")
    p.add_argument("--query-pool", choices=["inter", "nointer", "all"], default="inter",
                   help="situations the questions come from (tests are always conflict situations)")
    p.add_argument("--users", choices=["gauss", "sparse"], default="gauss")
    p.add_argument("--n-users", type=int, default=200)
    p.add_argument("--n-query-situations", type=int, default=10)
    p.add_argument("--n-test-situations", type=int, default=20)
    p.add_argument("--checkpoints", default="0,2,5,10,20")
    p.add_argument("--answer-scale", type=float, default=5.0)
    p.add_argument("--methods", default="Random,ActiveQuery,EUBO")
    p.add_argument("--dpo-beta", type=float, default=1.0)
    p.add_argument("--dpo-steps", type=int, default=100)
    p.add_argument("--lapse", type=float, default=0.0, help="probability of a random answer")
    p.add_argument("--nonlinear", type=float, default=0.0, help="weight of the pairwise-interaction term of the true utility")
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
        code_dev = vq.decode_codes().numpy()
    K, H = code_dev.shape[:2]

    base = np.where((d["split"] == 2) & (d["dest_t"] >= pa["min_dest_t"]))[0]
    _, x = situation(d, "heading", True, pa["people_input"], pa["goal"])
    assert pa["people_input"] == "summary", "needs a prior trained with --people-input summary"
    conflict = x[base][:, -4] < 8.0  # social_summary's TTC column
    assert x.shape[1] == 15, "mirror_situation expects speed, past, dest and the 8-number summary"
    with torch.no_grad():
        logp = torch.log_softmax(net(torch.tensor((x[base] - net.mu) / net.sd)), dim=1).numpy()
        if args.mirror:
            logp_m = torch.log_softmax(net(torch.tensor((mirror_situation(x[base]) - net.mu) / net.sd)), dim=1).numpy()
            logp = np.concatenate([logp, logp_m], axis=1) - np.log(2)
            code_dev = np.concatenate([code_dev, code_dev * np.array([1.0, -1.0])], axis=0)
            K = len(code_dev)
    t = 0.8 * np.arange(1, H + 1)
    paths = np.stack([d["speed"][base][:, None] * t, np.zeros((len(base), H))], axis=-1)[:, None] + code_dev[None]
    ppl, n_ppl, dest, speed = d["people"][base], d["n_people"][base], d["dest"][base], d["speed"][base]
    raw_old, dmin = code_features(paths, ppl, n_ppl, dest, logp)
    progress = raw_old[..., 2]
    safe = (dmin >= args.clearance) & (progress >= args.min_progress * progress.max(1, keepdims=True))
    keep = safe.sum(1) >= args.min_safe
    raw = style_features(paths, ppl, n_ppl, speed, dest, logp)[keep]
    safe, real, inter = safe[keep], d["future"][base][keep], conflict[keep]
    ade_real = np.linalg.norm(paths[keep] - real[:, None], axis=-1).mean(axis=2)
    S, C = len(raw), np.where(inter)[0]
    print(f"{conflict.sum()} conflict situations of {len(base)} (test people, dest >= {pa['min_dest_t']:g} s); "
          f"{len(C)} / {S} (conflict / all) keep >= {args.min_safe} "
          f"safe codes (clearance >= {args.clearance} m, progress >= {args.min_progress:g} x best); "
          f"safe codes median {np.median(safe[C].sum(1)):.0f}/{K}")

    thetas = sample_theta(rng, args.n_users, args.users)
    nf = len(FEATURES)
    ia, ib = np.triu_indices(nf, 1)
    Ws = np.random.default_rng(args.seed + 11).normal(size=(args.n_users, len(ia)))  # own stream: thetas/users unchanged
    Ws /= np.linalg.norm(Ws, axis=1, keepdims=True)
    W_of = {th.tobytes(): w for th, w in zip(thetas, Ws)}

    def true_u(F, th):
        u = F @ th
        if args.nonlinear:
            u = u + args.nonlinear * (F[:, ia] * F[:, ib]) @ W_of[th.tobytes()]
        return u
    users = []
    for _ in range(args.n_users):
        idx = C[rng.choice(len(C), args.n_query_situations + args.n_test_situations, replace=False)]
        users.append((idx[:args.n_query_situations], idx[args.n_query_situations:]))
    q_all = np.concatenate([q for q, _ in users])  # normalization of the inter run, kept for every pool
    if args.query_pool != "inter":
        qrng = np.random.default_rng(args.seed + 7)
        pool = np.where(~inter)[0] if args.query_pool == "nointer" else np.arange(S)
        users = [(qrng.choice(np.setdiff1d(pool, te), args.n_query_situations, replace=False), te) for _, te in users]
    pool_rows = raw[q_all][safe[q_all]]
    mu, sd = pool_rows.mean(0), pool_rows.std(0) + 1e-9
    X = (raw - mu) / sd
    cand = [np.where(safe[s])[0] for s in range(S)]
    within = np.mean([X[s][cand[s]].std(0) for s in C], axis=0)  # does each style vary among safe codes?
    corr = np.corrcoef(pool_rows.T)
    both_sides = np.mean([(raw[s][cand[s], 0].max() > 0.3) and (raw[s][cand[s], 0].min() < -0.3) for s in C])
    both_front = np.mean([(raw[s][cand[s], 1].max() > 0.3) and (raw[s][cand[s], 1].min() < -0.3) for s in C])
    print(f"  safe codes pass the conflict person on both sides in {both_sides:.0%} of situations, "
          f"both in front and behind in {both_front:.0%}")
    print("  feature spread among a situation's safe codes (normalized std): "
          + ", ".join(f"{f} {v:.2f}" for f, v in zip(FEATURES, within)))

    pairs = {}
    def pair_idx(n):
        if n not in pairs:
            pairs[n] = np.triu_indices(n, 1)
        return pairs[n]

    def evaluate(theta_hat, theta, test):
        out = []
        for s in test:
            F = X[s][cand[s]]
            u = true_u(F, theta)
            i, j = pair_idx(len(F))
            du = u[i] - u[j]
            if theta_hat is None:
                agree, acc, k = 0.5, 0.5, int(rng.integers(len(F)))
            else:
                v = F @ theta_hat
                dv = v[i] - v[j]
                right = np.sign(dv) == np.sign(du)
                agree = float(np.mean(right))
                p_true = 1 / (1 + np.exp(-args.answer_scale * np.abs(du)))  # chance the user picks their better one
                acc = float(np.mean(np.where(right, p_true, 1 - p_true)))
                k = int(np.argmax(v))
            out.append((agree, acc, (u.max() - u[k]) / max(np.ptp(u), 1e-9), ade_real[s, cand[s][k]]))
        return np.mean(out, axis=0)

    METRICS = ["agreement", "answer_acc", "regret", "ade_real"]
    rows = {}
    def add(name, vals):
        vals = np.asarray(vals)
        rows[name] = {m: boot(vals[:, i], rng) for i, m in enumerate(METRICS)}

    add("no preference (random safe code)", [np.mean([evaluate(None, th, te) for _ in range(5)], axis=0)
                                             for th, (_, te) in zip(thetas, users)])
    add("human prior (argmax log p)", [evaluate(np.eye(len(FEATURES))[-1], th, te) for th, (_, te) in zip(thetas, users)])
    add("oracle theta*", [evaluate(th, th, te) for th, (_, te) in zip(thetas, users)])
    curves, per_user = {}, {}
    for m in args.methods.split(","):
        per = {c: [] for c in checkpoints}
        for ui, (th, (qs, te)) in enumerate(zip(thetas, users)):
            srng = np.random.default_rng(args.seed * 100003 + ui)
            if m.startswith("DPO"):
                strat = DPOStrategy(sd[-1] * np.eye(len(FEATURES))[-1], "eubo" if m == "DPO-EUBO" else "random",
                                    beta=args.dpo_beta, steps=args.dpo_steps, rng=srng)
            else:
                strat = METHODS[m](srng)
            theta_hat = None
            for q in range(max(checkpoints) + 1):
                if q in per:
                    per[q].append(evaluate(theta_hat, th, te))
                if q == max(checkpoints):
                    break
                s = qs[q % len(qs)]
                F = X[s][cand[s]]
                i, j = strat.select(F)
                if strat.done():
                    continue
                u = args.answer_scale * true_u(F, th)
                p_i = 1 / (1 + np.exp(-(u[i] - u[j])))
                if args.lapse:
                    p_i = (1 - args.lapse) * p_i + args.lapse * 0.5
                strat.update(F, i, j, i if srng.random() < p_i else j)
                theta_hat = strat.theta_hat
        per = {c: np.asarray(v) for c, v in per.items()}
        curves[m] = {c: {mm: boot(v[:, k], rng) for k, mm in enumerate(METRICS[:3])} for c, v in per.items()}
        per_user[m] = per
        add(f"{m} ({max(checkpoints)} queries)", per[max(checkpoints)])
        print(f"  {m}: agreement / answer acc after " + ", ".join(
            f"{c}: {curves[m][c]['agreement'][0]:.3f}/{curves[m][c]['answer_acc'][0]:.3f}" for c in checkpoints))

    paired = {}
    for a, b in [("EUBO", "ActiveQuery"), ("EUBO", "Random"), ("EUBO", "InfoGain"), ("EUBO", "DPO"), ("EUBO", "DPO-EUBO"), ("DPO", "Random")]:
        if a in per_user and b in per_user:
            paired[f"{a} - {b}"] = {str(c): {mm: boot(per_user[a][c][:, k] - per_user[b][c][:, k], rng)
                                             for k, mm in enumerate(METRICS[:3])} for c in checkpoints if c > 0}
    print(f"=== held-out conflict situations, questions from {args.query_pool}, {args.n_users} synthetic users ({args.users}), answer scale {args.answer_scale:g} ===")
    print(f"  {'':>34} " + " ".join(f"{m:>22}" for m in METRICS))
    for name, r in rows.items():
        print(f"  {name:>34} " + " ".join(f"{r[m][0]:6.3f} [{r[m][1]:.3f},{r[m][2]:.3f}]" for m in METRICS))
    for k, v in paired.items():
        print(f"  {k}: " + "; ".join(f"{c} q agreement {x['agreement'][0]:+.3f} [{x['agreement'][1]:+.3f},{x['agreement'][2]:+.3f}]"
                                    for c, x in v.items()))
    res = {"args": vars(args), "n_situations": int(len(C)), "n_situations_all": S, "n_conflict": int(conflict.sum()), "features": FEATURES,
           "safe_median": float(np.median(safe.sum(1))), "within_spread": within.tolist(),
           "both_sides": float(both_sides), "both_front": float(both_front), "feature_corr": corr.tolist(),
           "rows": rows, "curves": {m: {str(c): v for c, v in cu.items()} for m, cu in curves.items()}, "paired": paired,
           "per_user": {m: {str(c): v.tolist() for c, v in pu.items()} for m, pu in per_user.items()}}
    out = args.out or args.prior.replace(".pt", "_style.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
