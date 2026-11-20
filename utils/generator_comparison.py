#!/usr/bin/env python
"""Candidate generator x query method: Crowd-FM vs the VQ-VAE codebook, each
with Random / AQ / EUBO questions, on the SAME scenes, users and features.

Scenes: the 900 per-person Grand Central scenes that already have Crowd-FM
candidates (crowd_fm_ws/scenarios_gcp/v3: 150 people x 6 frames, 20
candidates each after PRIEST, 3.0 s on a 30-point grid, people in Crowd-FM's
window ahead of the robot, goal = where the pedestrian really was 6.4 s
later). The VQ-VAE candidates for the same scene are the K codes from the
pedestrian's state (speed from the GC annotations, heading frame = the
scene's robot frame): constant velocity + decoded deviation.

Made comparable (11/20 design):
  horizon   both evaluated on the first 3.0 s (Crowd-FM's horizon): paths
            interpolated to t = 0.6, 1.2, ..., 3.0 s
  people    the scene's people (what Crowd-FM was given), constant velocity
  safety    min distance >= --clearance at those times, progress to the goal
            >= --min-progress x the best candidate of the SAME generator;
            scenes where either generator has < --min-safe safe candidates
            are dropped
  features  side, front, yield, detour (utils/style_preference_benchmark.py)
            with the scene goal as the destination; "human" (the codebook's
            log-prior) is left out because Crowd-FM candidates have none.
            One normalization over the safe candidates of both generators.
Users: synthetic theta* (gauss / sparse, unit norm, 4-d), answers
P(A) = sigmoid(c (u_A - u_B)). Per user, --n-query-situations query scenes
and --n-test-situations test scenes of other people; both generators get
the same scenes, users and answer noise seeds.

Metrics on the test scenes (95% bootstrap over users):
  union regret  (u_max - u_pick) / (u_max - u_min) with max/min over the safe
                candidates of BOTH generators -- the comparable number: a
                generator whose candidates miss what the user wants is
                penalized even if it picks its own best
  own regret    the same within the generator's own safe candidates (the
                usual regret; not comparable across generators)
  agreement     pairwise ordering agreement within the generator's own set
References per generator: random safe candidate, oracle (argmax theta*
within its own set, i.e. its coverage limit). Users whose theta* likes
detours favour wide candidate sets by construction, so the summary is also
given for the users with theta_detour < 0 only ("no-detour users").

Usage:
    PYTHONPATH=. .venv/bin/python utils/generator_comparison.py --users gauss \
        --out results/codebook/generator/gc_gauss.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

from src.preference.traj_vqvae import TrajVQVAE
from utils.build_gc_scenes import load_tracks
from utils.code_preference_benchmark import code_features
from utils.person_transfer_benchmark import METHODS, boot
from utils.style_preference_benchmark import style_features

FEATS = ["side", "front", "yield", "detour"]
GENS = ["Crowd-FM", "VQ-VAE"]


def interp(paths, t_src, t_dst):
    """paths (K, T, 2) sampled at t_src -> (K, len(t_dst), 2)."""
    return np.stack([np.stack([np.interp(t_dst, t_src, p[:, a]) for a in range(2)], -1) for p in paths])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/v3")
    p.add_argument("--codebook", default="results/codebook/vq16.pt")
    p.add_argument("--annot-dir", default="/home/nuri2/datasets/grandcentral/raw/opentraj/datasets/GC/Annotation")
    p.add_argument("--dt", type=float, default=0.6)
    p.add_argument("--horizon", type=float, default=3.0)
    p.add_argument("--clearance", type=float, default=0.5)
    p.add_argument("--min-progress", type=float, default=0.5)
    p.add_argument("--min-safe", type=int, default=4)
    p.add_argument("--users", choices=["gauss", "sparse"], default="gauss")
    p.add_argument("--n-users", type=int, default=200)
    p.add_argument("--n-query-situations", type=int, default=10)
    p.add_argument("--n-test-situations", type=int, default=20)
    p.add_argument("--checkpoints", default="0,2,5,10,20")
    p.add_argument("--answer-scale", type=float, default=5.0)
    p.add_argument("--methods", default="Random,ActiveQuery,EUBO")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/codebook/generator/gc_gauss.json")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    checkpoints = [int(c) for c in args.checkpoints.split(",")]

    ck = torch.load(args.codebook, weights_only=False)
    vq = TrajVQVAE(horizon=6, num_codes=ck["args"]["codes"], latent=ck["args"]["latent"])
    vq.load_state_dict(ck["state_dict"])
    vq.eval()
    with torch.no_grad():
        code_dev = vq.decode_codes().numpy()  # (K, 6, 2) at 0.8 ... 4.8 s
    t_vq = np.r_[0.0, 0.8 * np.arange(1, code_dev.shape[1] + 1)]
    T = args.dt * np.arange(1, int(round(args.horizon / args.dt)) + 1)

    frames = {}
    for f, peds in load_tracks(args.annot_dir).items():
        frames[f] = {pid: (vx, vy) for pid, x, y, vx, vy in peds}

    scenes = []  # per scene: person, {gen: paths (K, H, 2)}, ppl (N, 4), goal, speed
    for path in sorted(glob.glob(os.path.join(args.scenes, "scenario_gcp_*.json"))):
        s = json.load(open(path))
        src = s["source"]
        v = frames.get(src["frame"], {}).get(src["ego_ped_id"])
        if v is None:
            continue
        speed = float(np.hypot(*v))
        cf = np.asarray(s["candidates"], float)
        t_cf = np.linspace(0.0, s["traj_duration"], cf.shape[1])
        vqp = np.stack([speed * t_vq, np.zeros_like(t_vq)], -1)[None] + np.concatenate(
            [np.zeros((len(code_dev), 1, 2)), code_dev], axis=1)
        ppl = np.c_[np.asarray(s["people"], float).reshape(-1, 2), np.asarray(s["people_vel"], float).reshape(-1, 2)]
        scenes.append(dict(person=src["person"], name=os.path.basename(path), paths={
            "Crowd-FM": interp(cf, t_cf, T), "VQ-VAE": interp(vqp, t_vq, T)}, ppl=ppl,
            goal=np.asarray(s["goal"], float), speed=speed))
    n_max = max(len(s["ppl"]) for s in scenes)

    # safety filter and raw features per generator, on the common grid
    for s in scenes:
        ppl = np.zeros((1, n_max, 4))
        ppl[0, :len(s["ppl"])] = s["ppl"]
        n = np.array([len(s["ppl"])])
        s["safe"], s["raw"] = {}, {}
        for g in GENS:
            P = s["paths"][g][None]
            old, dmin = code_features(P, ppl, n, s["goal"][None], np.zeros((1, P.shape[1])), dt=args.dt)
            prog = old[0, :, 2]
            s["safe"][g] = (dmin[0] >= args.clearance) & (prog >= args.min_progress * prog.max())
            s["raw"][g] = style_features(P, ppl, n, np.array([s["speed"]]), s["goal"][None],
                                         np.zeros((1, P.shape[1])), dt=args.dt)[0, :, :4]
    ok = [s for s in scenes if all(s["safe"][g].sum() >= args.min_safe for g in GENS)]
    rows = np.concatenate([s["raw"][g][s["safe"][g]] for s in ok for g in GENS])
    mu, sd = rows.mean(0), rows.std(0) + 1e-9
    for s in ok:
        s["X"] = {g: (s["raw"][g][s["safe"][g]] - mu) / sd for g in GENS}
    spread = {g: np.mean([s["X"][g].std(0) for s in ok], axis=0) for g in GENS}
    lat = {g: float(np.median([np.ptp(s["paths"][g][s["safe"][g], -1, 1]) for s in ok])) for g in GENS}
    print(f"{len(scenes)} scenes, {len(ok)} with >= {args.min_safe} safe candidates for both generators "
          f"(clearance {args.clearance} m, first {args.horizon:g} s); people {len(set(s['person'] for s in ok))}")
    for g in GENS:
        print(f"  {g:8s} safe candidates median {np.median([s['safe'][g].sum() for s in ok]):.0f}, "
              f"lateral spread at {args.horizon:g} s median {lat[g]:.2f} m, feature spread "
              + ", ".join(f"{f} {v:.2f}" for f, v in zip(FEATS, spread[g])))

    D = len(FEATS)
    th = rng.standard_normal((args.n_users, D)) if args.users == "gauss" else \
        rng.dirichlet(0.3 * np.ones(D), size=args.n_users) * rng.choice([-1, 1], size=(args.n_users, D))
    thetas = th / np.linalg.norm(th, axis=1, keepdims=True)
    people = np.array([s["person"] for s in ok])
    users = []
    for _ in range(args.n_users):
        qs = rng.choice(len(ok), args.n_query_situations, replace=False)
        rest = np.where(~np.isin(people, people[qs]))[0]
        users.append((qs, rng.choice(rest, args.n_test_situations, replace=False)))

    def evaluate(g, theta_hat, theta, test):
        out = []
        for k in test:
            s = ok[k]
            F = s["X"][g]
            u = F @ theta
            u_all = np.r_[s["X"]["Crowd-FM"] @ theta, s["X"]["VQ-VAE"] @ theta]
            iu, ju = np.triu_indices(len(F), 1)
            if theta_hat is None:
                pick, agree = int(rng.integers(len(F))), 0.5
            else:
                v = F @ theta_hat
                pick = int(np.argmax(v))
                agree = float(np.mean(np.sign(v[iu] - v[ju]) == np.sign(u[iu] - u[ju])))
            out.append(((u_all.max() - u[pick]) / max(np.ptp(u_all), 1e-9),
                        (u.max() - u[pick]) / max(np.ptp(u), 1e-9), agree))
        return np.mean(out, axis=0)

    METRICS = ["union_regret", "own_regret", "agreement"]
    res, per_user = {}, {}
    for g in GENS:
        res[g] = {"random": [np.mean([evaluate(g, None, t, te) for _ in range(5)], axis=0) for t, (_, te) in zip(thetas, users)],
                  "oracle": [evaluate(g, t, t, te) for t, (_, te) in zip(thetas, users)]}
        for m in args.methods.split(","):
            per = {c: [] for c in checkpoints}
            for ui, (t, (qs, te)) in enumerate(zip(thetas, users)):
                srng = np.random.default_rng(args.seed * 100003 + ui)  # same answer noise for both generators
                strat = METHODS[m](srng)
                theta_hat = None
                for q in range(max(checkpoints) + 1):
                    if q in per:
                        per[q].append(evaluate(g, theta_hat, t, te))
                    if q == max(checkpoints):
                        break
                    F = ok[qs[q % len(qs)]]["X"][g]
                    i, j = strat.select(F)
                    if strat.done():
                        continue
                    u = args.answer_scale * (F @ t)
                    strat.update(F, i, j, i if srng.random() < 1 / (1 + np.exp(-(u[i] - u[j]))) else j)
                    theta_hat = strat.theta_hat
            per_user[f"{g}+{m}"] = {c: np.asarray(v) for c, v in per.items()}
        res[g] = {k: np.asarray(v) for k, v in res[g].items()}

    summary = {}
    for g in GENS:
        for k in ("random", "oracle"):
            summary[f"{g} {k}"] = {mm: boot(res[g][k][:, i], rng) for i, mm in enumerate(METRICS)}
    curves = {name: {str(c): {mm: boot(v[:, i], rng) for i, mm in enumerate(METRICS)} for c, v in per.items()}
              for name, per in per_user.items()}
    paired = {}
    pairs = [("VQ-VAE+EUBO", "Crowd-FM+EUBO"), ("VQ-VAE+Random", "Crowd-FM+Random"),
             ("Crowd-FM+EUBO", "Crowd-FM+Random"), ("VQ-VAE+EUBO", "VQ-VAE+Random"), ("VQ-VAE+EUBO", "VQ-VAE+ActiveQuery")]
    for a, b in pairs:
        if a in per_user and b in per_user:
            paired[f"{a} - {b}"] = {str(c): {mm: boot(per_user[a][c][:, i] - per_user[b][c][:, i], rng)
                                             for i, mm in enumerate(METRICS)} for c in checkpoints if c > 0}

    print(f"=== {len(ok)} GC scenes, {args.n_users} {args.users} users, test scenes of other people ===")
    print(f"  {'':>22} " + " ".join(f"{m:>22}" for m in METRICS))
    for name, r in summary.items():
        print(f"  {name:>22} " + " ".join(f"{r[m][0]:6.3f} [{r[m][1]:.3f},{r[m][2]:.3f}]" for m in METRICS))
    for c in checkpoints[1:]:
        for name, cu in curves.items():
            r = cu[str(c)]
            print(f"  {name + f' ({c}q)':>22} " + " ".join(f"{r[m][0]:6.3f} [{r[m][1]:.3f},{r[m][2]:.3f}]" for m in METRICS))
    nd = thetas[:, FEATS.index("detour")] < 0
    sub = {name: {str(c): boot(v[nd, 0], rng) for c, v in per.items()} for name, per in per_user.items()}
    for g in GENS:
        sub[f"{g} oracle"] = boot(res[g]["oracle"][nd, 0], rng)
    print(f"  no-detour users ({nd.sum()}): union regret at 5 / 20 q: " + "; ".join(
        f"{name} {v['5'][0]:.3f} / {v['20'][0]:.3f}" for name, v in sub.items() if isinstance(v, dict) and "5" in v)
        + "; oracle " + ", ".join(f"{g} {sub[f'{g} oracle'][0]:.3f}" for g in GENS))
    for k, v in paired.items():
        print(f"  {k}: " + "; ".join(f"{c} q union regret {x['union_regret'][0]:+.3f} [{x['union_regret'][1]:+.3f},"
                                    f"{x['union_regret'][2]:+.3f}]" for c, x in v.items()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"args": vars(args), "n_scenes": len(scenes), "n_ok": len(ok), "features": FEATS,
               "safe_median": {g: float(np.median([s["safe"][g].sum() for s in ok])) for g in GENS},
               "lateral_spread": lat, "feature_spread": {g: v.tolist() for g, v in spread.items()},
               "summary": summary, "curves": curves, "paired": paired, "no_detour_users": sub,
               "thetas": thetas.tolist(),
               "per_user": {n: {str(c): v.tolist() for c, v in pu.items()} for n, pu in per_user.items()}},
              open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
