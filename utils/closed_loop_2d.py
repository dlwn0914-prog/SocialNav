#!/usr/bin/env python
"""Closed-loop 2D replay of the style-preference pipeline on Grand Central.

A held-out (test) pedestrian in a predicted-conflict situation is replaced by
the robot: it starts where they were, with their heading, speed and the two
past positions, and has to reach their track's last position. Everyone else
is replayed from the annotations (0.8 s steps) and does not react to the
robot -- the usual replay setting, harder than real people who would also
give way. People walking with the pedestrian (within --companion m at the
start, velocity within 0.5 m/s, still within 2 m 0.8 and 1.6 s later) are
their party, not traffic: they are left out for the robot and for the
"human" reference alike (about a fifth of GC test people walk with someone).

Every 0.8 s the robot re-plans exactly as in utils/style_preference_benchmark.py:
the K codebook paths from its current state at its cruise speed (the
pedestrian's speed at the start: codes are deviations from constant velocity,
so planning from the current speed would let a slowing code ratchet the
speed down for good),
the safety filter (min distance >= --clearance with people at constant
velocity, progress >= --min-progress x best), the style features, and a
choice among the safe codes; it then executes the chosen path's first 0.8 s.
With no safe code it takes the one with the largest min distance (counted as
a fallback step). After the step its heading is the chosen path's tangent
there (codes bend late, so the step's own direction would barely turn the
robot). Within --terminal m of the goal it walks straight at it: the prior
only knows destinations >= 12 s away.

Choice rules (one synthetic user theta* per episode, as in the benchmark):
  random     a random safe code
  prior      the most human-like safe code (argmax log p(code | situation))
  pref-5q    argmax theta_hat . phi, theta_hat from 5 EUBO queries
  pref-20q   the same with 20 queries
  oracle     argmax theta* . phi
theta_hat is learned before the drive from A/B questions in other test
situations (the benchmark's conflict pool, without the episode's person);
features are normalized with the benchmark's statistics over that pool.

Per episode: reached the goal (within --goal-tol), time, distance left to the
goal at the end, collision, robot-caused collision (at the closest moment
the robot's own motion closes the gap faster than the person's), (anyone
closer than --collision at any 0.1 s substep, people interpolated between
annotations), near pass (< --clearance), minimum distance, fallback steps,
and the step regret under theta* among the safe codes (does the drive keep
choosing what the user prefers). The pedestrian's own recorded path is the
reference row ("human"). Means with a 95% bootstrap over episodes.

Usage:
    PYTHONPATH=. .venv/bin/python utils/closed_loop_2d.py --out results/codebook/closed_loop/gc_gauss.json
"""
import argparse
import json
import os

import numpy as np
import torch

from src.preference.traj_vqvae import TrajVQVAE
from utils.build_gc_scenes import FPS, load_tracks
from utils.build_gc_traj_dataset import STEP
from utils.code_preference_benchmark import code_features
from utils.person_transfer_benchmark import METHODS, boot
from utils.style_preference_benchmark import FEATURES, sample_theta, style_features
from utils.train_code_prior import load_prior, situation, social_summary

DT = STEP / FPS  # 0.8 s
RULES = ["random", "prior", "pref-5q", "pref-20q", "oracle"]


def party(tr_frames, pid, f0, radius):
    """Ids walking with pid at f0 (close, same velocity, still close next steps)."""
    e = tr_frames[f0][pid]
    out = set()
    for q, (x, y, vx, vy) in tr_frames[f0].items():
        if q == pid or np.hypot(x - e[0], y - e[1]) > radius or np.hypot(vx - e[2], vy - e[3]) > 0.5:
            continue
        later = [tr_frames.get(f0 + k * STEP, {}) for k in (1, 2)]
        if all(pid in fr and q in fr and np.hypot(fr[q][0] - fr[pid][0], fr[q][1] - fr[pid][1]) < 2.0 for fr in later):
            out.add(q)
    return out


def rot(head):
    c, s = np.cos(head), np.sin(head)
    return np.array([[c, s], [-s, c]])  # world -> robot


def view(pos, head, past, goal, peds, radius, n_max):
    """Robot-frame past (2, 2), destination (2,), people (n_max, 4) and their count."""
    R = rot(head)
    others = []
    for x, y, vx, vy in peds.values():
        q = R @ (np.array([x, y]) - pos)
        if np.hypot(*q) <= radius:
            others.append([*q, *(R @ np.array([vx, vy]))])
    others.sort(key=lambda r: np.hypot(r[0], r[1]))
    ppl = np.zeros((n_max, 4))
    others = others[:n_max]
    if others:
        ppl[:len(others)] = others
    return np.array([R @ (p - pos) for p in past]), R @ (goal - pos), ppl, len(others)


def min_dist_segment(a0, a1, peds0, peds1, sub=8):
    """Smallest robot-person distance while the robot goes a0 -> a1 and each
    person moves from their position now to their next annotation (or at
    constant velocity when it is missing), and whether at that moment the
    robot's own motion closes the gap faster than the person's (robot-caused)."""
    best, robot_caused = np.inf, False
    vr = (a1 - a0) / DT
    for pid, (x, y, vx, vy) in peds0.items():
        p0 = np.array([x, y])
        p1 = np.array(peds1[pid][:2]) if pid in peds1 else p0 + DT * np.array([vx, vy])
        vp = (p1 - p0) / DT
        for a in np.linspace(0, 1, sub + 1):
            gap = (p0 + a * (p1 - p0)) - (a0 + a * (a1 - a0))
            dist = np.linalg.norm(gap)
            if dist < best:
                u = gap / max(dist, 1e-9)
                best, robot_caused = dist, bool(vr @ u > -(vp @ u))
    return best, robot_caused


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default="results/codebook/goalfix/gc_dest_s1.pt")
    p.add_argument("--annot-dir", default="/home/nuri2/datasets/grandcentral/raw/opentraj/datasets/GC/Annotation")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--users", choices=["gauss", "sparse"], default="gauss")
    p.add_argument("--clearance", type=float, default=0.5)
    p.add_argument("--min-progress", type=float, default=0.5)
    p.add_argument("--min-safe", type=int, default=4, help="for the query pool, as in the benchmark")
    p.add_argument("--collision", type=float, default=0.4, help="two bodies of ~0.2 m radius")
    p.add_argument("--goal-tol", type=float, default=1.0)
    p.add_argument("--terminal", type=float, default=3.0)
    p.add_argument("--companion", type=float, default=1.5)
    p.add_argument("--max-time", type=float, default=60.0)
    p.add_argument("--answer-scale", type=float, default=5.0)
    p.add_argument("--n-query-situations", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/codebook/closed_loop/gc_gauss.json")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    net, pa = load_prior(args.prior)
    assert pa["people_input"] == "summary" and pa["goal"] == "dest"
    ck = torch.load(pa["ckpt"], weights_only=False)
    d = dict(np.load(pa["data"]))
    vq = TrajVQVAE(horizon=d["future"].shape[1], num_codes=ck["args"]["codes"], latent=ck["args"]["latent"])
    vq.load_state_dict(ck["state_dict"])
    vq.eval()
    with torch.no_grad():
        code_dev = vq.decode_codes().numpy()
    K, H = code_dev.shape[:2]
    t = DT * np.arange(1, H + 1)
    n_max = d["people"].shape[1]

    def prior_logp(x):
        with torch.no_grad():
            return torch.log_softmax(net(torch.tensor((x - net.mu) / net.sd, dtype=torch.float32)), dim=1).numpy()

    def candidates(speed, ppl, n, dest, logp):
        """(B,) situations -> paths (B, K, H, 2), safe (B, K), raw style features (B, K, 5), dmin (B, K)."""
        paths = np.stack([speed[:, None] * t, np.zeros((len(speed), H))], axis=-1)[:, None] + code_dev[None]
        old, dmin = code_features(paths, ppl, n, dest, logp)
        prog = old[..., 2]
        safe = (dmin >= args.clearance) & (prog >= args.min_progress * prog.max(1, keepdims=True))
        return paths, safe, style_features(paths, ppl, n, speed, dest, logp), dmin

    # the benchmark's conflict pool: normalization and the users' query situations
    base = np.where((d["split"] == 2) & (d["dest_t"] >= pa["min_dest_t"]))[0]
    _, x = situation(d, "heading", True, pa["people_input"], pa["goal"])
    base = base[x[base][:, -4] < 8.0]
    _, safe_b, raw_b, _ = candidates(d["speed"][base], d["people"][base], d["n_people"][base], d["dest"][base], prior_logp(x[base]))
    ok = safe_b.sum(1) >= args.min_safe
    pool, safe_b, raw_b = base[ok], safe_b[ok], raw_b[ok]
    rows = raw_b[safe_b]
    mu, sd = rows.mean(0), rows.std(0) + 1e-9
    X_pool = (raw_b - mu) / sd
    print(f"query pool: {len(pool)} conflict situations with >= {args.min_safe} safe codes")

    by_frame = load_tracks(args.annot_dir)
    frame = {f: {pid: (x_, y_, vx, vy) for pid, x_, y_, vx, vy in peds} for f, peds in by_frame.items()}
    track = {}
    for f, peds in frame.items():
        for pid, v in peds.items():
            track.setdefault(pid, {})[f] = v
    last = {pid: max(tr) for pid, tr in track.items()}

    # episodes: one start per held-out person, from the pool
    pids = d["pid"][pool]
    starts = [pool[pids == q][rng.integers((pids == q).sum())] for q in rng.permutation(np.unique(pids))[:args.episodes]]
    thetas = sample_theta(rng, len(starts), args.users)
    print(f"{len(starts)} episodes ({args.users} users); human time to goal median {np.median(d['dest_t'][starts]):.1f} s")

    def learn(theta, n_q, exclude, urng):
        cand = np.where(d["pid"][pool] != exclude)[0]
        qs = cand[urng.choice(len(cand), args.n_query_situations, replace=False)]
        strat = METHODS["EUBO"](urng)
        th_hat = None
        for q in range(n_q):
            F = X_pool[qs[q % len(qs)]][safe_b[qs[q % len(qs)]]]
            i, j = strat.select(F)
            if strat.done():
                continue
            u = args.answer_scale * (F @ theta)
            strat.update(F, i, j, i if urng.random() < 1 / (1 + np.exp(-(u[i] - u[j]))) else j)
            th_hat = strat.theta_hat
        return th_hat

    def drive(g, rule, w, erng):
        pid, f0 = int(d["pid"][g]), int(d["frame"][g])
        tr = track[pid]
        pos = np.array(tr[f0][:2], dtype=float)
        head = np.arctan2(tr[f0][3], tr[f0][2])
        sp = float(np.hypot(tr[f0][2], tr[f0][3]))  # cruise speed
        past = [np.array(tr[f0 - 2 * STEP][:2], float), np.array(tr[f0 - STEP][:2], float)]
        goal = np.array(tr[last[pid]][:2], float)
        skip = party(frame, pid, f0, args.companion) | {pid}
        t_max = min(args.max_time, 1.5 * float(d["dest_t"][g]) + 5.0)
        out = dict(reached=0.0, time=np.nan, goal_dist=np.nan, collision=0.0, coll_robot=0.0, near=0.0, min_dist=np.inf, fallback=0, steps=0, regret=[])
        for k in range(int(t_max / DT)):
            f = f0 + k * STEP
            peds = {q: v for q, v in frame.get(f, {}).items() if q not in skip}
            pst, dest, ppl, n = view(pos, head, past, goal, peds, 6.0, n_max)
            xs = np.r_[sp, pst.ravel(), dest, social_summary(ppl[None], (np.arange(n_max) < n)[None], np.array([sp]))[0]]
            logp = prior_logp(xs[None].astype(np.float32))
            paths, safe, raw, dmin = candidates(np.array([sp]), ppl[None], np.array([n]), dest[None], logp)
            paths, safe, raw, dmin = paths[0], safe[0], raw[0], dmin[0]
            Xn = (raw - mu) / sd
            c = np.where(safe)[0]
            if np.linalg.norm(dest) <= args.terminal:  # straight at the goal
                step = dest * min(1.0, sp * DT / np.linalg.norm(dest))
                tangent = dest
            else:
                if len(c) == 0:
                    pick = int(np.argmax(dmin))
                    out["fallback"] += 1
                else:
                    score = {"random": erng.random(len(c)), "prior": raw[c, -1]}.get(rule)
                    if score is None:
                        score = Xn[c] @ w
                    pick = int(c[np.argmax(score)])
                    if len(c) >= 2:
                        u = Xn[c] @ w_true
                        out["regret"].append((u.max() - Xn[pick] @ w_true) / max(np.ptp(u), 1e-9))
                step = paths[pick, 0]
                tangent = paths[pick, 1] - paths[pick, 0]
            new = pos + rot(head).T @ step
            md, rc = min_dist_segment(pos, new, peds, {q: v for q, v in frame.get(f + STEP, {}).items() if q not in skip})
            out["min_dist"] = min(out["min_dist"], md)
            out["collision"] = max(out["collision"], float(md < args.collision))
            out["coll_robot"] = max(out["coll_robot"], float(md < args.collision and rc))
            out["near"] = max(out["near"], float(md < args.clearance))
            out["steps"] += 1
            # reached the goal somewhere on this segment?
            seg = new - pos
            a = np.clip(((goal - pos) @ seg) / max(seg @ seg, 1e-9), 0, 1)
            past = [past[1], pos]
            pos = new
            if np.linalg.norm(tangent) > 0.05:
                tw = rot(head).T @ tangent
                head = np.arctan2(tw[1], tw[0])
            if np.linalg.norm(past[1] + a * seg - goal) <= args.goal_tol:
                out["reached"], out["time"] = 1.0, (k + a) * DT
                break
        out["goal_dist"] = float(np.linalg.norm(pos - goal)) if not out["reached"] else 0.0
        out["regret"] = float(np.mean(out["regret"])) if out["regret"] else np.nan
        out["fallback"] = out["fallback"] / max(out["steps"], 1)
        return out

    def human(g):
        """The pedestrian's own recorded path, measured the same way."""
        pid, f0 = int(d["pid"][g]), int(d["frame"][g])
        tr = track[pid]
        skip = party(frame, pid, f0, args.companion) | {pid}
        res = dict(reached=1.0, time=float(d["dest_t"][g]), goal_dist=0.0, collision=0.0, coll_robot=0.0, near=0.0, min_dist=np.inf, fallback=0.0, regret=np.nan)
        f = f0
        while f + STEP in tr and f + STEP <= last[pid]:
            peds = {q: v for q, v in frame.get(f, {}).items() if q not in skip}
            md, rc = min_dist_segment(np.array(tr[f][:2]), np.array(tr[f + STEP][:2]), peds,
                                      {q: v for q, v in frame.get(f + STEP, {}).items() if q not in skip})
            res["min_dist"] = min(res["min_dist"], md)
            res["collision"] = max(res["collision"], float(md < args.collision))
            res["coll_robot"] = max(res["coll_robot"], float(md < args.collision and rc))
            res["near"] = max(res["near"], float(md < args.clearance))
            f += STEP
        return res

    METRICS = ["reached", "time", "goal_dist", "collision", "coll_robot", "near", "min_dist", "fallback", "regret"]
    per = {r: [] for r in ["human"] + RULES}
    for e, (g, th) in enumerate(zip(starts, thetas)):
        urng = np.random.default_rng(args.seed * 100003 + e)
        w_hat = {"pref-5q": learn(th, 5, d["pid"][g], np.random.default_rng(args.seed * 100003 + e)),
                 "pref-20q": learn(th, 20, d["pid"][g], np.random.default_rng(args.seed * 100003 + e)),
                 "oracle": th}
        w_true = th
        per["human"].append([human(g)[m] for m in METRICS])
        for r in RULES:
            res = drive(g, r, w_hat.get(r), np.random.default_rng(urng.integers(1 << 31)))
            per[r].append([res[m] for m in METRICS])
        if (e + 1) % 20 == 0:
            print(f"  {e + 1}/{len(starts)} episodes", flush=True)

    rows = {}
    for r, v in per.items():
        v = np.asarray(v, float)
        rows[r] = {m: boot(v[:, i][np.isfinite(v[:, i])], rng) if np.isfinite(v[:, i]).any() else None
                   for i, m in enumerate(METRICS)}
    paired = {}
    for a, b in [("pref-5q", "prior"), ("pref-5q", "random"), ("pref-20q", "prior")]:
        va, vb = np.asarray(per[a], float), np.asarray(per[b], float)
        paired[f"{a} - {b}"] = {m: boot((va[:, i] - vb[:, i])[np.isfinite(va[:, i] - vb[:, i])], rng)
                                for i, m in enumerate(METRICS) if m not in ("time",)}
    print(f"=== closed loop, {len(starts)} GC episodes, {args.users} users, clearance {args.clearance} m, "
          f"collision < {args.collision} m, non-reactive replay ===")
    print(f"  {'':>9} " + " ".join(f"{m:>20}" for m in METRICS))
    for r, row in rows.items():
        print(f"  {r:>9} " + " ".join(f"{row[m][0]:7.3f} [{row[m][1]:.2f},{row[m][2]:.2f}]" if row[m] else f"{'-':>20}"
                                      for m in METRICS))
    for k, v in paired.items():
        print(f"  {k}: " + "; ".join(f"{m} {x[0]:+.3f} [{x[1]:+.3f},{x[2]:+.3f}]" for m, x in v.items()))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump({"args": vars(args), "n_episodes": len(starts), "metrics": METRICS, "rows": rows, "paired": paired,
               "per_episode": {r: np.asarray(v, float).tolist() for r, v in per.items()}}, open(args.out, "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
