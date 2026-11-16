#!/usr/bin/env python
"""Template scenes for the style-preference pilot: situations Grand Central
has few of, built by hand with randomized layouts (robot frame, x = heading,
robot at the origin walking at --speed):
  queue  -- 4-6 people standing in line across the robot's way with one gap
            (cut through the gap vs go round the end)
  talk   -- 3 people standing in a loose conversation circle ahead
            (through the circle vs round it)
  pair   -- 2-3 people walking side by side towards the robot
            (between them vs round the group)
  slow   -- one slow walker ahead in the same direction
            (overtake left vs right, follow vs overtake)
Candidates, safety filter, prior and style features as
utils/build_pilot_questionnaire.py, plus one feature for these scenes:
  group -- 1 if the path passes between two members of the same group
           (crosses the segment between them at the same time step), else 0
GC questions from build_pilot_questionnaire.py get group = 0.

Output: the GC questions plus --per-type questions per template, features
FEATURES + ["group"], same JSON layout; each question also has scene,
caption and groups (member index lists, drawn as links on the page).

Usage:
    PYTHONPATH=. .venv/bin/python utils/build_pilot_templates.py \
        --gc results/codebook/pilot_v2/questions_gc.json \
        --out results/codebook/pilot_v2/questions.json
"""
import argparse
import json
import os

import numpy as np
import torch

from src.preference.traj_vqvae import TrajVQVAE
from utils.style_preference_benchmark import FEATURES, style_features
from utils.train_code_prior import load_prior, social_summary

N_MAX = 12
CAPTIONS = {
    "queue": "사람들이 줄을 서 있습니다 (예: 카페 계산대 줄). 선으로 이어진 사람들이 한 줄입니다.",
    "talk": "사람들이 둥글게 서서 대화하고 있습니다. 선으로 이어진 사람들이 한 무리입니다.",
    "pair": "일행이 나란히 걸어오고 있습니다. 선으로 이어진 사람들이 일행입니다.",
    "slow": "앞에서 한 사람이 같은 방향으로 천천히 걷고 있습니다.",
}


def make_scene(kind, rng):
    """-> people (n, 4) [x, y, vx, vy], groups [[member indices in link order]]."""
    if kind == "queue":
        xq, sp, gap = rng.uniform(2.8, 4.0), rng.uniform(0.7, 0.9), rng.uniform(1.6, 2.0)
        yg = rng.uniform(-0.4, 0.4)
        n_r, n_l = rng.integers(1, 3), rng.integers(3, 5)
        if rng.random() < 0.5:
            n_r, n_l = n_l, n_r
        ys = [yg - gap / 2 - sp * i for i in range(n_r)][::-1] + [yg + gap / 2 + sp * i for i in range(n_l)]
        ppl = np.array([[xq + rng.normal(0, 0.05), y, 0, 0] for y in ys])
        return ppl, [list(range(len(ys)))]
    if kind == "talk":
        n, r = 3, rng.uniform(1.1, 1.3)  # loose circle: a 0.8 m-clear way through exists
        c = np.array([rng.uniform(3.2, 4.4), rng.choice([-1, 1]) * rng.uniform(0.5, 1.1)])  # off-centre: going round is reachable
        a = rng.uniform(0, 2 * np.pi) + 2 * np.pi * np.arange(n) / n + rng.normal(0, 0.15, n)
        ppl = np.c_[c + r * np.c_[np.cos(a), np.sin(a)], np.zeros((n, 2))]
        return ppl, [list(range(n)) + [0]]
    if kind == "pair":
        n, sp, v = rng.integers(2, 4), rng.uniform(1.6, 2.0), rng.uniform(1.0, 1.3)
        x0, yc = rng.uniform(5.0, 6.5), rng.uniform(-0.4, 0.4)
        ys = yc + sp * (np.arange(n) - (n - 1) / 2)
        ppl = np.array([[x0 + rng.normal(0, 0.1), y, -v, 0] for y in ys])
        return ppl, [list(range(n))]
    # slow
    ppl = np.array([[rng.uniform(1.5, 2.5), rng.uniform(-0.2, 0.2), rng.uniform(0.3, 0.5), 0]])
    return ppl, []


def group_cross(paths, ppl, groups, dt, sub=8):
    """paths (K, H, 2) -> (K,) 1 if a path crosses a link between linked members."""
    K, H, _ = paths.shape
    full = np.concatenate([np.zeros((K, 1, 2)), paths], axis=1)
    s = np.linspace(0, H, H * sub + 1)  # fine time grid in steps of dt
    fine = np.stack([np.interp(s, np.arange(H + 1), full[k, :, c]) for k in range(K) for c in (0, 1)]).reshape(K, 2, -1)
    fine = fine.transpose(0, 2, 1)  # (K, S, 2)
    pos = ppl[None, :, :2] + (s * dt)[:, None, None] * ppl[None, :, 2:]  # (S, n, 2)
    out = np.zeros(K)
    for g in groups:
        for a, b in zip(g[:-1], g[1:]):
            # robot's side of the line a->b, only while it is between a and b along that line
            pa, pb = pos[:, a], pos[:, b]
            ab = pb - pa
            rel = fine - pa[None]
            cross = ab[None, :, 0] * rel[..., 1] - ab[None, :, 1] * rel[..., 0]
            along = (rel * ab[None]).sum(-1) / (ab ** 2).sum(-1)[None]
            inside = (along > 0) & (along < 1)
            flip = (np.sign(cross[:, 1:]) != np.sign(cross[:, :-1])) & inside[:, 1:] & inside[:, :-1]
            out = np.maximum(out, flip.any(1))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gc", default="results/codebook/pilot_v2/questions_gc.json")
    p.add_argument("--prior", default="results/codebook/goalfix/gc_dest_s1.pt")
    p.add_argument("--clearance", type=float, default=0.8)
    p.add_argument("--min-progress", type=float, default=0.5)
    p.add_argument("--per-type", type=int, default=4)
    p.add_argument("--fit", type=int, default=3)
    p.add_argument("--tries", type=int, default=400, help="random layouts per template")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/codebook/pilot_v2/questions.json")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    gc = json.load(open(args.gc))
    net, pa = load_prior(args.prior)
    ck = torch.load(pa["ckpt"], weights_only=False)
    vq = TrajVQVAE(horizon=6, num_codes=ck["args"]["codes"], latent=ck["args"]["latent"])
    vq.load_state_dict(ck["state_dict"])
    vq.eval()
    with torch.no_grad():
        code_dev = vq.decode_codes().numpy()
    K, H = code_dev.shape[:2]
    dt = gc["dt"]
    t = dt * np.arange(1, H + 1)
    mu, sd = np.array(gc["mean"]), np.array(gc["std"])

    scenes = []  # per template: list of (people, groups, speed, dest, paths, raw, group, dmin, safe)
    for kind in CAPTIONS:
        for _ in range(args.tries):
            ppl, groups = make_scene(kind, rng)
            n = len(ppl)
            speed = rng.uniform(1.0, 1.3)
            dest = np.array([rng.uniform(15, 25), rng.uniform(-2, 2)])
            P = np.zeros((1, N_MAX, 4))
            P[0, :n] = ppl
            past = np.array([[-2 * dt * speed, 0], [-dt * speed, 0]])
            x = np.r_[speed, past.ravel(), dest, social_summary(P, (np.arange(N_MAX) < n)[None], np.array([speed]))[0]]
            with torch.no_grad():
                logp = torch.log_softmax(net(torch.tensor((x[None] - net.mu) / net.sd, dtype=torch.float32)), 1).numpy()
            paths = np.stack([speed * t, np.zeros(H)], -1)[None] + code_dev  # (K, H, 2)
            others = ppl[None, :, :2] + t[:, None, None] * ppl[None, :, 2:]  # (H, n, 2)
            dmin = np.linalg.norm(paths[:, :, None] - others[None], axis=-1).min((1, 2))
            prog = np.linalg.norm(dest) - np.linalg.norm(dest[None] - paths[:, -1], axis=-1)
            safe = (dmin >= args.clearance) & (prog >= args.min_progress * prog.max())
            raw = style_features(paths[None], P, np.array([n]), np.array([speed]), dest[None], logp)[0]
            grp = group_cross(paths, ppl, groups, dt)
            scenes.append(dict(kind=kind, ppl=ppl, groups=groups, speed=speed, dest=dest, paths=paths,
                               raw=raw, grp=grp, dmin=dmin, safe=safe))

    rows = np.concatenate([s["grp"][s["safe"]] for s in scenes if s["kind"] != "slow"])
    g_sd = rows.std() + 1e-9
    print(f"group feature: {rows.mean():.2f} of safe template candidates cross a group, std {g_sd:.2f}")

    questions = []
    for kind in CAPTIONS:
        pool = [s for s in scenes if s["kind"] == kind]
        found = []
        for s in pool:
            c = np.where(s["safe"])[0]
            if len(c) < 2:
                continue
            X = np.c_[(s["raw"] - mu) / sd, s["grp"] / g_sd]
            i, j = np.triu_indices(len(c), 1)
            dl = X[c[i]] - X[c[j]]
            if kind == "slow":  # alternate: overtake side, then follow vs overtake
                a = FEATURES.index("side") if len(found) % 2 == 0 else FEATURES.index("yield")
                axis = FEATURES[a]
            else:
                a, axis = len(FEATURES), "group"
            others = np.delete(np.abs(dl), a, axis=1).max(1)
            good = np.where((np.abs(dl[:, a]) >= 1.0))[0]
            if len(good) == 0:
                continue
            k = good[np.argmin(others[good])]
            found.append((s, int(c[i[k]]), int(c[j[k]]), axis, X))
            if len(found) == args.per_type:
                break
        print(f"  {kind}: {len(found)} questions (of {len(pool)} layouts)")
        for n, (s, ci, cj, axis, X) in enumerate(found):
            if rng.random() < 0.5:
                ci, cj = cj, ci
            questions.append({
                "id": f"{kind}{n + 1}", "axis": axis, "scene": kind, "caption": CAPTIONS[kind],
                "split": "fit" if n < args.fit else "test",
                "speed": float(s["speed"]), "dest": s["dest"].round(3).tolist(),
                "people": s["ppl"].round(3).tolist(), "groups": s["groups"],
                "codes": [ci, cj],
                "paths": [s["paths"][ci].round(3).tolist(), s["paths"][cj].round(3).tolist()],
                "raw": [np.r_[s["raw"][ci], s["grp"][ci]].round(4).tolist(), np.r_[s["raw"][cj], s["grp"][cj]].round(4).tolist()],
                "x": [X[ci].round(4).tolist(), X[cj].round(4).tolist()],
                "min_dist": [round(float(s["dmin"][ci]), 3), round(float(s["dmin"][cj]), 3)],
            })

    for q in gc["questions"]:
        q.update(scene="gc", caption="", groups=[])
        q["raw"] = [r + [0.0] for r in q["raw"]]
        q["x"] = [r + [0.0] for r in q["x"]]
    out = dict(gc, features=FEATURES + ["group"], mean=gc["mean"] + [0.0], std=gc["std"] + [float(g_sd)],
               args={"gc": gc["args"], "templates": vars(args)}, questions=gc["questions"] + questions)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    qs = out["questions"]
    print(f"{len(qs)} questions ({sum(q['split'] == 'fit' for q in qs)} fit, {sum(q['split'] == 'test' for q in qs)} test, "
          f"{sum(q['split'] == 'repeat' for q in qs)} repeats) -> {args.out}")


if __name__ == "__main__":
    main()
