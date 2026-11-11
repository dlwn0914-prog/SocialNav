#!/usr/bin/env python
"""Fixed A/B question set for the style-preference pilot (every participant
answers the same questions, so people can be compared with each other).

Situations and candidates as utils/style_preference_benchmark.py: held-out
GC people in a predicted conflict, safe codes only (both options are safe,
so an answer is about style, not safety).

Each question targets one style axis: a pair of safe candidates in one
situation that differs a lot on that axis (|delta| >= --big, normalized) and
little on the others (|delta| <= --small). side / front pairs come only from
situations where the safe codes go both ways. Per axis --per-axis questions;
the first --fit of them are for fitting a participant's theta, the rest are
held out for the agreement check; --repeats fit questions are asked again
(sides swapped) to measure answer consistency. No situation is used twice.

Output JSON: features, normalization, and per question the scene in the
robot's frame (x = heading): people x, y, vx, vy; destination; the two
paths (6 points every 0.8 s); their codes, raw and normalized features.

Usage:
    PYTHONPATH=. .venv/bin/python utils/build_pilot_questionnaire.py \
        --out results/codebook/pilot/questions.json
"""
import argparse
import json
import os

import numpy as np
import torch

from src.preference.traj_vqvae import TrajVQVAE
from utils.code_preference_benchmark import code_features
from utils.style_preference_benchmark import FEATURES, style_features
from utils.train_code_prior import load_prior, situation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default="results/codebook/goalfix/gc_dest_s1.pt")
    p.add_argument("--clearance", type=float, default=0.5)
    p.add_argument("--min-progress", type=float, default=0.5)
    p.add_argument("--min-safe", type=int, default=4)
    p.add_argument("--per-axis", type=int, default=8)
    p.add_argument("--fit", type=int, default=6)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--big", type=float, default=1.0)
    p.add_argument("--small", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/codebook/pilot/questions.json")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    net, pa = load_prior(args.prior)
    ck = torch.load(pa["ckpt"], weights_only=False)
    d = dict(np.load(pa["data"]))
    vq = TrajVQVAE(horizon=d["future"].shape[1], num_codes=ck["args"]["codes"], latent=ck["args"]["latent"])
    vq.load_state_dict(ck["state_dict"])
    vq.eval()
    with torch.no_grad():
        code_dev = vq.decode_codes().numpy()
    H = code_dev.shape[1]

    base = np.where((d["split"] == 2) & (d["dest_t"] >= pa["min_dest_t"]))[0]
    _, x = situation(d, "heading", True, pa["people_input"], pa["goal"])
    base = base[x[base][:, -4] < 8.0]  # predicted conflict
    with torch.no_grad():
        logp = torch.log_softmax(net(torch.tensor((x[base] - net.mu) / net.sd)), dim=1).numpy()
    t = 0.8 * np.arange(1, H + 1)
    paths = np.stack([d["speed"][base][:, None] * t, np.zeros((len(base), H))], axis=-1)[:, None] + code_dev[None]
    ppl, n_ppl, dest, speed = d["people"][base], d["n_people"][base], d["dest"][base], d["speed"][base]
    old, dmin = code_features(paths, ppl, n_ppl, dest, logp)
    prog = old[..., 2]
    safe = (dmin >= args.clearance) & (prog >= args.min_progress * prog.max(1, keepdims=True))
    raw = style_features(paths, ppl, n_ppl, speed, dest, logp)
    ok = safe.sum(1) >= args.min_safe
    rows = raw[ok][safe[ok]]
    mu, sd = rows.mean(0), rows.std(0) + 1e-9
    X = (raw - mu) / sd

    used, questions = set(), []
    for a, axis in enumerate(FEATURES):
        found = []
        for s in rng.permutation(np.where(ok)[0]):
            if s in used:
                continue
            c = np.where(safe[s])[0]
            if axis in ("side", "front") and not (raw[s, c, a].max() > 0.3 and raw[s, c, a].min() < -0.3):
                continue
            i, j = np.triu_indices(len(c), 1)
            dl = X[s, c[i]] - X[s, c[j]]
            others = np.delete(np.abs(dl), a, axis=1).max(1)
            good = np.where((np.abs(dl[:, a]) >= args.big) & (others <= args.small))[0]
            if len(good) == 0:
                continue
            k = good[np.argmax(np.abs(dl[good, a]) - others[good])]
            found.append((s, int(c[i[k]]), int(c[j[k]])))
            used.add(s)
            if len(found) == args.per_axis:
                break
        print(f"  {axis}: {len(found)} questions")
        for n, (s, ci, cj) in enumerate(found):
            if rng.random() < 0.5:
                ci, cj = cj, ci  # which one is shown as A
            g = base[s]
            m = int(n_ppl[s])
            questions.append({
                "id": f"{axis}{n + 1}", "axis": axis, "split": "fit" if n < args.fit else "test",
                "situation": int(g), "pid": int(d["pid"][g]), "frame": int(d["frame"][g]),
                "speed": float(speed[s]), "dest": dest[s].round(3).tolist(),
                "people": ppl[s, :m].round(3).tolist(),
                "codes": [ci, cj],
                "paths": [paths[s, ci].round(3).tolist(), paths[s, cj].round(3).tolist()],
                "raw": [raw[s, ci].round(4).tolist(), raw[s, cj].round(4).tolist()],
                "x": [X[s, ci].round(4).tolist(), X[s, cj].round(4).tolist()],
                "min_dist": [round(float(dmin[s, ci]), 3), round(float(dmin[s, cj]), 3)],
            })
    fit = [q for q in questions if q["split"] == "fit"]
    for q in [fit[k] for k in rng.choice(len(fit), args.repeats, replace=False)]:
        r = dict(q, id=q["id"] + "r", split="repeat", repeat_of=q["id"])
        for key in ("codes", "paths", "raw", "x", "min_dist"):
            r[key] = q[key][::-1]  # sides swapped
        questions.append(r)
    out = {"features": FEATURES, "mean": mu.tolist(), "std": sd.tolist(), "args": vars(args),
           "dt": 0.8, "questions": questions}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"{len(questions)} questions ({sum(q['split'] == 'fit' for q in questions)} fit, "
          f"{sum(q['split'] == 'test' for q in questions)} test, {args.repeats} repeats) -> {args.out}")


if __name__ == "__main__":
    main()
