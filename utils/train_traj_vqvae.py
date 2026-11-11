#!/usr/bin/env python
"""Train the walking-behaviour codebook (src/preference/traj_vqvae.py) on the
Grand Central paths of utils/build_gc_traj_dataset.py, then report how well
K codes describe held-out people's paths (val split, by person).

Reported on val:
  ADE of the code's path (constant velocity + decoded deviation) vs the real
  path, next to constant velocity alone and k-means on the deviation with
  the same K (the codebook should be at least as good as plain clustering);
  code usage (perplexity; K means all codes equally used).

Usage:
    PYTHONPATH=. .venv/bin/python utils/train_traj_vqvae.py --codes 16 \
        --out results/codebook/vq16.pt
"""
import argparse
import json
import os

import numpy as np
import torch

from src.preference.traj_vqvae import TrajVQVAE, deviation


def load(path):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def goal_frame(future, goal):
    """Rotate paths so +x points at the goal: deviation then measures how the
    walker departs from heading straight for where they are going."""
    a = np.arctan2(goal[:, 1], goal[:, 0])
    c, s = np.cos(a)[:, None], np.sin(a)[:, None]
    x, y = future[..., 0], future[..., 1]
    return np.stack([c * x + s * y, -s * x + c * y], axis=-1).astype(np.float32)


def ade(a, b):
    return float(np.linalg.norm(a - b, axis=-1).mean())


def kmeans(x, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        a = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.stack([x[a == j].mean(0) if np.any(a == j) else c[j] for j in range(k)])
    return c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/nuri2/datasets/grandcentral/traj_codebook.npz")
    p.add_argument("--codes", type=int, default=16)
    p.add_argument("--latent", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reference", choices=["heading", "goal"], default="heading",
                   help="deviation from straight on along the current heading, or along the goal direction")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/codebook/vq16.pt")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    d = load(args.data)
    fut, sp = torch.tensor(d["future"]), torch.tensor(d["speed"])
    if args.reference == "goal":
        fut = torch.tensor(goal_frame(d["future"], d["goal"]))
    dev = deviation(fut, sp)
    tr, va = d["split"] == 0, d["split"] == 1

    model = TrajVQVAE(horizon=fut.shape[1], num_codes=args.codes, latent=args.latent)
    model.scale.fill_(float(dev[tr].std()))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_tr = dev[tr]
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(x_tr))
        tot = 0.0
        for i in range(0, len(x_tr), args.batch):
            _, _, loss, rec = model(x_tr[perm[i:i + args.batch]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(rec) * len(perm[i:i + args.batch])
        if ep % 5 == 4 or ep == args.epochs - 1:
            print(f"epoch {ep + 1}: train rec (normalized MSE) {tot / len(x_tr):.4f}")

    model.eval()
    with torch.no_grad():
        rec, idx, _, _ = model(dev[va])
    real = fut[va].numpy()
    cv = real - dev[va].numpy()
    code_path = cv + rec.numpy()
    counts = np.bincount(idx.numpy(), minlength=args.codes) / len(idx)
    perplexity = float(np.exp(-(counts[counts > 0] * np.log(counts[counts > 0])).sum()))
    km = kmeans(dev[tr].flatten(1).numpy()[::5], args.codes, seed=args.seed)
    dv = dev[va].flatten(1).numpy()
    km_path = cv + km[((dv[:, None] - km[None]) ** 2).sum(-1).argmin(1)].reshape(-1, *real.shape[1:])
    res = {"codes": args.codes, "n_val": int(va.sum()),
           "ade_cv": ade(cv, real), "ade_kmeans": ade(km_path, real), "ade_vq": ade(code_path, real),
           "fde_cv": float(np.linalg.norm(cv[:, -1] - real[:, -1], axis=-1).mean()),
           "fde_vq": float(np.linalg.norm(code_path[:, -1] - real[:, -1], axis=-1).mean()),
           "perplexity": perplexity, "usage": counts.tolist()}
    print(f"val ({res['n_val']} paths, {res['codes']} codes): ADE constant velocity {res['ade_cv']:.3f} m, "
          f"k-means {res['ade_kmeans']:.3f} m, VQ-VAE {res['ade_vq']:.3f} m; "
          f"FDE {res['fde_cv']:.3f} -> {res['fde_vq']:.3f} m; perplexity {perplexity:.1f}/{args.codes}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "val": res}, args.out)
    json.dump(res, open(args.out.replace(".pt", ".json"), "w"), indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
