#!/usr/bin/env python
"""Which walking behaviour (codebook code) do people use in which situation?
Fits p(code | situation) on the codes a trained codebook assigns to each
Grand Central path, with and without the surrounding people, and measures
how much the people help -- the size of the situation-dependent (social)
part of the behaviour, as opposed to what the walker's own motion and goal
already decide.

Situation inputs (all in the same frame as the codebook's paths):
  self    speed, the two past positions, goal (--goal: the position 6.4 s
          ahead, the track's last position, or none)
  +people the --max-people nearest others: x, y, vx, vy and a present flag,
          sorted by distance (a set encoder would be order-free; with the
          sort a plain MLP is enough for this check)

Reported on val people (by person, as the codebook split):
  (early stopping on val NLL for both input sets, so both are slightly
  optimistic by the same rule)
  NLL and top-1 / top-3 accuracy of the code, and ADE of the most likely
  code's path to the real path (min over the top 3 as well), next to the
  code's own reconstruction (the ceiling) and the most frequent code.

Usage:
    PYTHONPATH=. .venv/bin/python utils/train_code_prior.py --ckpt results/codebook/vq16_goal.pt
"""
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.preference.traj_vqvae import TrajVQVAE, deviation
from utils.train_traj_vqvae import goal_frame


def social_summary(ppl, present, speed):
    """8 numbers per sample (frame: x = ahead): nearest person ahead (x, y,
    relative vx, vy), smallest time to closest approach < 1 m under constant
    velocity (capped at 8 s), number of people ahead within 4 m on the left
    and on the right, and that count's left-right difference."""
    rel_v = ppl[..., 2:] - np.stack([speed, np.zeros_like(speed)], axis=-1)[:, None]
    pos = ppl[..., :2]
    ahead = present & (pos[..., 0] > 0)
    d = np.where(ahead, np.linalg.norm(pos, axis=-1), np.inf)
    j = d.argmin(1)
    r = np.arange(len(ppl))
    has = np.isfinite(d[r, j])
    near = np.where(has[:, None], np.c_[pos[r, j], rel_v[r, j]], np.array([8.0, 0.0, 0.0, 0.0]))
    vv = (rel_v ** 2).sum(-1) + 1e-6
    t_ca = np.clip(-(pos * rel_v).sum(-1) / vv, 0, 8)
    d_ca = np.linalg.norm(pos + t_ca[..., None] * rel_v, axis=-1)
    ttc = np.where(present & (d_ca < 1.0), t_ca, 8.0).min(1)
    close = ahead & (d < 4.0)
    left, right = (close & (pos[..., 1] > 0)).sum(1), (close & (pos[..., 1] <= 0)).sum(1)
    return np.c_[near, ttc, left, right, left - right]


def situation(d, ref, with_people, people_input="list", goal_input="near"):
    fut, past, goal, ppl, n = d["future"], d["past"], d["goal"], d["people"], d["n_people"]
    if ref == "goal":
        fut, past = goal_frame(fut, goal), goal_frame(past, goal)
        ppl = np.concatenate([goal_frame(ppl[..., :2], goal), goal_frame(ppl[..., 2:], goal)], axis=-1)
        head = -np.arctan2(goal[:, 1], goal[:, 0])  # current heading in the goal frame
        selfx = np.c_[d["speed"], past.reshape(len(past), -1), np.linalg.norm(goal, axis=1), np.cos(head), np.sin(head)]
    elif goal_input == "none":
        selfx = np.c_[d["speed"], past.reshape(len(past), -1)]
    else:
        selfx = np.c_[d["speed"], past.reshape(len(past), -1), goal if goal_input == "near" else d["dest"]]
    if not with_people:
        return fut, selfx.astype(np.float32)
    present = (np.arange(ppl.shape[1])[None] < n[:, None]).astype(np.float32)
    if people_input == "summary":
        return fut, np.c_[selfx, social_summary(ppl, present > 0, d["speed"])].astype(np.float32)
    px = np.concatenate([ppl * present[..., None], present[..., None]], axis=-1).reshape(len(ppl), -1)
    return fut, np.c_[selfx, px].astype(np.float32)


def load_prior(path, name="self+people"):
    """The p(code | situation) net saved next to the .json: (net, args).
    Inputs are situation(...) rows with the saved args, normalized by net's own mu/sd."""
    ck = torch.load(path, weights_only=False)
    n = ck["nets"][name]
    net = nn.Sequential(nn.Linear(n["in_dim"], n["hidden"]), nn.ReLU(), nn.Linear(n["hidden"], n["hidden"]),
                        nn.ReLU(), nn.Linear(n["hidden"], ck["res"]["codes"]))
    net.load_state_dict(n["state_dict"])
    net.eval()
    net.mu, net.sd = n["mu"], n["sd"]
    return net, ck["args"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/codebook/vq16_goal.pt")
    p.add_argument("--data", default="/home/nuri2/datasets/grandcentral/traj_codebook.npz")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--goal", choices=["near", "dest", "none"], default="near",
                   help="goal input (heading-frame codebooks only): the walker's own position 6.4 s later "
                        "(carries any detour they make), the track's last position, or none")
    p.add_argument("--no-goal", action="store_true", help="same as --goal none")
    p.add_argument("--min-dest-t", type=float, default=0.0,
                   help="keep only samples whose track end is at least this many seconds ahead, so the "
                        "destination is far beyond the path (use the same value to compare --goal settings)")
    p.add_argument("--people-input", choices=["list", "summary"], default="list",
                   help="people as the sorted raw list, or 8 hand-made interaction numbers (social_summary)")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.no_goal:
        args.goal = "none"

    torch.manual_seed(args.seed)
    ck = torch.load(args.ckpt, weights_only=False)
    a = ck["args"]
    ref = a.get("reference", "heading")
    d = dict(np.load(args.data))
    vq = TrajVQVAE(horizon=d["future"].shape[1], num_codes=a["codes"], latent=a["latent"])
    vq.load_state_dict(ck["state_dict"])
    vq.eval()
    far = d["dest_t"] >= args.min_dest_t
    tr, va = (d["split"] == 0) & far, (d["split"] == 1) & far
    K = a["codes"]
    res = {"ckpt": args.ckpt, "codes": K, "goal": args.goal, "min_dest_t": args.min_dest_t,
           "n_train": int(tr.sum()), "n_val": int(va.sum())}
    with torch.no_grad():
        code_dev = vq.decode_codes()
    nets = {}

    for name, with_people in [("self", False), ("self+people", True)]:
        fut, x = situation(d, ref, with_people, args.people_input, args.goal)
        fut_t, sp = torch.tensor(fut), torch.tensor(d["speed"])
        with torch.no_grad():
            dev = deviation(fut_t, sp)
            y = vq.codebook.assign(vq.encode(dev))
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        X = torch.tensor((x - mu) / sd)
        net = nn.Sequential(nn.Linear(X.shape[1], args.hidden), nn.ReLU(), nn.Linear(args.hidden, args.hidden),
                            nn.ReLU(), nn.Linear(args.hidden, K))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        Xtr, ytr = X[tr], y[tr]
        best, best_state, best_ep = np.inf, None, 0
        for ep in range(args.epochs):
            net.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(Xtr), args.batch):
                b = perm[i:i + args.batch]
                loss = F.cross_entropy(net(Xtr[b]), ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():  # early stopping on val NLL, same rule for both input sets
                v = float(F.cross_entropy(net(X[va]), y[va]))
            if v < best:
                best, best_ep = v, ep + 1
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
        net.load_state_dict(best_state)
        net.eval()
        nets[name] = {"state_dict": best_state, "mu": mu, "sd": sd, "in_dim": X.shape[1], "hidden": args.hidden}
        with torch.no_grad():
            logits = net(X[va])
            nll = float(F.cross_entropy(logits, y[va]))
            top = logits.topk(3, dim=1).indices
        yv = y[va]
        real = fut[va]
        cv = real - dev[va].numpy()
        path_of = lambda k: cv + code_dev[k].numpy()
        ade_top1 = float(np.linalg.norm(path_of(top[:, 0]) - real, axis=-1).mean())
        ade_top3 = float(np.min(np.stack([np.linalg.norm(path_of(top[:, j]) - real, axis=-1).mean(1)
                                          for j in range(3)]), axis=0).mean())
        res[name] = {"nll": nll, "acc1": float((top[:, 0] == yv).float().mean()),
                     "acc3": float((top == yv[:, None]).any(1).float().mean()),
                     "ade_top1": ade_top1, "min_ade_top3": ade_top3, "best_epoch": best_ep}
        if name == "self":
            counts = np.bincount(y[tr].numpy(), minlength=K) / tr.sum()
            res["prior_only"] = {"nll": float(-np.log(counts[yv.numpy()] + 1e-12).mean()),
                                 "acc1": float((yv == int(counts.argmax())).float().mean()),
                                 "ade_top1": float(np.linalg.norm(path_of(torch.full_like(yv, int(counts.argmax())))
                                                                  - real, axis=-1).mean())}
            res["ceiling_own_code"] = float(np.linalg.norm(path_of(yv) - real, axis=-1).mean())
            res["constant_velocity"] = float(np.linalg.norm(cv - real, axis=-1).mean())
    print(f"=== p(code | situation), {args.ckpt} ({K} codes, reference {ref}, people as {args.people_input}, goal {args.goal}, dest >= {args.min_dest_t:g} s), {res['n_val']} val samples ===")
    print(f"  {'':>14} {'NLL':>6} {'top1':>6} {'top3':>6} {'ADE top1':>9} {'minADE top3':>12}")
    pr = res["prior_only"]
    print(f"  {'most frequent':>14} {pr['nll']:6.3f} {pr['acc1']:6.3f} {'':>6} {pr['ade_top1']:9.3f}")
    for name in ["self", "self+people"]:
        r = res[name]
        print(f"  {name:>14} {r['nll']:6.3f} {r['acc1']:6.3f} {r['acc3']:6.3f} {r['ade_top1']:9.3f} {r['min_ade_top3']:12.3f}  (epoch {r['best_epoch']})")
    print(f"  constant velocity ADE {res['constant_velocity']:.3f}; own code (ceiling) ADE {res['ceiling_own_code']:.3f}")
    out = args.out or args.ckpt.replace(".pt", f"_prior_{args.people_input}{'' if args.goal == 'near' else '_' + args.goal + 'goal'}"
                                             f"{f'_dest{args.min_dest_t:g}' if args.min_dest_t else ''}.json")
    json.dump(res, open(out, "w"), indent=1)
    torch.save({"nets": nets, "args": vars(args), "res": res}, out.replace(".json", ".pt"))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
