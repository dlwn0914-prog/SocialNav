#!/usr/bin/env python
"""Figure 1 (paper/figures/fig_overview.pdf): the pipeline on one real
Grand Central test situation.

  (a) codebook     the K = 16 decoded paths from the robot's state
  (b) safety       the safe ones (min distance >= 0.5 m at constant velocity,
                   progress >= 0.5 x best); unsafe ones greyed
  (c) A/B query    the first EUBO question in this situation (prior posterior)
  (d) preference   theta* of an illustrative user vs theta_hat after 5 EUBO
                   answers given in 5 OTHER situations
  (e) execute      argmax theta_hat . phi among the safe paths, next to the
                   human prior's choice (argmax log p)

Situation, candidates, features and normalization are those of
utils/style_preference_benchmark.py (GC, dest >= 12 s, predicted conflict).
The situation is the first (seeded order) with 6-10 safe codes on both sides
of the conflict person where the learned and the prior choice differ.

Usage:
    PYTHONPATH=. .venv/bin/python utils/figure_overview.py --out paper/figures/fig_overview
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
import torch

from src.preference.bayesian_query import EUBOStrategy
from src.preference.traj_vqvae import TrajVQVAE
from utils.code_preference_benchmark import code_features
from utils.style_preference_benchmark import FEATURES, style_features
from utils.train_code_prior import load_prior, situation

INK, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # reference categorical slots 1-3
PERSON = "#7a7974"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prior", default="results/codebook/goalfix/gc_dest_s1.pt")
    p.add_argument("--theta", default="0.2,-0.8,0.1,-0.5,0.2", help="illustrative user: side, front, yield, detour, human")
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default="paper/figures/fig_overview")
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
    K, H = code_dev.shape[:2]
    t = 0.8 * np.arange(1, H + 1)

    base = np.where((d["split"] == 2) & (d["dest_t"] >= pa["min_dest_t"]))[0]
    _, x = situation(d, "heading", True, pa["people_input"], pa["goal"])
    base = base[x[base][:, -4] < 8.0]
    with torch.no_grad():
        logp = torch.log_softmax(net(torch.tensor((x[base] - net.mu) / net.sd)), dim=1).numpy()
    paths = np.stack([d["speed"][base][:, None] * t, np.zeros((len(base), H))], axis=-1)[:, None] + code_dev[None]
    ppl, n_ppl, dest, speed = d["people"][base], d["n_people"][base], d["dest"][base], d["speed"][base]
    old, dmin = code_features(paths, ppl, n_ppl, dest, logp)
    prog = old[..., 2]
    safe = (dmin >= 0.5) & (prog >= 0.5 * prog.max(1, keepdims=True))
    raw = style_features(paths, ppl, n_ppl, speed, dest, logp)
    ok = np.where(safe.sum(1) >= 4)[0]
    rows = raw[ok][safe[ok]]
    mu, sd = rows.mean(0), rows.std(0) + 1e-9
    X = (raw - mu) / sd

    theta = np.array([float(v) for v in args.theta.split(",")])
    theta /= np.linalg.norm(theta)
    qs = rng.choice(ok, 5, replace=False)  # query situations of this user
    strat = EUBOStrategy(rng=np.random.default_rng(args.seed))
    for s in qs:
        F = X[s][safe[s]]
        i, j = strat.select(F)
        u = 5.0 * (F @ theta)
        strat.update(F, i, j, i if rng.random() < 1 / (1 + np.exp(-(u[i] - u[j]))) else j)
    th_hat = strat.theta_hat

    pick = None
    for s in rng.permutation(np.setdiff1d(ok, qs)):
        c = np.where(safe[s])[0]
        side = raw[s][c, 0]
        if not (6 <= len(c) <= 10 and side.max() > 0.3 and side.min() < -0.3 and n_ppl[s] <= 8):
            continue
        k_hat, k_pri = c[np.argmax(X[s][c] @ th_hat)], c[np.argmax(raw[s][c, -1])]
        if k_hat != k_pri:
            pick = s
            break
    s = pick
    c = np.where(safe[s])[0]
    k_hat, k_pri = c[np.argmax(X[s][c] @ th_hat)], c[np.argmax(raw[s][c, -1])]
    qi, qj = EUBOStrategy(rng=np.random.default_rng(0)).select(X[s][c])
    k_a, k_b = c[qi], c[qj]
    P = paths[s]
    people = ppl[s][:n_ppl[s]]
    print(f"situation {s} (index into the conflict pool), {len(c)}/{K} safe, {n_ppl[s]} people; "
          f"query {k_a} vs {k_b}; prior picks {k_pri}, learned picks {k_hat}")

    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED})
    fig = plt.figure(figsize=(12, 3.1))
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1], wspace=0.62)
    axes = [fig.add_subplot(gs[i]) for i in range(5)]

    def scene(ax, title):
        ax.set_xlim(-4.2, 4.2)
        ax.set_ylim(-1.2, 7.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        for m in range(-4, 5, 2):
            ax.axvline(m, color=GRID, lw=0.5, zorder=0)
        for m in range(0, 8, 2):
            ax.axhline(m, color=GRID, lw=0.5, zorder=0)
        u = dest[s] / np.linalg.norm(dest[s])  # destination direction (robot frame: x ahead -> up, y left -> left)
        ax.annotate("", xy=(-u[1] * 6.6, u[0] * 6.6), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=AQUA, lw=1.2, ls=(0, (4, 3)), alpha=0.8), zorder=1)
        for px, py, vx, vy in people:
            ax.add_patch(Circle((-py, px), 0.2, color=PERSON, zorder=4))
            ax.plot([-py, -py - vy * 0.8], [px, px + vx * 0.8], color=PERSON, lw=1.2, zorder=4)
        ax.add_patch(Circle((0, 0), 0.22, color=INK, zorder=5))
        ax.set_title(title, fontsize=9, color=INK, loc="left")

    def path(ax, k, **kw):
        pts = np.r_[[[0.0, 0.0]], P[k]]
        ax.plot(-pts[:, 1], pts[:, 0], **kw)

    scene(axes[0], "(a) Codebook: 16 paths")
    for k in range(K):
        path(axes[0], k, color=BLUE, lw=1.1, alpha=0.75, zorder=3)
    scene(axes[1], f"(b) Safety filter: {len(c)} safe")
    for k in range(K):
        if k in c:
            path(axes[1], k, color=BLUE, lw=1.3, zorder=3)
        else:
            path(axes[1], k, color=FAINT, lw=1.0, ls=(0, (2, 2)), zorder=2)
    scene(axes[2], "(c) A/B question (EUBO)")
    for k in c:
        path(axes[2], k, color=FAINT, lw=0.9, zorder=2)
    path(axes[2], k_a, color=BLUE, lw=2.2, zorder=3)
    path(axes[2], k_b, color=ORANGE, lw=2.2, zorder=3)
    for k, lab, col in [(k_a, "A", BLUE), (k_b, "B", ORANGE)]:
        e = P[k][-1]
        axes[2].text(-e[1], e[0] + 0.25, lab, color=col, fontsize=10, fontweight="bold", ha="center", va="bottom")

    ax = axes[3]
    yy = np.arange(len(FEATURES))[::-1]
    th_n = th_hat / np.linalg.norm(th_hat)
    ax.barh(yy + 0.18, theta, height=0.34, color=MUTED, label="user $\\theta^*$")
    ax.barh(yy - 0.18, th_n, height=0.34, color=BLUE, label="$\\hat\\theta$ after 5 answers")
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(yy)
    ax.set_yticklabels(FEATURES)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, len(FEATURES) - 0.4)
    ax.set_box_aspect(1)
    ax.tick_params(axis="x", labelsize=7, colors=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), fontsize=7, frameon=False, ncol=2, handlelength=1)
    ax.set_title("(d) Bayesian Bradley-Terry", fontsize=9, color=INK, loc="left")

    scene(axes[4], "(e) Execute: best under $\\hat\\theta$")
    for k in c:
        path(axes[4], k, color=FAINT, lw=0.9, zorder=2)
    path(axes[4], k_pri, color=MUTED, lw=1.6, ls=(0, (4, 2)), zorder=3, label="human prior")
    path(axes[4], k_hat, color=BLUE, lw=2.4, zorder=3, label="learned preference")
    axes[4].legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=7, frameon=False, ncol=2, handlelength=1.6)

    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        l, r = axes[a].get_position(), axes[b].get_position()
        y = (l.y0 + l.y1) / 2
        x1 = r.x0 - (0.045 if b == 3 else 0.004)  # (d) has its feature names left of the axis
        fig.patches.append(FancyArrowPatch((l.x1 + 0.004, y), (x1, y), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=10, color=MUTED, lw=1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=220, bbox_inches="tight")
    print(f"saved {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
