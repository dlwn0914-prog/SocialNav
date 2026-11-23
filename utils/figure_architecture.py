#!/usr/bin/env python
"""Architecture figure (paper/figures/fig_architecture.pdf): the framework
as a block diagram, from the pedestrian data to the executed path.

  offline      pedestrian trajectories -> VQ-VAE codebook + prior p(code | s)
  per situation  decode K paths -> safety filter C(s) -> style features phi
  query phase  (a few query situations) EUBO pair -> user's A/B answer ->
               Bayesian Bradley-Terry posterior, repeated n times
  deployment   (new situations) execute argmax_{k in C(s)} theta_hat . phi

The small path fans are the real decoded codes of the GC codebook from a
1.2 m/s start (unsafe ones in the filter box are an illustrative subset).

Usage:
    PYTHONPATH=. .venv/bin/python utils/figure_architecture.py --out paper/figures/fig_architecture
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch

from src.preference.traj_vqvae import TrajVQVAE

INK, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FILL, FILL_Q, FILL_D = "#f6f5f2", "#eef4fc", "#ecf8f2"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=None, help="VQ-VAE checkpoint (default: the one behind the GC prior)")
    p.add_argument("--prior", default="results/codebook/goalfix/gc_dest_s1.pt")
    p.add_argument("--out", default="paper/figures/fig_architecture")
    args = p.parse_args()

    ck_path = args.ckpt or torch.load(args.prior, weights_only=False)["args"]["ckpt"]
    ck = torch.load(ck_path, weights_only=False)
    vq = TrajVQVAE(horizon=6, num_codes=ck["args"]["codes"], latent=ck["args"]["latent"])
    vq.load_state_dict(ck["state_dict"])
    vq.eval()
    with torch.no_grad():
        dev = vq.decode_codes().numpy()
    t = 0.8 * np.arange(1, dev.shape[1] + 1)
    paths = np.stack([1.2 * t, np.zeros_like(t)], -1)[None] + dev  # (K, H, 2), x ahead, y left
    paths = np.concatenate([np.zeros((len(paths), 1, 2)), paths], 1)
    unsafe = set(np.argsort(paths[:, -1, 1])[-4:]) | {int(np.argmin(paths[:, -1, 0]))}  # illustrative

    plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(7.2, 2.75))
    ax.set_xlim(0, 144)
    ax.set_ylim(0, 55)
    ax.axis("off")
    T, S = 7.0, 5.9  # title / subtitle font sizes

    def box(x, y, w, h, title, sub=None, fill=FILL, edge=FAINT, bold=True):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=fill, ec=edge, lw=0.8, zorder=1))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=T,
                fontweight="bold" if bold else "normal", color=INK, zorder=3)
        if sub:
            ax.text(x + w / 2, y + 1.6, sub, ha="center", va="bottom", fontsize=S, color=MUTED, zorder=3,
                    linespacing=1.15)

    def arrow(a, b, color=MUTED, rad=0.0):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, color=color, lw=0.9,
                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=4))

    def fan(cx, cy, scale, ks, colors, lws):
        for k, c, w in zip(ks, colors, lws):
            pts = paths[k]
            ax.plot(cx - pts[:, 1] * scale, cy + pts[:, 0] * scale, color=c, lw=w, solid_capstyle="round", zorder=3)
        ax.add_patch(plt.Circle((cx, cy), 0.8, color=INK, zorder=5))

    def header(x, y, text, color=MUTED):
        ax.text(x, y, text, fontsize=S, color=color, va="top", style="italic")

    K = len(paths)
    # offline: data -> codebook
    header(1, 54.5, "Offline, once")
    box(1, 30, 24, 20, "Pedestrian data", "Grand Central,\nETH/UCY")
    for i, off in enumerate(np.linspace(-6, 6, 5)):  # a few stylized tracks
        yy = np.linspace(37.5, 44.5, 12)
        ax.plot(13 + off + 1.1 * np.sin(yy * 0.6 + i), yy, color=FAINT, lw=1.0, zorder=2)
    box(1, 5, 24, 20, "VQ-VAE codebook", "+ prior $p(\\mathrm{code}\\mid s)$")
    ax.text(13, 15, f"K = {K}\nwalking behaviours", ha="center", va="center", fontsize=S, color=INK, zorder=3)
    arrow((13, 30), (13, 25))

    # per-situation pipeline
    header(30, 54.5, "In every situation $s$")
    box(30, 12, 21, 38, "Candidates", "decode all\n%d codes" % K)
    fan(40.5, 21, 2.1, range(K), [BLUE] * K, [0.8] * K)
    box(55, 12, 21, 38, "Safety filter", "$\\mathcal{C}(s)$: clearance\n0.5 m, progress")
    fan(65.5, 21, 2.1, range(K), [FAINT if k in unsafe else BLUE for k in range(K)],
        [0.8 if k in unsafe else 0.9 for k in range(K)])
    box(80, 12, 20, 38, "Style features", "$\\phi(s,k)\\in\\mathbb{R}^5$")
    for i, (f, v) in enumerate(zip(["side", "front", "yield", "detour", "human"], [0.5, -0.7, 0.3, -0.4, 0.8])):
        y = 40 - i * 4.6
        ax.text(89.5, y, f, ha="right", va="center", fontsize=S, color=MUTED, zorder=3)
        ax.plot([93, 93], [y - 1.6, y + 1.6], color=FAINT, lw=0.6, zorder=3)
        ax.add_patch(plt.Rectangle((93, y - 1.1), v * 5.5, 2.2, color=MUTED, lw=0, zorder=3))
    arrow((25, 15), (30, 22))
    arrow((51, 31), (55, 31))
    arrow((76, 31), (80, 31))

    # query phase: EUBO -> user -> posterior (blue); deployment: execute (green)
    ax.add_patch(FancyBboxPatch((104, 19.5), 40, 35.5, boxstyle="round,pad=0,rounding_size=2",
                                fc=FILL_Q, ec="none", zorder=0))
    header(105.5, 54.2, "Query phase: $n\\approx5$ query situations", BLUE)
    box(106, 35.5, 16, 13, "EUBO", "picks the\npair (A, B)", fill="white", edge=BLUE)
    box(126, 35.5, 16, 13, "User", "picks one", fill="white", edge=BLUE)
    for x, lab, c in [(130.6, "A", BLUE), (137.4, "B", ORANGE)]:
        ax.text(x, 41.2, lab, color=c, fontsize=8, fontweight="bold", ha="center", va="center", zorder=3)
    ax.text(134, 41.2, "vs", color=MUTED, fontsize=S, ha="center", va="center", zorder=3)
    arrow((122, 42), (126, 42), color=BLUE)
    box(106, 21.5, 36, 9.5, "Bayesian Bradley\u2013Terry", "posterior $p(\\theta\\mid\\mathcal{D})$ (Laplace)",
        fill="white", edge=BLUE)
    arrow((134, 35.5), (134, 31), color=BLUE)
    arrow((114, 31), (114, 35.5), color=BLUE)
    ax.text(113, 33.3, "next query", fontsize=5.4, color=BLUE, ha="right", va="center")
    ax.text(135, 33.3, "update", fontsize=5.4, color=BLUE, ha="left", va="center")

    ax.add_patch(FancyBboxPatch((104, 0), 40, 17.5, boxstyle="round,pad=0,rounding_size=2",
                                fc=FILL_D, ec="none", zorder=0))
    header(105.5, 16.9, "Deployment: new situations", AQUA)
    box(106, 1.8, 36, 9.6, "Execute best safe path", "$\\mathrm{argmax}_{k\\in\\mathcal{C}(s)}\\,\\hat\\theta^{\\top}\\phi(s,k)$",
        fill="white", edge=AQUA)
    arrow((137, 21.5), (137, 11.4), color=AQUA)
    ax.text(138, 18.6, "$\\hat\\theta$", fontsize=S + 0.5, color=AQUA, ha="left", va="center")

    # features feed both phases
    arrow((100, 38), (106, 42), color=BLUE)
    arrow((100, 20), (106, 6.6), color=AQUA)

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=240, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
