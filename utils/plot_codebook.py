#!/usr/bin/env python
"""One panel per code of a trained walking-behaviour codebook
(utils/train_traj_vqvae.py): held-out (val) paths assigned to the code in
light grey, the code's own path (constant velocity at the members' median
speed + decoded deviation) in bold, and what the situation looked like when
people used it -- share of paths, median speed, people within 3 m ahead,
goal direction. Panels are ordered by where the code ends up (left to right).

Usage:
    PYTHONPATH=. .venv/bin/python utils/plot_codebook.py --ckpt results/codebook/vq16.pt \
        --out results/codebook/vq16.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager

from src.preference.traj_vqvae import TrajVQVAE, deviation
from utils.train_traj_vqvae import goal_frame

for f in font_manager.findSystemFonts():
    if "NotoSansCJK" in f or "NanumGothic" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "NanumGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="results/codebook/vq16.pt")
    p.add_argument("--data", default="/home/nuri2/datasets/grandcentral/traj_codebook.npz")
    p.add_argument("--out", default="results/codebook/vq16.png")
    p.add_argument("--per-code", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ck = torch.load(args.ckpt, weights_only=False)
    a = ck["args"]
    d = np.load(args.data)
    va = d["split"] == 1
    fut, sp = d["future"][va], d["speed"][va]
    goal, ppl, n_ppl = d["goal"][va], d["people"][va], d["n_people"][va]
    ref = a.get("reference", "heading")
    if ref == "goal":  # people and goal into the same goal-aligned frame as the paths
        fut = goal_frame(fut, goal)
        ppl = np.concatenate([goal_frame(ppl[..., :2], goal), goal_frame(ppl[..., 2:], goal)], axis=-1)
        goal = goal_frame(goal[:, None], goal)[:, 0]
    model = TrajVQVAE(horizon=fut.shape[1], num_codes=a["codes"], latent=a["latent"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        dev = deviation(torch.tensor(fut), torch.tensor(sp))
        idx = model.codebook.assign(model.encode(dev)).numpy()
        code_dev = model.decode_codes().numpy()
    K, H = a["codes"], fut.shape[1]
    t = 0.8 * np.arange(1, H + 1)

    # situation features per path
    ahead = ((ppl[..., 0] > 0) & (np.hypot(ppl[..., 0], ppl[..., 1]) < 3.0)
             & (np.arange(ppl.shape[1])[None] < n_ppl[:, None])).sum(1)
    gang = np.degrees(np.arctan2(goal[:, 1], goal[:, 0]))
    head = -np.degrees(np.arctan2(d["goal"][va][:, 1], d["goal"][va][:, 0]))  # current heading in the goal frame

    order = np.argsort([code_dev[k, -1, 1] for k in range(K)])[::-1]  # most-left first
    cols = 4 if K <= 16 else 8
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 2.9 * rows), squeeze=False)
    rng = np.random.default_rng(args.seed)
    for n, k in enumerate(order):
        ax = axes[n // cols][n % cols]
        m = np.where(idx == k)[0]
        share = len(m) / len(idx)
        v = float(np.median(sp[m])) if len(m) else 1.0
        for j in rng.choice(m, size=min(args.per_code, len(m)), replace=False) if len(m) else []:
            ax.plot(np.r_[0, fut[j, :, 0]], np.r_[0, fut[j, :, 1]], color="#b9c2bd", lw=0.6, alpha=0.6, zorder=1)
        path = np.c_[v * t, np.zeros(H)] + code_dev[k]
        ax.plot([0, v * t[-1]], [0, 0], color="#8a948f", lw=0.8, ls=":", zorder=2)
        ax.plot(np.r_[0, path[:, 0]], np.r_[0, path[:, 1]], color="#1f7a55", lw=2.4, zorder=3)
        ax.scatter(path[:, 0], path[:, 1], s=10, color="#1f7a55", zorder=4)
        ax.scatter([0], [0], marker="s", s=24, color="#1d2421", zorder=5)
        ax.set_xlim(-0.5, 8.5)
        ax.set_ylim(-3.2, 3.2)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25)
        dx = code_dev[k, -1]
        ax.set_title(f"코드 {k} · {share:.1%}", fontsize=10, loc="left", fontweight="bold")
        if len(m):
            ax.text(0.02, 0.03,
                    f"속도 {v:.2f} m/s · 4.8초 뒤 앞뒤 {dx[0]:+.1f}, 좌우 {dx[1]:+.1f} m\n" +
                    (f"앞 3 m 안 사람 {np.mean(ahead[m]):.1f}명 · 목표 대비 현재 진행 방향 {np.median(head[m]):+.0f}°" if ref == "goal" else
                     f"앞 3 m 안 사람 {np.mean(ahead[m]):.1f}명 · 목표 각도 중앙값 {np.median(gang[m]):+.0f}°"),
                    transform=ax.transAxes, fontsize=7, va="bottom", color="#3d4743")
    for n in range(K, rows * cols):
        axes[n // cols][n % cols].axis("off")
    res = ck["val"]
    fig.suptitle(f"Grand Central 보행 코드북 (VQ-VAE, 코드 {K}개) · 검증 사람 {res['n_val']}개 경로 · "
                 f"평균 오차 등속 {res['ade_cv']:.2f} m → 코드 {res['ade_vq']:.2f} m\n"
                 f"초록 = 코드 경로(그 코드 사람들의 중앙 속도로 등속 + 편차), 회색 = 그 코드에 배정된 실제 경로, " +
                 (f"점선 = 목표 방향 등속 직진, 검은 네모 = 현재 위치(+x = 목표 방향)" if ref == "goal" else
                  f"점선 = 등속 직진, 검은 네모 = 현재 위치(진행 방향 = +x)"), fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(args.out, dpi=130)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
