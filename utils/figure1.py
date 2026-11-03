#!/usr/bin/env python
"""Figure 1 draft: (a) a Grand Central scene with the generator's candidates,
(b) A/B queries narrowing them to the path the replaced pedestrian actually
walked, (c) regret vs number of queries on the whole benchmark.

(a)/(b) reuse utils/animate_query_session.py (same scene loading, same EUBO
session loop, seed 0). (c) reads a unified_query_benchmark.py result json;
the shared-theta line is the 5-fold cross-validated number passed in
--shared (it is not stored in that json).

Usage:
    PYTHONPATH=. .venv/bin/python utils/figure1.py --scene gc_b4to6_19 --out fig1_b4to6_19
    PYTHONPATH=. .venv/bin/python utils/figure1.py --scene gc_b4to6_19 --lang ko --out fig1_b4to6_19_ko
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.animate_query_session import load_scene, run, neighbour_tracks

INK, MUTED = "#1C2230", "#7A8090"
C_CAND, C_REAL, C_A, C_B = "#B8C2D6", "#E07B39", "#2A6FD6", "#8E44AD"
C_PICK0, C_PICK = "#9FD3B5", "#1E9E5A"
C_EUBO, C_AQ, C_RAND = "#2A6FD6", "#8A8F99", "#C4C8D0"

TEXT = {
    "en": dict(a="(a) Safe, diverse candidates", b="(b) A few A/B questions", c="(c) Regret vs. queries",
               cand="candidates", ped="pedestrians at t=0", tick="positions at 1, 2, 3 s",
               dmin="same-time min gap {d:.2f} m", real="path the person actually walked",
               q1="query 1: A or B?", pick0="pick after 1 query", pick="final pick ({n} queries)",
               xq="number of queries", yr="normalized simple regret", shared="shared preference",
               eubo="EUBO (ours)", aq="ActiveQuery", rand="Random",
               eq="{a} queries ≈ ActiveQuery's {b}"),
    "ko": dict(a="(a) 안전하고 서로 다른 후보", b="(b) A/B 질문 몇 개", c="(c) 질문 수와 regret",
               cand="후보", ped="주변 보행자 (t=0)", tick="1·2·3초 뒤 위치",
               dmin="같은 시각 최소 거리 {d:.2f} m", real="실제로 걸은 경로",
               q1="질문 1: A? B?", pick0="질문 1개 후 선택", pick="최종 선택 (질문 {n}개)",
               xq="질문 수", yr="정규화 simple regret", shared="공통 가중치",
               eubo="EUBO (제안)", aq="ActiveQuery", rand="무작위 질문",
               eq="질문 {a}개 ≈ ActiveQuery {b}개"),
}


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#BBBBBB")
    ax.tick_params(labelsize=7, colors=MUTED, length=2)


def scene_axes(ax, sc, tracks, lim, tick_idx):
    ax.set_xlim(*lim[0])
    ax.set_ylim(*lim[1])
    ax.set_aspect("equal", adjustable="datalim")
    style(ax)
    ax.set_xlabel("x (m)", fontsize=7, color=MUTED, labelpad=1)
    # t=0 positions are drawn as hollow rings under the candidates: the robot only reaches that spot
    # later, so a solid disc there reads as a collision. Dots mark where each person is at 1/2/3 s.
    for tr in tracks:
        ax.plot(tr[:, 0], tr[:, 1], color="#C9CCD2", lw=0.8, zorder=1)
        ax.add_patch(plt.Circle(tr[0], 0.22, facecolor="white", edgecolor="#9AA0AA", lw=0.8, zorder=1.5))
        ax.scatter(tr[tick_idx, 0], tr[tick_idx, 1], s=[5, 7, 9], color="#6B717C", lw=0, zorder=5)
    ax.plot(0, 0, marker="s", color=INK, ms=5, zorder=6)


def sync_min_dist(paths, tracks):
    """Closest robot-pedestrian distance at the same time step (paths and tracks share the time grid)."""
    return min(np.linalg.norm(paths - tr[None], axis=-1).min() for tr in tracks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="gc_b4to6_19")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=17.0)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--curve", default="results/overnight_1028/v3_candidates_shape_social.json")
    p.add_argument("--shared", type=float, default=0.353, help="5-fold CV shared-theta regret for --curve's setting")
    p.add_argument("--lang", choices=["en", "ko"], default="en")
    p.add_argument("--out", default="figure1")
    args = p.parse_args()
    T = TEXT[args.lang]
    plt.rcParams["font.family"] = "Noto Sans CJK KR" if args.lang == "ko" else "DejaVu Sans"

    sc = load_scene(args.scene, args.beta)
    safe = sc["cands"][sc["safe_idx"]]
    fut = sc["fut"]
    steps = run("EUBO", sc["feats"], sc["u"], args.seed, args.n_queries)
    t_grid = np.linspace(0.0, 3.0, safe.shape[1])
    tracks = neighbour_tracks(sc["d"], t_grid)
    tick_idx = [int(np.argmin(np.abs(t_grid - s))) for s in (1.0, 2.0, 3.0)]
    dmin = sync_min_dist(safe, tracks)

    pts = np.concatenate([safe.reshape(-1, 2), fut, tracks[:, 0]])
    lim = ((pts[:, 0].min() - 0.4, pts[:, 0].max() + 0.4), (pts[:, 1].min() - 0.4, pts[:, 1].max() + 0.4))

    fig = plt.figure(figsize=(7.2, 2.35), dpi=300, facecolor="white")
    axa = fig.add_axes([0.04, 0.17, 0.28, 0.68])
    axb = fig.add_axes([0.355, 0.17, 0.28, 0.68])
    axc = fig.add_axes([0.70, 0.17, 0.20, 0.68])

    scene_axes(axa, sc, tracks, lim, tick_idx)
    for c in safe:
        axa.plot(c[:, 0], c[:, 1], color=C_EUBO, lw=0.9, alpha=0.55, zorder=2)
    axa.set_title(T["a"], fontsize=7.5, color=INK, loc="left", weight="bold")
    axa.legend(handles=[plt.Line2D([], [], color=C_EUBO, alpha=0.55, lw=1, label=f"{T['cand']} (K={len(safe)})"),
                        plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#9AA0AA", ms=5, label=T["ped"]),
                        plt.Line2D([], [], marker="o", ls="", color="#6B717C", ms=2.5, label=T["tick"])],
               loc="lower left", fontsize=6, frameon=False, handlelength=1.4, borderaxespad=0.2)

    axa.text(lim[0][1] - 0.1, lim[1][1] - 0.1, T["dmin"].format(d=dmin), fontsize=6, color=MUTED, ha="right", va="top")

    scene_axes(axb, sc, tracks, lim, tick_idx)
    for c in safe:
        axb.plot(c[:, 0], c[:, 1], color=C_CAND, lw=0.7, zorder=2)
    i, j = steps[0]["pair"]
    axb.plot(safe[i][:, 0], safe[i][:, 1], color=C_A, lw=1.4, zorder=3)
    axb.plot(safe[j][:, 0], safe[j][:, 1], color=C_B, lw=1.4, zorder=3)
    axb.text(safe[i][-1, 0] + 0.08, safe[i][-1, 1], "A", color=C_A, fontsize=7, weight="bold", va="center")
    axb.text(safe[j][-1, 0] + 0.08, safe[j][-1, 1], "B", color=C_B, fontsize=7, weight="bold", va="center")
    p0, pf = safe[steps[0]["rec"]], safe[steps[-1]["rec"]]
    axb.plot(p0[:, 0], p0[:, 1], color=C_PICK0, lw=1.8, zorder=4)
    axb.plot(pf[:, 0], pf[:, 1], color=C_PICK, lw=2.2, zorder=5, solid_capstyle="round")
    axb.scatter(pf[tick_idx, 0], pf[tick_idx, 1], s=[5, 7, 9], color=C_PICK, edgecolor="white", lw=0.4, zorder=6)
    axb.plot(fut[:, 0], fut[:, 1], color=C_REAL, lw=1.4, ls=(0, (3, 1.5)), zorder=6)
    axb.set_title(T["b"], fontsize=7.5, color=INK, loc="left", weight="bold")
    axb.legend(handles=[plt.Line2D([], [], color=C_A, lw=1.4, label=T["q1"]),
                        plt.Line2D([], [], color=C_PICK0, lw=1.8, label=T["pick0"]),
                        plt.Line2D([], [], color=C_PICK, lw=2.2, label=T["pick"].format(n=args.n_queries)),
                        plt.Line2D([], [], color=C_REAL, lw=1.4, ls=(0, (3, 1.5)), label=T["real"])],
               loc="lower left", fontsize=6, frameon=False, handlelength=1.4, borderaxespad=0.2)

    s = json.load(open(args.curve))["summary"]
    n = len(s["EUBO"]["regret_per_query"])
    x = np.arange(1, n + 1)
    style(axc)
    axc.axhline(args.shared, color=INK, lw=0.9, ls=(0, (4, 2)))
    axc.text(n, args.shared + 0.008, T["shared"], fontsize=6, color=INK, ha="right", va="bottom")
    for m, c, lw, lab in (("Random", C_RAND, 1.1, T["rand"]), ("ActiveQuery", C_AQ, 1.3, T["aq"]),
                          ("EUBO", C_EUBO, 1.8, T["eubo"])):
        y = s[m]["regret_per_query"]
        axc.plot(x, y, color=c, lw=lw, marker="o", ms=2.2)
        axc.text(n + 0.25, y[-1], lab, fontsize=6, color=INK if m != "EUBO" else C_EUBO, va="center",
                 weight="bold" if m == "EUBO" else "normal")
    aq_end = s["ActiveQuery"]["regret_per_query"][-1]
    k = int(np.argmax(np.array(s["EUBO"]["regret_per_query"]) <= aq_end))
    axc.annotate("", (k + 1, aq_end), (n, aq_end), arrowprops=dict(arrowstyle="-|>", color=C_EUBO, lw=0.8, mutation_scale=6))
    axc.text((k + 1 + n) / 2, 0.128, T["eq"].format(a=k + 1, b=n), fontsize=6, color=C_EUBO, ha="center", va="top")
    axc.set_xlim(0.5, n + 0.3)
    axc.set_ylim(0.1, max(s["Random"]["regret_per_query"][0], args.shared) + 0.04)
    axc.set_xticks([1, 5, 10] if n == 10 else [1, 5, 10, 15, 20])
    axc.set_xlabel(T["xq"], fontsize=7, color=MUTED, labelpad=1)
    axc.set_ylabel(T["yr"], fontsize=7, color=MUTED, labelpad=1)
    axc.set_title(T["c"], fontsize=7.5, color=INK, loc="left", weight="bold")

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", facecolor="white")
    print(f"saved {args.out}.pdf/.png  scene {args.scene}: EUBO regret " + " ".join(f"{t['regret']:.2f}" for t in steps)
          + f"  same-time min gap {dmin:.2f} m  (EUBO {k + 1} queries <= ActiveQuery {n}: {aq_end:.3f})")


if __name__ == "__main__":
    main()
