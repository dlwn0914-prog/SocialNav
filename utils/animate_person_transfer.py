#!/usr/bin/env python
"""One person of utils/person_transfer_benchmark.py as a video: A/B queries in
their first scenes (top row), and what the learned preference picks in their
later scenes, which were never asked about (bottom row), next to what one
shared preference for everyone picks there.

Same pipeline as the benchmark: v3 candidates on the per-person scenes
(real-future goals), draft5 features with one normalization for all scenes,
EUBO cycling through the query scenes, gc-ego shape answers (beta 17), and
the shared direction chosen on the OTHER people's scenes (noise-free, by
direction search). Pedestrians are drawn as in Figure 1: hollow ring at
t=0, dots at 1/2/3 s on their real track.

Pass --pick-from <benchmark json> to choose the person by a stated rule
instead of by eye: the person whose held-out EUBO gain over the shared
preference is closest to the median gain.

Usage:
    PYTHONPATH=. .venv/bin/python utils/animate_person_transfer.py \
        --pick-from results/person_transfer/v3_draft5.json --out results/videos/person_transfer.mp4
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from src.preference.bayesian_query import EUBOStrategy
from utils.animate_query_session import neighbour_tracks
from utils.fisher_optimal_on_real_candidates import load_scenario_feats
from utils.person_transfer_benchmark import load_people, regret, scene_regrets

INK, MUTED = "#1C2230", "#7A8090"
C_CAND, C_REAL, C_A, C_B = "#C4CCDC", "#E07B39", "#2A6FD6", "#8E44AD"
C_PICK, C_SHARED, C_GOAL = "#1E9E5A", "#6B717C", "#C0392B"


def pick_person(path):
    """Person whose held-out EUBO gain over the shared preference is closest to the median gain."""
    r = json.load(open(path))["per_person"]
    gain = {int(p): r["ref"]["shared (cv)"][p] - r["methods"]["EUBO"][p] for p in r["methods"]["EUBO"]}
    med = np.median(list(gain.values()))
    pid = min(gain, key=lambda p: abs(gain[p] - med))
    return pid, gain[pid], med


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/v3")
    p.add_argument("--ego-future", default="/home/nuri2/crowd_fm_ws/scenarios_gcp/ego_future.json")
    p.add_argument("--features", default="draft5")
    p.add_argument("--person", type=int, default=None)
    p.add_argument("--pick-from", default=None, help="benchmark json; picks the median-gain person")
    p.add_argument("--n-train", type=int, default=3)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--beta", type=float, default=17.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-dirs", type=int, default=20000)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--out", default="person_transfer.mp4")
    args = p.parse_args()

    note = ""
    if args.pick_from:
        args.person, g, med = pick_person(args.pick_from)
        note = f"대표 사람 선정: EUBO의 공통 가중치 대비 개선이 전체 중앙값({med:+.3f})에 가장 가까운 사람({g:+.3f})"
    futures = json.load(open(args.ego_future))
    people = load_people(args.scene_dir, futures, args.features, args.beta)
    split = {pid: ([sc[k] for k in sorted(sc) if k < args.n_train], [sc[k] for k in sorted(sc) if k >= args.n_train])
             for pid, sc in people.items()}
    split = {pid: v for pid, v in split.items() if v[0] and v[1]}
    tr, te = split[args.person]

    pool = np.vstack([s["raw"] for v in split.values() for s in v[0]])  # as the benchmark: query scenes only
    mu, sd = pool.mean(axis=0), np.where(pool.std(axis=0) < 1e-9, 1.0, pool.std(axis=0))
    for v in split.values():
        for s in v[0] + v[1]:
            s["X"] = (s["raw"] - mu) / sd
    rng = np.random.default_rng(args.seed)
    dirs = rng.standard_normal((args.n_dirs, len(mu)))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    others = [np.mean([scene_regrets(s, dirs) for s in v[0] + v[1]], axis=0) for pid, v in split.items() if pid != args.person]
    th_shared = dirs[int(np.argmin(np.mean(others, axis=0)))]

    # one EUBO session, recording every step
    srng = np.random.default_rng([args.seed, args.person, 0])
    strat = EUBOStrategy(rng=srng)
    steps = []
    for q in range(args.n_queries):
        k = q % len(tr)
        s = tr[k]
        i, j = strat.select(s["X"])
        p_i = 1.0 / (1.0 + np.exp(-(s["u"][i] - s["u"][j])))
        win = i if srng.random() < p_i else j
        strat.update(s["X"], i, j, win)
        th = strat.theta_hat
        steps.append({"scene": k, "pair": (i, j), "win": win,
                      "picks": [int(np.argmax(t["X"] @ th)) for t in te],
                      "reg": float(np.mean([regret(t["u"], int(np.argmax(t["X"] @ th))) for t in te]))})
    shared_picks = [int(np.argmax(t["X"] @ th_shared)) for t in te]
    shared_reg = float(np.mean([regret(t["u"], k) for t, k in zip(te, shared_picks)]))
    rand_reg = float(np.mean([np.mean((t["u"].max() - t["u"]) / max(np.ptp(t["u"]), 1e-9)) for t in te]))

    # scene geometry
    scenes = tr + te
    t3 = np.array([0.0, 1.0, 2.0, 3.0])
    for s in scenes:
        d = json.load(open(f"{args.scene_dir}/scenario_{s['name']}.json"))
        s["d"] = d
        s["cands"] = load_scenario_feats(f"{args.scene_dir}/scenario_{s['name']}.json", "candidates",
                                         moving_people=True, return_candidates=True, feature_set=args.features)[1]
        s["fut"] = np.array(futures[s["name"]])
        s["tracks"] = neighbour_tracks(d, t3)

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor="#FBFAF7")
    gs = GridSpec(3, 3, height_ratios=[2.3, 2.3, 1.15], hspace=0.42, wspace=0.12,
                  left=0.035, right=0.985, top=0.87, bottom=0.1)
    fig.text(0.035, 0.955, "앞 장면에서 물어 배운 선호로, 처음 보는 장면의 경로 고르기", fontsize=21, weight="bold", color=INK)
    sub = fig.text(0.035, 0.915, "", fontsize=13.5, color=MUTED)
    fig.text(0.985, 0.955, f"Grand Central · 사람 {args.person:03d} · 장면 {len(scenes)}개", fontsize=11, color=MUTED, ha="right")
    if note:
        fig.text(0.985, 0.925, note, fontsize=9.5, color=MUTED, ha="right")

    art = []
    for n, s in enumerate(scenes):
        held = n >= len(tr)
        ax = fig.add_subplot(gs[int(held), n - len(tr) if held else n])  # query scenes on top, held-out below
        c, fut, tracks = s["cands"], s["fut"], s["tracks"]
        pts = np.concatenate([c.reshape(-1, 2), fut, tracks[:, 0]])
        lo, hi = pts.min(axis=0) - 0.4, pts.max(axis=0) + 0.4
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_facecolor("#FFFFFF")
        ax.tick_params(labelsize=8, colors=MUTED)
        for sp in ax.spines.values():
            sp.set_color("#DDDDDD")
        order = int(s["name"].rsplit("_", 1)[1]) + 1  # position in the person's walk
        ax.set_title(("처음 보는 장면 " if held else "질문 장면 ") + f"(걸음 순서 {order})", fontsize=13, loc="left",
                     weight="bold", color=C_PICK if held else C_A)
        for cc in c:
            ax.plot(cc[:, 0], cc[:, 1], color=C_CAND, lw=1.1, zorder=2)
        for trk in tracks:
            ax.plot(trk[:, 0], trk[:, 1], color="#D3D6DC", lw=0.8, zorder=1)
            ax.add_patch(plt.Circle(trk[0], 0.22, facecolor="white", edgecolor="#9AA0AA", lw=1, zorder=1.5))
            ax.scatter(trk[1:, 0], trk[1:, 1], s=[8, 12, 16], color="#6B717C", lw=0, zorder=5)
        ax.plot(fut[:, 0], fut[:, 1], color=C_REAL, lw=2, ls=(0, (4, 2)), zorder=6)
        g = np.array(s["d"]["goal"])
        gd = g / np.linalg.norm(g)
        tip = np.clip(g, lo + 0.25, hi - 0.25)
        ax.annotate("", tip, tip - 0.6 * gd, arrowprops=dict(arrowstyle="-|>", color=C_GOAL, lw=1.6, mutation_scale=12), zorder=7)
        ax.plot(0, 0, marker="s", color=INK, ms=7, zorder=8)
        a = {"ax": ax, "held": held}
        if held:
            k = n - len(tr)
            a["shared"], = ax.plot(*c[shared_picks[k]].T, color=C_SHARED, lw=2.4, ls=(0, (2, 1.5)), zorder=7)
            a["pick"], = ax.plot([], [], color=C_PICK, lw=3.4, zorder=9, solid_capstyle="round")
            a["label"] = ax.text(0.02, 0.03, "", transform=ax.transAxes, fontsize=10.5, color=INK, va="bottom",
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD"), zorder=10)
        else:
            a["A"], = ax.plot([], [], color=C_A, lw=3, zorder=9)
            a["B"], = ax.plot([], [], color=C_B, lw=3, zorder=9)
            a["label"] = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=10.5, color=INK, va="top",
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#DDDDDD"), zorder=10)
        art.append(a)

    axr = fig.add_subplot(gs[2, :2])
    axr.set_xlim(-0.2, args.n_queries + 0.2)
    top = max(rand_reg, shared_reg, max(st["reg"] for st in steps)) + 0.08
    axr.set_ylim(0, top)
    axr.set_xticks(range(args.n_queries + 1))
    axr.set_xlabel("질문 수", fontsize=10, color=MUTED, labelpad=1)
    axr.set_ylabel("평균 regret", fontsize=10, color=MUTED)
    axr.set_title("처음 보는 장면의 평균 regret (낮을수록 실제 경로에 가까움)", fontsize=11, color=INK, loc="left")
    axr.tick_params(labelsize=9, colors=MUTED)
    axr.grid(alpha=0.25)
    for sp in ("top", "right"):
        axr.spines[sp].set_visible(False)
    axr.axhline(shared_reg, color=C_SHARED, lw=1.6, ls=(0, (3, 2)))
    axr.text(0.05, shared_reg, f"공통 가중치 {shared_reg:.2f}", color=C_SHARED, fontsize=10, va="bottom", ha="left")
    axr.axhline(rand_reg, color="#C4C8D0", lw=1.2, ls=":")
    axr.text(0.05, rand_reg, f"무작위 {rand_reg:.2f}", color=MUTED, fontsize=10, va="bottom", ha="left")
    curve, = axr.plot([], [], color=C_PICK, lw=2.4, marker="o", ms=5)

    handles = [plt.Line2D([], [], color=C_CAND, lw=1.5, label="후보"),
               plt.Line2D([], [], color=C_A, lw=3, label="질문 A"), plt.Line2D([], [], color=C_B, lw=3, label="질문 B"),
               plt.Line2D([], [], color=C_PICK, lw=3.4, label="배운 개인 선호로 고른 경로"),
               plt.Line2D([], [], color=C_SHARED, lw=2.4, ls=(0, (2, 1.5)), label="공통 가중치로 고른 경로"),
               plt.Line2D([], [], color=C_REAL, lw=2, ls=(0, (4, 2)), label="그 사람이 실제로 걸은 경로 (정답)"),
               plt.Line2D([], [], marker=r"$\rightarrow$", ls="", color=C_GOAL, ms=12, label="목표 (실제 목적지 방향)"),
               plt.Line2D([], [], marker="o", ls="", mfc="white", mec="#9AA0AA", ms=8, label="보행자 t=0 (점: 1·2·3초 뒤)")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.69, 0.165), fontsize=10, frameon=False, ncol=1)

    fps = args.fps
    n_intro, n_ask, n_ans, n_hold = int(2.5 * fps), int(0.9 * fps), int(0.9 * fps), int(4 * fps)
    n_frames = n_intro + args.n_queries * (n_ask + n_ans) + n_hold

    def show_step(q, answered):
        st = steps[q]
        for k, a in enumerate(art[:len(tr)]):
            on = k == st["scene"]
            for sp in a["ax"].spines.values():
                sp.set_color(C_A if on else "#DDDDDD")
                sp.set_linewidth(2.2 if on else 1)
            if not on:
                a["A"].set_data([], [])
                a["B"].set_data([], [])
                continue
            c = scenes[k]["cands"]
            i, j = st["pair"]
            a["A"].set_data(*c[i].T)
            a["B"].set_data(*c[j].T)
            if answered:
                a["A" if st["win"] == i else "B"].set_linewidth(4.2)
                a["B" if st["win"] == i else "A"].set_alpha(0.25)
                a["label"].set_text(f"질문 {q + 1}: {'A' if st['win'] == i else 'B'}가 더 좋다")
            else:
                a["A"].set_linewidth(3); a["B"].set_linewidth(3)
                a["A"].set_alpha(1); a["B"].set_alpha(1)
                a["label"].set_text(f"질문 {q + 1}: A? B?")
        if answered:
            for k, a in enumerate(art[len(tr):]):
                c = scenes[len(tr) + k]["cands"]
                pk = st["picks"][k]
                a["pick"].set_data(*c[pk].T)
                a["label"].set_text(f"개인 {regret(te[k]['u'], pk):.2f} · 공통 {regret(te[k]['u'], shared_picks[k]):.2f}")
            curve.set_data(range(1, q + 2), [s["reg"] for s in steps[:q + 1]])
            sub.set_text(f"질문 {q + 1}개 뒤: 처음 보는 장면 평균 regret 개인 {st['reg']:.2f} vs 공통 가중치 {shared_reg:.2f}")

    def frame_fn(f):
        if f < n_intro:
            sub.set_text("위: 질문하는 장면 3개 · 아래: 질문하지 않는 장면 3개. 정답(주황)은 로봇이 모른다.")
            for k, a in enumerate(art[len(tr):]):
                a["label"].set_text(f"공통 {regret(te[k]['u'], shared_picks[k]):.2f}")
            return
        f -= n_intro
        q = f // (n_ask + n_ans)
        if q < args.n_queries:
            show_step(q, f % (n_ask + n_ans) >= n_ask)
            return
        show_step(args.n_queries - 1, True)
        sub.set_text(f"질문 {args.n_queries}개로 배운 선호를 처음 보는 장면에 사용: 평균 regret 개인 {steps[-1]['reg']:.2f} · "
                     f"공통 가중치 {shared_reg:.2f} · 무작위 {rand_reg:.2f}")

    anim = animation.FuncAnimation(fig, frame_fn, frames=n_frames, interval=1000 / fps)
    anim.save(args.out, writer=animation.FFMpegWriter(fps=fps, bitrate=4000))
    print(f"saved {args.out} ({n_frames / fps:.1f} s) person {args.person}: held-out regret per query "
          + " ".join(f"{s['reg']:.2f}" for s in steps) + f" | shared {shared_reg:.2f} random {rand_reg:.2f}")


if __name__ == "__main__":
    main()
