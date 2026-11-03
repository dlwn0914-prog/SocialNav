#!/usr/bin/env python
"""One Grand Central query session as a video: EUBO vs ActiveQuery on the
same scene, same simulated user (the pedestrian the robot replaced), same
setting as the main benchmark run (utils/unified_query_benchmark.py --user
gc-ego --answer shape --social --moving-people --beta 17, PRIEST v3
candidates with per-candidate endpoints).

Timeline: (1) the real 3 s of the scene -- every pedestrian on their
annotated track, the ego's real path dashed; (2) N A/B queries, each shown
as the asked pair, then the answer and the updated pick; (3) the robot
driving the final pick while the crowd replays.

Neighbour tracks are not stored in the scene files, so they are recovered
from the OpenTraj annotations by matching robot-frame positions at the
scene's frame (utils/build_gc_scenes.to_robot_frame); a track that ends
before 3 s continues at its last velocity.

Usage:
    PYTHONPATH=. .venv/bin/python utils/animate_query_session.py --scene gc_4to6_00 --out session.mp4
    PYTHONPATH=. .venv/bin/python utils/animate_query_session.py --scan   # per-scene final regret, to pick one
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

from src.preference.active_query import ActiveQueryStrategy
from src.preference.bayesian_query import EUBOStrategy
from src.preference.safety_filter import filter_safe_candidates
from utils.build_gc_scenes import FPS, read_track, velocity_at
from utils.fisher_optimal_on_real_candidates import load_scenario_feats
from utils.unified_query_benchmark import normalize, shape_distance

SCENE_DIR = "/home/nuri2/crowd_fm_ws/scenarios_v3/gc_all"
EGO_FUTURE = "/home/nuri2/crowd_fm_ws/scenarios_gc/ego_future_all.json"
ANNOT_DIR = "/home/nuri2/datasets/grandcentral/raw/opentraj/datasets/GC/Annotation"
METHODS = {"EUBO": EUBOStrategy, "ActiveQuery": ActiveQueryStrategy}
TITLES = {"EUBO": "EUBO (제안)", "ActiveQuery": "ActiveQuery (UserAlign)"}

INK, MUTED = "#1C2230", "#7A8090"
C_CAND, C_UNSAFE, C_REAL = "#B8C2D6", "#D5D5D5", "#E07B39"
C_A, C_B, C_PICK = "#2A6FD6", "#8E44AD", "#1E9E5A"
C_METHOD = {"EUBO": "#2A6FD6", "ActiveQuery": "#8A8F99"}


def load_scene(name, beta):
    path = os.path.join(SCENE_DIR, f"scenario_{name}.json")
    d = json.load(open(path))
    fut = json.load(open(EGO_FUTURE)).get(name)
    loaded = load_scenario_feats(path, "candidates", True, return_candidates=True, social=True)
    if fut is None or loaded is None:
        return None
    feats_raw, safe = loaded
    cands = np.array(d["candidates"])
    people = np.array(d["people"])
    real = np.linalg.norm(people, axis=1) < 500
    static = np.array(d["static_obstacles"])
    static = static[np.linalg.norm(static, axis=1) > 1e-5]
    filt = filter_safe_candidates(list(cands), static_obs=static, dynamic_obs=people[real],
                                  dynamic_vel=np.array(d["people_vel"]).reshape(-1, 2)[real],
                                  duration=float(d["traj_duration"]))
    fut = np.array(fut)
    dist = shape_distance(safe, fut)
    return {"d": d, "cands": cands, "safe_idx": list(filt["safe_indices"]), "feats": normalize(feats_raw),
            "fut": fut, "dist": dist, "u": -beta * dist}


def run(method, feats, u, seed, n_queries):
    """Same loop as unified_query_benchmark.run_session (native estimator),
    keeping every query for the animation."""
    rng = np.random.default_rng(seed)
    strat = METHODS[method](rng=rng)
    u_best, u_range = u.max(), max(u.max() - u.min(), 1e-8)
    steps, theta, stopped = [], None, False
    for _ in range(n_queries):
        pair = winner = None
        if not stopped:
            i, j = strat.select(feats)
            if strat.done():
                stopped = True
            else:
                p_i = 1.0 / (1.0 + np.exp(-(u[i] - u[j])))
                winner = i if rng.random() < p_i else j
                strat.update(feats, i, j, winner)
                theta, pair = strat.theta_hat, (i, j)
        rec = int(np.argmax(feats @ theta)) if theta is not None else strat.best(feats)
        steps.append({"pair": pair, "winner": winner, "rec": rec, "regret": float((u_best - u[rec]) / u_range)})
    return steps


def neighbour_tracks(d, t):
    """Real positions of the scene's people at times t (robot frame), matched
    to annotation tracks at the scene frame."""
    frame, ego_id = d["source"]["frame"], d["source"]["ego_ped_id"]
    ef, ew = read_track(os.path.join(ANNOT_DIR, f"{ego_id:06d}.txt"))
    k = int(np.flatnonzero(ef == frame)[0])
    v = velocity_at(ef, ew, k)
    th = np.arctan2(v[1], v[0])
    R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    origin = ew[k]

    people = np.array(d["people"])
    vel = np.array(d["people_vel"]).reshape(-1, 2)
    real = np.linalg.norm(people, axis=1) < 500
    people, vel = people[real], vel[real]
    found = [None] * len(people)
    for path in glob.glob(os.path.join(ANNOT_DIR, "*.txt")):
        frames, w = read_track(path)
        hit = np.flatnonzero(frames == frame)
        if not hit.size or int(os.path.basename(path)[:-4]) == ego_id:
            continue
        m = hit[0]
        p = R @ (w[m] - origin)
        dists = np.linalg.norm(people - p, axis=1)
        n = int(np.argmin(dists))
        if dists[n] < 1e-3:
            ok = np.flatnonzero((frames >= frame) & (frames <= frame + FPS * 3.2))
            found[n] = ((frames[ok] - frame) / FPS, (w[ok] - origin) @ R.T)

    out = []
    for n in range(len(people)):
        if found[n] is None or len(found[n][0]) < 2:
            out.append(people[n] + t[:, None] * vel[n])
            continue
        tt, pp = found[n]
        pos = np.stack([np.interp(t, tt, pp[:, 0]), np.interp(t, tt, pp[:, 1])], axis=1)
        late = t > tt[-1]
        pos[late] = pp[-1] + (t[late, None] - tt[-1]) * vel[n]
        out.append(pos)
    return np.array(out)  # (people, T, 2)


def scan(args):
    for path in sorted(glob.glob(os.path.join(SCENE_DIR, "scenario_*.json"))):
        name = os.path.basename(path)[len("scenario_"):-5]
        sc = load_scene(name, args.beta)
        if sc is None:
            continue
        r = {m: run(m, sc["feats"], sc["u"], args.seed, args.n_queries) for m in METHODS}
        print(f"{name:>14} K={len(sc['u']):2d} people={len(sc['d']['people']):2d}  "
              + "  ".join(f"{m} " + " ".join(f"{s['regret']:.2f}" for s in r[m]) for m in METHODS))


def animate(args):
    sc = load_scene(args.scene, args.beta)
    d, fut, safe_idx = sc["d"], sc["fut"], sc["safe_idx"]
    cands = sc["cands"]
    safe = cands[safe_idx]
    res = {m: run(m, sc["feats"], sc["u"], args.seed, args.n_queries) for m in METHODS}
    best = int(np.argmax(sc["u"]))

    fps = args.fps
    T_play = np.linspace(0.0, 3.0, int(3.0 * fps / args.slow))  # real-time 3 s replay, slowed
    n_intro, n_q, n_out, n_hold = len(T_play), int(1.6 * fps), len(T_play), int(2.5 * fps)
    n_frames = n_intro + int(0.8 * fps) + args.n_queries * n_q + n_out + n_hold
    t_grid = np.linspace(0.0, 3.0, cands.shape[1])
    tracks = neighbour_tracks(d, T_play)

    plt.rcParams["font.family"] = "Noto Sans CJK KR"
    fig = plt.figure(figsize=(16, 9), dpi=args.dpi, facecolor="#FBFAF7")
    gs = GridSpec(2, 2, height_ratios=[3.2, 1], hspace=0.32, wspace=0.08, left=0.04, right=0.98, top=0.86, bottom=0.13)
    axes = {m: fig.add_subplot(gs[0, k]) for k, m in enumerate(METHODS)}
    axr = fig.add_subplot(gs[1, :])
    fig.text(0.04, 0.95, "실제 보행자 장면에서 A/B 질문으로 선호 경로 찾기", fontsize=22, weight="bold", color=INK)
    sub = fig.text(0.04, 0.905, "", fontsize=14, color=MUTED)
    src = d["source"]
    fig.text(0.98, 0.95, f"Grand Central · {args.scene} · frame {src['frame']}", fontsize=11, color=MUTED, ha="right")

    allpts = np.concatenate([cands.reshape(-1, 2), fut, tracks[:, 0]])  # crowd may walk out of view later
    lo, hi = allpts.min(axis=0) - 0.5, allpts.max(axis=0) + 0.5
    art = {}
    for m, ax in axes.items():
        ax.set_facecolor("#FFFFFF")
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.tick_params(labelsize=9, colors=MUTED)
        for s in ax.spines.values():
            s.set_color("#DDDDDD")
        ax.set_title(TITLES[m], fontsize=17, weight="bold", color=C_METHOD[m] if m == "EUBO" else INK, loc="left")
        for k, c in enumerate(cands):
            if k not in safe_idx:
                ax.plot(c[:, 0], c[:, 1], color=C_UNSAFE, lw=1, ls=(0, (3, 2)), zorder=1)
        cand_lines = [ax.plot(c[:, 0], c[:, 1], color=C_CAND, lw=1.3, zorder=2)[0] for c in safe]
        ax.plot(fut[:, 0], fut[:, 1], color=C_REAL, lw=2.2, ls=(0, (4, 2)), zorder=3)
        ax.plot(0, 0, marker="s", color=INK, ms=9, zorder=6)
        peds = [ax.add_patch(plt.Circle(tr[0], 0.25, color="#9AA0AA", alpha=0.85, zorder=4)) for tr in tracks]
        lineA, = ax.plot([], [], color=C_A, lw=3.2, zorder=7)
        lineB, = ax.plot([], [], color=C_B, lw=3.2, zorder=7)
        pick, = ax.plot([], [], color=C_PICK, lw=4.2, zorder=8, solid_capstyle="round")
        robot = ax.add_patch(plt.Circle((0, 0), 0.22, color=C_PICK, zorder=9, visible=False))
        ghost = ax.add_patch(plt.Circle((0, 0), 0.22, color=C_REAL, alpha=0.6, zorder=9, visible=False))
        label = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=14, va="top", color=INK,
                        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#DDDDDD"), zorder=10)
        art[m] = dict(cand_lines=cand_lines, peds=peds, A=lineA, B=lineB, pick=pick, robot=robot, ghost=ghost,
                      label=label)
    # legend under the left panel
    handles = [plt.Line2D([], [], color=C_CAND, lw=1.5, label="후보 (안전)"),
               plt.Line2D([], [], color=C_UNSAFE, lw=1.5, ls=(0, (3, 2)), label="안전 필터 탈락"),
               plt.Line2D([], [], color=C_REAL, lw=2.2, ls=(0, (4, 2)), label="실제 보행자 경로 (정답)"),
               plt.Line2D([], [], color=C_A, lw=3, label="질문 A"),
               plt.Line2D([], [], color=C_B, lw=3, label="질문 B"),
               plt.Line2D([], [], color=C_PICK, lw=4, label="현재 선택 경로"),
               plt.Line2D([], [], marker="o", ls="", color="#9AA0AA", ms=10, label="주변 보행자 (실제 궤적)")]
    fig.legend(handles=handles, loc="lower center", ncol=7, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.005))

    axr.set_xlim(0, args.n_queries)
    axr.set_ylim(-0.03, 1.03)
    axr.set_xlabel("질문 수", fontsize=11, color=MUTED)
    axr.set_ylabel("정규화 regret", fontsize=11, color=MUTED)
    axr.set_xticks(range(args.n_queries + 1))
    axr.tick_params(labelsize=9, colors=MUTED)
    axr.grid(alpha=0.25)
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)
    curves = {m: axr.plot([], [], color=C_METHOD[m], lw=2.2, marker="o", ms=6, label=TITLES[m])[0] for m in METHODS}
    axr.legend(loc="upper right", fontsize=11, frameon=False)
    reg0 = {m: res[m][0]["regret"] for m in METHODS}  # placeholder; t=0 pick = strategy.best before any answer

    def set_pick(m, idx):
        c = safe[idx]
        art[m]["pick"].set_data(c[:, 0], c[:, 1])

    def frame_fn(f):
        for m in METHODS:
            a = art[m]
            a["robot"].set_visible(False)
            a["ghost"].set_visible(False)
        if f < n_intro:
            sub.set_text(f"1. 장면: 주변 보행자 {len(tracks)}명이 실제로 움직인 3초 · 후보 {len(cands)}개 중 {len(safe)}개가 안전")
            for m in METHODS:
                for p, tr in zip(art[m]["peds"], tracks):
                    p.center = tr[f]
                k = np.searchsorted(t_grid, T_play[f])
                art[m]["ghost"].center = fut[min(k, len(fut) - 1)]
                art[m]["ghost"].set_visible(True)
                art[m]["label"].set_text("실제 보행자가 걸은 경로가 정답")
            return
        f -= n_intro
        if f < int(0.8 * fps):
            for m in METHODS:
                for p, tr in zip(art[m]["peds"], tracks):
                    p.center = tr[0]
            return
        f -= int(0.8 * fps)
        if f < args.n_queries * n_q:
            q, ph = divmod(f, n_q)
            sub.set_text(f"2. 질문 {q + 1}/{args.n_queries}: 두 후보 중 실제 경로와 모양이 더 가까운 쪽을 고른다 (응답 노이즈 포함)")
            for m in METHODS:
                st = res[m][q]
                a = art[m]
                if st["pair"] is None:
                    a["A"].set_data([], [])
                    a["B"].set_data([], [])
                    a["label"].set_text(f"질문 {q + 1}: 조기 종료")
                else:
                    i, j = st["pair"]
                    a["A"].set_data(safe[i][:, 0], safe[i][:, 1])
                    a["B"].set_data(safe[j][:, 0], safe[j][:, 1])
                    if ph < n_q // 2:
                        a["A"].set_alpha(1.0)
                        a["B"].set_alpha(1.0)
                        a["label"].set_text(f"질문 {q + 1}: A와 B 중 어느 쪽?")
                    else:
                        win_a = st["winner"] == i
                        a["A"].set_alpha(1.0 if win_a else 0.2)
                        a["B"].set_alpha(0.2 if win_a else 1.0)
                        a["label"].set_text(f"질문 {q + 1}: {'A' if win_a else 'B'} 선택 → regret {st['regret']:.2f}")
                if ph >= n_q // 2 or q > 0:
                    shown = q + 1 if ph >= n_q // 2 else q
                    set_pick(m, res[m][shown - 1]["rec"])
                    xs = np.arange(1, shown + 1)
                    curves[m].set_data(xs, [s["regret"] for s in res[m][:shown]])
            return
        f -= args.n_queries * n_q
        for m in METHODS:
            art[m]["A"].set_data([], [])
            art[m]["B"].set_data([], [])
        k_play = min(f, n_out - 1)
        sub.set_text(f"3. 질문 {args.n_queries}개 후 고른 경로로 주행 (초록 = 로봇, 주황 = 실제 보행자)")
        for m in METHODS:
            a = art[m]
            rec = res[m][-1]["rec"]
            for p, tr in zip(a["peds"], tracks):
                p.center = tr[k_play]
            k = min(np.searchsorted(t_grid, T_play[k_play]), len(fut) - 1)
            a["robot"].center = safe[rec][k]
            a["ghost"].center = fut[k]
            a["robot"].set_visible(True)
            a["ghost"].set_visible(True)
            a["label"].set_text(f"최종 regret {res[m][-1]['regret']:.2f} · 경로 모양 거리 {sc['dist'][rec]:.2f} m"
                                f"\n(가능한 최선 {sc['dist'][best]:.2f} m)")

    anim = animation.FuncAnimation(fig, frame_fn, frames=n_frames, interval=1000 / fps)
    anim.save(args.out, writer=animation.FFMpegWriter(fps=fps, bitrate=4000))
    print(f"saved {args.out} ({n_frames / fps:.1f} s)")
    for m in METHODS:
        print(f"  {m}: regret " + " ".join(f"{s['regret']:.2f}" for s in res[m]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="gc_4to6_00")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--beta", type=float, default=17.0)
    p.add_argument("--n-queries", type=int, default=10)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--slow", type=float, default=0.75, help="replay speed factor (<1 = slower than real time)")
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--scan", action="store_true")
    p.add_argument("--out", default="query_session.mp4")
    args = p.parse_args()
    if args.scan:
        scan(args)
    else:
        animate(args)


if __name__ == "__main__":
    main()
