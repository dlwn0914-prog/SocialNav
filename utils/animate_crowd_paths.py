#!/usr/bin/env python
"""Animate a robot moving along the 3 rule-based candidate paths (direct,
moderate_detour, wide_detour from src/preference/crowd_path_generator.py)
around utils/crowd_scenario_demo.py's default 6-person corridor scenario,
saved as a GIF. Purely offline/geometric -- no live Gazebo simulation needed
(see utils/crowd_scenario_demo.py's own docstring for why: sidesteps the
nav2 controller instability found earlier this session).

Usage:
    PYTHONPATH=. python utils/animate_crowd_paths.py [output.gif]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.preference.crowd_path_generator import generate_candidate_paths
import utils.crowd_scenario_demo as csd

STYLES = {
    "direct": dict(color="dimgray", marker="o", label="direct"),
    "moderate_detour": dict(color="tab:blue", marker="s", label="moderate_detour"),
    "wide_detour": dict(color="tab:orange", marker="^", label="wide_detour"),
}


def interpolate(path_world, start, n_per_seg=12):
    """Smooth the 5 discrete waypoints into a walk-able point sequence for animation."""
    full = np.vstack([start[None, :], path_world])
    pts = []
    for i in range(len(full) - 1):
        for t in np.linspace(0, 1, n_per_seg, endpoint=False):
            pts.append(full[i] * (1 - t) + full[i + 1] * t)
    pts.append(full[-1])
    return np.array(pts)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/home/nuri2/Desktop/robot_crowd_paths.gif"

    start, goal, agents = csd.load_scenario(None)  # default hardcoded corridor scenario
    candidates_world = generate_candidate_paths(start, goal, agents, num_steps=5)
    interp = {name: interpolate(candidates_world[name], start) for name in candidates_world}
    n_frames = max(len(v) for v in interp.values())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 30)
    ax.set_ylim(6, 17)
    ax.set_aspect("equal")
    ax.set_title("Robot navigating 3 candidate paths around a 6-person crowd")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)

    ax.scatter(agents[:, 0], agents[:, 1], color="red", marker="x", s=100, zorder=5, label="crowd (6)")
    ax.scatter(*start, color="black", marker="o", s=60, zorder=5)
    ax.scatter(*goal, color="black", marker="*", s=200, zorder=5, label="goal")

    trail_lines, robot_dots = {}, {}
    for name, st in STYLES.items():
        (trail,) = ax.plot([], [], color=st["color"], lw=1.5, alpha=0.5)
        (dot,) = ax.plot([], [], marker=st["marker"], color=st["color"], markersize=12, label=st["label"], linestyle="None")
        trail_lines[name] = trail
        robot_dots[name] = dot
    ax.legend(loc="upper left", fontsize=9)

    def update(frame):
        for name in interp:
            pts = interp[name]
            idx = min(frame, len(pts) - 1)
            trail_lines[name].set_data(pts[: idx + 1, 0], pts[: idx + 1, 1])
            robot_dots[name].set_data([pts[idx, 0]], [pts[idx, 1]])
        return list(trail_lines.values()) + list(robot_dots.values())

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=60, blit=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=16))
    print("saved to", out_path)


if __name__ == "__main__":
    main()
