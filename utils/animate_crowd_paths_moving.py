#!/usr/bin/env python
"""Same scenario as utils/animate_crowd_paths.py, but the 6-person crowd also
wanders (small per-agent sinusoidal motion, different phase/freq/radius each)
instead of standing frozen -- a "moving crowd" companion to that script's
"stationary crowd" animation, matching the two scenario types (crowd
stationary / crowd moving) used in the earlier 2D CrowdNav validation this
research direction builds on. Still purely offline/geometric: the 3 candidate
paths themselves are NOT replanned around the moving crowd (that would need
the live guidance pipeline, not this illustrative animation) -- only the
crowd markers move while the paths stay fixed.

Usage:
    PYTHONPATH=. python utils/animate_crowd_paths_moving.py [output.gif]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from src.preference.crowd_path_generator import generate_candidate_paths
import utils.crowd_scenario_demo as csd
from utils.animate_crowd_paths import STYLES, interpolate


def agent_positions(agents0, frame, phases, freqs, radii):
    dx = radii[:, 0] * np.sin(freqs * frame + phases)
    dy = radii[:, 1] * np.cos(freqs * frame * 1.3 + phases)
    return agents0 + np.stack([dx, dy], axis=1)


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/home/nuri2/Desktop/robot_crowd_paths_moving.gif"

    start, goal, agents0 = csd.load_scenario(None)
    candidates_world = generate_candidate_paths(start, goal, agents0, num_steps=5)
    interp = {name: interpolate(candidates_world[name], start) for name in candidates_world}
    n_frames = max(len(v) for v in interp.values())

    rng = np.random.default_rng(3)
    n_agents = len(agents0)
    phases = rng.uniform(0, 2 * np.pi, n_agents)
    freqs = rng.uniform(0.06, 0.12, n_agents)
    radii = rng.uniform(0.6, 1.4, (n_agents, 2))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlim(0, 30)
    ax.set_ylim(6, 17)
    ax.set_aspect("equal")
    ax.set_title("Robot navigating 3 candidate paths around a MOVING 6-person crowd")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)

    ax.scatter(*start, color="black", marker="o", s=60, zorder=5)
    ax.scatter(*goal, color="black", marker="*", s=200, zorder=5, label="goal")
    (crowd_scatter,) = ax.plot([], [], "x", color="red", markersize=11, label="crowd (6, moving)")

    trail_lines, robot_dots = {}, {}
    for name, st in STYLES.items():
        (trail,) = ax.plot([], [], color=st["color"], lw=1.5, alpha=0.5)
        (dot,) = ax.plot([], [], marker=st["marker"], color=st["color"], markersize=12, label=st["label"], linestyle="None")
        trail_lines[name] = trail
        robot_dots[name] = dot
    ax.legend(loc="upper left", fontsize=9)

    def update(frame):
        pos = agent_positions(agents0, frame, phases, freqs, radii)
        crowd_scatter.set_data(pos[:, 0], pos[:, 1])
        for name in interp:
            pts = interp[name]
            idx = min(frame, len(pts) - 1)
            trail_lines[name].set_data(pts[: idx + 1, 0], pts[: idx + 1, 1])
            robot_dots[name].set_data([pts[idx, 0]], [pts[idx, 1]])
        return [crowd_scatter] + list(trail_lines.values()) + list(robot_dots.values())

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=60, blit=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=16))
    print("saved to", out_path)


if __name__ == "__main__":
    main()
