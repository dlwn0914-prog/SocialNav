#!/usr/bin/env python
"""Static (non-animated) version of utils/animate_crowd_paths_moving.py: the
3 candidate paths around the default corridor scenario, with the moving
crowd's wander trails (8 sampled snapshots per agent) drawn instead of a
single frozen position, so the "moving crowd" aspect is visible in a still
image -- companion to utils/plot_crowd_paths.py's stationary-crowd version.

Usage:
    PYTHONPATH=. python utils/plot_crowd_paths_moving.py [output.png]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preference.crowd_path_generator import generate_candidate_paths
import utils.crowd_scenario_demo as csd
from utils.animate_crowd_paths import STYLES
from utils.animate_crowd_paths_moving import agent_positions


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/home/nuri2/Desktop/robot_crowd_paths_moving.png"

    start, goal, agents0 = csd.load_scenario(None)
    candidates_world = generate_candidate_paths(start, goal, agents0, num_steps=5)

    rng = np.random.default_rng(3)
    n_agents = len(agents0)
    phases = rng.uniform(0, 2 * np.pi, n_agents)
    freqs = rng.uniform(0.06, 0.12, n_agents)
    radii = rng.uniform(0.6, 1.4, (n_agents, 2))

    n_frames = 60
    sample_frames = np.linspace(0, n_frames, 8, dtype=int)
    trails = np.stack(
        [agent_positions(agents0, f, phases, freqs, radii) for f in sample_frames], axis=0
    )  # (T, n_agents, 2)

    fig, ax = plt.subplots(figsize=(9, 6))

    for a in range(n_agents):
        ax.plot(trails[:, a, 0], trails[:, a, 1], color="salmon", lw=1.2, alpha=0.6, zorder=3)
        ax.scatter(trails[:-1, a, 0], trails[:-1, a, 1], color="salmon", marker=".", s=25, alpha=0.5, zorder=3)
        ax.scatter(trails[-1, a, 0], trails[-1, a, 1], color="red", marker="x", s=120, zorder=5)
    ax.scatter([], [], color="red", marker="x", s=120, label="crowd (6, current pos.)")
    ax.plot([], [], color="salmon", lw=1.2, alpha=0.6, label="wander trail")

    ax.scatter(*start, color="black", marker="o", s=70, zorder=5, label="start")
    ax.scatter(*goal, color="black", marker="*", s=220, zorder=5, label="goal")

    for name, st in STYLES.items():
        full = np.vstack([start[None, :], candidates_world[name]])
        ax.plot(full[:, 0], full[:, 1], color=st["color"], marker=st["marker"], lw=2.5, markersize=8, label=st["label"])

    ax.set_xlim(0, 30)
    ax.set_ylim(6, 17)
    ax.set_aspect("equal")
    ax.set_title("3 candidate paths around a MOVING 6-person crowd (wander trails shown)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("saved to", out_path)


if __name__ == "__main__":
    main()
