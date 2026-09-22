#!/usr/bin/env python
"""Static (non-animated) version of utils/animate_crowd_paths.py: the same 3
candidate paths (direct/moderate_detour/wide_detour) around
utils/crowd_scenario_demo.py's default 6-person corridor scenario, all drawn
at once as a single PNG for quick reference/slides.

Usage:
    PYTHONPATH=. python utils/plot_crowd_paths.py [output.png]
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preference.crowd_path_generator import generate_candidate_paths
import utils.crowd_scenario_demo as csd
from utils.animate_crowd_paths import STYLES


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "/home/nuri2/Desktop/robot_crowd_paths.png"

    start, goal, agents = csd.load_scenario(None)
    candidates_world = generate_candidate_paths(start, goal, agents, num_steps=5)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(agents[:, 0], agents[:, 1], color="red", marker="x", s=120, zorder=5, label="crowd (6)")
    ax.scatter(*start, color="black", marker="o", s=70, zorder=5, label="start")
    ax.scatter(*goal, color="black", marker="*", s=220, zorder=5, label="goal")

    for name, st in STYLES.items():
        full = np.vstack([start[None, :], candidates_world[name]])
        ax.plot(full[:, 0], full[:, 1], color=st["color"], marker=st["marker"], lw=2.5, markersize=8, label=st["label"])

    ax.set_xlim(0, 30)
    ax.set_ylim(6, 17)
    ax.set_aspect("equal")
    ax.set_title("3 candidate paths around a 6-person crowd (utils/crowd_scenario_demo.py)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print("saved to", out_path)


if __name__ == "__main__":
    main()
