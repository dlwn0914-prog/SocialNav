#!/usr/bin/env python
"""Simple regret vs number of queries (reports/method_regret.tex, Fig. regret),
one panel per setting, from utils/style_preference_benchmark.py results.

Usage:
    PYTHONPATH=. .venv/bin/python utils/plot_regret_curves.py --out reports/fig_regret
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# reference categorical palette, fixed order (dataviz skill)
SERIES = [("EUBO", "BT + EUBO (ours)", "#2a78d6"), ("ActiveQuery", "BT + AQ", "#eb6834"),
          ("Random", "BT + Random", "#1baf7a"), ("DPO", "DPO + Random", "#eda100"),
          ("DPO-EUBO", "DPO + EUBO queries", "#e87ba4")]
PANELS = [("gc_gauss", "Grand Central, dense $\\theta^*$"), ("gc_sparse", "Grand Central, sparse $\\theta^*$"),
          ("eth_gauss", "ETH/UCY, dense $\\theta^*$"), ("eth_sparse", "ETH/UCY, sparse $\\theta^*$")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/codebook/dpo")
    p.add_argument("--out", default="reports/fig_regret")
    args = p.parse_args()

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        r = json.load(open(f"{args.dir}/{key}.json"))
        prior = r["rows"]["human prior (argmax log p)"]["regret"][0]
        ax.axhline(prior, color=MUTED, lw=1, ls=":")
        ax.axhline(0, color=MUTED, lw=0.8)
        for m, label, c in SERIES:
            cu = r["curves"][m]
            q = sorted(int(k) for k in cu)
            mean = [cu[str(k)]["regret"][0] for k in q]
            ax.fill_between(q, [cu[str(k)]["regret"][1] for k in q], [cu[str(k)]["regret"][2] for k in q],
                            color=c, alpha=0.15, lw=0)
            ax.plot(q, mean, color=c, lw=2, marker="o", ms=4, label=label)
        ax.set_title(title, color=INK, fontsize=9)
        ax.set_xticks(q)
        ax.set_xlabel("number of queries $n$")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(q[-1], prior, "human prior", color=MUTED, fontsize=7, ha="right", va="bottom")
    axes[0].set_ylabel("simple regret $\\mathrm{SR}_n$")
    axes[0].set_ylim(0, None)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {args.out}.pdf/.png")


if __name__ == "__main__":
    main()
