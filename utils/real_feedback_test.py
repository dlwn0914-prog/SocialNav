#!/usr/bin/env python
"""Does the few-shot Bradley-Terry pipeline actually work with a REAL human's
pairwise judgments (not the synthetic/simulated profiles every other script in
this repo uses)? Reuses a cached crowd_citywalker_integration_demo.json run
(candidates + synthetic crowd already generated -- no GPU needed) so a person
can just answer a handful of "A vs B" questions.

Session result (2026-09-22, recorded below as the default --answers so this is
reproducible without re-asking): picked 4 of the item's 10 real candidates via
farthest-point sampling in feature space, asked a human 4 pairwise judgments
among them, fit Bradley-Terry, then asked about a 5th candidate (cand_6) the
scorer picked as best *without ever having been directly compared* -- the
human preferred it too, over their own previously-stated #1 choice. That is
the core validation this repo has been building toward: N=4 real comparisons
generalized correctly to an unseen candidate, not just reproduced the
examples it was fit on.

Caveat found along the way: the fitted w_hat did NOT cleanly track any single
feature (total_turn_angle_deg's weight came out positive -- "more turning
preferred" -- because the 4 real judgments were not monotonic in turn angle:
the smallest-turn candidate was ranked *last*). A real human's judgment is not
a clean linear function of these 3 features even though the overall ranking
it produces still generalizes usefully. Worth reporting honestly rather than
overclaiming interpretability of the fitted weights.

Usage:
    # replay today's recorded session (no input needed):
    PYTHONPATH=. python utils/real_feedback_test.py

    # ask a real person fresh, interactively:
    PYTHONPATH=. python utils/real_feedback_test.py --interactive \
        --cache crowd_citywalker_integration_demo.json --image forward_1224.jpg
"""
import argparse
import json

import numpy as np

import utils.crowd_citywalker_integration_demo as ccd
from src.preference.bradley_terry import fit_preference_weights, score as bt_score

# Exactly what was asked and answered in the 2026-09-22 session (see module
# docstring). Each entry is (winner_label, loser_label) among the 4
# farthest-point-sampled candidates, in the order asked.
SESSION_ANSWERS = [
    ("cand_0", "cand_2"),
    ("cand_0", "cand_4"),
    ("cand_5", "cand_0"),
    ("cand_4", "cand_2"),
]
SESSION_FOLLOWUP = ("cand_6", "cand_5")  # the unseen-candidate generalization check


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="crowd_citywalker_integration_demo.json")
    p.add_argument("--image", default="forward_1224.jpg", help="Substring matching the item's image filename.")
    p.add_argument("--interactive", action="store_true", help="Ask a real person via the terminal instead of replaying SESSION_ANSWERS.")
    return p.parse_args()


def load_item(cache_path, image_substr):
    d = json.load(open(cache_path))
    rec = next(r for r in d["per_item"] if image_substr in r["geometry"]["image"])
    g = rec["geometry"]
    return {
        "target": np.array(g["target"]),
        "agents": np.array(g["agents"]),
        "candidates": [np.array(c) for c in g["all_candidates"]],
        "gt": np.array(g["gt"]),
        "image": g["image"],
    }


def farthest_point_sample(feats, k=4, start=0):
    chosen = [start]
    for _ in range(k - 1):
        dists = np.min([np.linalg.norm(feats - feats[i], axis=1) for i in chosen], axis=0)
        dists[chosen] = -1
        chosen.append(int(np.argmax(dists)))
    return chosen


def ask_pairwise(label_a, feats_a, label_b, feats_b, interactive):
    names_f = ["total_turn_angle_deg", "target_alignment_cos", "min_dist_to_agents(rel)"]
    stats_a = dict(zip(names_f, np.round(feats_a, 2)))
    stats_b = dict(zip(names_f, np.round(feats_b, 2)))
    if not interactive:
        return None  # caller substitutes the recorded session answer
    print(f"\n  {label_a}: {stats_a}")
    print(f"  {label_b}: {stats_b}")
    while True:
        choice = input(f"  Which do you prefer, {label_a} or {label_b}? ").strip()
        if choice in (label_a, label_b):
            return choice
        print(f"  (type exactly '{label_a}' or '{label_b}')")


def main():
    args = parse_args()
    item = load_item(args.cache, args.image)
    candidates, target, agents, gt = item["candidates"], item["target"], item["agents"], item["gt"]

    reach = ccd.candidate_reach_scale(candidates)
    spread = ccd.candidate_spread(candidates)
    feats = [ccd.reduced_features(c, target, agents, reach) for c in candidates]

    chosen_idx = farthest_point_sample(np.array(feats), k=4)
    label_of = {i: f"cand_{i}" for i in chosen_idx}
    print(f">>> item image: {item['image']}")
    print(f">>> 4 farthest-point-sampled candidates: {list(label_of.values())}")

    pairs_to_ask = [(chosen_idx[0], chosen_idx[1]), (chosen_idx[0], chosen_idx[2]),
                     (chosen_idx[3], chosen_idx[0]), (chosen_idx[2], chosen_idx[1])]

    results = []
    for k, (a, b) in enumerate(pairs_to_ask):
        la, lb = label_of[a], label_of[b]
        winner_label = ask_pairwise(la, feats[a], lb, feats[b], args.interactive)
        if winner_label is None:
            winner_label, loser_label = SESSION_ANSWERS[k]
        else:
            loser_label = lb if winner_label == la else la
        results.append((winner_label, loser_label))
    print(">>> pairwise judgments:", results)

    label_to_idx = {v: k for k, v in label_of.items()}
    diff_feats = [feats[label_to_idx[w]] - feats[label_to_idx[l]] for w, l in results]
    w_hat, scale = fit_preference_weights(np.array(diff_feats), np.array([1.0] * len(results)), l2=1.0, lr=0.2, num_steps=800)
    print("\nw_hat (normalized space):", dict(zip(ccd.REDUCED_FEATURE_NAMES, np.round(w_hat, 3))))

    all_scores = {f"cand_{i}": bt_score(w_hat, feats[i], scale=scale) for i in range(len(candidates))}
    ranked = sorted(all_scores, key=lambda n: -all_scores[n])
    print("Fitted-scorer full ranking (all candidates):", ranked)

    best_idx, best_score = ccd.local_select_best(w_hat, candidates, target, agents, reach, scale=scale)
    best_label = f"cand_{best_idx}"
    print(f"\nselect_best picks: {best_label}")

    if best_label not in label_of.values():
        print(f">>> {best_label} was never directly compared -- asking about the generalization")
        if args.interactive:
            # compare against the human's own implied favorite: the judged
            # candidate that lost none of its comparisons.
            losers = {l for _, l in results}
            human_top = next(w for w, _ in results if w not in losers)
            winner = ask_pairwise(best_label, feats[best_idx], human_top, feats[label_to_idx[human_top]], True)
            print(f">>> generalization check: {winner} preferred")
        else:
            winner, loser = SESSION_FOLLOWUP
            print(f">>> replaying session follow-up: {winner} preferred over {loser}")

    max_step = 2.0 * spread
    refined = ccd.local_refine(candidates[best_idx], target, w_hat, agents, reach, num_steps=8, scale=scale, max_step_per_waypoint=max_step)
    refined_score = bt_score(w_hat, ccd.reduced_features(refined, target, agents, reach), scale=scale)
    print(f"\nrefine: score {best_score:.3f} -> {refined_score:.3f}")
    print("endpoint displacement:", round(float(np.linalg.norm(refined[-1] - candidates[best_idx][-1])), 3))


if __name__ == "__main__":
    main()
