"""Select the best candidate trajectory under a fitted preference scorer, and
optionally nudge it a little further toward higher preference score
(numeric-gradient ascent on the feature score) -- the "select, then slightly
adjust" behavior the advisor described, and what the turn-category sanity check
(utils/turn_multimodal_check.py) showed plain best-of-N alone under-delivers on.
"""
import numpy as np

from src.preference.features import extract_features, agent_avoidance_feature
from src.preference.bradley_terry import score as bt_score


def _features(waypoints, target, agent_positions):
    """Base 7-dim features, plus min_dist_to_agents appended when
    agent_positions is given (must match how `w` was fit: bradley_terry weight
    vectors fit on FULL_FEATURE_NAMES in utils/crowd_scenario_demo.py are
    8-dim, so len(w) tells select_best/refine_towards_preference whether an
    8th "clearance" feature is expected)."""
    feats = extract_features(waypoints, target)
    if agent_positions is None:
        return feats
    dist = agent_avoidance_feature(waypoints, agent_positions)
    return np.concatenate([feats, [dist]])


def select_best(w, candidates, target, agent_positions=None):
    """candidates: list of (T, 2) waypoint arrays. Returns (best_index, best_score, all_scores).
    Pass agent_positions when `w` was fit with the crowd-avoidance feature
    (len(w) == 8, see utils/crowd_scenario_demo.py) so scoring uses the same
    feature space it was trained on."""
    scores = [bt_score(w, _features(c, target, agent_positions)) for c in candidates]
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx], scores


def refine_towards_preference(waypoints, target, w, agent_positions=None, num_steps=5, lr=None, eps=1e-3):
    """Numeric-gradient ascent of `w . features(waypoints)` w.r.t. the waypoints
    themselves, a few small steps. This is a cheap stand-in for classifier-guidance
    during the flow ODE (guiding the *sampling process*); here we guide the already
    -sampled discrete waypoints directly, which needs no access to the flow model.

    When agent_positions is given, min_dist_to_agents is recomputed against the
    *current* (perturbed) waypoints each step, so a preference weight on
    clearance actively nudges waypoints away from the crowd -- not just picks
    among fixed candidates.

    lr=None auto-scales the *initial* per-step move distance to ~2% of the
    start->target distance instead of a fixed absolute lr in meters (a fixed
    lr=0.05 is a rounding error on a 20-70m real-world path but wildly too
    large on a sub-meter synthetic one). That initial step is then refined by
    backtracking line search each iteration: a step is only taken if it
    actually increases `w . features(wp)`, halving the step size (up to 6
    times) otherwise, and refinement stops early once no improving step can be
    found. Without this, a naive fixed-size gradient step routinely
    overshoots past the local optimum on these nonlinear (turn-angle,
    direction-cosine) features and *decreases* the score instead -- caught by
    running this against utils/crowd_scenario_demo.py's real captured
    scenarios, where several profiles got negative deltas before this fix.
    Guarantees the returned waypoints never score worse than the input.
    """
    wp = np.array(waypoints, dtype=np.float64)
    if lr is None:
        scale = float(np.linalg.norm(np.asarray(target, dtype=np.float64) - wp[0])) if len(wp) else 1.0
        lr = 0.02 * max(scale, 1e-3)

    best_wp = wp
    best_score = bt_score(w, _features(best_wp, target, agent_positions))
    step = lr
    for _ in range(num_steps):
        grad = np.zeros_like(best_wp)
        for i in range(best_wp.shape[0]):
            for j in range(best_wp.shape[1]):
                perturbed = best_wp.copy()
                perturbed[i, j] += eps
                s = bt_score(w, _features(perturbed, target, agent_positions))
                grad[i, j] = (s - best_score) / eps
        norm = np.linalg.norm(grad)
        if norm < 1e-8:
            break

        direction = grad / norm
        improved = False
        cur_step = step
        for _ in range(6):
            candidate = best_wp + cur_step * direction
            cand_score = bt_score(w, _features(candidate, target, agent_positions))
            if cand_score > best_score:
                best_wp, best_score = candidate, cand_score
                step = cur_step  # keep this scale as the starting point next iteration
                improved = True
                break
            cur_step *= 0.5
        if not improved:
            break  # no improving step at any scale tried -- converged
    return best_wp


def select_and_refine(w, candidates, target, agent_positions=None, num_refine_steps=5, refine_lr=None):
    best_idx, best_score, scores = select_best(w, candidates, target, agent_positions=agent_positions)
    refined = refine_towards_preference(
        candidates[best_idx], target, w,
        agent_positions=agent_positions, num_steps=num_refine_steps, lr=refine_lr,
    )
    return refined, best_idx, scores
