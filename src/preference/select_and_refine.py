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


def candidate_spread(candidates):
    """Per-waypoint-index diversity (std of Euclidean distance from the
    across-candidate centroid, one value per step t) across a list of
    candidate trajectories -- how much they actually varied at each step.
    Used to bound refine_towards_preference's displacement: past this,
    refine is no longer nudging within what was actually observed, it is
    extrapolating into territory none of the candidates expressed."""
    arr = np.stack([np.asarray(c, dtype=np.float64) for c in candidates], axis=0)  # (N, T, 2)
    centroid = arr.mean(axis=0)  # (T, 2)
    dists = np.linalg.norm(arr - centroid[None, :, :], axis=-1)  # (N, T)
    return dists.std(axis=0)  # (T,)


def _clip_displacement(wp, origin, max_step):
    """Clip each waypoint's displacement from `origin` to at most
    `max_step[t]` (per-index), preserving direction -- a soft projection back
    onto the allowed ball rather than rejecting the whole step."""
    disp = wp - origin
    dist = np.linalg.norm(disp, axis=-1, keepdims=True)
    factor = np.minimum(1.0, max_step[:, None] / np.maximum(dist, 1e-9))
    return origin + disp * factor


def select_best(w, candidates, target, agent_positions=None, scale=None):
    """candidates: list of (T, 2) waypoint arrays. Returns (best_index, best_score, all_scores).
    Pass agent_positions when `w` was fit with the crowd-avoidance feature
    (len(w) == 8, see utils/crowd_scenario_demo.py) so scoring uses the same
    feature space it was trained on. Pass `scale` (as returned by
    bradley_terry.fit_preference_weights) when `w` is a *fitted* weight vector
    in normalized space -- omit it (default) for a hand-specified raw-space
    w_true. See bradley_terry.py's score()/fit_preference_weights() docstrings."""
    scores = [bt_score(w, _features(c, target, agent_positions), scale=scale) for c in candidates]
    best_idx = int(np.argmax(scores))
    return best_idx, scores[best_idx], scores


def refine_towards_preference(
    waypoints, target, w, agent_positions=None, num_steps=5, lr=None, eps=1e-3, scale=None,
    candidates=None, bound_scale=2.0,
):
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
    Pass `scale` (as returned by bradley_terry.fit_preference_weights) when
    `w` is a *fitted* weight vector in normalized space -- see select_best().

    candidates: optional list of the alternative trajectories `w` was scored
    against (e.g. the full pool select_best chose among). When given, each
    waypoint's total displacement from its starting position is clipped to
    bound_scale (default 2) std-devs of candidate_spread(candidates) at that
    index -- i.e. refine may only interpolate within the diversity those
    candidates actually expressed, never extrapolate past it. Without this, a
    noisy few-shot `w` (small N, or one feature's weight sign flipped by
    underdetermined fitting) lets backtracking line search keep taking
    "improving" steps indefinitely: caught testing with 3 *real* human
    pairwise judgments in utils/crowd_scenario_demo.py, where an
    unconstrained refine moved the endpoint 3.2m further from the target,
    chasing a spuriously negative min_dist_to_agents weight. Pass
    candidates=None (default) to keep the old unconstrained behavior.
    """
    wp = np.array(waypoints, dtype=np.float64)
    origin = wp.copy()
    if lr is None:
        target_dist = float(np.linalg.norm(np.asarray(target, dtype=np.float64) - wp[0])) if len(wp) else 1.0
        lr = 0.02 * max(target_dist, 1e-3)

    max_step_per_waypoint = None
    if candidates is not None and bound_scale is not None:
        max_step_per_waypoint = bound_scale * candidate_spread(candidates)

    best_wp = wp
    best_score = bt_score(w, _features(best_wp, target, agent_positions), scale=scale)
    step = lr
    for _ in range(num_steps):
        grad = np.zeros_like(best_wp)
        for i in range(best_wp.shape[0]):
            for j in range(best_wp.shape[1]):
                perturbed = best_wp.copy()
                perturbed[i, j] += eps
                s = bt_score(w, _features(perturbed, target, agent_positions), scale=scale)
                grad[i, j] = (s - best_score) / eps
        norm = np.linalg.norm(grad)
        if norm < 1e-8:
            break

        direction = grad / norm
        improved = False
        cur_step = step
        for _ in range(6):
            candidate = best_wp + cur_step * direction
            if max_step_per_waypoint is not None:
                candidate = _clip_displacement(candidate, origin, max_step_per_waypoint)
            cand_score = bt_score(w, _features(candidate, target, agent_positions), scale=scale)
            if cand_score > best_score:
                best_wp, best_score = candidate, cand_score
                step = cur_step  # keep this scale as the starting point next iteration
                improved = True
                break
            cur_step *= 0.5
        if not improved:
            break  # no improving step at any scale tried -- converged
    return best_wp


def select_and_refine(
    w, candidates, target, agent_positions=None, num_refine_steps=5, refine_lr=None, scale=None,
    refine_bound_scale=2.0,
):
    best_idx, best_score, scores = select_best(w, candidates, target, agent_positions=agent_positions, scale=scale)
    refined = refine_towards_preference(
        candidates[best_idx], target, w,
        agent_positions=agent_positions, num_steps=num_refine_steps, lr=refine_lr, scale=scale,
        candidates=candidates, bound_scale=refine_bound_scale,
    )
    return refined, best_idx, scores
