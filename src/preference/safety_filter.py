"""Hard geometric safety filter for the proposed research pipeline (CFM
candidate generation -> safety filtering -> N<=10 A/B personalized
preference scorer): a candidate trajectory is "safe" if no waypoint comes
within the robot's physical collision ellipse of any static or dynamic
obstacle, anywhere along the path.

Uses PRIEST's own calibrated collision radii (static 0.5m, dynamic 0.68m
semi-axis -- see crowd_fm_ws/extract_real_candidates.py's PriestPlanner(...)
call, same values Crowd-FM's own pipeline already uses) as the threshold,
rather than inventing a new cutoff on PRIEST's aggregate goal/smoothness/
collision weighted-sum cost: that cost mixes unrelated terms, so a scalar
threshold on it would need its own separate justification. A pass/fail
geometric distance check needs no new hyperparameter -- it reuses a value
already physically grounded in the robot's footprint + safety margin.

Why filter *before* the N<=10 preference step rather than after: this
keeps "is this candidate safe" (hard constraint) and "do I like this
candidate's style" (soft, personalized) as two separate questions. The
session's earlier active_vs_random_query.py / crowd_citywalker_integration_demo.py
work found that when a path-shape preference and a clearance/safety
preference genuinely trade off within the *same* fit, Bradley-Terry
weight recovery becomes unstable (cosine(w_true, w_hat) collapsed
0.92 -> 0.19 in that experiment). Filtering unsafe candidates out first
means the preference step only ever compares already-safe candidates, so
it never has to learn that trade-off.

Fallback: in a crowded scene, every candidate can violate the ellipse
somewhere (no safe option exists). Returning an empty candidate set would
stall the downstream preference step entirely -- instead this keeps the
single least-unsafe candidate (smallest worst-case violation) rather than
returning nothing.
"""
import numpy as np

STATIC_RADIUS = 0.5    # PRIEST's static_obstacle_semi_{minor,major}_axis
DYNAMIC_RADIUS = 0.68  # PRIEST's dynamic_obstacle_semi_{minor,major}_axis


def _min_clearance_margin(waypoints, static_obs, dynamic_obs, static_radius, dynamic_radius):
    """(clearance - required_radius) minimized over every waypoint x every
    obstacle, for one candidate. >=0 means this candidate never enters
    the safety ellipse of any obstacle anywhere along its path (safe).
    The most negative value is this candidate's worst single violation
    (used both to decide pass/fail and, if nothing passes, to rank
    candidates for the fallback)."""
    wp = np.asarray(waypoints, dtype=np.float64)
    margins = []

    if static_obs is not None and len(static_obs) > 0:
        so = np.asarray(static_obs, dtype=np.float64)
        d = np.linalg.norm(wp[:, None, :] - so[None, :, :], axis=-1)  # (T, S)
        margins.append((d - static_radius).min())

    if dynamic_obs is not None and len(dynamic_obs) > 0:
        do = np.asarray(dynamic_obs, dtype=np.float64)
        d = np.linalg.norm(wp[:, None, :] - do[None, :, :], axis=-1)  # (T, D)
        margins.append((d - dynamic_radius).min())

    if not margins:
        return float("inf")
    return float(min(margins))


def filter_safe_candidates(candidates, static_obs=None, dynamic_obs=None,
                            static_radius=STATIC_RADIUS, dynamic_radius=DYNAMIC_RADIUS):
    """Hard safety filter + crowded-scene fallback.

    candidates: list of (T, 2) waypoint arrays, one per trajectory candidate.
    static_obs / dynamic_obs: (N, 2) obstacle position arrays, or None/empty
        (e.g. padded "no obstacle" rows like Crowd-FM's x=y=1000 convention
        should be filtered out by the caller before passing in here).
    static_radius / dynamic_radius: collision ellipse semi-axes; defaults
        match PRIEST's own values (see module docstring) -- override only
        if a different robot footprint or PRIEST config is in use.

    Returns dict:
      safe_indices: indices into `candidates` that pass the hard filter.
                    If none pass, this is a single-element list holding
                    the least-unsafe index instead (fallback) -- never
                    empty, so the downstream preference step never stalls.
      margins: per-candidate _min_clearance_margin (for logging/plots).
      used_fallback: True iff every candidate violated the filter and the
                     fallback path was taken.
    """
    margins = np.array([
        _min_clearance_margin(c, static_obs, dynamic_obs, static_radius, dynamic_radius)
        for c in candidates
    ])
    safe = np.where(margins >= 0)[0]

    if len(safe) > 0:
        return {"safe_indices": safe.tolist(), "margins": margins.tolist(), "used_fallback": False}

    fallback_idx = int(np.argmax(margins))
    return {"safe_indices": [fallback_idx], "margins": margins.tolist(), "used_fallback": True}
