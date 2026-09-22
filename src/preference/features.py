"""Interpretable geometric features of a candidate trajectory, used by the
Bradley-Terry preference scorer (see bradley_terry.py). Kept deliberately simple
and closed-form -- computable from waypoints + target alone, no image features --
so the scorer needs no VLM/backbone access and can be fit from a handful of
pairwise comparisons.

Feature order matters (it is the order fitted weight vectors are indexed by):
    0: path_length            -- sum of consecutive step norms
    1: directness              -- net displacement / path length, in [0, 1]
    2: total_turn_angle_deg    -- sum of absolute heading changes between steps
    3: final_heading_error_deg -- angle between the trajectory's final heading
                                  and the direction straight to the target
    4: lateral_deviation       -- max perpendicular distance from the
                                  straight start->target line
    5: endpoint_dist_to_target -- distance from the trajectory's last waypoint
                                  to the target
    6: target_alignment_cos    -- cosine similarity between the trajectory's
                                  overall displacement and the direction to the
                                  target, in [-1, 1]. Directly mirrors the
                                  angle-based eval metric (mean_angle), unlike
                                  the Euclidean endpoint_dist_to_target above.
"""
import numpy as np

FEATURE_NAMES = [
    "path_length",
    "directness",
    "total_turn_angle_deg",
    "final_heading_error_deg",
    "lateral_deviation",
    "endpoint_dist_to_target",
    "target_alignment_cos",
]
NUM_FEATURES = len(FEATURE_NAMES)


def _angle_between(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_sim = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_sim)))


def extract_features(waypoints, target, fade_radius=2.0):
    """waypoints: (T, 2) array, ego-frame relative positions, oldest->newest.
    target: (2,) array, the goal position in the same ego frame.
    fade_radius: final_heading_error_deg is scaled by min(1, dist_to_target /
    fade_radius) -- 0 right at the target, ramping linearly up to its full raw
    value by the time the endpoint is fade_radius away (and unscaled beyond
    that). Without this, the angle between the (typically several-meter)
    final step and the (typically sub-meter) residual vector still remaining
    to target is numerically unstable -- two nearly-orthogonal vectors of very
    different magnitude can report 60-90 degrees of "heading error" even
    though the trajectory has essentially arrived. This was previously masked
    only because src/preference/crowd_path_generator.py's synthetic
    candidates used to force their endpoint to *exactly* equal target
    (residual = 0); real model-predicted waypoints (e.g.
    utils/preference_integration_demo.py) never land exactly on target
    either, so this instability was likely already silently affecting those
    fits too.
    A *hard* 0/raw cutoff at some tolerance (an earlier version of this fix)
    creates a discontinuity in the score surface right at that radius --
    select_and_refine.py's refine_towards_preference does finite-difference
    gradient ascent on these waypoints, and a step landing near that boundary
    saw a huge spurious slope, causing large, physically meaningless score
    jumps (+30-50) instead of genuine small improvements. The linear ramp above
    is smooth (bounded slope 1/fade_radius everywhere), which fixes that.
    Returns a (NUM_FEATURES,) float array.
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    start = np.zeros(2)
    full = np.vstack([start[None, :], wp])

    steps = np.diff(full, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    path_length = float(step_norms.sum())

    net_disp = float(np.linalg.norm(full[-1] - full[0]))
    directness = net_disp / path_length if path_length > 1e-6 else 0.0

    total_turn = 0.0
    for i in range(1, len(steps)):
        total_turn += _angle_between(steps[i - 1], steps[i])

    final_heading = steps[-1] if len(steps) > 0 and np.linalg.norm(steps[-1]) > 1e-8 else (full[-1] - full[0])
    to_target = target - full[-1]
    dist_to_target = float(np.linalg.norm(to_target))
    raw_final_heading_error = _angle_between(final_heading, to_target) if dist_to_target > 1e-6 else 0.0
    fade = min(1.0, dist_to_target / fade_radius) if fade_radius > 1e-8 else 1.0
    final_heading_error = raw_final_heading_error * fade

    line_vec = target - start
    line_norm = np.linalg.norm(line_vec)
    if line_norm > 1e-6:
        line_dir = line_vec / line_norm
        deviations = []
        for p in full[1:]:
            proj_len = np.dot(p - start, line_dir)
            proj_point = start + proj_len * line_dir
            deviations.append(np.linalg.norm(p - proj_point))
        lateral_deviation = float(max(deviations)) if deviations else 0.0
    else:
        lateral_deviation = 0.0

    endpoint_dist_to_target = dist_to_target

    overall_disp = full[-1] - full[0]
    disp_norm = float(np.linalg.norm(overall_disp))
    if disp_norm <= 1e-8:
        # trajectory never moved -- alignment with the target is genuinely
        # undefined here, not "arrived".
        target_alignment_cos = 0.0
    elif dist_to_target > 1e-6:
        target_alignment_cos = float(np.dot(overall_disp, to_target) / (disp_norm * dist_to_target))
    else:
        # endpoint exactly reaches target: the *best* possible outcome for
        # this feature, not the "else 0.0" this used to fall back to. That
        # wrong default made an exact arrival (e.g.
        # src/preference/crowd_path_generator.py's "direct" candidate, whose
        # endpoint is target by construction) look artificially *worse* than
        # a near-miss on this feature, and Bradley-Terry fitting exploited
        # that artificial discontinuity: comparing "direct" (wrongly 0.0)
        # against near-target candidates (correctly ~1.0) inflated a fitted
        # w_hat['target_alignment_cos'] to 54.5 in
        # utils/crowd_scenario_demo.py, which then blew up refine's score by
        # +53 for a barely-perturbed waypoint set.
        target_alignment_cos = 1.0

    return np.array(
        [
            path_length,
            directness,
            total_turn,
            final_heading_error,
            lateral_deviation,
            endpoint_dist_to_target,
            target_alignment_cos,
        ],
        dtype=np.float64,
    )


def agent_avoidance_feature(waypoints, agent_positions):
    """Placeholder for CrowdNav integration: minimum distance from any waypoint
    to any pedestrian/agent position over the rollout, so a "wide berth" vs
    "short distance" tradeoff (e.g. 6 people blocking a corridor) can be scored.
    Not wired into `extract_features` yet -- CityWalker has no agent position
    labels; append this to FEATURE_NAMES/extract_features once testing moves to
    the CrowdNav simulator, which does expose ground-truth agent positions.

    waypoints: (T, 2) ego-frame trajectory. agent_positions: (num_agents, 2).
    Returns the minimum waypoint-to-agent distance (larger = wider berth).
    """
    wp = np.asarray(waypoints, dtype=np.float64)
    agents = np.asarray(agent_positions, dtype=np.float64)
    if agents.size == 0:
        return float("inf")
    dists = np.linalg.norm(wp[:, None, :] - agents[None, :, :], axis=-1)
    return float(dists.min())


def normalize_features(feats, mean=None, std=None):
    """Z-score normalization so Bradley-Terry weights are comparable across
    features with very different natural scales (degrees vs. meters)."""
    feats = np.asarray(feats, dtype=np.float64)
    if mean is None:
        mean = feats.mean(axis=0)
    if std is None:
        std = feats.std(axis=0)
        std = np.where(std < 1e-6, 1.0, std)
    return (feats - mean) / std, mean, std
