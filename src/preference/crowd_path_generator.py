"""Rule-based candidate trajectory generation for a start->goal robot motion
around a crowd, so the Bradley-Terry preference pipeline can be validated
without needing nav2/the robot to actually execute anything in Gazebo --
matches the advisor's "6 people blocking a corridor" scenario: given real
(or realistic) crowd positions, generate a few named path *styles* and let the
feature extractor + preference scorer discriminate between them.

Coordinates are in the same 2D world frame as the map/crowd positions (not the
ego-relative frame src/preference/features.py's extract_features() expects --
call `to_ego_frame` first).
"""
import numpy as np


def _crowd_centroid_and_spread(agent_positions, start, goal):
    """Only agents near the start->goal corridor should influence detours."""
    agents = np.asarray(agent_positions, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    line = goal - start
    line_len = np.linalg.norm(line)
    line_dir = line / (line_len + 1e-8)
    normal = np.array([-line_dir[1], line_dir[0]])

    rel = agents - start
    along = rel @ line_dir
    across = rel @ normal
    in_corridor = (along > -2.0) & (along < line_len + 2.0)
    if not np.any(in_corridor):
        return None, 0.0, normal
    blocking = agents[in_corridor]
    centroid = blocking.mean(axis=0)
    spread = float(np.max(np.abs(across[in_corridor]))) + 1.0
    return centroid, spread, normal


def generate_candidate_paths(start, goal, agent_positions, num_steps=5):
    """Returns {name: (num_steps, 2) world-frame waypoint array}.

    - "direct": straight line start->goal (shortest, may pass through the crowd)
    - "wide_detour": bulges away from the crowd's centroid by ~2x its spread
    - "moderate_detour": bulges away by ~1x its spread
    """
    start = np.asarray(start, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_steps + 1)[1:]  # exclude t=0 (the start itself)
    direct = start[None, :] + t[:, None] * (goal - start)[None, :]

    centroid, spread, normal = _crowd_centroid_and_spread(agent_positions, start, goal)
    if centroid is None:
        # no crowd in the way: all three styles collapse to ~direct
        return {"direct": direct, "moderate_detour": direct.copy(), "wide_detour": direct.copy()}

    # bulge magnitude peaks at the midpoint (t=0.5), zero at the endpoints,
    # pushed away from the crowd centroid along the corridor normal
    mid_vec = centroid - (start + goal) / 2.0
    side = np.sign(np.dot(mid_vec, normal)) or 1.0
    bulge_profile = np.sin(np.pi * t)  # 0 at t=0/1, 1 at t=0.5

    def bulge(mag):
        return direct - side * mag * bulge_profile[:, None] * normal[None, :]

    return {
        "direct": direct,
        "moderate_detour": bulge(spread * 1.0),
        "wide_detour": bulge(spread * 2.0),
    }


def to_ego_frame(waypoints_world, start):
    """Shift world-frame waypoints so `start` is the origin, matching what
    src/preference/features.py's extract_features()/agent_avoidance_feature()
    expect (ego-frame waypoints + ego-frame agent positions)."""
    return np.asarray(waypoints_world, dtype=np.float64) - np.asarray(start, dtype=np.float64)
