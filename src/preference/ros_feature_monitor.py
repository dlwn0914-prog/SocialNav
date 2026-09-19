#!/usr/bin/env python3
"""Live integration test: subscribe to the running Gazebo+HuNav simulation
(arena-rosnav) and run src/preference/features.py's feature extraction on real
simulated data, instead of synthetic candidates or precomputed CityWalker data.

Confirms the Bradley-Terry preference pipeline's feature extraction works against
a live robot (/task_generator_node/jackal/odom, nav_msgs/Odometry) and a live crowd
(/task_generator_node/people, hunav_msgs/Agents) -- no camera/image topics needed,
since every feature here is geometric (positions only).

Run inside the sourced arena4_ws environment (needs rclpy + hunav_msgs), e.g.:
    source /opt/ros/jazzy/setup.bash
    source /home/nuri2/arena4_ws/install/setup.bash
    PYTHONPATH=/home/nuri2/SocialNav:$PYTHONPATH python3 src/preference/ros_feature_monitor.py
"""
import collections
import sys

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from people_msgs.msg import People

sys.path.insert(0, "/home/nuri2/SocialNav")
from src.preference.features import extract_features, agent_avoidance_feature, FEATURE_NAMES  # noqa: E402

ODOM_TOPIC = "/task_generator_node/jackal/odom"
PEOPLE_TOPIC = "/task_generator_node/people"
WINDOW_SIZE = 5  # number of recent odom samples to treat as a "trajectory" for features()


class FeatureMonitor(Node):
    def __init__(self):
        super().__init__("preference_feature_monitor")
        self.robot_history = collections.deque(maxlen=WINDOW_SIZE)
        self.latest_agents = []

        self.create_subscription(Odometry, ODOM_TOPIC, self.on_odom, 10)
        self.create_subscription(People, PEOPLE_TOPIC, self.on_people, 10)
        self.create_timer(2.0, self.on_timer)

        self.get_logger().info(f"Subscribed to {ODOM_TOPIC} and {PEOPLE_TOPIC}")

    def on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.robot_history.append((p.x, p.y))

    def on_people(self, msg: People):
        self.latest_agents = [(p.position.x, p.position.y) for p in msg.people]

    def on_timer(self):
        if len(self.robot_history) < 2:
            self.get_logger().info("waiting for odom samples...")
            return

        origin = np.array(self.robot_history[0])
        waypoints = np.array([np.array(p) - origin for p in list(self.robot_history)[1:]])
        target = waypoints[-1] if len(waypoints) > 0 else np.zeros(2)

        feats = extract_features(waypoints, target)
        min_dist = agent_avoidance_feature(waypoints, self.latest_agents) if self.latest_agents else float("nan")

        self.get_logger().info(
            f"n_agents={len(self.latest_agents)} min_dist_to_agent={min_dist:.2f}m | "
            + ", ".join(f"{n}={v:.3f}" for n, v in zip(FEATURE_NAMES, feats))
        )


def main():
    rclpy.init()
    node = FeatureMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
