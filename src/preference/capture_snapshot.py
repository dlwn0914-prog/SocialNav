#!/usr/bin/env python3
"""One-shot: capture a single live (robot odom, crowd positions) snapshot from
the running Gazebo+HuNav simulation and save it to JSON, so downstream
candidate-path generation / Bradley-Terry validation can run fully offline
without needing the simulation alive.
"""
import json
import sys

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from people_msgs.msg import People

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/crowd_snapshot.json"


class Snapshotter(Node):
    def __init__(self):
        super().__init__("crowd_snapshotter")
        self.odom = None
        self.people = None
        self.create_subscription(Odometry, "/task_generator_node/jackal/odom", self.on_odom, 10)
        self.create_subscription(People, "/task_generator_node/people", self.on_people, 10)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom = (p.x, p.y)

    def on_people(self, msg):
        self.people = [(p.position.x, p.position.y) for p in msg.people]

    def done(self):
        return self.odom is not None and self.people is not None


def main():
    rclpy.init()
    node = Snapshotter()
    for _ in range(200):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.done():
            break
    if not node.done():
        print("FAILED: did not receive both odom and people within timeout")
        sys.exit(1)
    with open(OUT_PATH, "w") as f:
        json.dump({"robot": node.odom, "agents": node.people}, f)
    print(f"saved robot={node.odom} n_agents={len(node.people)} -> {OUT_PATH}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
