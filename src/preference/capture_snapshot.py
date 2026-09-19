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
from geometry_msgs.msg import PoseStamped

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/crowd_snapshot.json"

# goal_pose is only published once the task_generator has assigned a goal to
# the robot, so it may never arrive if the scenario hasn't started a task yet.
# GOAL_REQUIRED=0 lets the snapshot save with robot+agents only (caller then
# supplies GOAL manually, e.g. for a hand-picked corridor scenario).
GOAL_REQUIRED = sys.argv[2] != "0" if len(sys.argv) > 2 else True


class Snapshotter(Node):
    def __init__(self):
        super().__init__("crowd_snapshotter")
        self.odom = None
        self.people = None
        self.goal = None
        self.create_subscription(Odometry, "/task_generator_node/jackal/odom", self.on_odom, 10)
        self.create_subscription(People, "/task_generator_node/people", self.on_people, 10)
        self.create_subscription(PoseStamped, "/task_generator_node/jackal/goal_pose", self.on_goal, 10)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        self.odom = (p.x, p.y)

    def on_people(self, msg):
        self.people = [(p.position.x, p.position.y) for p in msg.people]

    def on_goal(self, msg):
        p = msg.pose.position
        self.goal = (p.x, p.y)

    def done(self):
        have_goal = self.goal is not None or not GOAL_REQUIRED
        return self.odom is not None and self.people is not None and have_goal


def main():
    rclpy.init()
    node = Snapshotter()
    for _ in range(200):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.done():
            break
    if not node.done():
        missing = [n for n, v in [("odom", node.odom), ("people", node.people)] if v is None]
        if GOAL_REQUIRED and node.goal is None:
            missing.append("goal_pose (pass GOAL_REQUIRED=0 as 2nd arg to skip)")
        print(f"FAILED: did not receive {missing} within timeout")
        sys.exit(1)
    with open(OUT_PATH, "w") as f:
        json.dump({"robot": node.odom, "agents": node.people, "goal": node.goal}, f)
    print(f"saved robot={node.odom} goal={node.goal} n_agents={len(node.people)} -> {OUT_PATH}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
