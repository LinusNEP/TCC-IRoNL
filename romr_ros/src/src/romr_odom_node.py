#!/usr/bin/env python
"""
Subscribes:
  /wheel_vel (geometry_msgs/Vector3Stamped)
      vector.x = left wheel linear velocity  (m/s)
      vector.y = right wheel linear velocity (m/s)

Publishes:
  /odom (nav_msgs/Odometry)
  TF:  odom -> base_link
  
"""

import math
import rospy
import tf2_ros
from geometry_msgs.msg import Vector3Stamped, TransformStamped, Quaternion
from nav_msgs.msg import Odometry


def yaw_to_quat(yaw):
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


class RomrOdom(object):
    def __init__(self):
        self.wheel_base = rospy.get_param("~wheel_base", 0.300)   # meters
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.publish_tf = rospy.get_param("~publish_tf", True)

        # Pose state
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0
        self.last_stamp = None

        self.odom_pub = rospy.Publisher("odom", Odometry, queue_size=20)
        self.tf_bcast = tf2_ros.TransformBroadcaster()

        rospy.Subscriber("wheel_vel", Vector3Stamped, self.wheel_cb, queue_size=50)

        rospy.loginfo("romr_odom_node up. wheel_base=%.3f m", self.wheel_base)

    def wheel_cb(self, msg):
        stamp = msg.header.stamp
        if self.last_stamp is None:
            self.last_stamp = stamp
            return

        dt = (stamp - self.last_stamp).to_sec()
        self.last_stamp = stamp
        if dt <= 0.0 or dt > 0.5:
            # Skip garbage dt (clock jump, dropped packets, etc.)
            return

        v_left = msg.vector.x
        v_right = msg.vector.y

        v = 0.5 * (v_left + v_right)
        w = (v_right - v_left) / self.wheel_base

        # Midpoint integration keeps yaw error bounded under turns
        dth = w * dt
        th_mid = self.th + 0.5 * dth
        self.x += v * math.cos(th_mid) * dt
        self.y += v * math.sin(th_mid) * dt
        self.th += dth

        # Wrap yaw to (-pi, pi]
        if self.th > math.pi:
            self.th -= 2.0 * math.pi
        elif self.th <= -math.pi:
            self.th += 2.0 * math.pi

        quat = yaw_to_quat(self.th)

        # --- TF ---
        if self.publish_tf:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = self.odom_frame
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.translation.z = 0.0
            t.transform.rotation = quat
            self.tf_bcast.sendTransform(t)

        # --- Odometry msg ---
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation = quat

        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w

        # Reasonable defaults. Tune these to your robot for nav stack use.
        odom.pose.covariance[0]  = 0.01   # x
        odom.pose.covariance[7]  = 0.01   # y
        odom.pose.covariance[35] = 0.05   # yaw
        odom.twist.covariance[0]  = 0.01
        odom.twist.covariance[35] = 0.05

        self.odom_pub.publish(odom)


def main():
    rospy.init_node("romr_odom_node")
    RomrOdom()
    rospy.spin()


if __name__ == "__main__":
    main()
