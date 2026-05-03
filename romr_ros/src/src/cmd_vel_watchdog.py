#!/usr/bin/env python
"""
Forwards cmd_vel_in -> cmd_vel_out normally, but if no message is received within `timeout` seconds, publishes a zero Twist at 10 Hz until input resumes.
This is important for WiFi teleop: if the phone loses signal, the app crashes, the last non-zero command would otherwise be held by the Arduino and the robot would keep moving.
"""

import rospy
from geometry_msgs.msg import Twist


class Watchdog(object):
    def __init__(self):
        self.timeout = rospy.get_param("~timeout", 0.5)
        self.last_stamp = rospy.Time(0)
        self.last_msg = Twist()

        self.pub = rospy.Publisher("cmd_vel_out", Twist, queue_size=10)
        rospy.Subscriber("cmd_vel_in", Twist, self.cb, queue_size=10)

        # Publish at 10 Hz regardless, so the Arduino always sees a recent cmd
        rospy.Timer(rospy.Duration(0.1), self.tick)
        rospy.loginfo("cmd_vel watchdog up, timeout=%.2fs", self.timeout)

    def cb(self, msg):
        self.last_stamp = rospy.Time.now()
        self.last_msg = msg

    def tick(self, _evt):
        age = (rospy.Time.now() - self.last_stamp).to_sec()
        if age > self.timeout:
            self.pub.publish(Twist())   # zeros
        else:
            self.pub.publish(self.last_msg)


if __name__ == "__main__":
    rospy.init_node("cmd_vel_watchdog")
    Watchdog()
    rospy.spin()
