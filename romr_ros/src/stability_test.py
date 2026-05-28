#!/usr/bin/env python
"""
ROMR stability / characterization test.

Sweeps commanded (turning_radius, linear_velocity) combinations and logs:
  - commanded linear and angular velocity
  - measured wheel velocity (from /wheel_vel)
  - pose (from /odom)
  - IMU roll/pitch (optional)

Usage:
  rosrun romr_bringup stability_test.py                  # default sweep
  rosrun romr_bringup stability_test.py --quick          # small sweep for sanity
  rosrun romr_bringup stability_test.py --radii 1.0 2.0 --velocities 0.3 0.5
  rosrun romr_bringup stability_test.py --trial-duration 3.0 --settle 1.0

Safety:
  Ctrl-C triggers controlled stop (zero cmd_vel, disarm).
  Any /odom timeout during a trial aborts the whole test.
  Max linear velocity is clamped to MAX_SAFE_LINEAR_VEL regardless of args.
"""

from __future__ import print_function

import argparse
import csv
import math
import os
import signal
import sys
import time

import rospy
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from tf.transformations import euler_from_quaternion

try:
    from sensor_msgs.msg import Imu
    HAVE_IMU_MSG = True
except ImportError:
    HAVE_IMU_MSG = False


# ---- Safety clamps (do not remove) ----
MAX_SAFE_LINEAR_VEL = 1.0    # m/s; 
MAX_SAFE_ANGULAR_VEL = 2.0   # rad/s


class Recorder(object):

    def __init__(self):
        self.odom = None
        self.wheel = None
        self.imu = None
        self.last_odom_stamp = None

        rospy.Subscriber("/odom", Odometry, self._odom_cb, queue_size=10)
        rospy.Subscriber("/wheel_vel", Vector3Stamped, self._wheel_cb,
                         queue_size=10)
        if HAVE_IMU_MSG:
            rospy.Subscriber("/imu/data", Imu, self._imu_cb, queue_size=10)

    def _odom_cb(self, msg):
        self.odom = msg
        self.last_odom_stamp = rospy.Time.now()

    def _wheel_cb(self, msg):
        self.wheel = msg

    def _imu_cb(self, msg):
        self.imu = msg

    def snapshot(self):
        row = {"t_ros": rospy.Time.now().to_sec()}

        if self.odom is not None:
            p = self.odom.pose.pose.position
            q = self.odom.pose.pose.orientation
            _, _, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
            row.update(dict(
                odom_x=p.x, odom_y=p.y, odom_yaw=yaw,
                odom_vx=self.odom.twist.twist.linear.x,
                odom_wz=self.odom.twist.twist.angular.z,
            ))
        else:
            row.update(dict(odom_x=None, odom_y=None, odom_yaw=None,
                            odom_vx=None, odom_wz=None))

        if self.wheel is not None:
            row["wheel_left"] = self.wheel.vector.x
            row["wheel_right"] = self.wheel.vector.y
        else:
            row["wheel_left"] = None
            row["wheel_right"] = None

        if self.imu is not None:
            q = self.imu.orientation
            if any([q.x, q.y, q.z, q.w]):
                roll, pitch, _ = euler_from_quaternion([q.x, q.y, q.z, q.w])
                row["imu_roll"] = roll
                row["imu_pitch"] = pitch
            else:
                row["imu_roll"] = None
                row["imu_pitch"] = None
        else:
            row["imu_roll"] = None
            row["imu_pitch"] = None

        return row


class StabilityTest(object):
    def __init__(self, args):
        self.args = args
        self.cmd_pub = rospy.Publisher("/cmd_vel_raw", Twist, queue_size=10)
        self.arm_pub = rospy.Publisher("/robot/arm", Bool, queue_size=1,
                                       latch=True)
        self.recorder = Recorder()

        self._stopping = False
        signal.signal(signal.SIGINT, self._on_sigint)

        # Give publishers a moment to register with the master.
        rospy.sleep(1.0)

    def _on_sigint(self, *_):
        rospy.logwarn("Ctrl-C received, stopping cleanly...")
        self._stopping = True

    def arm(self):
        rospy.loginfo("Arming")
        self.arm_pub.publish(Bool(data=True))
        rospy.sleep(0.3)

    def disarm(self):
        rospy.loginfo("Disarming")
        # Send zero cmd_vel first, then disarm
        self.cmd_pub.publish(Twist())
        rospy.sleep(0.1)
        self.arm_pub.publish(Bool(data=False))
        rospy.sleep(0.2)

    def _publish_cmd(self, lin, ang):
        msg = Twist()
        msg.linear.x = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)

    def _check_feedback_alive(self):
        if self.recorder.last_odom_stamp is None:
            return True   # give it a bit more time on first trial
        age = (rospy.Time.now() - self.recorder.last_odom_stamp).to_sec()
        if age > 1.0:
            rospy.logerr("No /odom for %.1fs; aborting test.", age)
            return False
        return True

    def run_trial(self, turning_radius, linear_velocity, trial_idx,
                  payload_kg, cog_m, writer):
        """
        Drive one trial and log time-series data.
        Returns False to abort the whole test.
        """
        if self._stopping:
            return False

        # Derive angular velocity from radius.
        if turning_radius == float("inf"):
            ang = 0.0
        else:
            ang = linear_velocity / turning_radius

        # Safety clamps.
        lin = max(-MAX_SAFE_LINEAR_VEL, min(MAX_SAFE_LINEAR_VEL, linear_velocity))
        ang = max(-MAX_SAFE_ANGULAR_VEL, min(MAX_SAFE_ANGULAR_VEL, ang))
        if (lin, ang) != (linear_velocity, linear_velocity / turning_radius
                          if turning_radius != float("inf") else 0.0):
            rospy.logwarn("Clamped to (lin=%.2f ang=%.2f) for safety",
                          lin, ang)

        rospy.loginfo("Trial #%d: r=%.2fm v=%.2fm/s (ang=%.2frad/s) "
                      "payload=%.1fkg cog=%.2fm",
                      trial_idx, turning_radius, lin, ang, payload_kg, cog_m)

        rate = rospy.Rate(self.args.sample_hz)
        start_t = rospy.Time.now()
        duration = rospy.Duration(self.args.trial_duration)
        settle_end = start_t + rospy.Duration(self.args.settle)

        while not rospy.is_shutdown() and not self._stopping:
            now = rospy.Time.now()
            elapsed = (now - start_t).to_sec()

            if (now - start_t) >= duration:
                break

            # Publish cmd at every sample to keep watchdog happy.
            self._publish_cmd(lin, ang)

            if not self._check_feedback_alive():
                self._publish_cmd(0.0, 0.0)
                return False

            row = self.recorder.snapshot()
            row.update({
                "trial_idx": trial_idx,
                "t_elapsed": elapsed,
                "in_settle": 1 if now < settle_end else 0,
                "cmd_lin": lin,
                "cmd_ang": ang,
                "cmd_radius": turning_radius,
                "payload_kg": payload_kg,
                "cog_m": cog_m,
            })
            writer.writerow(row)
            rate.sleep()

        # Stop between trials
        self._publish_cmd(0.0, 0.0)
        rospy.sleep(self.args.inter_trial_pause)
        return True

    def run(self):
        # Build output path
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.abspath(self.args.output.replace("{ts}", ts))
        rospy.loginfo("Writing CSV to %s", out_path)

        fieldnames = [
            "trial_idx", "t_ros", "t_elapsed", "in_settle",
            "cmd_radius", "cmd_lin", "cmd_ang",
            "payload_kg", "cog_m",
            "odom_x", "odom_y", "odom_yaw", "odom_vx", "odom_wz",
            "wheel_left", "wheel_right",
            "imu_roll", "imu_pitch",
        ]

        with open(out_path, "w") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            self.arm()
            try:
                trial_idx = 0
                for pay in self.args.payloads:
                    for cog in self.args.cogs:
                        for r in self.args.radii:
                            for v in self.args.velocities:
                                for _ in range(self.args.repeats):
                                    trial_idx += 1
                                    ok = self.run_trial(
                                        turning_radius=r,
                                        linear_velocity=v,
                                        trial_idx=trial_idx,
                                        payload_kg=pay,
                                        cog_m=cog,
                                        writer=writer,
                                    )
                                    if not ok:
                                        return
                                    if self._stopping:
                                        return
            finally:
                self.disarm()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", nargs="+", type=float,
                    default=[1.0, 2.0, float("inf")],
                    help="Turning radii (m). Use 'inf' for straight.")
    ap.add_argument("--velocities", nargs="+", type=float,
                    default=[0.2, 0.4, 0.6])
    ap.add_argument("--payloads", nargs="+", type=float, default=[0.0],
                    help="Payload mass in kg (for logging only)")
    ap.add_argument("--cogs", nargs="+", type=float, default=[0.0],
                    help="CoG offset in m (for logging only)")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Trials per (radius, velocity, payload, cog)")
    ap.add_argument("--trial-duration", type=float, default=2.5,
                    help="Seconds of motion per trial")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="Settle time at start of trial (logged with in_settle=1)")
    ap.add_argument("--inter-trial-pause", type=float, default=0.5)
    ap.add_argument("--sample-hz", type=float, default=20.0)
    ap.add_argument("--output", default="romr_stability_{ts}.csv")
    ap.add_argument("--quick", action="store_true",
                    help="Override with tiny sweep for sanity checking")

    args = ap.parse_args(rospy.myargv()[1:])

    if args.quick:
        args.radii = [2.0, float("inf")]
        args.velocities = [0.2, 0.4]
        args.repeats = 1
        args.trial_duration = 2.0

    # Parse 'inf' from string form if someone passed --radii inf
    args.radii = [float("inf") if (isinstance(x, float) and math.isinf(x))
                  else x for x in args.radii]
    return args


def main():
    rospy.init_node("romr_stability_test", anonymous=False)
    args = parse_args()

    rospy.loginfo("Stability test starting")
    rospy.loginfo("  radii:      %s", args.radii)
    rospy.loginfo("  velocities: %s", args.velocities)
    rospy.loginfo("  payloads:   %s", args.payloads)
    rospy.loginfo("  cogs:       %s", args.cogs)
    rospy.loginfo("  repeats:    %d", args.repeats)
    total = (len(args.radii) * len(args.velocities) * len(args.payloads)
             * len(args.cogs) * args.repeats)
    rospy.loginfo("  total trials: %d (~%.1fs each -> %.1f min)",
                  total, args.trial_duration + args.inter_trial_pause,
                  total * (args.trial_duration + args.inter_trial_pause) / 60.0)

    if total > 100 and not args.quick:
        rospy.logwarn("Large sweep requested. Press Ctrl-C in 5s to abort.")
        rospy.sleep(5.0)

    tester = StabilityTest(args)
    tester.run()
    rospy.loginfo("Test complete.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
