#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist, Point, Quaternion
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler
import tf
import pandas as pd
import math
import matplotlib.pyplot as plt

class RobotMotionTester:
    def __init__(self):
        self.payload_mass = 0.0 # in kg
        self.payload_distance = 0.0 # in meters
        self.center_of_gravity = 0.0 # in meters
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    def move_robot(self, turning_radius, linear_velocity):
        # Set up Twist message with specified turning radius and linear velocity
        cmd_vel_msg = Twist()
        cmd_vel_msg.linear.x = linear_velocity
        cmd_vel_msg.angular.z = linear_velocity / turning_radius

        # Publish cmd_vel message and sleep for a short time
        self.cmd_vel_pub.publish(cmd_vel_msg)
        rospy.sleep(0.5)

    def get_payload_torque(self):
        # Calculate torque due to payload weight
        torque = self.payload_mass * 9.81 * self.payload_distance
        return torque

    def get_cog_torque(self, turning_radius):
        # Calculate torque due to center of gravity offset
        cog_torque = self.payload_mass * 9.81 * self.center_of_gravity * math.sin(math.atan(1/turning_radius))
        return cog_torque

    def get_motion_data(self, turning_radius, linear_velocity):
        # Move the robot and record its motion data
        self.move_robot(turning_radius, linear_velocity)
        position_data = rospy.wait_for_message('/odom', Odometry)
        x_pos = position_data.pose.pose.position.x
        y_pos = position_data.pose.pose.position.y
        z_pos = position_data.pose.pose.position.z
        orientation = position_data.pose.pose.orientation
        quaternion = [orientation.x, orientation.y, orientation.z, orientation.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
        timestamp = rospy.Time.now().to_sec()

        # Calculate payload torque and center of gravity torque
        payload_torque = self.get_payload_torque()
        cog_torque = self.get_cog_torque(turning_radius)

        # Write the motion data to the CSV file
        data = {'timestamp': timestamp, 'turning_radius': turning_radius, 'linear_velocity': linear_velocity,
                'x_pos': x_pos, 'y_pos': y_pos, 'z_pos': z_pos, 'payload_torque': payload_torque, 'cog_torque': cog_torque, 'payload_mass': self.payload_mass, 'center_of_gravity': self.center_of_gravity}
        return data

if __name__ == '__main__':
    # Initialize ROS node and publisher for cmd_vel topic
    rospy.init_node('move_robot_node', anonymous=True)

    # Define range of turning radii, linear velocities, and payload masses to test
    turning_radii = [0.5, 1.0, 1.5, 2.0, 2.5] # in m
    linear_velocities = [0.5, 1.0, 1.5, 2.5] # in m/s
    payload_masses = [17.1, 25, 85] # in kg
    center_of_gravities = [-0.1, 0.0, 0.1]  # in m

    # Create CSV file to save data
    data = {'timestamp': [], 'turning_radius': [], 'linear_velocity': [], 'x_pos': [], 'y_pos': [], 'z_pos': [],
            'payload_mass': [], 'payload_distance': [], 'center_of_gravity': [], 'payload_torque': [], 'cog_torque': []}
    df = pd.DataFrame(data)
    df.to_csv('robot_motion_data.csv', index=False)

    # Loop through all combinations of turning radius, linear velocity, and payload mass
    for radius in turning_radii:
        for velocity in linear_velocities:
            for mass in payload_masses:
               for cog in center_of_gravities:
                # Initialize RobotMotionTester instance and set payload and center of gravity values
                motion_tester = RobotMotionTester()
                motion_tester.payload_mass = mass # in kg
                motion_tester.payload_distance = 0.1 # in meters
                motion_tester.center_of_gravity = cog # in meters

                # Collect data for specified turning radius, linear velocity, and payload mass
                for i in range(5):
                    motion_data = motion_tester.get_motion_data(radius, velocity)
                    motion_data['payload_mass'] = mass
                    motion_data['payload_distance'] = motion_tester.payload_distance
                    motion_data['center_of_gravity'] = motion_tester.center_of_gravity
                    df = df.append(motion_data, ignore_index=True)
                    print(f"Collected data for turning radius {radius}, linear velocity {velocity}, payload mass {mass}, and center_of_gravity {cog}")
                
                # Stop the robot after data collection is complete
                cmd_vel_msg = Twist()
                motion_tester.cmd_vel_pub.publish(cmd_vel_msg)
                rospy.sleep(0)

    # Save motion data to CSV file
    df.to_csv('robot_motion_data.csv', index=False)   
    plt.plot(df['x_pos'], df['y_pos'])
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()
    
    print("Motion data saved to CSV file.")
    
    
