
/*
  the following code is based on the following code from the ROS website: http://wiki.ros.org/navigation/Tutorials/RobotSetup/Odom
  The code reads the cmd_vel_message (the measured linear velocity and the rotation speed of the robot) and computes the transform form odom to base_footprint based on that data.
*/



#include <ros/ros.h>
//#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>  

//variable needed for the subscription
geometry_msgs::Twist cmd_vel_msg;


//variables needed for the publishing 
ros::Publisher odom_pub;

double x = 0.0;
double y = 0.0;
double th = 0.0;

double vx = 0.0;
double vy = 0.0;
double vth = 0.0;
double dt = 0.0;

ros::Time current_time;
ros::Time last_time;
//variables needed for the publishing- ENDE


void thisNodeCallback(const geometry_msgs::Twist& cmd_vel_msg)
{
  static tf::TransformBroadcaster odom_broadcaster;

  //printing the data to the terminal
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "linear x: ";
  std::cout << cmd_vel_msg.linear.x << std::endl;
  std::cout << "linear y: ";
  std::cout << cmd_vel_msg.linear.y << std::endl;
  std::cout << "linear z: ";
  std::cout << cmd_vel_msg.linear.z << std::endl;
  std::cout << "angular z: ";
  std::cout << cmd_vel_msg.angular.z << std::endl;
  std::cout << "------------------------------------------------" << std::endl;


  current_time = ros::Time::now();

  //reading the velocity valuen from the cmd_vel topic
  vx = cmd_vel_msg.linear.x * cos(th);
  vy = cmd_vel_msg.linear.x * sin(th);
  vth = cmd_vel_msg.angular.z;

  //compute odometry based on the topic cmd_vel
  double dt = (current_time - last_time).toSec();
  double delta_x = vx * dt;
  double delta_y = vy * dt;
  double delta_th = vth * dt;

  x += delta_x;
  y += delta_y;
  th += delta_th;

  //since all odometry is 6DOF we'll need a quaternion created from yaw
  geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(th);

  //first, we'll publish the transform over tf
  geometry_msgs::TransformStamped odom_trans;
  odom_trans.header.stamp = current_time;
  odom_trans.header.frame_id = "odom";
  odom_trans.child_frame_id = "base_link";

  odom_trans.transform.translation.x = x;
  odom_trans.transform.translation.y = y;
  odom_trans.transform.translation.z = 0.0;
  odom_trans.transform.rotation = odom_quat;

  //send the transform
  odom_broadcaster.sendTransform(odom_trans);

  //next, we'll publish the odometry message over ROS
  nav_msgs::Odometry odom;
  odom.header.stamp = current_time;
  odom.header.frame_id = "odom";

  //set the position
  odom.pose.pose.position.x = x;
  odom.pose.pose.position.y = y;
  odom.pose.pose.position.z = 0.0;
  odom.pose.pose.orientation = odom_quat;

  //set the velocity
  odom.child_frame_id = "base_link";
  odom.twist.twist.linear.x = vx;
  odom.twist.twist.linear.y = vy;
  odom.twist.twist.angular.z = vth;

  //publish the message
  odom_pub.publish(odom);

  last_time = current_time;

}


int main(int argc, char** argv){

  std::cout << "odom to robot generator is started..." << std::endl;
  ros::init(argc, argv, "odomtorobottfgenerator");

  ros::NodeHandle nh;
  ros::Subscriber imuOrientationSubscriber = nh.subscribe("measured_vel", 1000, thisNodeCallback);

  odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 1000);  //viell 50 statt 1000 als Wert
  
  current_time = ros::Time::now();
  last_time = ros::Time::now();

  ros::spin(); 

  return 0;

}

