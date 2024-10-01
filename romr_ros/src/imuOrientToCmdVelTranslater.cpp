/*
  This code computes the linear velocity and rotation speed of o2s based on the orientation of the IMU sensor (P19). At first step, the orientation is received from the "imu/data" topic in the form of a quaternion. Then, in the second step, the quaternions are used to calculate the angular rotation (the roll, pitch and yaw angles) of the sensor from the neutral position.  Based on the roll and pitch values the command velocity of the robot is calculated and published to the cmd_vel topic of ROS.
  
 The code for computation of the roll, pitch and yaw values from the quaternion are based on the code from the following website: https://blog.karatos.in/a?ID=00750-4151b77d-39bf-4067-a2e2-9543486eabc4
 For more information about writing a publisher/subscriber nodes, visit: http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28c%2B%2B%29
*/

#include "ros/ros.h"
#include <sensor_msgs/Imu.h>        
#include <geometry_msgs/Twist.h>    
#include "std_msgs/String.h"
#include <tf/tf.h>                  
#include <sstream>

//defining the variables
double linearVelFactor = 0.3;       
double rotSpeedFactor = 1.5;      

double roll, pitch, yaw;
double linearVelocity, angularRotationSpeed; 
float pi = 3.1415926535897932384626433832795;

geometry_msgs::Twist cmd_vel_msg;   
ros::Publisher cmd_vel_Publisher;


void chatterCallback(const sensor_msgs::Imu& imuSub_msg)
{
  //printing the data of the quaternion in the terminal:
  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Quaternation x: ";
  std::cout << imuSub_msg.orientation.x << std::endl;
  std::cout << "Quaternation y: ";
  std::cout << imuSub_msg.orientation.y << std::endl;
  std::cout << "Quaternation z: ";
  std::cout << imuSub_msg.orientation.z << std::endl;
  std::cout << "Quaternation w: ";
  std::cout << imuSub_msg.orientation.w << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  //converting the quaternion into roll, pitch and yaw angles (in the "unit" rad)
  tf::Quaternion quat;
  tf::quaternionMsgToTF(imuSub_msg.orientation, quat);
  tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);

  //converting the roll, pitch and yaw values from the "unit" rad into the unit degrees 
  roll = (roll * 360)/(2*pi);
  pitch = (pitch * 360)/(2*pi);
  yaw = (yaw * 360)/(2*pi);

  //printing the roll, pitch and yaw values in the terminal
  std::cout << "roll: ";
  std::cout << roll << std::endl;
  std::cout << "pitch: ";
  std::cout << pitch << std::endl;
  std::cout << "yaw: ";
  std::cout << yaw << std::endl;

  //calculating the linear velocity and the rotation speed values based on the angle the sensor is tilted (pitch and roll angles)
  linearVelocity = (linearVelFactor*pitch)/90;             
  angularRotationSpeed = (-1)*(rotSpeedFactor*roll)/90;
  
  //writing the previously calculated linear velocity and rotationspeed values in the message in order to publish the message to the cmd_vel topic (geometry_msgs/Twist)
  cmd_vel_msg.linear.x = linearVelocity;          
  cmd_vel_msg.angular.z = angularRotationSpeed;  

  //publishing the message to the topic  
  cmd_vel_Publisher.publish(cmd_vel_msg);
}

int main(int argc, char **argv)
{
    //printing information to the terminal
    std::cout << "The imuOrientToCmdVelTranslater was started ..." << std::endl;
    std::cout << "The linearVelFactor is: ";
    std::cout << linearVelFactor << std::endl;
    std::cout << "The rotSpeedFactor is: ";
    std::cout << rotSpeedFactor << std::endl;  

    //setting up the node, subscriber and publisher
    ros::init(argc, argv, "imuOrientToCmdVelTranslater_Node");
    ros::NodeHandle nh;
    ros::Subscriber imuOrientationSubscriber = nh.subscribe("imu/data", 1000, chatterCallback);
    cmd_vel_Publisher = nh.advertise<geometry_msgs::Twist>("cmd_vel", 1000);

    ros::spin(); 

    return 0;
}
