/*
 # ROS node to send romr to goal location on a map.

 # 1 = prof_office
 # 2 = secretary_office
 # 3 = phd1_office
 # 4 = phd2_office
 # 5 = technician_office
 # 6 = student_room
 # 7 = workshop
 # 8 = conference_room
 # 9 = resource_room
 # 10 = printer_room
 # 11 = kitchen
 # 12 = passage
 # 13 = elevator
 # 14 = toilet
*/

#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>
#include <geometry_msgs/Twist.h>
#include <iostream>
#include <vector>
 
using namespace std;

// Action specification for move_base
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

int main(int argc, char** argv){

    // Connect to ROS
  ros::init(argc, argv, "simple_navigation_goals");
  
  // Create a ROS node handle
  ros::NodeHandle nh;

    // Create a move_base action client
  MoveBaseClient ac("move_base", true);
  
    // Wait for the action server to come up so that we can begin processing goals.
  while(!ac.waitForServer(ros::Duration(5.0))){
  ROS_INFO("Waiting for the move_base action server to come up");
  }
  
  vector<int> goal_locations; // Stores the selected goal locations
  char wish_to_continue = 'Y';
  bool run = true;

     
  while(run) {
    // Ask the user where he wants the robot to go?
  
  cout << "==================================" << endl;
  cout << " Where do you want the mubot to go?  " << endl;
  cout << "==================================" << endl;
  cout << "1 = Prof_office" << endl;
  cout << "2 = Secretary_office" << endl;
  cout << "3 = PhD1_office" << endl;
  cout << "4 = PhD2_office" << endl;
  cout << "5 = Technician_office" << endl;
  cout << "6 = Student_room" << endl;
  cout << "7 = Workshop" << endl;
  cout << "8 = Conference_room" << endl;
  cout << "9 = Server_room" << endl;
  cout << "10 = Printer_room" << endl;
  cout << "11 = Kitchen" << endl;
  cout << "12 = Passage" << endl;
  cout << "13 = Elevator" << endl;
  cout << "14 = Toilet" << endl; 
  
  cout << "Please select up to four numbers from 0~14 (separated by spaces): ";
        goal_locations.clear(); // Clear the previously selected locations

        int goal_location;
        for (int i = 0; i < 4; ++i) {
            cin >> goal_location;
            goal_locations.push_back(goal_location);
        }

        // Loop through the selected goal locations
        for (int i = 0; i < goal_locations.size(); ++i) {
            int current_location = goal_locations[i];
            // Create a new goal to send to move_base
            move_base_msgs::MoveBaseGoal goal;

            // Set the goal based on the selected location
  switch (current_location) {
    case 1: // Room 1
      cout << "\nGoal Location: Prof_office\n" << endl;
      goal.target_pose.pose.position.x = -1.946;
      goal.target_pose.pose.position.y = -13.070;
      goal.target_pose.pose.position.z = 0.19;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 2: // Room 2
      cout << "\nGoal Location:Secretary_office\n" << endl;
      goal.target_pose.pose.position.x = 4.845;
      goal.target_pose.pose.position.y = -15.308;
      goal.target_pose.pose.position.z = 0.003;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
   case 3:
      cout << "\nGoal Location: PhD1_office\n" << endl;
      goal.target_pose.pose.position.x = 9.744;
      goal.target_pose.pose.position.y = -14.097;
      goal.target_pose.pose.position.z = 0.005;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
     case 4:
      cout << "\nGoal Location: PhD2_office\n" << endl;
      goal.target_pose.pose.position.x = 4.999;
      goal.target_pose.pose.position.y = -2.885;
      goal.target_pose.pose.position.z = 0.003;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 5:
      cout << "\nGoal Location: Technician_office\n" << endl;
      goal.target_pose.pose.position.x = 7.514;
      goal.target_pose.pose.position.y = -15.181;
      goal.target_pose.pose.position.z = 0.001;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 6:
      cout << "\nGoal Location: Student_room\n" << endl;
      goal.target_pose.pose.position.x = 9.720;
      goal.target_pose.pose.position.y = -2.819;
      goal.target_pose.pose.position.z = 0.001;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 7:
      cout << "\nGoal Location: Workshop\n" << endl;
      goal.target_pose.pose.position.x = 7.0147;
      goal.target_pose.pose.position.y = -3.5259;
      goal.target_pose.pose.position.z = 0.001;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 8:
      cout << "\nGoal Location: Conference_room\n" << endl;
      goal.target_pose.pose.position.x = 0.274;
      goal.target_pose.pose.position.y = -4.140;
      goal.target_pose.pose.position.z = 0.002;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 9:
      cout << "\nGoal Location: Server_room\n" << endl;
      goal.target_pose.pose.position.x = -4.2263;
      goal.target_pose.pose.position.y = -2.827;
      goal.target_pose.pose.position.z = -0.001;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 10:
      cout << "\nGoal Location: Printer_room\n" << endl;
      goal.target_pose.pose.position.x = 8.8047;
      goal.target_pose.pose.position.y = -8.9809;
      goal.target_pose.pose.position.z = 0.177;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 11:
      cout << "\nGoal Location: Kitchen\n" << endl;
      goal.target_pose.pose.position.x = -4.314;
      goal.target_pose.pose.position.y = -16.413;
      goal.target_pose.pose.position.z = 0.001;
      goal.target_pose.pose.orientation.w = 1.0;
      break; 
    case 12:
      cout << "\nGoal Location: Passage\n" << endl;
      goal.target_pose.pose.position.x = -7.723;
      goal.target_pose.pose.position.y = 2.589;
      goal.target_pose.pose.position.z = 0.000;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 13:
      cout << "\nGoal Location: Elevator\n" << endl;
      goal.target_pose.pose.position.x = -7.869;
      goal.target_pose.pose.position.y = -15.927;
      goal.target_pose.pose.position.z = 0.000;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
    case 14:
      cout << "\nGoal Location: Toilet\n" << endl;
      goal.target_pose.pose.position.x = -12.083;
      goal.target_pose.pose.position.y = -8.519;
      goal.target_pose.pose.position.z = 0.001;
      goal.target_pose.pose.orientation.w = 1.0;
      break;
   }   
      
     // Send the goal to the robot
            goal.target_pose.header.frame_id = "map";
            goal.target_pose.header.stamp = ros::Time::now();
            ROS_INFO("Sending goal");
            ac.sendGoal(goal);

            // Wait until the robot reaches the goal
            ac.waitForResult();

            if (ac.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
                ROS_INFO("Successfully arrived at the goal location");
            else
                ROS_INFO("Failed to reach the goal location");
        }

        // Ask if the user wants to continue to another goal
        do {
            cout << "\nDo you wish to go to another goal location? (Y/N)" << endl;
            cin >> wish_to_continue;
            wish_to_continue = tolower(wish_to_continue);
        } while (wish_to_continue != 'n' && wish_to_continue != 'y');

        if (wish_to_continue == 'n') {
            run = false;
        }
    }

    return 0;
}

