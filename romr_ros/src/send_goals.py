#!/usr/bin/env python
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal


def send_goal(goal_location):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()

    switch (goal_location) {
        case 1: // Room 1
            goal.target_pose.pose.position.x = -1.946;
            goal.target_pose.pose.position.y = -13.070;
            goal.target_pose.pose.position.z = 0.19;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 2: // Room 2
            goal.target_pose.pose.position.x = 4.845;
            goal.target_pose.pose.position.y = -15.308;
            goal.target_pose.pose.position.z = 0.003;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 3:
            goal.target_pose.pose.position.x = 9.744;
            goal.target_pose.pose.position.y = -14.097;
            goal.target_pose.pose.position.z = 0.005;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 4:
            goal.target_pose.pose.position.x = 4.999;
            goal.target_pose.pose.position.y = -2.885;
            goal.target_pose.pose.position.z = 0.003;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 5:
            goal.target_pose.pose.position.x = 7.514;
            goal.target_pose.pose.position.y = -15.181;
            goal.target_pose.pose.position.z = 0.001;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 6:
            goal.target_pose.pose.position.x = 9.720;
            goal.target_pose.pose.position.y = -2.819;
            goal.target_pose.pose.position.z = 0.001;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 7:
            goal.target_pose.pose.position.x = 7.0147;
            goal.target_pose.pose.position.y = -3.5259;
            goal.target_pose.pose.position.z = 0.001;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 8:
            goal.target_pose.pose.position.x = 0.274;
            goal.target_pose.pose.position.y = -4.140;
            goal.target_pose.pose.position.z = 0.002;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 9:
            goal.target_pose.pose.position.x = -4.2263;
            goal.target_pose.pose.position.y = -2.827;
            goal.target_pose.pose.position.z = -0.001;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 10:
            goal.target_pose.pose.position.x = 8.8047;
            goal.target_pose.pose.position.y = -8.9809;
            goal.target_pose.pose.position.z = 0.177
        case 11:
            goal.target_pose.pose.position.x = -4.314;
            goal.target_pose.pose.position.y = -16.413;
            goal.target_pose.pose.position.z = 0.001;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 12:
            goal.target_pose.pose.position.x = -7.723;
            goal.target_pose.pose.position.y = 2.589;
            goal.target_pose.pose.position.z = 0.000;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 13:
            goal.target_pose.pose.position.x = -7.869;
            goal.target_pose.pose.position.y = -15.927;
            goal.target_pose.pose.position.z = 0.000;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
        case 14:
            goal.target_pose.pose.position.x = -12.083;
            goal.target_pose.pose.position.y = -8.519;
            goal.target_pose.pose.position.z = 0.001;
            goal.target_pose.pose.orientation.w = 1.0;
            break;
    }

    return goal


def main():
    rospy.init_node("simple_navigation_goals")
    ac = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    ac.wait_for_server()

    while True:
        goal_location = int(input("Enter the goal location (1-10): "))
        goal = send_goal(goal_location)

        ac.send_goal(goal)
        ac.wait_for_result()

        if ac.get_state() == actionlib.SimpleClientGoalState.SUCCEEDED:
            print("Successfully reached the goal location")
        else:
            print("Failed to reach the goal location")

        do_again = input("Do you want to go to another goal location? (Y/N): ")
        do_again = do_again.lower()
        if do_again != "y":
            break


if __name__ == "__main__":
    main()

