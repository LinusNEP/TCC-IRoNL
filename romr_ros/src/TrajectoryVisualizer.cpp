#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  class TrajectoryVisualizer : public ModelPlugin
  {
    private: physics::ModelPtr model;
    private: event::ConnectionPtr updateConnection;
    private: ros::NodeHandle *rosNode;
    private: ros::Subscriber poseSubscriber;
    private: std::vector<ignition::math::Vector3d> points;

    public: void Load(physics::ModelPtr _model, sdf::ElementPtr /*_sdf*/)
    {
      // Initialize ROS
      if (!ros::isInitialized())
      {
        int argc = 0;
        char **argv = NULL;
        ros::init(argc, argv, "gazebo_client", ros::init_options::NoSigintHandler);
      }

      // Store the model pointer for convenience
      this->model = _model;

      // Create a new ROS node
      this->rosNode = new ros::NodeHandle("gazebo_client");

      // Subscribe to the robot's pose topic
      this->poseSubscriber = this->rosNode->subscribe<geometry_msgs::PoseStamped>(
          "/amcl_pose", 10, &TrajectoryVisualizer::OnPoseReceived, this);

      // Connect to the update event
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(
          std::bind(&TrajectoryVisualizer::OnUpdate, this));
    }

    public: void OnPoseReceived(const geometry_msgs::PoseStampedConstPtr& msg)
    {
      // Add the received pose to the points vector
      ignition::math::Vector3d point(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
      this->points.push_back(point);
    }

    public: void OnUpdate()
    {
      if (this->points.size() > 1)
      {
        // Draw lines between points
        for (size_t i = 1; i < this->points.size(); ++i)
        {
          this->model->GetWorld()->DrawLine(this->points[i - 1], this->points[i],
                                            ignition::math::Color(1, 0, 0, 1));
        }
      }
    }
  };

  // Register this plugin with the simulator
  GZ_REGISTER_MODEL_PLUGIN(TrajectoryVisualizer)
}

