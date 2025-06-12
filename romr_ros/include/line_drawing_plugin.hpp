#ifndef LINE_DRAWING_PLUGIN_HPP
#define LINE_DRAWING_PLUGIN_HPP

#include <gazebo/common/Plugin.hh>
#include <gazebo/rendering/rendering.hh>
#include <gazebo/transport/transport.hh>
#include <ignition/math/Vector3.hh>

namespace gazebo
{
  class GZ_PLUGIN_VISIBLE LineDrawingPlugin : public RenderingPlugin
  {
  public:
    LineDrawingPlugin() = default;
    ~LineDrawingPlugin() override = default;
    
    // Called when the plugin is loaded
    void Load(rendering::ScenePtr scene, sdf::ElementPtr sdf) override;

  private:
    // Callback when we receive a new robot pose (if subscribing to a topic)
    void OnPoseMsg(ConstPoseStampedPtr &msg);

    // Internal helper: create the dynamic line
    void CreateLine(rendering::ScenePtr scene);

  private:
    rendering::ScenePtr scene_;
    rendering::VisualPtr visual_;
    rendering::DynamicLines *line_;
    
    // Gazebo transport for subscribing to a topic
    transport::NodePtr node_;
    transport::SubscriberPtr sub_;
  };
}

#endif // LINE_DRAWING_PLUGIN_HPP

