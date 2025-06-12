# Gazebo models and worlds collection

The repository is a ROS package based on the repository [gazebo_models_worlds_collection](https://github.com/leonhartyao/gazebo_models_worlds_collection). It contains models and worlds files for [Gazebo](http://gazebosim.org/), which are collected from several public projects.

## Usage

Clone the repository in the `src` folrder of your catkin workspace
```
git clone https://github.com/macc-n/gazebo_worlds
```

Then you can use `catkin_make` to build:
```
cd ~/catkin_ws
catkin_make
```

You can load the worlds in Gazebo using a launch file with the following configuration:
```
<include file="$(find gazebo_ros)/launch/empty_world.launch">
	<arg name="world_name" value="$(find gazebo_worlds)/gazebo/worlds/office_small.world"/>
	<arg name="debug" value="$(arg debug)"/>
	<arg name="gui" value="$(arg gui)"/>
	<arg name="paused" value="$(arg paused)"/>
	<arg name="use_sim_time" value="$(arg use_sim_time)"/>
	<arg name="headless" value="$(arg headless)"/>
</include>
```

The argument `world_name` indicates which world to load in Gazebo. In the example, the `office_small.world` file is loaded. You can change the file name with the ones in the folder `gazebo/worlds`.

## Preview
Gazebo screenshots are provided in `gazebo/screenshots` to preview the worlds.
