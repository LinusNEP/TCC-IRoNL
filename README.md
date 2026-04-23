# The Conversation is the Command: Interacting with Real-World Autonomous Robots through Natural Language (TCC-IRoNL)

[![TCC IRoNL](https://img.shields.io/badge/TCC%20IRoNL-Website-lightblue?style=flat&logo=globe&logoColor=white)](https://linusnep.github.io/TCC-IRoNL/)
[![ROS 1](https://img.shields.io/badge/ROS-Noetic-brightgreen.svg)](http://www.ros.org/)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://index.ros.org/doc/ros2/)
[![Python](https://img.shields.io/badge/Python-≥3.8-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![GitHub Stars](https://img.shields.io/github/stars/LinusNEP/TCC-IRoNL?style=social)](https://github.com/LinusNEP/TCC-IRoNL/stargazers)
[![Last Commit](https://img.shields.io/github/last-commit/LinusNEP/TCC-IRoNL.svg)](https://github.com/LinusNEP/TCC-IRoNL/commits)

<p align="center">
  <a href="https://doi.org/10.1145/3610978.3640723">Paper</a> •
  <a href="https://arxiv.org/abs/2401.11838">ArXiv</a> •
  <a href="https://linusnep.github.io/TCC-IRoNL/">Project Website</a> •
  <a href="https://osf.io/cmbw6/">Data &amp; Videos</a> •
  <a href="https://creativecommons.org/licenses/by/4.0/">License</a>
</p>

<div align="center">
  <img src="https://github.com/LinusNEP/TCC-IRoNL/blob/main/Figures/gifAnimation1.gif" width="400px" alt="Simulation demo" />
  <img src="https://github.com/LinusNEP/TCC-IRoNL/blob/main/Figures/real_world_optimize.gif" width="400px" alt="Real-world demo" />
</div>

> **Status:** This work is in active development, expect occasional bugs. Bug reports and contributions via GitHub Issues are very welcome.

## Contents

- [Overview](#overview)
- [Updates since publication](#updates-since-publication)
- [Citation](#citation)
- [Installation](#installation)
- [Running the Example Demos](#running-the-example-demos)
  - [Simulation](#simulation)
  - [Real-world robot](#real-world-robot)
  - [Running TCC-IRoNL on your own robot](#running-tcc-ironl-on-your-own-robot)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

TCC-IRoNL is a framework that combines pre-trained large language models (LLMs) with a multimodal vision-language model (VLM) to enable humans to interact naturally with autonomous robots through conversational dialogue. It utilises the LLM to decode high-level natural-language instructions and abstract them into precise robot-actionable commands or queries, and the VLM to provide visual and semantic understanding of the robot's task environment. See the [paper](https://dl.acm.org/doi/10.1145/3610978.3640723) for full details.

## Updates since publication

In the published paper, we built on YOLO and GPT-2. Since then, the repository has been updated to use:

- **Vision:** [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) in place of YOLO
- **Language models:** the latest GPT models (e.g. `gpt-4o`, `gpt-4o-mini`), `deepseek-chat`, and `llama-2-7b-chat` in place of GPT-2

You will therefore need an API key for your chosen LLM provider and the SAM pre-trained weights, see [Installation](#installation).

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{10.1145/3610978.3640723,
author = {Linus, Nwankwo and Elmar, Rueckert},
title = {The Conversation is the Command: Interacting with Real-World Autonomous Robot Through Natural Language},
year = {2024},
isbn = {979-8-4007-0323-2/24/03},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3610978.3640723},
doi = {10.1145/3610978.3640723},
booktitle = {Companion of the 2024 ACM/IEEE International Conference on Human-Robot Interaction},
numpages = {5},
keywords = {Human-robot interaction, LLMs, VLMs, ChatGPT, ROS, autonomous robots, natural language interaction},
location = {Boulder, CO, USA},
series = {HRI '24}
}
```

## Installation

### Requirements

- Ubuntu with ROS Noetic (primary target). ROS Melodic has been tested inside a Docker environment. For ROS 2, a `ros1_bridge` is required to forward the relevant topics.
- Python 3.8 or above
- NVIDIA GPU with CUDA (required for SAM and local LLM inference)

### 1. Install ROS and the navigation stack

TCC-IRoNL works with any ROS-based mobile robot that publishes the standard ROS topics listed in [Real-world robot](#real-world-robot). Install ROS by following the [ROS Wiki instructions](http://wiki.ros.org/ROS/Installation). You also need the ROS navigation planner and its dependencies, install them with the provided script:

```bash
./planner_dependencies.sh
```

### 2. Create a ROS workspace and clone the repository

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/LinusNEP/TCC-IRoNL.git
```

### 3. Install TCC-IRoNL dependencies

```bash
cd TCC-IRoNL/
mv install_TCC-IRoNL_deps.sh ~/catkin_ws/
cd ~/catkin_ws
bash install_TCC-IRoNL_deps.sh
```

The dependency script creates a Python virtual environment (`TCC-IRoNLEnv`) that you will activate in the terminals that run the LLM/VLM nodes.

### 4. Build the workspace

```bash
cd ~/catkin_ws
catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 5. Download the SAM weights

Download the SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`) from the [Segment Anything model-checkpoints page](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and place it at:

```
~/catkin_ws/src/TCC-IRoNL/tcc_ros/src/tcc_ros/sam_vit_b_01ec64.pth
```

Without this file, `roslaunch tcc_ros tcc_ros.launch` will fail to start.

### 6. Set your LLM API key

Export the API key for your chosen provider before launching the LLM node:

```bash
export API_KEY="your_api_key_here"
```

To make it persistent, add the line to your `~/.bashrc`.

## Running the Example Demos

Open four terminals (T1–T4) in your workspace directory. **Source the workspace in every terminal** before running the commands below:

```bash
source ~/catkin_ws/devel/setup.bash
```

In **T3** and **T4**, also activate the Python virtual environment created during installation:

```bash
source TCC-IRoNLEnv/bin/activate
```

### Simulation

#### Quadruped robot (Unitree Go1)

**T1 — Gazebo world:**
```bash
roslaunch unitree_gazebo sim_bringup.launch rname:=go1 wname:=cps_world rviz:=false
```

**T2 — Navigation stack:**
```bash
roslaunch unitree_navigation navigation.launch rname:=go1 rviz:=true
```

After launching T1 and T2 the robot will be lying on the floor of the Gazebo world. In the T1 terminal, press:

- `2` to switch the robot from **Passive** (initial state) to **FixedStand**
- `5` to switch from **FixedStand** to **MoveBase**

The robot is now ready to receive navigation commands.

#### Wheeled robot (ROMR)

**T1 — Navigation:**
```bash
roslaunch romr_ros romr_navigation.launch
```

No state-switching is needed for the wheeled robot. Proceed directly to T3 and T4.

#### T3 and T4 — TCC-IRoNL nodes (both robot types)

**T3 — LLM/VLM backend:**
```bash
roslaunch tcc_ros tcc_ros.launch
```

**T4 — Chat GUI:**
```bash
roslaunch tcc_ros chatGUI_SR.launch
```

A menu will appear allowing you to send either textual or vocal (audio) instructions. You can now interact with the simulated robot through natural language.

**Example non-goal-directed commands:**

- *"Move forward 1.5 m at 0.2 m/s."*
- *"Move backwards 2 m and thereafter move in a circular pattern of diameter 2 m."*

**Example general queries:**

- *"Tell me about your capabilities."*
- *"Report your current orientation."*

**Example goal-directed commands:**

- *"Go between the Secretary's office and the kitchen twice."*
- *"Head to the location (0, 3, 0) and return to your start location afterwards."*

Use the environment layout below as a reference, or adapt your own Gazebo world by adding the location names and coordinates to `config.yaml`.

<p align="center">
  <img src="https://github.com/LinusNEP/TCC-IRoNL/blob/main/Figures/gazebo-world.jpg" alt="Gazebo world layout" />
</p>

### Real-world robot

Launch your robot and ensure that the ROS topics listed below — and the parameters in `config.yaml` — are available. Simple movement commands and perception queries (e.g. *"move forward"*, *"what can you see around you?"*, *"where are you now?"*) work out of the box. Goal-directed navigation to a named location (e.g. *"head to location x, y, z"*) requires you to add the approximate `x, y, z` coordinates of the target to `config.yaml`; you can obtain these from LiDAR point-cloud data.

**Required topics:**

| Topic                           | Publisher          | Subscribers       | Description                                   | Msg type                                                                                         |
|---------------------------------|--------------------|-------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------|
| `/odom`                         | REM                | MoveBase, LLMNode | Robot's odometry data                         | [nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)           |
| `/cmd_vel`                      | MoveBase, LLMNode  | REM               | Robot's command velocity data                 | [geometry_msgs/Twist](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Twist.html)       |
| `/clip_node/recognized_objects` | CLIPNode           | LLMNode           | CLIPNode object descriptions                  | [std_msgs/String](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html)               |
| `/llm_input`                    | ChatGUI            | LLMNode           | User's input commands, queries, and tasks    | [std_msgs/String](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html)               |
| `/llm_output`                   | LLMNode            | ChatGUI           | LLMNode's interpretation of the input command | [std_msgs/String](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html)               |
| `/depth/image`, `/rgb/image`    | Observation source | CLIPNode, LLMNode | Image stream from RGB-D camera                | [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)           |
| `/depth/points`                 | Observation source | LLMNode           | Point cloud from 3D LiDAR or RGB-D camera     | [sensor_msgs/PointCloud2](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/PointCloud2.html) |

**Tested observation sources:**

- Ouster 3D LiDAR — `sensor_msgs/PointCloud2`, `sensor_msgs/LaserScan`
- Intel RealSense D435i — `sensor_msgs/Image`, `sensor_msgs/PointCloud2`

**Frames:**

- Robot base frame: `base_link`
- Map frame: `map`

### Running TCC-IRoNL on your own robot

Configure your robot to publish/subscribe to the topics in the table above, then launch **T3** and **T4** as in the simulation section. Update `config.yaml` with the spatial coordinates of your task environment. Non-goal-directed commands and queries that do not require obstacle avoidance or path planning need no additional configuration.

## License

[![Creative Commons licenses 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by.png)](https://creativecommons.org/licenses/by/4.0/)

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgements

Thanks to the following repositories, which this project builds on:

- [Unitree_ros](https://github.com/macc-n/ros_unitree.git)
- [ROMR_ros](https://github.com/LinusNEP/ROMR.git)
- [Ultralytics_ros](https://github.com/Alpaca-zip/ultralytics_ros.git)
