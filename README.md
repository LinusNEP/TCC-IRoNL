<p align="center">
  <h2 align="center">The Conversation is the Command: Interacting with Real-World Autonomous Robots through Natural Language (TCC-IRoNL)</h2>
</p>
  
<p align="center">
  <h3 align="center"> | <a href="https://doi.org/10.1145/3610978.3640723">Paper</a> | <a href="https://arxiv.org/abs/2401.11838">ArXiv</a> | <a href="https://linusnep.github.io/TCC-IRoNL/">Project Website</a> | <a href="https://osf.io/cmbw6/">Data & Videos</a> | <a href="https://creativecommons.org/licenses/by/4.0/">License</a> | </h3>
  <div align="center"></div>
</p>

|                                                                                                   |                                                                                                         |
|---------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
|<img src="https://github.com/LinusNEP/TCC_IRoNL/blob/main/Figures/gifAnimation1.gif" width="500px"> | <img src="https://github.com/LinusNEP/TCC_IRoNL/blob/main/Figures/real_world_optimize.gif" width="500px">|

## Contents

- [TCC-IRoNL](https://github.com/LinusNEP/TCC-IRoNL#TCC-IRoNL)
- [Citation](https://github.com/LinusNEP/TCC-IRoNL#Citation)
- [TCC-IRoNL installation](https://github.com/LinusNEP/TCC-IRoNL#TCC-IRoNL-installation)
- [Run TCC-IRoNL Example Demos](https://github.com/LinusNEP/TCC-IRoNL#Run-TCC-IRoNL-Example-Demos)
- [License](https://github.com/LinusNEP/TCC-IRoNL#License)
- [Acknowledgement](https://github.com/LinusNEP/TCC-IRoNL#Acknowledgement)

## TCC-IRoNL
TCC-IRoNL is a framework that synergically exploits the capabilities of pre-trained large language models (LLMs) and a multimodal vision-language model (VLM) to enable humans to interact naturally with autonomous robots through conversational dialogue. It leverages the LLMs to decode the high-level natural language instructions from humans and abstract them into precise robot-actionable commands or queries. Further, it utilised the VLM to provide a visual and semantic understanding of the robot’s task environment. Refer to the [paper here](https://dl.acm.org/doi/10.1145/3610978.3640723) for more details.

## Citation
If you use this work in your research, please cite it using the following BibTeX entry:
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

## TCC-IRoNL installation
The following instructions are necessary to set up TCC-IRoNL. Please note that CUDA and Python 3.8 or above are required.

**1.  Install ROS and the navigation planner:**

TCC-IRoNL can work with any ROS-based mobile robot publishing standard ROS topics. The whole framework is implemented using ROS Noetic. It was also tested using ROS Melodic in a Docker environment. For ROS2, you will need a ROS bridge to bridge the ROS2 topics. To install ROS, follow the instructions at the [ROS Wiki](http://wiki.ros.org/ROS/Installation). You will need to install or ensure that you have the ROS navigation planner and its dependencies installed. Install the navigation planner and the dependencies by running the `./planner_dependencies.sh` script. After successful installation, follow the next steps to install TCC-IRoNL.

**Create a ROS workspace:**
 ```bash
  mkdir -p ~/catkin_ws/src
  cd ~/catkin_ws/src
  ```
 **Clone the TCC-IRoNL repository to your workspace:**
```bash
git clone https://github.com/LinusNEP/TCC-IRoNL.git
```
 **2.  Install TCC-IRoNL dependencies:**
```bash
cd TCC-IRoNL/
mv install_TCC-IRoNL_deps.sh ~/catkin_ws/
cd ~/catkin_ws
bash install_TCC-IRoNL_deps.sh
```
**Build the workspace:**
```bash
catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
source devel/setup.bash
```

## Run TCC-IRoNL Example Demos
### Simulation
Open four terminal windows (T1-T4) in your workspace directory and run the following:

**T1 - T2 (if quadruped robot):**

First, make sure to source all the opened terminals `source devel/setup.bash`.

**T1:** 
```bash
source devel/setup.bash
roslaunch unitree_gazebo sim_bringup.launch rname:=go1 wname:=cps_world rviz:=false
```
**T2:**
```bash
source devel/setup.bash
roslaunch unitree_navigation navigation.launch rname:=go1 rviz:=true
```
After running T1 -T2 above, the robot will lie on the floor of the Gazebo world. At the terminal where you ran `roslaunch unitree_gazebo sim_bringup.launch rname:=go1 wname:=cps_world rviz:=false`, press the key '2' on the keyboard to switch the robot's state from Passive(initial state) to FixedStand. After, you can press the '5' key to switch from FixedStand to MoveBase. At this point, the robot is ready to receive navigation commands.

**T1 (if wheeled robot):**
```bash
source devel/setup.bash
roslaunch romr_ros romr_navigation.launch
```
For the wheeled robot, you do not need to switch states. After launching `roslaunch romr_ros romr_navigation.launch`, execute T3 - T4 below, and start interacting with the robot.

**T3 - T4:**

Ensure that the virtual environment that was created after installing the TCC-IRoNL and its dependencies is activated `source TCC-IRoNLEnv/bin/activate` in each of T3 - T4. Upon running `roslaunch tcc_ros chatGUI_SR.launch` below, a menu will appear, allowing you to input either textual or vocal (audio) instructions. Note, we have updated both the visual perception node and the large language models (LLMs) to utilise the Segment Anything Model (SAM), as well as the latest GPTs (gpt-40, gpt-40-mini, etc.), deepseek-chat, and llama-2-7b-chat. This is a change from the models (YOLO, GPT-2, ...) mentioned in the original paper. Therefore, you will need to export your LLM API key to ensure compatibility with the updated models: `export API_KEY="your_api_key_here"`. Further, it is also required to include the SAM pre-trained weights (`sam_vit_b_01ec64.pth`) at `<your_workspace_path>/TCC-IRoNL/tcc_ros/src/tcc_ros/sam_vit_b_01ec64.pth`, otherwise `roslaunch tcc_ros tcc_ros.launch` will throw an error. You can download the checkpoint [here](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints).

**T3:**
```bash
source devel/setup.bash
roslaunch tcc_ros tcc_ros.launch
```
**T4:**
```bash
source devel/setup.bash
roslaunch tcc_ros chatGUI_SR.launch
```
Interact with the simulated robot through natural language. You can start with non-goal-directed commands like *"Move forward 1.5m at 0.2m/s", "Move backwards 2m and thereafter move in a circular pattern of diameter 2m", etc*. You could also ask general questions or retrieve data from the robot, e.g., *"Tell me about your capabilities", "Report your current orientation", etc*. For goal-directed commands, e.g., *Go between the Secretary's office and the kitchen twice, Head to the location (0,3,0) and return to your start location afterwards, etc*, use the following environment layout as a reference. You could also adapt your own Gazebo world. For this, you will have to define the location's name in the configuration file (`config.yaml`).

 <img src="https://github.com/LinusNEP/TCC_IRoNL/blob/main/Figures/gazebo-world.jpg">

### Real-World Robot
Launch your robot! Ensure that the ROS topics and parametric configurations in the table below and `config.yaml` are available. Sending custom movement commands and queries such as "move forward, backwards, right, what can you see around you? where are you now? etc" may not require further configuration. However, sending goal navigation tasks other than specified coordinates, e.g., head to the location x,y,z or (x,y,z), would require you to update the config file with the approximate `x, y, z` coordinates of the task location. You can obtain such coordinate information from LiDAR point cloud data.
- Configurations: 
  | Topics                         | Publisher           | Subscribers                 | Description                                   | Msg Type                     |
  |--------------------------------|---------------------|-----------------------------|-----------------------------------------------|------------------------------|
  | `/odom`                        | REM                 | MoveBase, LLMNode           | Robot's odometry data                         | [nav_msgs/Odometry](http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/Odometry.html)|
  | `/cmd_vel`                     | MoveBase, LLMNode   | REM                         | Robot's command velocity data                 | [geometry_msgs/Twist](http://docs.ros.org/en/noetic/api/geometry_msgs/html/msg/Twist.html)|
  | `/clip_node/recognized_objects`| CLIPNode            | LLMNode                     | CLIPNode objects descriptions                  | [std_msgs/String](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html)|
  | `/llm_input`                   | ChatGUI             | LLMNode                     | User's input commands, queries and tasks      | [std_msgs/String](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html)|
  | `/llm_output`                  | LLMNode             | ChatGUI                     | LLMNode's interpretation of the input command | [std_msgs/String](http://docs.ros.org/en/noetic/api/std_msgs/html/msg/String.html)|
  | `/depth/image`, `/rgb/image`*  | Observation Source  | CLIPNode, LLMNode           | Image stream from RGB-D camera                | [sensor_msgs/Image](http://docs.ros.org/en/noetic/api/sensor_msgs/html/msg/Image.html)|
  | `/depth/points`                | Observation Source  | LLMNode                     | Point cloud from 3D LiDAR or RGB-D camera     | [sensor_msgs/PointCloud2](http://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html)|
  
- Observation sources:
    - Ouster 3D LiDAR - `sensor_msgs/PointCloud2`, `sensor_msgs/LaserScan`
    - Intel Realsense D435i Camera - `sensor_msgs/Image`, `sensor_msgs/PointCloud2`
- Robot's base frame: `base_link`
- Map frame: `map`

### Run TCC-IRoNL on your Own Robot
Configure your robot using the ROS topic configurations described above. Then, follow the instructions to launch **T3 - T4** as shown above and begin interacting with the robot. Keep in mind to update the configuration file to adapt to the spatial coordinates of your task environment. For non-goal-directed commands and queries that require no obstacle avoidance and path planning, no additional configurations are needed."

## License
![Creative Commons licenses 4.0](https://mirrors.creativecommons.org/presskit/buttons/88x31/png/by.png)

[This work is licensed under a Creative Commons Attribution International 4.0 License.](https://creativecommons.org/licenses/by/4.0/)

## Acknowledgement
This work is still in progress; therefore, expect some bugs. However, we would appreciate your kind contribution or raising an issue for such a bug.

**Thanks to the following repository:**
- [Unitree_ros](https://github.com/macc-n/ros_unitree.git)
- [ROMR_ros](https://github.com/LinusNEP/ROMR.git)
- [Ultralytics_ros](https://github.com/Alpaca-zip/ultralytics_ros.git)

