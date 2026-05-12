# TCC-IRoNL: The Conversation is the Command

> **TCC-IRoNL** leverages pre-trained large language models (LLMs) with a multimodal vision-language model (VLM) to enable humans to interact naturally with autonomous robots through conversational dialogue. It utilises the LLM to decode high-level natural-language instructions and abstract them into precise robot-actionable commands or queries, and the VLM to provide visual and semantic understanding of the robot's task environment.

[![TCC IRoNL](https://img.shields.io/badge/TCC%20IRoNL-Website-lightblue?style=flat&logo=globe&logoColor=white)](https://linusnep.github.io/TCC-IRoNL/)
[![ROS](https://img.shields.io/badge/ROS-Noetic-brightgreen.svg)](http://www.ros.org/)
[![Python 3.8+](https://img.shields.io/badge/Python-≥3.8-blue.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Required-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

<p align="center">
  <a href="https://doi.org/10.1145/3610978.3640723">📄 Paper (HRI '24)</a> &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2401.11838">ArXiv</a> &nbsp;|&nbsp;
  <a href="https://linusnep.github.io/TCC-IRoNL/">🌐 Project Website</a> &nbsp;|&nbsp;
  <a href="https://osf.io/cmbw6/">🎬 Demo Videos</a> &nbsp;|&nbsp;
  <a href="https://creativecommons.org/licenses/by/4.0/">License</a>
</p>

<div align="center">
  <img src="https://github.com/LinusNEP/TCC-IRoNL/blob/main/Figures/gifAnimation1.gif" width="400px" alt="Simulation demo" />
  <img src="https://github.com/LinusNEP/TCC-IRoNL/blob/main/Figures/real_world_optimize.gif" width="400px" alt="Real-world demo" />
</div>

> **Status:** This work is in active development, expect occasional bugs. Bug reports and contributions via GitHub Issues are very welcome.

---

## Contents

- [Quick Install](#quick-install)
  - [Prerequisites](#prerequisites)
- [Advanced Installation](#advanced-installation)
  - [System Requirements](#system-requirements)
  - [Step-by-Step Setup](#step-by-step-setup)
- [Run Demos](#run-demos)
  - [Simulation: Unitree Go1 Quadruped](#simulation-unitree-go1-quadruped)
  - [Simulation: ROMR Wheeled Robot](#simulation-romr-wheeled-robot)
  - [Example Commands](#example-commands)
- [Real-World Robot Deployment](#real-world-robot-deployment)
- [Troubleshooting](#troubleshooting)
  - [SAM Model Not Found](#sam-model-not-found)
  - [CUDA Out of Memory](#cuda-out-of-memory)
  - [LLM API Errors](#llm-api-errors)
  - [ROS Topics Not Connected](#ros-topics-not-connected)
  - [Docker: GPU Not Available](#docker-gpu-not-available)
- [Updates Since Publication](#updates-since-publication)
- [Citation](#citation)
- [Contributing](#contributing)
  - [Development Setup](#development-setup)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Quick Install

The fastest way to get started is using our pre-configured Docker environment.

### Prerequisites

- Ubuntu 20.04+ (or any Linux with Docker)
- NVIDIA GPU with CUDA 11.8+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed
- Docker and Docker Compose installed
- Your user added to the `docker` group, e.g.
  ```bash
  sudo usermod -aG docker $USER && newgrp docker
  ```

### 1. Clone the Repository

```bash
git clone https://github.com/LinusNEP/TCC-IRoNL.git
cd TCC-IRoNL
```

### 2. Configure Your API Key

```bash
cp env.tcc-ironl-denv .env
# Edit .env and add your LLM API key:
# OPENAI_API_KEY=sk-...
# Or one of: DEEPSEEK_API_KEY, GEMINI_API_KEY, ANTHROPIC_API_KEY
nano .env
```

### 3. Create the models directory

The Docker services mount a local `./models` directory into the containers for SAM weights and other assets. Create it before launching compose:

```bash
mkdir -p models
```

### 4. Launch with Docker Compose

```bash
docker compose up --build
```

This starts all services automatically:
- ROS core
- Gazebo simulation world
- ROS navigation stack
- LLM/VLM processing node
- Chat GUI

### 5. Start Chatting

Once everything is up, interact with the robot using the chat GUI launched by
the `chat-gui` service, e.g.:

> **"Move forward 1.5 meters at 0.2 m/s"**
>
> **"Go between the Secretary's office and the kitchen twice"**
>
> **"What can you see around you?"**

---

## Advanced Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Ubuntu 20.04 | Ubuntu 20.04 LTS |
| ROS | Noetic | Noetic (full-desktop) |
| Python | 3.8 | 3.10 |
| GPU | NVIDIA GTX 1060 6GB | NVIDIA RTX 3060+ |
| CUDA | 11.3 | 11.8 or 12.x |
| RAM | 16 GB | 32 GB |
| Storage | 20 GB free | 50 GB SSD |

### Step-by-Step Setup

#### 1. Install ROS Noetic

Follow the [official ROS Noetic installation guide](http://wiki.ros.org/noetic/Installation/Ubuntu), then install navigation dependencies:

```bash
sudo apt update
sudo apt install -y \
    ros-noetic-navigation \
    ros-noetic-move-base \
    ros-noetic-amcl \
    ros-noetic-map-server \
    ros-noetic-gmapping \
    ros-noetic-teb-local-planner
```

#### 2. Create Workspace & Clone

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/LinusNEP/TCC-IRoNL.git
cd TCC-IRoNL
```

#### 3. Install Python Dependencies

```bash
cd ~/catkin_ws
bash src/TCC-IRoNL/scripts/install_python_deps.sh
```

This creates a virtual environment at `~/catkin_ws/TCC-IRoNLEnv/` with all required packages.

#### 4. Download SAM Weights

```bash
mkdir -p ~/catkin_ws/src/TCC-IRoNL/tcc_ros/src/tcc_ros/models
cd ~/catkin_ws/src/TCC-IRoNL/tcc_ros/src/tcc_ros/models

# Download SAM ViT-B checkpoint (~350 MB)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

> ⚠️ **Required**: Without this file, `roslaunch tcc_ros tcc_ros.launch` will fail.

#### 5. Configure API Keys

```bash
# Option A: Environment variable (session only)
export OPENAI_API_KEY="sk-your-key-here"

# Option B: Persistent configuration (recommended)
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc

# Option C: Use .env file
cp ~/catkin_ws/src/TCC-IRoNL/env.tcc-ironl-denv ~/catkin_ws/.env
nano ~/catkin_ws/.env
```

Supported providers: `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`

#### 6. Build the Workspace

```bash
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

Add to `~/.bashrc` for persistence:
```bash
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
```

---

## Run Demos

### Simulation: Unitree Go1 Quadruped

Open **four terminals** in your workspace directory. Source the workspace in **every terminal**:

```bash
source ~/catkin_ws/devel/setup.bash
```

In **Terminals 3 & 4**, also activate the Python environment:

```bash
source ~/catkin_ws/TCC-IRoNLEnv/bin/activate
```

| Terminal | Command | Purpose |
|----------|---------|---------|
| **T1** | `roslaunch unitree_gazebo sim_bringup.launch rname:=go1 wname:=cps_world rviz:=false` | Gazebo world |
| **T2** | `roslaunch unitree_navigation navigation.launch rname:=go1 rviz:=true` | Navigation stack |
| **T3** | `roslaunch tcc_ros tcc_ros.launch` | LLM/VLM backend |
| **T4** | `roslaunch tcc_ros chatGUI_SR.launch` | Chat interface |

**Robot State Setup** (in T1):
- Press `2` → Switch from **Passive** to **FixedStand**
- Press `5` → Switch from **FixedStand** to **MoveBase**

The robot is now ready to receive commands!

### Simulation: ROMR Wheeled Robot

| Terminal | Command |
|----------|---------|
| **T1** | `roslaunch romr_ros romr_navigation.launch` |
| **T3** | `roslaunch tcc_ros tcc_ros.launch` |
| **T4** | `roslaunch tcc_ros chatGUI_SR.launch` |

No state-switching needed for wheeled robots.

### Example Commands

**Non-goal-directed:**
- `"Move forward 1.5 m at 0.2 m/s"`
- `"Move backwards 2 m and thereafter move in a circular pattern of diameter 2 m"`

**Information queries:**
- `"Tell me about your capabilities"`
- `"Report your current orientation"`
- `"What can you see around you?"`

**Goal-directed navigation:**
- `"Go between the Secretary's office and the kitchen twice"`
- `"Head to the location (0, 3, 0) and return to your start location afterwards"`

Use the environment layout below as a reference, or adapt your own Gazebo world by adding the location names and coordinates to `config.yaml`.

<p align="center">
  <img src="https://github.com/LinusNEP/TCC-IRoNL/blob/main/Figures/gazebo-world.jpg" alt="Gazebo world layout" />
</p>

---

## Real-World Robot Deployment

To deploy on your own ROS-based robot, ensure these topics are published:

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
- Ouster OS1-32 LiDAR - `sensor_msgs/PointCloud2`, `sensor_msgs/LaserScan`
- Intel RealSense D435i - `sensor_msgs/Image`, `sensor_msgs/PointCloud2`

**Frames:**

- Robot base frame: `base_link`
- Map frame: `map`

**Configuration:**
Simple movement commands and perception queries (e.g. *"move forward"*, *"what can you see around you?"*, *"where are you now?"*) work out of the box. Goal-directed navigation to a named location (e.g. *"head to location x, y, z"*) requires you to add the approximate `x, y, z` coordinates of the target location to `config.yaml`. You can obtain these coordinates from LiDAR point-cloud data. Example config:

```yaml
locations:
  kitchen: [2.5, 4.0, 0.0]
  office: [8.2, 3.1, 0.0]
  elevator: [1.0, 1.0, 0.0]
```

Then launch only the TCC-IRoNL nodes (T3 and T4 from above).

---

## Troubleshooting

### SAM Model Not Found
```
[ERROR] Cannot find sam_vit_b_01ec64.pth
```
**Fix:** Ensure the model is downloaded to `tcc_ros/src/tcc_ros/models/`:
```bash
ls ~/catkin_ws/src/TCC-IRoNL/tcc_ros/src/tcc_ros/models/sam_vit_b_01ec64.pth
```

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Fix:** Use a smaller SAM model or reduce image resolution in `config.yaml`:
```yaml
sam_model_type: "vit_b"  # Options: vit_b (smallest), vit_l, vit_h
image_resize: [640, 480]
```

### LLM API Errors
```
[ERROR] OpenAI API request failed: 401 Unauthorized
```
**Fix:** Verify your API key is set:
```bash
echo $OPENAI_API_KEY
# If empty, re-run: export OPENAI_API_KEY="sk-..."
```

### ROS Topics Not Connected
```
[ WARN] [timestamp]: /odom topic not received
```
**Fix:** Check topic availability:
```bash
rostopic list | grep odom
rostopic hz /odom
```

### Docker: GPU Not Available
```
Failed to initialize NVML: Unknown Error
```
**Fix:** Ensure NVIDIA Container Toolkit is installed:
```bash
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

---

## Updates Since Publication

| Component | Paper | Current |
|-----------|-----------------|---------|
| Vision | YOLOv8 | **SAM (Segment Anything)** |
| Language Model | GPT-2 | **GPT-4o, Gemini, DeepSeek, Llama-2-7B** |
| Segmentation | Bounding boxes | **Pixel-level masks** |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{10.1145/3610978.3640723,
author	={Nwankwo, Linus and Rueckert, Elmar}
title = {The Conversation is the Command: Interacting with Real-World Autonomous Robots Through Natural Language},
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

---

## Contributing

Bug reports and contributions are welcome! Please open an [issue](https://github.com/LinusNEP/TCC-IRoNL/issues) or submit a pull request.

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

## Acknowledgements

Thanks to the following open-source projects:
- [Unitree_ros](https://github.com/unitreerobotics/unitree_ros)
- [ROS Mobile Robot](https://github.com/LinusNEP/ROS-Mobile-Robot)
- [Ultralytics_ros](https://github.com/Alpaca-zip/ultralytics_ros)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)

