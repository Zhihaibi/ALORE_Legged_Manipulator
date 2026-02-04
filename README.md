# ALORE: Autonomous Large-Object Rearrangement with a Legged Manipulator

<div align="center">

[![Project Page](https://img.shields.io/badge/Project%20Page-6DE1D2?style=for-the-badge&logo=safari&labelColor=555555)]( https://zhihaibi.github.io/Alore/)
[![arXiv](https://img.shields.io/badge/arXiv-F75A5A?style=for-the-badge&logo=arxiv&labelColor=555555)](https://arxiv.org/pdf/2503.01474)

</div>

In this work, we present ALORE, an autonomous large-object rearrangement system for a legged manipulator that can rearrange various large objects across diverse scenarios.The proposed system is characterized by three main features: 
- A hierarchical reinforcement learning training pipeline for multi-object environment learning, where a high-level object velocity controller is trained on top of a low-level whole-body controller to achieve efficient and stable joint learning across multiple objects; 
- Two key modules, a unified interaction configuration representation and an object velocity estimator, that allow a single policy to regulate planar velocity of diverse objects accurately;  
- A task-and-motion planning framework that jointly optimizes object visitation order and object-to-target assignment, improving task efficiency while enabling online replanning. 
- Extensive simulations and real-world experiments are conducted to validate the robustness and effectiveness of the entire system, which successfully completes 8 continuous loops to rearrange 32 chairs over nearly 40 minutes without a single failure, and executes long-distance autonomous rearrangement over an approximately 40 m route.

![The proposed system, ALORE, seamlessly integrates perception, planning, locomotion, and grasping, enabling a legged manipulator to perform diverse large-object rearrangement tasks. (b) It achieves stable grasping and accurate planar velocity control across various objects with different masses, while preventing self-collision and object toppling. (c) The system also demonstrates long-term autonomy, running continuously for 40 minutes to complete 8 loops and totally rearrange 32 chairs without a single failure. (d) Moreover, ALORE supports long-distance autonomous rearrangement, moving a chair along an approximately 40\,m route through narrow passages without collisions, over uneven terrain with protrusions, and across floor surfaces of varying materials.](/images/head.png)




# Content Table
- [Installation](#installation)
- [Training in IssacLab](#training-in-issaclab)
- [Simulation in IssacSim](#simulation-in-issac-sim)
- [Real-World Deployment](#object-rearrangement-sim-to-real-deploy)

# Installation
#### 1. Create conda environment:
```
conda env remove -n alore
conda create -n alore python=3.8
conda activate alore
```

#### 2. Download and Install Issac Lab
We use [Isaac Sim 4.5](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html) and [Issac Lab V1.20](https://isaac-sim.github.io/IsaacLab/v1.2.0/index.html). Please install them according to the official guidance. After installation, run example below to check if everthing is ready.
```
# install python module (for rsl-rl)
./isaaclab.sh -i rsl_rl
# run script for rl training of the teacher agent
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Anymal-D-v0 --headless
```


#### 3. Download Pinocchio
```
 conda install pinocchio -c conda-forge
```
#### 4. Download FoundationPose
```
git clone https://github.com/NVlabs/FoundationPose
```
Follow the installation commands in [Here](https://github.com/NVlabs/FoundationPose).

#### 5. DDR-OPT
Using the file Planning_ddr_opt in this repo. Follow the commands in [ddr-opt](https://github.com/ZJU-FAST-Lab/DDR-opt)
```
mkdir -p DDRopt_ws/src
cd DDRopt_ws/src
cd ..
catkin build
```

# Training in IssacLab
If you want to modify the training objects, revise the file **b2z1_multiobj_wbc_gnn_plan_env_cfg.py**
```
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-B2Z1MultiObjWBCGNNPLAN-Direct-v0
```
- Note: We use the direct env in issac lab with a modified rsl-rl package.

# Simulation in Issac Sim
```
cd isaac_b2_controller/b2z1/
python b2z1_highlevel_controller.py
roslaunch plan_manager planner_sim.launch # planning module
python b2z1_object_fsm.py # FSM for various object arrangement
```


# Object Rearrangement Sim-to-Real Deploy

#### Note: change your own dir.

### 1. For obj perception and localization
open the perception module, get the object pose and robot pose
```python
# if use motion capture
source /home/unitree/robot_ws/devel_isolated/setup.bash
roslaunch vrpn_client_ros sample.launch 

conda activate z1_obj
python /home/unitree/work2/Whole_Body_Object_Rearrangement/real_experiment/perception/env_perception_mocap.py
```

```python
# Auto perception
## 1) Open camera
export ROS_MASTER_URI=http://192.168.10.167:11311
export ROS_IP=192.168.10.167
roslaunch realsense2_camera rs_camera.launch align_depth:=true

## 2) Localization using Lidar
source /home/unitree/robot_ws/devel_isolated/setup.bash

roslaunch livox_ros_driver2 rviz_MID360.launch 

source /home/unitree/robot_ws/devel_isolated/setup.bash

roslaunch hdl_localization hdl_localization.launch 

conda activate z1_obj
python /home/unitree/work2/Whole_Body_Object_Rearrangement/real_experiment/perception/env_perception_auto.py

## 3) April_tag pose(using system python3, but not conda)
python3 /home/unitree/work2/Whole_Body_Object_Rearrangement/real_experiment/perception/apriltag_pose.py

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
conda activate yolo
python3 /home/unitree/work2/Whole_Body_Object_Rearrangement/real_experiment/perception/yolo_pose.py
```




### 2. For Unitree Z1 manipulator
```python
cd /home/unitree/z1_arm/bin/
./z1_ctrl  # open the z1 sdk 

conda activate z1_obj
python /home/unitree/work2/Whole_Body_Object_Rearrangement/real_experiment/Z1_deploy/z1_control.py # start to sub the target position of z1
```

### 3. For Unitree B2 robot
start the sim to real deploy code
```python
cd /home/unitree/work2/Whole_Body_Object_Rearrangement/real_experiment/B2_deploy/
conda activate z1_obj
taskset -c 1 python deploy_real_b2z1_obj.py eth0 b2z1.yaml 
```

### 4. For planner and object rearrangement
```python
source /home/unitree/robot_ws/devel_isolated/setup.bash
roslaunch plan_manager planner_sim.launch 
python /home/unitree/work2/Whole_Body_Object_Rearrangement/real_experiment/object_arrangement_fsm_auto.py
```

#### Note
- use taskset -c 0/1/2 python XXX.py, to keep the running code stable (stable HZ).
- once the z1_controller is on, it must be kept running and send the angle to the z1_sdk, or it will lose connect.


# Acknowledgements
We modify [ddr-opt](https://github.com/ZJU-FAST-Lab/DDR-opt) for task and planning. 

We use [foundationPose]((https://github.com/NVlabs/FoundationPose)) for pose estimation. 

We modify the [VBC](https://github.com/Ericonaldo/visual_wholebody) for our low-level Whole-Body Controller.
