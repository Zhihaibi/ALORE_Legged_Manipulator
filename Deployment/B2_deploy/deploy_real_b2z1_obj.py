# -*- coding: utf-8 -*-
from typing import Union
import numpy as np
import time
import torch
from pynput import keyboard
import rospy
from std_msgs.msg import Float32MultiArray

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.b2.sport.sport_client import SportClient


from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import  get_body_orientation, quat_inverse_safe, quat_mul_2, quat_rotate
from common.remote_controller import RemoteController, KeyMap
from config import Config


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        # Initialize the policy network
        self.low_policy = torch.jit.load(config.low_level_policy_path)
        self.qj = np.zeros(config.num_actions_low + 1, dtype=np.float32) # add gripper
        self.dqj = np.zeros(config.num_actions_low, dtype=np.float32)
        self.arm_theta = np.zeros(7, dtype=np.float32)
        self.arm_theta_vel = np.zeros(6, dtype=np.float32)

        self.action_low = np.zeros(config.num_actions_low, dtype=np.float32)
        self.obs_low = np.zeros(config.num_obs_low, dtype=np.float32)
        self.obs_proprio = np.zeros(config.num_proprio_low, dtype=np.float32)
        self.obs_history_buf = np.zeros((config.history_len_low, config.num_proprio_low))
        self.cmd_low = np.array([0.0, 0.0, 0.0])
        self.cmd_low_previous = np.array([0.0, 0.0, 0.0])
        self.counter_low = 0

        self.step_value = 0.00
        self.kp_add = 0

        self.arm_default_angles = config.arm_default_angles_low

        self.gait_indices = 0.0
        self.clock_inputs = np.zeros(4)

        self.curr_ee_goal_cart = np.array([0., 0.,  0.]) # x, y, z

        if config.msg_type == "hg":
            # g1 and h1_2 use the hg msg type
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            # h1 uses the go msg type
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
            self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)
        else:
            raise ValueError("Invalid msg_type")

        # wait for the subscriber to receive data
        rospy.init_node('b2z1_arm_listen', anonymous=True)
        self.rate = rospy.Rate(50) # 50HZ

        self.base_quat_pub = rospy.Publisher('base_quat', Float32MultiArray, queue_size=2)
        self.target_dof_pos_pub = rospy.Publisher('target_dof_pos', Float32MultiArray, queue_size=2)
        self.robot_leg_joints_pub = rospy.Publisher('robot_leg_joints', Float32MultiArray, queue_size=2)

        self.b2z1_cur_joint_pub = rospy.Publisher("/b2z1_cur_joint", Float32MultiArray, queue_size=10)
        rospy.Timer(rospy.Duration(0.02), self.publish_b2z1_cur_joint) # 10Hz


        # Initialize the command msg
        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)
        
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)
        print("B2 controller closed")

        # high level state
        self.high_policy = torch.jit.load(config.high_level_policy_path)

        self.object_height_offset = 0.0
        self.arm_default_angles_high = np.zeros(7, dtype=np.float32)
        self.default_angles_high = np.zeros(18, dtype=np.float32)

        self.counter_high = 0
        self.obs_proprio_high = np.zeros(config.num_proprio_high, dtype=np.float32)
        self.obs_history_buf_high = np.zeros((config.history_len_high, config.num_proprio_high))
        self.obs_high = np.zeros(config.num_obs_high, dtype=np.float32)
        self.action_high = np.zeros(config.num_actions_high, dtype=np.float32)
        self.last_action_high = np.zeros(config.num_actions_high, dtype=np.float32)

        self.obj_vel_cmd = np.array([0.0, 0.0, 0.0])
        self.robot_pose_world_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # x, y, z, qx,qy,qz,w
        self.object_pose_world_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # x, y, z, qx,qy,qz,w
        self.object_pose = np.zeros((4, 8))  # assuming 4 objects, each with (x, y, z, yaw) + (qx,qy,qz,w)
        self.ee_pose_in_robot_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # x, y, z, x, y, z, w TODO: order matter
        self.obj_pose_in_robot_frame = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) # x, y, z, x, y, z, w TODO: order matter
        
        # FSM WAIT_TASK_PLANNING, WAIT_ROBOT_PATH, ROBOT_TRACKING, GRASPING, WAIT_OBJECT_PATH, OBJECT_TRACKING, RELEASING
        self.task_state = "WAIT_TASK_PLANNING"  

        self.task_state_mapping = {
                0.0: "WAIT_TASK_PLANNING",
                1.0: "WAIT_ROBOT_PATH",
                2.0: "ROBOT_TRACKING",
                3.0: "GRASPING",
                4.0: "WAIT_OBJECT_PATH",
                5.0: "OBJECT_TRACKING",
                6.0: "RELEASING"
            }

        self.object_category_mapping = {
            0.0: "box",
            1.0: "table",
            2.0: "chair"
        }
        self.object_category = "UNKNOWN"
        
        # z1
        self.z1_target_pos = config.arm_default_angles_initial  # 7 dof
        self.z1_target_pos_pub = rospy.Publisher('arm_target_pos', Float32MultiArray, queue_size=2)

        self.robot_vel_cmd = np.array([0.0, 0.0, 0.0]) # vx, vy, omega
        self.z1_joint_cmd = np.zeros(7)
        rospy.Subscriber('/b2z1_control_data', Float32MultiArray, self.sub_b2z1_control_data_callback, queue_size=1)

        self.STOP_SIGN = False
        self.START_Tracking = False
        self.HIGH_LEVEL = False

        rospy.Subscriber('/arm_current_state', Float32MultiArray, self.sub_arm_state_callback, queue_size=1)
        rospy.Subscriber('/hand_current_state', Float32MultiArray, self.sub_ee_state_callback, queue_size=1)
        rospy.Subscriber("/env_obs", Float32MultiArray, self.sub_env_obs_callback, queue_size=1) # from motion capture, robot global pose, object global pose

        self.wait_for_low_state()


    def sub_env_obs_callback(self, msg):
        data = msg.data
        robot_obs = data[:8]   # 8
        object_obs = data[8:]  # 4*8 = 32

        # update current poses
        if robot_obs is not None: 
            self.robot_pose_world_frame[0:3] = robot_obs[0:3] # x,y,z
            # self.robot_pose_world_frame[3:7] = robot_obs[4:8] # qx,qy,qz,w
            # w,x,y,z 
            self.robot_pose_world_frame[3:7] = [robot_obs[7], robot_obs[4], robot_obs[5], robot_obs[6]]

        ee_posi_world_frame = self.ee_pose_in_robot_frame[0:3] + self.robot_pose_world_frame[0:3]

        if object_obs is not None:
            self.object_pose[0][0:3] = object_obs[0:3]
            self.object_pose[0][4:8] = object_obs[4:8]

            self.object_pose[1][0:3] = object_obs[8:11]
            self.object_pose[1][4:8] = object_obs[12:16]

            self.object_pose[2][0:3] = object_obs[16:19]
            self.object_pose[2][4:8] = object_obs[20:24]

            self.object_pose[3][0:3] = object_obs[24:27]
            self.object_pose[3][4:8] = object_obs[28:32]
        
        # select the closest object to the ee
        dists = np.linalg.norm(self.object_pose[:, 0:3] - ee_posi_world_frame, axis=1)
        closest_obj_idx = np.argmin(dists)
        self.object_pose_world_frame[0:3] = self.object_pose[closest_obj_idx, 0:3]
        # x,y,z,w to w,x,y,z
        self.object_pose_world_frame[3:7] = [self.object_pose[closest_obj_idx, 7], self.object_pose[closest_obj_idx, 4], self.object_pose[closest_obj_idx, 5], self.object_pose[closest_obj_idx, 6]]
        # print("closest_obj_idx", closest_obj_idx)
        # self.rate.sleep()

    def publish_b2z1_cur_joint(self, event):
        self.b2z1_cur_joint_pub.publish((Float32MultiArray(data=self.qj.tolist())))

    def sub_b2z1_control_data_callback(self, msg):
        self.robot_vel_cmd = np.array(msg.data[0:3])
        self.obj_vel_cmd = np.array(msg.data[3:6])
        self.z1_joint_cmd = np.array(msg.data[6:13])        
        self.task_state = self.task_state_mapping.get(msg.data[13], "UNKNOWN")
        self.object_category = self.object_category_mapping.get(msg.data[14], "UNKNOWN")
        # self.rate.sleep()

    def sub_arm_state_callback(self, msg):
        self.arm_theta = np.array(msg.data[:7])
        self.arm_theta_vel = np.array(msg.data[7:13])
        # self.rate.sleep()
    
    def sub_ee_state_callback(self, msg):
        self.curr_ee_goal_cart = np.array(msg.data[:3])
        ee_pose_in_arm_base_frame = np.array(msg.data[:]) # x, y, z, qx, qy, qz, w

        self.ee_pose_in_robot_frame[0:3] = ee_pose_in_arm_base_frame[0:3] + np.array([0.2, 0.0, 0.15])

        # x,y,z,w to w,x,y,z
        # self.ee_pose_in_robot_frame[3:7] = [ee_pose_in_arm_base_frame[6], ee_pose_in_arm_base_frame[3], ee_pose_in_arm_base_frame[4], ee_pose_in_arm_base_frame[5]]  # w， qx, qy, qz
        self.ee_pose_in_robot_frame[3:7] = [0.0, 0.0, 0.0, 0.0]  # w， qx, qy, qz

        # print("curr_ee_goal_cart", curr_ee_goal_cart)
        # self.rate.sleep()

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0 or self.arm_theta[1] == 0 or self.curr_ee_goal_cart[0] == 0:
            time.sleep(self.config.control_dt)
            print("arm_theta", self.arm_theta[1])
            print("Waiting for the low state...")
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)
        
        dof_idx = self.config.leg_joint2motor_idx
        kps_stand = self.config.kps_stand
        kds_stand = self.config.kds_stand 
        default_pos = self.config.default_angles_low
        dof_size = len(dof_idx)
        
        # record the current pos
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q
        
        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps_stand[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds_stand[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
        # self.arm_target_pos_pub.publish(Float32MultiArray(data=self.arm_default_angles.tolist()))  


    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles_low[i]
                self.low_cmd.motor_cmd[motor_idx].qd = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps_stand[i]
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds_stand[i]
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def get_gait_obs(self):
        frequencies = 2.0
        phases = 0.5
        offsets = 0
        bounds = 0
        self.gait_indices = np.remainder(self.gait_indices + 0.02 * frequencies, 1.0)

        walking_mask0 = np.abs(self.cmd_low[0]) > 0.1
        walking_mask1 = np.abs(self.cmd_low[1]) > 0.1
        walking_mask2 = np.abs(self.cmd_low[2]) > 0.1
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2

        if not walking_mask:
            self.gait_indices = 0  # reset gait indices for non-walking commands

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        foot_indices = np.remainder(np.array(foot_indices), 1.0)

        self.clock_inputs[0] = np.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[1] = np.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[2] = np.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[3] = np.sin(2 * np.pi * foot_indices[3])


    def reindex_all(vec):
        return np.hstack((vec[[3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]], vec[12:]))



    def low_level_controller(self):
        self.counter_low += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
        self.qj[12:19] = self.arm_theta.copy() # 6 joints and 1 gripper
        self.dqj[12:18] = self.arm_theta_vel.copy()

        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion

        # quat_msg = Float32MultiArray(data=quat)
        # self.base_quat_pub.publish(quat_msg)
    
        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)

        # create observation
        body_orientation_rp = get_body_orientation(quat)
        ang_vel = ang_vel * self.config.ang_vel_scale_low

        self.get_gait_obs()

        qj_obs = self.qj[0:18].copy()
        dqj_obs = self.dqj.copy()

        qj_obs = (qj_obs - self.config.default_angles_low) * self.config.dof_pos_scale_low
        dqj_obs = dqj_obs * self.config.dof_vel_scale_low

        if self.STOP_SIGN and self.HIGH_LEVEL == False:
            self.cmd_low[0] = self.remote_controller.ly / 2.0
            self.cmd_low[1] = (self.remote_controller.lx * -1) / 2.0
            self.cmd_low[2] = (self.remote_controller.rx * -1) / 2.0

        self.cmd_low[0] = np.clip(self.cmd_low[0], -0.2, 0.2)
        self.cmd_low[1] = np.clip(self.cmd_low[1], -0.0, 0.0)
        self.cmd_low[2] = np.clip(self.cmd_low[2], -0.3, 0.3)

        # per-step rate limit (max change per control step)
        # desired_cmd = self.cmd_low.copy()
        # max_delta = 0.01
        # delta = desired_cmd - self.cmd_low_previous
        # delta = np.clip(delta, -max_delta, max_delta)
        # self.cmd_low = self.cmd_low_previous + delta
        # # update previous applied command
        # self.cmd_low_previous = self.cmd_low.copy()

        priv_buf = [ 0.0000,  0.0000,  0.0000,  0.0000,  0.0,  1.0,   
                    0.,  0.,  0.,  0.,  0.,  0., 
                    0.,  0.,  0.,  0.,  0.,  0.] # 18

        # qj_obs[-6:] = 0.
        # dqj_obs[-6:] = 0.
        
        # print("cmd: ", self.cmd_low)
        self.obs_proprio[:2] = body_orientation_rp     # dim 2
        self.obs_proprio[2:2+3] = ang_vel              # dim 3
        self.obs_proprio[5:5+18] = qj_obs              # dim 18
        self.obs_proprio[23:23+18] = dqj_obs           # dim 18
        self.obs_proprio[41:41+12] = self.action_low[:12]  # dim 12
        self.obs_proprio[53:53+4] = np.zeros(4)        # dim 4
        self.obs_proprio[57:57+3] = self.cmd_low * self.config.cmd_scale_low # dim 3
        self.obs_proprio[60:60+3] = self.curr_ee_goal_cart  # dim 3 TODO: ee goal cart
        self.obs_proprio[63:63+3] = np.zeros(3)        # dim 3
        self.obs_proprio[66:66+1] = self.gait_indices  # dim 4
        self.obs_proprio[67:67+4] = self.clock_inputs  # dim 4

        self.obs_low = np.concatenate([self.obs_proprio, priv_buf, self.obs_history_buf.reshape(-1)], axis=-1)

        # print("self.curr_ee_goal_cart", self.curr_ee_goal_cart)

        if self.counter_low <= 1:
            self.obs_history_buf = np.stack([self.obs_proprio] * 10, axis=0)  # shape: (10, 71)
        else:
            self.obs_history_buf = np.concatenate([
                self.obs_history_buf[1:, :],
                self.obs_proprio[np.newaxis, :]
            ], axis=0)

        # Get the action from the low_policy network
        # print("obs shape: ", self.obs_low.shape)

        obs_tensor = torch.from_numpy(self.obs_low).unsqueeze(0).float()
        self.action_low = self.low_policy(obs_tensor).detach().numpy().squeeze()
        self.action_low = np.clip(self.action_low, -100, 100)

        # transform action to target_dof_pos
        if self.counter_low > 10:
            target_dof_pos = self.config.default_angles_low + self.action_low * self.config.action_scale_low
            self.counter_low = 11
        else:
            target_dof_pos = self.config.default_angles_low
        
        # publishs the command and the current state for debug
        # self.target_dof_pos_pub.publish(Float32MultiArray(data=target_dof_pos.tolist()))
        # self.robot_leg_joints_pub.publish(Float32MultiArray(data=self.qj.tolist()))

        # protect the target_dof_pos within the joint limits
        target_dof_pos[:12] = np.clip(target_dof_pos[:12], self.config.joint_limits_min, self.config.joint_limits_max)

        # Build low cmd
        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].qd = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.kps[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.kds[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        # send the command to b2
        self.send_cmd(self.low_cmd)
        # send the command to z1
        z1_target_pos = self.z1_target_pos.copy()
        self.z1_target_pos_pub.publish(Float32MultiArray(data=z1_target_pos.tolist()))

    
    def high_level_controller(self):
        print(self.object_category)
        if self.object_category == "box":
            self.object_height_offset = self.config.object_height_offset_box
            self.default_angles_high = self.config.default_angles_high_box
            self.arm_default_angles_high = self.config.arm_default_angles_high_box
            # print("object_category: box")
        elif self.object_category == "table":
            self.object_height_offset = self.config.object_height_offset_table
            self.default_angles_high = self.config.default_angles_high_table
            self.arm_default_angles_high = self.config.arm_default_angles_high_table
            # print("object_category: table")
        else:
            self.object_height_offset = self.config.object_height_offset_chair
            self.default_angles_high = self.config.default_angles_high_chair
            self.arm_default_angles_high = self.config.arm_default_angles_high_chair
            # print("object_category: chair")

        self.counter_high += 1
        # Get the current joint position and velocity
        for i in range(len(self.config.leg_joint2motor_idx)):
            self.qj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].q
            self.dqj[i] = self.low_state.motor_state[self.config.leg_joint2motor_idx[i]].dq
        self.qj[12:19] = self.arm_theta.copy() # 6 joints and 1 gripper
        self.dqj[12:18] = self.arm_theta_vel.copy()

        if self.HIGH_LEVEL:
            self.obj_vel_cmd[0] = self.remote_controller.ly / 2.0
            self.obj_vel_cmd[1] = 0.0
            self.obj_vel_cmd[2] = (self.remote_controller.rx * -1) / 2.0
        
        # imu_state quaternion: w, x, y, z
        quat = self.low_state.imu_state.quaternion

        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32)
        body_orientation_rp = get_body_orientation(quat)
        ang_vel = ang_vel * self.config.ang_vel_scale_high
        
        qj_obs = self.qj[0:18].copy()
        dqj_obs = self.dqj.copy()

        # print(f"qj_obs1: {np.array2string(qj_obs, precision=3, suppress_small=True)}")

        qj_obs = (qj_obs - self.default_angles_high[:18]) * self.config.dof_pos_scale_high
        dqj_obs = dqj_obs * self.config.dof_vel_scale_high

        obj_posi_relative = self.object_pose_world_frame[:3] - self.robot_pose_world_frame[:3]  # (num_envs, 3) 
        robot_quat_inv = quat_inverse_safe(self.robot_pose_world_frame[3:7])  # (num_envs, 4)
        obj_posi_in_robot_frame = quat_rotate(robot_quat_inv, obj_posi_relative)  # (num_envs, 3)
        obj_posi_in_robot_frame[:] = obj_posi_in_robot_frame[:] - self.object_height_offset # adjust the object height
        obj_quat_in_robot_frame = quat_mul_2(robot_quat_inv, self.object_pose_world_frame[3:7])  # (num_envs, 4)

        self.obj_pose_in_robot_frame[:3] = obj_posi_in_robot_frame
        if self.object_category == "table":
            self.obj_pose_in_robot_frame[:3] = [0.89, -0.47, -0.54]
        if self.object_category == "box":
            self.obj_pose_in_robot_frame[:3] = [0.62, -0.29, -0.47]
        self.obj_pose_in_robot_frame[3:7] = obj_quat_in_robot_frame
        self.obj_pose_in_robot_frame[3:7] = [0, 0, 0, 0]

        # self.obj_pose_in_robot_frame[:7] = [ 0.9059,  0.0358, -0.570, 1, 0, 0, 0]  # temp fix TODO

        self.obs_proprio_high[:18] = qj_obs
        self.obs_proprio_high[18:18+18] = dqj_obs           # dim 18
        self.obs_proprio_high[36:36+2] = body_orientation_rp     # dim 2
        self.obs_proprio_high[38:38+3] = ang_vel              # dim 3
        self.obs_proprio_high[41:41+9] = self.last_action_high  # dim 9
        self.obs_proprio_high[50:50+3] = self.obj_vel_cmd * self.config.cmd_scale_high # dim 3
        self.obs_proprio_high[53:53+7] = self.ee_pose_in_robot_frame   # dim 7 TODO: (x, y, z,  x, y, z, w)
        self.obs_proprio_high[60:60+7] = self.obj_pose_in_robot_frame  # dim 7 TODO: (x, y, z,  x, y, z, w)
        self.obs_proprio_high[67:67+3] = np.zeros(3) 

        # print("qj_obs", qj_obs)
        # print("default_angle_high", self.config.default_angles_high[:18])
        # print(f"qj_obs: {np.array2string(qj_obs, precision=3, suppress_small=True)}")
        # print(f"dqj_obs: {np.array2string(dqj_obs, precision=3, suppress_small=True)}")
        # print(f"body_orientation_rp: {np.array2string(body_orientation_rp, precision=3, suppress_small=True)}")
        # print(f"ang_vel: {np.array2string(ang_vel, precision=3, suppress_small=True)}")
        # print(f"obj_vel_cmd: {np.array2string(self.obj_vel_cmd, precision=3, suppress_small=True)}")
        # print(f"ee_pose_in_robot_frame: {np.array2string(self.ee_pose_in_robot_frame, precision=3, suppress_small=True)}")
        print(f"obj_pose_in_robot_frame: {np.array2string(self.obj_pose_in_robot_frame, precision=3, suppress_small=True)}")

        self.obs_high = np.concatenate([self.obs_proprio_high, self.obs_history_buf_high.reshape(-1)], axis=-1)

        if self.counter_high <= 1:
            self.obs_history_buf_high = np.stack([self.obs_proprio_high] * 10, axis=0)  # shape: (10, 70)
        else:
            self.obs_history_buf_high = np.concatenate([
                self.obs_history_buf_high[1:, :],
                self.obs_proprio_high[np.newaxis, :]
            ], axis=0)

        obs_tensor = torch.from_numpy(self.obs_high).unsqueeze(0).float()
        # print("action_high shape 1:", self.action_high.shape)
        self.action_high = self.high_policy(obs_tensor).detach().numpy().squeeze() # 9 dim, body vx,vy,omega, 6 joint angles
        # print("action_high shape 2:", self.action_high.shape)
        self.action_high = self.action_high * self.config.action_scale_high
        action_clip_high = np.array(self.config.action_clip_high)
        self.action_high = np.clip(self.action_high, -action_clip_high, action_clip_high)

        desired_action_high = self.action_high.copy()
        max_delta = 0.02
        delta = desired_action_high - self.last_action_high
        delta = np.clip(delta, -max_delta, max_delta)
        self.action_high = self.last_action_high + delta

        self.last_action_high = self.action_high.copy()


        if self.task_state == "OBJECT_TRACKING" and self.START_Tracking:
            self.z1_target_pos[:6] = self.arm_default_angles_high[:6] + self.action_high[3:9]

        if self.HIGH_LEVEL:
            self.cmd_low[0] = self.action_high[0]
            self.cmd_low[1] = self.action_high[1]
            self.cmd_low[2] = self.action_high[2]
            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument("config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()

    # Load config
    config_path = f"/real_experiment/B2_deploy/configs/{args.config}"
    config = Config(config_path)
    print("Config loaded.")

    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)
    print("DDS communication finished.")

    controller = Controller(config)
    print("controller finished.")

    # Enter the zero torque state, press the start key to continue executing
    controller.zero_torque_state()

    # Move to the default position
    controller.move_to_default_pos()

    # Enter the default position state, press the A key to continue executing
    controller.default_pos_state()


    button_pressed = False

    while True:
        time_start = time.time()
        try:
            controller.high_level_controller()

            if controller.START_Tracking and controller.STOP_SIGN == False and controller.HIGH_LEVEL == False:

                if controller.task_state == "WAIT_ROBOT_PATH" or controller.task_state == "WAIT_TASK_PLANNING":
                    controller.cmd_low = np.array([0.0, 0.0, 0.0])
                    controller.z1_target_pos = controller.config.arm_default_angles_tracking.copy()
                    print("==========env_WAIT_ROBOT_PATH==========")
                
                if controller.task_state == "ROBOT_TRACKING":
                    controller.cmd_low = controller.robot_vel_cmd.copy()
                    controller.z1_target_pos = controller.config.arm_default_angles_tracking.copy()
                    print("==========env_ROBOT_TRACKING==========")

                if controller.task_state == "GRASPING":
                    controller.cmd_low = controller.robot_vel_cmd.copy()
                    controller.z1_target_pos = controller.z1_joint_cmd.copy()
                    print("==========env_GRASPING==========")
                
                if controller.task_state == "WAIT_OBJECT_PATH":
                    controller.cmd_low = np.array([0.0, 0.0, 0.0])
                    controller.z1_target_pos = controller.z1_joint_cmd.copy()
                    print("==========env_WAIT_OBJECT_PATH==========")

                if controller.task_state == "OBJECT_TRACKING":
                    controller.cmd_low = controller.action_high[:3]  # body vx, vy, omega
                    print("==========env_OBJECT_TRACKING==========")
                
                if controller.task_state == "RELEASING":
                    controller.cmd_low = controller.robot_vel_cmd.copy()
                    controller.z1_target_pos = controller.z1_joint_cmd.copy()
                    print("==========env_RELEASING==========")

            controller.low_level_controller()

            if controller.remote_controller.button[KeyMap.X] == 1:
                controller.STOP_SIGN = True
            if controller.remote_controller.button[KeyMap.Y] == 1:
                controller.STOP_SIGN = False

            if controller.remote_controller.button[KeyMap.B] == 1:
                controller.START_Tracking = True

            if controller.remote_controller.button[KeyMap.R2] == 1:
                controller.START_Tracking = False
                controller.STOP_SIGN = False
                controller.HIGH_LEVEL = False
                controller.task_state = "WAIT_TASK_PLANNING"
                controller.z1_target_pos = controller.config.arm_default_angles_tracking.copy()
            
            if controller.remote_controller.button[KeyMap.L1] == 1:
                controller.HIGH_LEVEL = True

            if controller.remote_controller.button[KeyMap.R1] == 1 and not button_pressed:
                if controller.z1_target_pos[6] < -0.5:
                    controller.z1_target_pos[6] = 0.0  # close the gripper
                else:
                    controller.z1_target_pos[6] = -1.5  # open the gripper
                button_pressed = True
            elif controller.remote_controller.button[KeyMap.R1] == 0:
                button_pressed = False

            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
        time_end = time.time()
        # frequence
        sleep_length = max(0, config.control_dt - (time_end - time_start))

        time.sleep(sleep_length)

        time_end2 = time.time()
        # print("frequence", 1 / (time_end2 - time_start))
        
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")

