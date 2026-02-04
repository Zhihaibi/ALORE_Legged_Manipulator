# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import numpy as np
import time

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.assets import RigidObject, RigidObjectCfg
import omni.physics.tensors as tensors

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

from omni.physx.scripts import physicsUtils
from pxr import Usd, UsdPhysics, PhysxSchema, UsdShade
import random
from scipy.spatial.transform import Rotation as R

import torch.nn.functional as F
from isaaclab.utils.math import quat_from_matrix

from isaacsim.core.prims import SingleArticulation
from isaacsim.core.prims import Articulation as ArticulationSim
import asyncio
from isaacsim.core.api import World
from isaacsim.core.utils.stage import (
    add_reference_to_stage,
    create_new_stage_async,
    get_current_stage,
)
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics

from isaacsim.core.prims import RigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaaclab.utils.math import quat_rotate
from copy import deepcopy


import omni.kit.app
import pinocchio as pin
from pathlib import Path

from .b2z1_multiobj_wbc_gnn_plan_env_cfg import B2Z1MultiObjWBCGNNPLANFlatEnvCfg, B2Z1MultiObjWBCGNNPLANRoughEnvCfg
from .low_level_model import ActorCriticLow


class B2Z1MultiObjWBCGNNPLANEnv(DirectRLEnv):
    cfg: B2Z1MultiObjWBCGNNPLANFlatEnvCfg | B2Z1MultiObjWBCGNNPLANRoughEnvCfg

    def __init__(self, cfg: B2Z1MultiObjWBCGNNPLANFlatEnvCfg | B2Z1MultiObjWBCGNNPLANRoughEnvCfg, render_mode: str | None = None, **kwargs):
        
        self.object = "mix"  # TODO mix, table1, table2, movechair1
        self.vel_log_file = f"log_direct_828_all_{self.object}"

        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        self._preprevious_actions = torch.zeros( 
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )
        
        self._low_action = torch.zeros(self.num_envs, 18, device=self.device, dtype=torch.float32)  # low level action
        self._previous_low_actions = torch.zeros(self.num_envs, 18, device=self.device, dtype=torch.float32)


        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device) 
        self.commands_scale = torch.tensor([2.0, 2.0, 0.25], device=self.device)[:3]

        self._low_commands = torch.zeros(self.num_envs, 3, device=self.device)  # low level command

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "alive",
                "pos_alignment_reward",
                "yaw_alignment_reward",

                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "flat_orientation_l2",
                "lin_vel_change_penalty",
                "ang_vel_change_penalty",
                "distance_penalty",

                "action_rate_l2",
                "action_rate2_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "joint_default_pos_l2",
                "joint_efforts_arm_l2",
                "undesired_contacts",

                "track_lin_vel_l2",
                "track_ang_vel_l2",
                "lin_vel_penalty",
                "yaw_rate_penalty",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*calf")
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*thigh")
        self._undesired_contact_body_ids = self._base_id + self._calf_ids + self._thigh_ids

        self._gripperMover_ids, _ = self._contact_sensor.find_bodies("gripperMover")
        self._gripperStator_ids, _ = self._contact_sensor.find_bodies("gripperStator")

        # helper variables
        self.initial_target_wrench = torch.zeros((self.num_envs, 3), device=self.device)  # (fx, fy, fz)
        self.initial_moment_arm = torch.zeros((self.num_envs, 3), device=self.device)  # (mx, my, mz)

        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device,
                                        requires_grad=False)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device, requires_grad=False)
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.history_len, self.cfg.num_proprio, device=self.device, dtype=torch.float)
        self.obs_history_buf_low_level = torch.zeros(self.num_envs, self.cfg.history_len, self.cfg.num_proprio_low_level, device=self.device, dtype=torch.float)

        self.processed_actions_with_fixed_joint = torch.zeros((self.num_envs, 19), device=self.device) 
        self.processed_actions_low_level = torch.zeros((self.num_envs, 19), device=self.device)  # (num_envs, 19)  # joint position target
        self.gripper_vel = 0.008
        self.gripper_angle = torch.full((self.num_envs, 1), -1.2, device=self.device)
        self.reset_steps = 1./self.gripper_vel + 5 
        self.arm_indices = [8, 13, 14, 15, 16, 17] # joint 1 ~ 6
        self.arm_indices_with_gripper = [8, 13, 14, 15, 16, 17, 18] # joint 1 ~ 7
        self.last_action = torch.zeros((self.num_envs, 9), device=self.device)  # (num_envs, 19)  # last action for logging
        
        # system ID
        self.robot_view_sim = ArticulationSim("/World/envs/env_.*/Robot/base")

        self.iter_time = 0

        # For test logger
        self.IF_LOG = False
        self.log_obj_vel = []  # (vx, vy, omega)
        self.log_obj_commands = []  # (vx, vy, omega)

        self.random_y = []
        self._previous_objects_vel = torch.zeros((self.num_envs, 3), device=self.device)  # (vx, vy, omega)

        self.obj_euler_in_robot_frame = torch.zeros((self.num_envs, 3), device=self.device)  # (roll, pitch, yaw)
        self.ee_euler_in_robot_frame = torch.zeros((self.num_envs, 3), device=self.device)  # (roll, pitch, yaw)

        # load low level policy
        # import sys
        # sys.path.append("/home/v1/IsaacLab/logs/rsl_rl/b2z1_flat_direct/2025-05-07_19-55-15/exported")
        self.policy_path = "/home/v1/Projects/visual_wholebody/low-level/logs/b2z1-low/20250408_kp300_kd3/model_78000.pt"
        # print("self.low_level_policy.code", self.low_level_policy.code)  #
        self.low_level_policy = self._load_low_level_model()
        # print("=======self.low_level_policy=========")
        self.ee_goal_local_cart = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # (num_envs, 3)

        self.gripper_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  #gripper_fail mask

    
        if self.num_envs % 3 != 0:
            raise ValueError(
                f"Number of environments ({self.num_envs}) must be divisible by 3 for multi-object setup. "
                f"Please set num_envs to a multiple of 3 (e.g., 3, 6, 9, 12, ...)."
            )
        self.num_env_obj = self.num_envs // 3

        self.default_joint_pos_robot = torch.tensor([[ 0.1000, -0.1000,  0.1000, -0.1000,  
                                                       0.8000,  0.8000,  0.8000,  0.8000,
                                                       0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                                       1.48,   -0.63,   -0.84,   0.0,     1.57, 0]], device=self.device).repeat(self.num_envs, 1)
        self.default_joint_pos_objs_reset = torch.zeros((self.num_envs, 19), device=self.device)  # (num_envs, 19)

        ## object 3: move chair
        if self.object == "table1":
            self.default_joint_pos_obj1 = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  
                                                        0.8000,  0.8000,  0.8000,  0.8000,
                                                        0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                                        2.9, -1.4, -0.0, 0.0, 0.0, -1.2], device=self.device)
            
            self.default_joint_pos_obj1_reset= torch.tensor([[ 0.1000, -0.1000,  0.1000, -0.1000,  
                                                0.8000,  0.8000,  0.8000,  0.8000,
                                                0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                                2.75, -0.78, -0.5, 0.0, 0.0, -1.2]],device=self.device)
                    
            self.default_joint_pos_obj2 =  self.default_joint_pos_obj1.clone() 
            self.default_joint_pos_obj3 = self.default_joint_pos_obj1.clone() 



        elif self.object == "table2":
            self.default_joint_pos_obj1 = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  
                                                        0.8000,  0.8000,  0.8000,  0.8000,
                                                        0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                                        2.8, -1.15, -1.4, 0.0, 0.0, -1.2], device=self.device) # TODO
            self.default_joint_pos_obj2 =  self.default_joint_pos_obj1.clone() 
            self.default_joint_pos_obj3 = self.default_joint_pos_obj1.clone()     

        elif self.object == "movechair1":
            self.default_joint_pos_obj1 = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  
                                                        0.8000,  0.8000,  0.8000,  0.8000,
                                                        0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                                        1.9, -1.72, 0.72, 0.0, 0.0, -1.2], device=self.device)
            self.default_joint_pos_obj2 =  self.default_joint_pos_obj1.clone() 
            self.default_joint_pos_obj3 = self.default_joint_pos_obj1.clone() 

        else:
            # self.default_joint_pos_obj1 = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  
            #                                         0.8000,  0.8000,  0.8000,  0.8000,
            #                                         0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
            #                                         2.9, -1.4, 0.0, 0.0, 0.0, -1.2], device=self.device)
            
            self.default_joint_pos_obj1= torch.tensor([[ 0.1000, -0.1000,  0.1000, -0.1000,  
                                        0.8000,  0.8000,  0.8000,  0.8000,
                                        0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                        2.71, -0.8, -0.5, 0.0, 0.0, -1.2]],device=self.device)
            
            self.default_joint_pos_obj1_reset= torch.tensor([[ 0.1000, -0.1000,  0.1000, -0.1000,  
                                        0.8000,  0.8000,  0.8000,  0.8000,
                                        0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                        2.75, -0.78, -0.5, 0.0, 0.0, -1.2]],device=self.device)

            # self.default_joint_pos_obj1= torch.tensor([[ 0.1000, -0.1000,  0.1000, -0.1000,  
            #                                 0.8000,  0.8000,  0.8000,  0.8000,
            #                                 0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
            #                                 2.7, -1.2, -0.3, 0.0, 0.0, -1.5]],device=self.device)

            self.default_joint_pos_obj2 = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  
                                                    0.8000,  0.8000,  0.8000,  0.8000,
                                                    0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                                    2.8, -1.15, -1.4, 0.0, 0.0, -1.5], device=self.device) # TODO
        
            self.default_joint_pos_obj3 = torch.tensor([0.1000, -0.1000,  0.1000, -0.1000,  
                                                    0.8000,  0.8000,  0.8000,  0.8000,
                                                    0.0000, -1.5000, -1.5000, -1.5000, -1.5000, 
                                                    1.9, -1.72, 0.72, 0.0, 0.0, -1.5], device=self.device)

        self.default_joint_pos_objs = torch.zeros((self.num_envs, 19), device=self.device)  # (num_envs, 19)
        self.default_joint_pos_objs[:self.num_env_obj, :] = self.default_joint_pos_obj1.clone()  # (num_envs, 19)
        self.default_joint_pos_objs[self.num_env_obj:2*self.num_env_obj, :] = self.default_joint_pos_obj2.clone()  # (num_envs, 19)
        self.default_joint_pos_objs[2*self.num_env_obj:, :] = self.default_joint_pos_obj3.clone()  # (num_envs, 19)

        self.default_joint_pos_objs_reset[:self.num_env_obj, :] = self.default_joint_pos_obj1.clone()  # (num_envs, 19)
        self.default_joint_pos_objs_reset[self.num_env_obj:2*self.num_env_obj, :] = self.default_joint_pos_obj2.clone()  # self.default_joint_pos_obj2.clone()  # (num_envs, 19)
        self.default_joint_pos_objs_reset[2*self.num_env_obj:, :] = self.default_joint_pos_obj3.clone()  # self.default_joint_pos_obj3.clone()  # (num_envs, 19)

        self.joint_low_limit = torch.zeros((self.num_envs, 19), device=self.device, dtype=torch.float32) 
        self.joint_high_limit = torch.zeros((self.num_envs, 19), device=self.device, dtype=torch.float32)
        self.joint_low_limit = self.default_joint_pos_objs - 1.0
        self.joint_high_limit = self.default_joint_pos_objs + 1.0

        
        if self.object == "table1":
            self._obj1_gripper_pos = torch.tensor([0.54, -0.3, 0.0], device=self.device) 
            self._obj2_gripper_pos = torch.tensor([0.54, -0.3, 0.0], device=self.device)  # TODO
            self._obj3_gripper_pos = torch.tensor([0.54, -0.3, 0.0], device=self.device)
        elif self.object == "table2":
            self._obj1_gripper_pos = torch.tensor([0.82, -0.5, 0.0], device=self.device) 
            self._obj2_gripper_pos = torch.tensor([0.82, -0.5, 0.0], device=self.device)
            self._obj3_gripper_pos = torch.tensor([0.82, -0.5, 0.0], device=self.device)
        elif self.object == "movechair1":
            self._obj1_gripper_pos = torch.tensor([0.91, 0.0, 0.0], device=self.device) 
            self._obj2_gripper_pos = torch.tensor([0.91, 0.0, 0.0], device=self.device)   # TODO
            self._obj3_gripper_pos = torch.tensor([0.91, 0.0, 0.0], device=self.device)
        else:
            self._obj1_gripper_pos = torch.tensor([0.54, -0.3, 0.0], device=self.device) 
            self._obj2_gripper_pos = torch.tensor([0.82, -0.5, 0.0], device=self.device)  # TODO
            self._obj3_gripper_pos = torch.tensor([0.91, 0.0, 0.0], device=self.device)
        
        self._obj1_root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self._obj2_root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        self._obj3_root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        
        self.object_lin_vel_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # (num_envs, 3) for object linear velocity
        self.object_ang_vel_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # (num_envs, 3) for object angular velocity
        self.object_root_pos_w = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # (num_envs, 3) for object root position in world frame
        self.object_root_quat_w = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)  # (num_envs, 3) for object root position in world frame
        self.object_mass = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)  # (num_envs, 1) for object mass
        self.object_coms = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # (num_envs, 3) for object center of mass
        self.object_dyn_fric_coef = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)  # (num_envs, 1) for object dynamic friction coefficient
        self.object_static_fric_coef = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)  # (num_envs, 1) for object static friction coefficient
        self.object_projected_gravity_b = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # (num_envs, 3) for object projected gravity in body frame
        
        self.object_euler_w = torch.zeros((self.num_envs,3), device=self.device, dtype=torch.float32)  # (num_envs, 1) for object pitch angle

        self.catelogy_encode = torch.zeros((self.num_envs, self.cfg.catelogy_num), device=self.device, dtype=torch.float32)  # (num_envs, N)

        # For multi object WBC
        self.instance_geom = None

        self._ee_posi_in_robot_frame = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)  # (num_envs, 3)


    def _update_selected_objects_state(self):
        self.object_lin_vel_b[:self.num_env_obj, :] = self._object1.data.root_lin_vel_b[:self.num_env_obj, :3]  # (num_envs, 3)
        self.object_lin_vel_b[self.num_env_obj:2*self.num_env_obj, :] = self._object2.data.root_lin_vel_b[self.num_env_obj:2*self.num_env_obj, :3]  # (num_envs, 3)
        self.object_lin_vel_b[2*self.num_env_obj:, :] = self._object3.data.root_lin_vel_b[2*self.num_env_obj:, :3]  # (num_envs, 3)

        self.object_ang_vel_b[:self.num_env_obj, :] = self._object1.data.root_ang_vel_b[:self.num_env_obj, :3]  # (num_envs, 3)
        self.object_ang_vel_b[self.num_env_obj:2*self.num_env_obj, :] = self._object2.data.root_ang_vel_b[self.num_env_obj:2*self.num_env_obj, :3]  # (num_envs, 3)
        self.object_ang_vel_b[2*self.num_env_obj:, :] = self._object3.data.root_ang_vel_b[2*self.num_env_obj:, :3]  # (num_envs, 3)
        ## TODO: Hardcode, depend on the objects inital pose

        # print("object_ang_vel_b", self.object_ang_vel_b)

        self.object_root_pos_w[:self.num_env_obj, :] = self._object1.data.root_pos_w[:self.num_env_obj, :3]  # (num_envs, 3)
        self.object_root_pos_w[self.num_env_obj:2*self.num_env_obj, :] = self._object2.data.root_pos_w[self.num_env_obj:2*self.num_env_obj, :3]  # (num_envs, 3)
        self.object_root_pos_w[2*self.num_env_obj:, :] = self._object3.data.root_pos_w[2*self.num_env_obj:, :3]  # (num_envs, 3)
        # print("object_root_pos_w", self.object_root_pos_w)

        self.object_root_quat_w[:self.num_env_obj, :] = self._object1.data.root_quat_w[:self.num_env_obj, :4]  # (num_envs, 4)
        self.object_root_quat_w[self.num_env_obj:2*self.num_env_obj, :] = self._object2.data.root_quat_w[self.num_env_obj:2*self.num_env_obj, :4]  # (num_envs, 4)
        self.object_root_quat_w[2*self.num_env_obj:, :] = self._object3.data.root_quat_w[2*self.num_env_obj:, :4]  # (num_envs, 4)
        # print("object_root_quat_w", self.object_root_quat_w)

        obj1_material = self._object1.root_physx_view.get_material_properties()
        obj2_material = self._object2.root_physx_view.get_material_properties()
        obj3_material = self._object3.root_physx_view.get_material_properties()

        self.object_mass[:self.num_env_obj] = torch.sum(self._object1.root_physx_view.get_masses()[:self.num_env_obj, :], dim=1)  # 对所有链接求和
        self.object_mass[self.num_env_obj:2*self.num_env_obj] = torch.sum(self._object2.root_physx_view.get_masses()[:self.num_env_obj, :], dim=1)  # 对所有链接求和
        self.object_mass[2*self.num_env_obj:] = torch.sum(self._object3.root_physx_view.get_masses()[:self.num_env_obj, :], dim=1)  # 对所有链接求和
        # print("object_mass", self._object1.root_physx_view.get_masses())

        if self.object == "movechair1":
            self.object_coms[:self.num_env_obj, :] = self._object1.root_physx_view.get_coms()[:self.num_env_obj, 0, :3]  # (num_envs, 3)
            self.object_coms[self.num_env_obj:2*self.num_env_obj, :] = self._object2.root_physx_view.get_coms()[self.num_env_obj:2*self.num_env_obj, 0, :3]  # (num_envs, 3)
            self.object_coms[2*self.num_env_obj:, :] = self._object3.root_physx_view.get_coms()[2*self.num_env_obj:, 0, :3]  # (num_envs, 3)
        elif self.object == "mix":
            self.object_coms[:self.num_env_obj, :] = self._object1.root_physx_view.get_coms()[:self.num_env_obj, :3]  # (num_envs, 3)
            self.object_coms[self.num_env_obj:2*self.num_env_obj, :] = self._object2.root_physx_view.get_coms()[self.num_env_obj:2*self.num_env_obj, :3]  # (num_envs, 3)
            self.object_coms[2*self.num_env_obj:, :] = self._object3.root_physx_view.get_coms()[2*self.num_env_obj:, 0, :3]  # (num_envs, 3)
        else:
            self.object_coms[:self.num_env_obj, :] = self._object1.root_physx_view.get_coms()[:self.num_env_obj, :3]  # (num_envs, 3)
            self.object_coms[self.num_env_obj:2*self.num_env_obj, :] = self._object2.root_physx_view.get_coms()[self.num_env_obj:2*self.num_env_obj, :3]  # (num_envs, 3)
            self.object_coms[2*self.num_env_obj:, :] = self._object3.root_physx_view.get_coms()[2*self.num_env_obj:, :3]  # (num_envs, 3)

        self.object_static_fric_coef[:self.num_env_obj] = obj1_material[:self.num_env_obj, 0, 0]  # static friction coefficient
        self.object_static_fric_coef[self.num_env_obj:2*self.num_env_obj] = obj2_material[self.num_env_obj:2*self.num_env_obj, 0, 0]  # static friction coefficient
        self.object_static_fric_coef[2*self.num_env_obj:] = obj3_material[2*self.num_env_obj:, 0, 0]  # static friction coefficient

        self.object_dyn_fric_coef[:self.num_env_obj] = obj1_material[:self.num_env_obj, 0, 1]  # dynamic friction coefficient
        self.object_dyn_fric_coef[self.num_env_obj:2*self.num_env_obj] = obj2_material[self.num_env_obj:2*self.num_env_obj, 0, 1]  # dynamic friction coefficient
        self.object_dyn_fric_coef[2*self.num_env_obj:] = obj3_material[2*self.num_env_obj:, 0, 1]  # dynamic friction coefficient

        self.object_projected_gravity_b[:self.num_env_obj, :] = self._object1.data.projected_gravity_b[:self.num_env_obj, :3]  # (num_envs, 3)
        self.object_projected_gravity_b[self.num_env_obj:2*self.num_env_obj, :] = self._object2.data.projected_gravity_b[self.num_env_obj:2*self.num_env_obj, :3]  # (num_envs, 3)
        self.object_projected_gravity_b[2*self.num_env_obj:, :] = self._object3.data.projected_gravity_b[2*self.num_env_obj:, :3]  # (num_envs, 3)


    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        if isinstance(self.cfg, B2Z1MultiObjWBCGNNPLANRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        

        if self.object == "table1":
            self._object1 = RigidObject(self.cfg.table1_table1)
            self._object2 = RigidObject(self.cfg.table2_table1)
            self._object3 = RigidObject(self.cfg.movechair1_table1)
            self.scene.rigid_objects["object1"] = self._object1
            self.scene.rigid_objects["object2"] = self._object2
            self.scene.rigid_objects["object3"] = self._object3
        elif self.object == "table2":
            self._object1 = RigidObject(self.cfg.table1_table2)
            self._object2 = RigidObject(self.cfg.table2_table2)
            self._object3 = RigidObject(self.cfg.movechair1_table2)
            self.scene.rigid_objects["object1"] = self._object1
            self.scene.rigid_objects["object2"] = self._object2
            self.scene.rigid_objects["object3"] = self._object3
        elif self.object == "movechair1":
            self._object1 = Articulation(self.cfg.table1_chair)
            self._object2 = Articulation(self.cfg.table2_chair)
            self._object3 = Articulation(self.cfg.movechair1_chair)
            self.scene.articulations["object1"] = self._object1
            self.scene.articulations["object2"] = self._object2
            self.scene.articulations["object3"] = self._object3
        else:
            self._object1 = RigidObject(self.cfg.table1_mix)
            self._object2 = RigidObject(self.cfg.table2_mix)
            self._object3 = Articulation(self.cfg.movechair1_mix)
            self.scene.rigid_objects["object1"] = self._object1
            self.scene.rigid_objects["object2"] = self._object2
            self.scene.articulations["object3"] = self._object3

        # self.marker_goal_vel = VisualizationMarkers(self.cfg.marker_goal_vel)
        # self.marker_cur_vel = VisualizationMarkers(self.cfg.marker_cur_vel)

        
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone() # x,y,yaw, 6 arm delta_theta
        # scale action
        action_scale_tensor = torch.tensor(self.cfg.action_scale, device=self.device)
        self._processed_actions = action_scale_tensor * self._actions 
        # clip action
        action_clip_tensor = torch.tensor(self.cfg.action_clip, device=self.device)
        self._processed_actions = torch.clamp(self._processed_actions, -action_clip_tensor, action_clip_tensor)
        # wait for gripper
        self._processed_actions[self.episode_length_buf < self.reset_steps] = 0

        self.last_action = self._processed_actions.clone()  # for logging
        # print("_processed_actions", self._processed_actions[:, 3])


  
    def _apply_action(self):
        # Gripper the object in the beginning
        gripper_mask = self.episode_length_buf > 1 
        if gripper_mask.any(): 
            self.gripper_angle[gripper_mask] = torch.clamp(
                -1.2 + self.episode_length_buf[gripper_mask].unsqueeze(-1) * self.gripper_vel, max= -0.05
            )

        # print("default joint", self._robot.data.default_joint_pos)
        self.processed_actions_with_fixed_joint = self.default_joint_pos_objs.clone()  # (num_envs, 19)  
        # Set default joint positions for the objects

        ## Set Joint 1 ~ 6
        start_mask = self.episode_length_buf > self.reset_steps
        start_mask_2 = self.episode_length_buf > (self.reset_steps + 20)

        un_start_mask = self.episode_length_buf <= self.reset_steps
        self.processed_actions_with_fixed_joint[un_start_mask, :] = self.default_joint_pos_objs_reset[un_start_mask, :]  # (num_envs, 19)  

        self.processed_actions_with_fixed_joint[:, 18] = self.gripper_angle.squeeze(1) 

        if start_mask.any():  # 
            # if start_mask_2.any():
            self.processed_actions_with_fixed_joint[start_mask_2, 8] = self._processed_actions[start_mask_2, 3] + \
                                                                        self._robot.data.joint_pos[start_mask_2, 8]  # joint 1    
            self.processed_actions_with_fixed_joint[start_mask_2, 13:18] = self._processed_actions[start_mask_2, 4:9] + \
                                                                            self._robot.data.joint_pos[start_mask_2, 13:18]  # joint 2 ~ 6
            # else:
            # self.processed_actions_with_fixed_joint[start_mask, 8] += self._processed_actions[start_mask, 3] 
            # self.processed_actions_with_fixed_joint[start_mask, 13:18] += self._processed_actions[start_mask, 4:9]
            
            
            # self.processed_actions_with_fixed_joint[start_mask, 8] = torch.clamp(self.processed_actions_with_fixed_joint[start_mask, 8], 
            #                                                                 self.joint_low_limit[start_mask, 8], self.joint_high_limit[start_mask, 8])
            # self.processed_actions_with_fixed_joint[start_mask, 13:18] = torch.clamp(self.processed_actions_with_fixed_joint[start_mask, 13:18], 
            #                                                                     self.joint_low_limit[start_mask, 13:18], self.joint_high_limit[start_mask, 13:18])
            
        ## low level policy
        vx_body = self._processed_actions[:, 0]
        vy_body = self._processed_actions[:, 1]
        ang_vel = self._processed_actions[:, 2]
        self._low_commands = torch.stack([vx_body, vy_body, ang_vel], dim=-1)  # (num_envs, 3)
        # self._low_commands = self._commands[:, :3].clone()
        # print("self._low_commands", self._low_commands)

        # self.cfg.object_vel_cmd = [0.4, 0.0, 0.0]
    
    #     # ===============================================================================================
    #     # ===============================================================================================
    #     # TODO: planner
    #     # WAIT_ROBOT_PATH, ROBOT_TRACKING, GRASPING, WAIT_OBJECT_PATH, OBJECT_TRACKING, RELEASING
    #     if self.cfg.task_state == "WAIT_ROBOT_PATH":
    #         self._low_commands[:, :3] = torch.tensor([0.0, 0.0, 0.0], device=self.device)  # robot fix
    #         self.processed_actions_with_fixed_joint = self._robot.data.default_joint_pos   # arm default fix

    #     if self.cfg.task_state == "ROBOT_TRACKING":
    #         self._low_commands[:, :3] = torch.tensor(self.cfg.robot_vel_cmd, device=self.device) # robot move
    #         self.processed_actions_with_fixed_joint = self._robot.data.default_joint_pos   # arm default fix

    #     if self.cfg.task_state == "GRASPING" or self.cfg.task_state == "RELEASING":
    #         self._low_commands[:, :3] = torch.tensor([0.0, 0.0, 0.0], device=self.device)  # robot fix
    #         grasping_joint = torch.tensor(self.cfg.joint_cmd, device=self.device)
    #         self.processed_actions_with_fixed_joint[start_mask, 8] = grasping_joint[0]     # arm move
    #         self.processed_actions_with_fixed_joint[start_mask, 13:19] = grasping_joint[1:7]
        
    #     if self.cfg.task_state == "WAIT_OBJECT_PATH":
    #         self._low_commands[:, :3] = torch.tensor([0.0, 0.0, 0.0], device=self.device)   # robot fix
    #         self.processed_actions_with_fixed_joint = self.default_joint_pos_objs.clone()   # arm object fix

    #     if self.cfg.task_state == "OBJECT_TRACKING":
    #         self._commands[:, :3] = torch.tensor(self.cfg.object_vel_cmd, device=self.device)

    #    # ================================================================================================
    #    # ================================================================================================

        # compute obsevation
        # print("low-level command:",  self._low_commands)

        low_level_obs = self._compute_low_level_observation()

        action_low_level = self.low_level_policy(low_level_obs.detach(), hist_encoding=True)
        action_low_level[:, 12:] = 0.0  # real

        actions_low_level = self._reindex_real2Isaacsim(action_low_level)
        self._low_action = actions_low_level.clone()  # for logging
        self._previous_low_actions[:] = actions_low_level[:] 

        action_scale_tensor = torch.tensor(self.cfg.action_scale_low_level, device=self.device)
        self.processed_actions_low_level[:,:18] = action_scale_tensor * self._low_action + self.default_joint_pos_objs[:,:18]


        # print("joint_names", self._robot.data.joint_names)
        mask = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12]
        self.processed_actions_with_fixed_joint[:, mask] = self.processed_actions_low_level[:, mask]

        # Set joint position target
        self._robot.set_joint_position_target(self.processed_actions_with_fixed_joint)

        # print("arm joint action",self._processed_actions[0, 4:9] )  # (num_envs, 19)


        # ======= For online sysID ========
        self.iter_time =  self.iter_time + 1
        if  self.iter_time > 4000:
            self.iter_time = 4010


    def _compute_low_level_observation(self):
        
        dof_pos = self._reindex_Isaacsim2real((self._robot.data.joint_pos - self.default_joint_pos_robot) * 1.0)[:, :-1] # 18
        
        # dof_pos = self._reindex_Isaacsim2real((self._robot.data.joint_pos - self.default_joint_pos_objs) * 1.0)[:, :-1] # 18

        dof_vel = self._reindex_Isaacsim2real(self._robot.data.joint_vel * 0.05)[:, :-1] # 18

        # default_joint_pos_reindex = self._reindex_Isaacsim2real(self._robot.data.default_joint_pos)
        # print("=== default_joint_pos_reindex ===", default_joint_pos_reindex)

        # ee_goal_local_cart = torch.tensor([[0.3991, -0.0004,  0.047]], device = self._commands.device).repeat(self.num_envs, 1)
        ee_goal_local_cart = self.ee_goal_local_cart.clone()  # (num_envs, 3)


        # ee_goal_local_cart[:self.num_env_obj, 2] = ee_goal_local_cart[:self.num_env_obj, 2] - 0.1

        priv_buf = torch.tensor(
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0795, 0.5203, -0.1516, -0.0065,
            0.0467, 0.2631, 0.1297, 0.1543, -0.1086, -0.1943, 0.0883, 0.2819,
            0.2323, -0.0110], device=self.device, dtype=torch.float32
            ).repeat(self.num_envs, 1)  
        
        self._step_contact_targets()

        obs = torch.cat(
        [
            tensor
            for tensor in (
                self._get_body_orientation(), # 2
                self._robot.data.root_ang_vel_b * 0.25, # angular velocity # 3
                dof_pos, # 18
                dof_vel, # 18
                self._reindex_Isaacsim2real(self._low_action )[:, :12], # 12
                torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32),  #
                self._low_commands[:, :3] * self.commands_scale, # 3
                ee_goal_local_cart, # 3
                torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32),  # 
                self.gait_indices.unsqueeze(1), # 1
                self.clock_inputs, # 4
            )
            if tensor is not None
        ],
        dim=-1,
        )

        # print("body_orientation():", self._get_body_orientation())
        # print("ang_vel_base", self._robot.data.root_ang_vel_b * 0.25)
        # print("dof_pos:", dof_pos)
        # print("dof_vel:", dof_vel)
        # print("self._actions:", self._reindex_Isaacsim2real(self._low_action)[:, :12])
        # print("self._commands:", command_vel_base * self.commands_scale)
        # print("ee_goal_local_cart", ee_goal_local_cart)
        # print("self.gait_indices.unsqueeze(1):", self.gait_indices.unsqueeze(1))
        # print("self.clock_inputs:", self.clock_inputs)

        obs_buf = torch.cat([obs, priv_buf, self.obs_history_buf_low_level.view(self.num_envs, -1)], dim=-1)

        self.obs_history_buf_low_level = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs] * self.cfg.history_len, dim=1),
            torch.cat([
                self.obs_history_buf_low_level[:, 1:],
                obs.unsqueeze(1)
            ], dim=1)
        )

        return obs_buf
   


    def _get_observations(self) -> dict:
        self._update_selected_objects_state()  # Update the state of selected objects

        self._preprevious_actions = self._previous_actions.clone()
        self._previous_actions = self._actions.clone()
        height_data = None

        if isinstance(self.cfg, B2Z1MultiObjWBCGNNPLANRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)

        dof_pos = self._reindex_Isaacsim2real((self._robot.data.joint_pos - self.default_joint_pos_objs))[:, :-1] # 18
        dof_vel = self._reindex_Isaacsim2real(self._robot.data.joint_vel)[:, :-1] # 18
        dof_pos_arm = dof_pos[:, 12:18]
        dof_vel_arm = dof_vel[:, 12:18]

        # object pose in robot frame
        obj_posi_relative = self.object_root_pos_w - self._robot.data.root_pos_w  # (num_envs, 3) # self._object1.data.root_pos_w  
        robot_quat_inv = self.quat_inverse_safe(self._robot.data.root_quat_w)  # (num_envs, 4)
        obj_posi_in_robot_frame = quat_rotate(robot_quat_inv, obj_posi_relative)  # (num_envs, 3)
        obj_quat_in_robot_frame = quat_mul(robot_quat_inv, self.object_root_quat_w) # self._object1.data.root_quat_w)
        # print("object position", obj_posi_in_robot_frame)
        roll, pitch, yaw = self._euler_from_quat(obj_quat_in_robot_frame)
        roll = (roll + np.pi) % (2 * np.pi)
        self.obj_euler_in_robot_frame = torch.stack([roll, pitch, yaw], dim=-1)  # (num_envs, 3)
        # print("object orient", self.obj_euler_in_robot_frame)
        
        ## link pose in robot frame
        # metadata = self.robot_view_sim._metadata
        # joint_indices = metadata.link_indices["link06"]
        # #link00:5, link02:18, link03:23, link04:24, link05:25, link06:26
        # print("joint_indices", joint_indices) 

        link_transforms = self._robot.root_physx_view.get_link_transforms()  # shape (N_envs, max_links, 7)
        link_indices = torch.tensor([5, 18, 23, 24, 25, 26, 28], device=self.device) # link00, 02, 03, 04, 05, 06, ee
        link_pose_global = link_transforms[:, link_indices]  # shape (N_envs, 7, 7)
        

        # 
        robot_base_pos = self._robot.data.root_pos_w  # (num_envs, 3)
        num_links = link_pose_global.shape[1]  # 7
        robot_pos_expanded = robot_base_pos.unsqueeze(1).expand(-1, num_links, -1)  # (num_envs, 6, 3)
        robot_quat_inv_expanded = robot_quat_inv.unsqueeze(1).expand(-1, num_links, -1)  # (num_envs, 6, 4)

        link_posi_relative = link_pose_global[:, :, :3] - robot_pos_expanded  # (num_envs, 6, 3)
        link_posi_base = quat_rotate(robot_quat_inv_expanded, link_posi_relative)  # (num_envs, 6, 3)
        link_quat_in_robot_frame = quat_mul(robot_quat_inv_expanded, link_pose_global[:, :, 3:])  # (num_envs, 6, 4)
        link_pose_in_robot_frame = torch.cat([link_posi_base, link_quat_in_robot_frame], dim=-1)  # (num_envs, 6, 7)
        # print("link_pose_in_robot_frame", link_pose_in_robot_frame[:, :, :7])  # (num_envs, 6, 7)

        self._ee_posi_in_robot_frame = link_pose_in_robot_frame[:, -1, :3]  # (num_envs, 3)  # end-effector position in robot frame
        ee_quat_in_robot_frame = link_pose_in_robot_frame[:, -1, 3:]  # (num_envs, 4)  # end-effector quaternion in robot frame
        # print("_ee_posi_in_robot_frame", self._ee_posi_in_robot_frame)
        # print("ee_quat_in_robot_frame", ee_quat_in_robot_frame) 

        self.ee_goal_local_cart[:] = self._ee_posi_in_robot_frame[:]

        link_pose_in_robot_frame_obs = link_pose_in_robot_frame.reshape(-1, 49)  # (num_envs, 49)
        robot_joint_pos = self._reindex_Isaacsim2real(self._robot.data.joint_pos)[:, :-1] # 18
        gripper_fail_expanded = ~self.gripper_fail.unsqueeze(-1).expand(self.num_envs, -1)  # [num_env, 1]

        lin_vel_scale = 2.0
        ang_vel_scale = 0.25
        dof_vel_scale = 0.05
        
        # self.cfg.object_vel_cmd = [0.4, 0, 0]
        # # if self.cfg.task_state == "OBJECT_TRACKING":
        # self._commands[:, :3] = torch.tensor(self.cfg.object_vel_cmd, device=self.device)

        ## ===================== TODO: Our observation num: 70 =======================
        actor_obs = torch.cat(
            [
                tensor
                for tensor in (
                    dof_pos, # 18
                    dof_vel * dof_vel_scale, # 18
                    self._get_body_orientation(), # 2
                    self._robot.data.root_ang_vel_b * ang_vel_scale, # 3
                    self.last_action, # 9 
                    self._commands[:, :3] * self.commands_scale, # 3 (x,y,yaw)
                    self._ee_posi_in_robot_frame,  # 3
                    ee_quat_in_robot_frame,  # 4 
                    obj_posi_in_robot_frame, # 3, 
                    obj_quat_in_robot_frame, # 4,
                    self.catelogy_encode,    # 3 kinds of catelogy
                )
                if tensor is not None
            ],
            dim=-1,
        )
        
        # print("self.default_joint_pos_objs", self.default_joint_pos_objs)
        # print("dof_pos", dof_pos[0,:])  # (num_envs, 18)
        # print("dof_vel", dof_vel[0,:]* dof_vel_scale)  # (num_envs, 18)
        # print("_commands", self._commands[0, :3])  # (num_envs, 70)
        # print("ee_posi_in_robot_frame", self._ee_posi_in_robot_frame[0,:])  # (num_envs, 3)
        # print("ee_quat_in_robot_frame", ee_quat_in_robot_frame[0,:])  # (num_envs, 4)
        # print("obj_posi_in_robot_frame", obj_posi_in_robot_frame[0,:])  # (num_envs, 3)
        # print("obj_quat_in_robot_frame", obj_quat_in_robot_frame[0,:])  # (num_envs, 4)


# dof_pos tensor([-0.0056,  0.0414,  0.1087,  0.0692, -0.1335,  0.1202,  0.0352, -0.2319,
#          0.1726, -0.0276,  0.0864, -0.0082,  0.0236,  0.1886, -0.3583,  0.1055,
#          0.0201,  0.0249], device='cuda:0')

# dof_pos tensor([-0.0082,  0.0580,  0.1591,  0.0817, -0.1304,  0.1341,  0.0035, -0.2193,
#          0.0974, -0.0317,  0.0721,  0.2010,  0.0073, -0.0074, -0.0078, -0.0551,
#         -0.0233,  0.0496], device='cuda:0')


        # dof_pos tensor([ 0.0311,  0.0325, -0.1000,  0.0136, -0.0344,  0.0954, -0.0247, -0.0284,
        #  0.0288,  0.0406,  0.0486, -0.1059,  0.0645, -0.0002,  0.0397, -0.0617,
        # -0.0579,  0.0320], device='cuda:0')
        # dof_vel tensor([ 0.0146,  0.0716, -0.1722, -0.0015,  0.0306, -0.0081, -0.0112,  0.0028,
        #         0.0296, -0.0403,  0.0724, -0.2029,  0.0164, -0.0181,  0.0042, -0.0085,
        #         0.0005,  0.0225], device='cuda:0')
        # _commands tensor([0.8000, 0.0000, 0.0000], device='cuda:0')
        # ee_posi_in_robot_frame tensor([0.6840, 0.0220, 0.3669], device='cuda:0')
        # ee_quat_in_robot_frame tensor([-0.0393,  0.2981, -0.0308,  0.9532], device='cuda:0')
        # obj_posi_in_robot_frame tensor([ 0.8794,  0.0307, -0.5744], device='cuda:0')
        # obj_quat_in_robot_frame tensor([0.9997, 0.0029, 0.0149, 0.0206], device='cuda:0')



        # critic: 161 
        tensor_list = []
        for tensor in (
            dof_pos,  # 18
            dof_vel * dof_vel_scale,  # 18
            self.default_joint_pos_objs[:, :18], # 18
            robot_joint_pos,  # 18

            self._get_body_orientation(), # 2
            self._robot.data.root_ang_vel_b * ang_vel_scale, # 3
            self.last_action, # 9 

            self._commands[:, :3] * self.commands_scale,  # 3 (x,y,yaw)
            link_pose_in_robot_frame_obs,  # 49
            gripper_fail_expanded,  # ee contact state

            obj_posi_in_robot_frame,  # 3
            obj_quat_in_robot_frame,  # 4
 
            self._robot.data.root_lin_vel_b * lin_vel_scale,  # 3
            self._robot.data.root_ang_vel_b * ang_vel_scale,  # 3 

            self.object_lin_vel_b * lin_vel_scale,   # 3
            self.object_ang_vel_b * ang_vel_scale,   # 3

            self.object_static_fric_coef.unsqueeze(-1),  # 1obj_coms
            self.object_mass.unsqueeze(-1),  # 1
            self.object_dyn_fric_coef.unsqueeze(-1),  # 1
        ):
            if tensor is not None:
            
                tensor_list.append(tensor.to(self.device))

        critic_obs = torch.cat(tensor_list, dim=-1)

        obs_buf = torch.cat([actor_obs, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([actor_obs] * self.cfg.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                actor_obs.unsqueeze(1)
            ], dim=1)
        )

        observations = {
            "policy": obs_buf,
            "critic": critic_obs,
        }


        ## ===================== TODO: Vallian method: 70 =======================
        # actor obs
        # actor_obs = torch.cat(
        #     [
        #         tensor
        #         for tensor in (
        #             # dof_pos, # 18
        #             # dof_vel * dof_vel_scale, # 18
        #             # self._get_body_orientation(), # 2
        #             # self._robot.data.root_ang_vel_b * ang_vel_scale, # 3
        #             # self.last_action, # 9 
        #             dof_pos_arm, # 6
        #             dof_vel_arm * dof_vel_scale, # 6
        #             self._commands[:, :3] * self.commands_scale, # 3 (x,y,yaw)

        #             torch.zeros(self.num_envs, 55, device=self.device, dtype=torch.float32),  
        #         )
        #         if tensor is not None
        #     ],
        #     dim=-1,
        # )
        # observations = {"policy": actor_obs,  "critic": actor_obs} # no history



        if self.IF_LOG:
            self.log_vel_tracking_result()

        # self.log_joint_effort()

        # self.log_joint_position()

        return observations


    def _get_rewards(self) -> torch.Tensor:
        self._update_selected_objects_state()  # Update the state of selected objects

        # ===== Object =====
        ## linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self.object_lin_vel_b[:, :2]), dim=1) # self._object1.data.root_lin_vel_b[:, :2]
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        ## lin_vel_penalty        
        lin_vel_l2 = torch.norm(self.object_lin_vel_b[:, :2] - self._commands[:, :2], dim=1) # self._object1.data.root_lin_vel_b[:, :2] 
        track_lin_vel_reward_l2 = 1.0 / (1.0 + lin_vel_l2)  # 
        lin_vel_penalty = lin_vel_l2 

        ## ang vel tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self.object_ang_vel_b[:, 2]) # self._object1.data.root_ang_vel_b[:, 2]
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        ## ang vel penalty
        yaw_rate_l2 = torch.abs(self.object_ang_vel_b[:, 2] - self._commands[:, 2])  # shape: [N]
        track_ang_vel_reward_l2 = 1.0 / (1.0 + yaw_rate_l2)  # 
        yaw_rate_penalty = yaw_rate_l2

        ## alive
        alive = torch.ones(self.num_envs, dtype=torch.float, device=self.device) * 0.5

        ## alignment reward: keep the objects in front of the robot
        rel_pos_robot = torch.bmm(
            self.quat_to_rot_matrix(self._robot.data.root_quat_w).transpose(1, 2),
            (self.object_root_pos_w - self._robot.data.root_pos_w).unsqueeze(-1)  # self._object1.data.root_pos_w 
        ).squeeze(-1)  # (num_envs, 3)

        yaw_diff = (
            (self._euler_from_quat(self.object_root_quat_w)[2] -         # self._object1.data.root_quat_w
            self._euler_from_quat(self._robot.data.root_quat_w)[2] + torch.pi)
            % (2 * torch.pi) - torch.pi
        )
        # print("yaw_diff1", yaw_diff)

        x_error = rel_pos_robot[:, 0] - 1.0  # 
        y_error = rel_pos_robot[:, 1]        #
        pos_alignment_reward = torch.exp(-y_error**2 / 0.2) + torch.exp(-x_error**2 / 0.2)
        yaw_alignment_reward = -torch.abs(yaw_diff) / torch.pi

        # print("alignment_reward", alignment_reward)
        # print("x_error", x_error)
        # print("y_error", y_error)
        # print("yaw_diff", yaw_diff)

        # distance keep the objects in front of the robot
        distance_ee2base_x = self._ee_posi_in_robot_frame[:, 0]
        distance_threshold = 0.6
        distance_penalty = 1.0 / (1.0 + torch.exp(200 * (distance_ee2base_x - distance_threshold)))
        # print("distance_ee2base_x", distance_ee2base_x)
        # print("distance_penalty", distance_penalty)
        
        ## z velocity tracking
        z_vel_error = torch.square(self.object_lin_vel_b[:, 2])  # self._object1.data.root_lin_vel_b[:, 2]
        # print("z_vel_error:", z_vel_error)

        ## angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self.object_ang_vel_b[:, :2]), dim=1)  # self._object1.data.root_ang_vel_b[:, :2]
        # print("ang_vel_error:", ang_vel_error)

        ## flat orientation
        flat_orientation = torch.sum(torch.square(self.object_projected_gravity_b[:, :2]), dim=1) # self._object1.data.projected_gravity_b[:, :2]
        # print("flat_orientation:", flat_orientation)

        ## vel smooth penalty
        current_lin_vel = self.object_lin_vel_b[:, :2]  # (vx, vy)
        current_ang_vel = self.object_ang_vel_b[:, 2:3]  # omega
        current_vel = torch.cat([current_lin_vel, current_ang_vel], dim=-1)  # (vx, vy, omega)

        # print("current_vel", current_vel)
        lin_vel_change_penalty = torch.norm(current_lin_vel - self._previous_objects_vel[:, :2], dim=-1)
        ang_vel_change_penalty = torch.abs(current_ang_vel.squeeze(-1) - self._previous_objects_vel[:, 2])

        self._previous_objects_vel = current_vel.clone()

        # ===== Robot =====
        # base velocity smooth
        robot_base_lin_vel = self._robot.data.root_lin_vel_b[:, :2]

        ## action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        action_rate2 = torch.sum(
                            torch.square(self._actions - 2 * self._previous_actions + self._preprevious_actions),
                            dim=1)
        ## joint
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque[:, self.arm_indices]), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc[:, self.arm_indices]), dim=1)

        
        joint_diff = self._robot.data.joint_pos[:, self.arm_indices] - self.default_joint_pos_objs[:, self.arm_indices]
        joint_default_pos = torch.sum(torch.abs(joint_diff), dim=1)


        # joint efforts

        joint_efforts = self.robot_view_sim.get_measured_joint_efforts()  # (num_envs, 19)

        # joint_forces = self.robot_view_sim.get_measured_joint_forces()  # (num_envs, 19)
        # joint_force = joint_forces[:, joint_indices+1] # link 1 ~ 6 + gripper

        joint_efforts_arm = joint_efforts[:, self.arm_indices_with_gripper] # (num_envs, 7)
        joint_efforts_arm = torch.sum(torch.square(joint_efforts[:, self.arm_indices_with_gripper]), dim=1) # (num_envs,)
        # joint_efforts_arm = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)  # (num_envs,)

        ## undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_undesired_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contact = torch.sum(is_undesired_contact, dim=1)
        

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_exp_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_exp_reward_scale * self.step_dt,
            "alive": alive * self.cfg.alive_reward_scale * self.step_dt,
            "pos_alignment_reward": pos_alignment_reward * self.cfg.pos_alignment_reward_scale * self.step_dt,
            "yaw_alignment_reward": yaw_alignment_reward * self.cfg.yaw_alignment_reward_scale * self.step_dt,

            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "lin_vel_change_penalty": lin_vel_change_penalty * self.cfg.lin_vel_change_penalty_scale * self.step_dt,
            "ang_vel_change_penalty": ang_vel_change_penalty * self.cfg.ang_vel_change_penalty_scale * self.step_dt,
            "distance_penalty": distance_penalty * self.cfg.distance_penalty_scale * self.step_dt,

            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "action_rate2_l2": action_rate2 * self.cfg.action_rate2_reward_scale * self.step_dt,

            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt, 
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "joint_default_pos_l2": joint_default_pos * self.cfg.joint_default_pos_reward_scale * self.step_dt,  #
            "joint_efforts_arm_l2": joint_efforts_arm * self.cfg.joint_efforts_arm_reward_scale * self.step_dt,  #
            
            "undesired_contacts": contact * self.cfg.undesired_contact_reward_scale * self.step_dt,     

            "track_lin_vel_l2": track_lin_vel_reward_l2 * self.cfg.lin_vel_l2_reward_scale * self.step_dt,
            "track_ang_vel_l2": track_ang_vel_reward_l2 * self.cfg.ang_vel_l2_reward_scale * self.step_dt,
            "lin_vel_penalty": lin_vel_penalty * self.cfg.lin_vel_penalty_scale * self.step_dt,
            "yaw_rate_penalty": yaw_rate_penalty * self.cfg.yaw_rate_penalty_scale * self.step_dt,
        }
        for key, val in rewards.items():
            if val.ndim == 0:
                print(f"[ERROR] reward '{key}' has no batch dimension! shape: {val.shape}")
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            # if value.ndim == 0:
            #     print(f"[ERROR] reward '{key}' has no batch dimension! shape: {value.shape}")
            self._episode_sums[key] += value
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._update_selected_objects_state()  # Update the state of selected objects

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # base contact
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        # print("died", died)

        # reset if objects fall down
        obj_roll, obj_pitch, obj_yaw = self._euler_from_quat(self.object_root_quat_w)  # (num_envs, 3)
        obj_roll_mask = torch.abs(obj_roll) > 1.  # roll > 1
        obj_pitch_mask = torch.abs(obj_pitch) > 1.  # pitch > 1
        orientation_mask = obj_roll_mask | obj_pitch_mask

        # gripper fail
        start_mask = self.episode_length_buf > self.reset_steps

        # gripperMover_force = net_contact_forces[:, :, self._gripperMover_ids]
        # print("gripperMover_force", gripperMover_force)

        gripper_fail = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        if start_mask.any():  # 
            gripperMover_contact = torch.max(torch.norm(net_contact_forces[start_mask, :, self._gripperMover_ids], dim=-1), dim=1)[0] > 1.0
            gripperStator_connect = torch.max(torch.norm(net_contact_forces[start_mask, :, self._gripperStator_ids], dim=-1), dim=1)[0] > 1.0
            # print("gripperMover_contact", gripperMover_contact)
            gripper_fail[start_mask] = (gripperMover_contact == False) & (gripperStator_connect == False)  # 
        
        self.gripper_fail = gripper_fail.clone()  # Update the gripper fail state
        # contact between objects and robot

        done = gripper_fail | died | orientation_mask

        # done = False
        return done, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        # self._object3.reset(env_ids)

        super()._reset_idx(env_ids)

        if env_ids is not None and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            # self.episode_length_buf[:] = 0
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(20))
        else:
            self.episode_length_buf[env_ids] = torch.randint_like(self.episode_length_buf[env_ids], high=int(20))

        self._actions[env_ids, :] = 0.0
        self._previous_actions[env_ids, :] = 0.0
        self._preprevious_actions[env_ids, :] = 0.0
        self._previous_low_actions[env_ids, :] = 0.0

        # Sample new commands
        batch_size = len(env_ids)
        self._commands[env_ids, 0] = torch.rand(batch_size, device=self.device) * 1.0 - 0.5  # x
        self._commands[env_ids, 1] = 0.0  # y
        self._commands[env_ids, 2] = torch.rand(batch_size, device=self.device) * 1.0 - 0.5  # z

        # Reset robot state        
        joint_pos = self.default_joint_pos_objs_reset[env_ids].clone()  # (num_envs, 19)
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # print("===joint pos===", joint_pos.shape)

        # Reset objects state
        default_objects_pos_obj1 = self._object1.data.default_root_state[env_ids].clone()  # (num_envs, 13)
        default_objects_pos_obj2 = self._object2.data.default_root_state[env_ids].clone()  # (num_envs, 13)
        default_objects_pos_obj3 = self._object3.data.default_root_state[env_ids].clone()  # (num_envs, 13)

        default_objects_pos_obj1[:, :3] = self._obj1_gripper_pos     
        default_objects_pos_obj2[:, :3] = self._obj2_gripper_pos   
        default_objects_pos_obj3[:, :3] = self._obj3_gripper_pos     

        default_objects_pos_obj1[:, :3] += self._terrain.env_origins[env_ids]
        default_objects_pos_obj2[:, :3] += self._terrain.env_origins[env_ids]
        default_objects_pos_obj3[:, :3] += self._terrain.env_origins[env_ids]

        # put the objects aside
        remote_pos_1 = torch.tensor([10000.0, 100.0, 0.05], device=self.device)
        remote_pos_2 = torch.tensor([100.0, 10000.0, 0.00], device=self.device)
        remote_pos_3 = torch.tensor([10000.0, 10000.0, 0.28], device=self.device) 


        group1_mask = env_ids < self.num_env_obj                           # 
        group2_mask = (env_ids >= self.num_env_obj) & (env_ids < 2 * self.num_env_obj)  #  N/3  
        group3_mask = env_ids >= 2 * self.num_env_obj                     #  N/3


        if group1_mask.any():
            default_objects_pos_obj2[group1_mask, :3] = remote_pos_2
            default_objects_pos_obj3[group1_mask, :3] = remote_pos_3
            # print("joint_pos", joint_pos)
        if group2_mask.any():
            default_objects_pos_obj1[group2_mask, :3] = remote_pos_1
            default_objects_pos_obj3[group2_mask, :3] = remote_pos_3
        if group3_mask.any():
            default_objects_pos_obj1[group3_mask, :3] = remote_pos_1
            default_objects_pos_obj2[group3_mask, :3] = remote_pos_2

        # reset robot
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # import ipdb; ipdb.set_trace()
        # reset objects
        self._object1.write_root_pose_to_sim(default_objects_pos_obj1[:, :7], env_ids)
        self._object2.write_root_pose_to_sim(default_objects_pos_obj2[:, :7], env_ids)
        self._object3.write_root_pose_to_sim(default_objects_pos_obj3[:, :7], env_ids)
        self._object1.write_root_velocity_to_sim(default_objects_pos_obj1[:, 7:], env_ids)
        self._object2.write_root_velocity_to_sim(default_objects_pos_obj2[:, 7:], env_ids)
        self._object3.write_root_velocity_to_sim(default_objects_pos_obj3[:, 7:], env_ids)

        # reset buffer
        self.obs_history_buf[env_ids, :, :] = 0.
        self.obs_history_buf_low_level[env_ids, :, :] = 0.

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

    def _get_body_orientation(self) -> torch.Tensor:
        r, p, y = self._euler_from_quat(self._robot.data.root_quat_w)
        body_angles = torch.stack([r, p, y], dim=-1)
        return body_angles[:, :-1]
    
    def _get_robot_yaw(self) -> torch.Tensor:
        r, p, y = self._euler_from_quat(self._robot.data.root_quat_w)
        return y

    def _get_object_yaw(self) -> torch.Tensor:
        r, p, y = self._euler_from_quat(self._object3.data.root_quat_w)
        return y


    # ================================  Helper functions ================================
    def _euler_from_quat(self, quat_angle: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to Euler angles (roll, pitch, yaw).
        """
        w = quat_angle[:,0]
        x = quat_angle[:,1]
        y = quat_angle[:,2]
        z = quat_angle[:,3]
 
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = torch.atan2(t0, t1)
        
        t2 = 2.0 * (w * y - z * x)
        t2 = torch.clip(t2, -1, 1)
        pitch_y = torch.asin(t2)
        
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw_z = torch.atan2(t3, t4)
        
        return roll_x, pitch_y, yaw_z # in radians


    def quat_to_rot_matrix(self, quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        # Assumes input is (w, x, y, z)
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        B = quat.size(0)
        rot = torch.zeros((B, 3, 3), device=quat.device)

        rot[:, 0, 0] = 1 - 2 * (y ** 2 + z ** 2)
        rot[:, 0, 1] = 2 * (x * y - z * w)
        rot[:, 0, 2] = 2 * (x * z + y * w)
        rot[:, 1, 0] = 2 * (x * y + z * w)
        rot[:, 1, 1] = 1 - 2 * (x ** 2 + z ** 2)
        rot[:, 1, 2] = 2 * (y * z - x * w)
        rot[:, 2, 0] = 2 * (x * z - y * w)
        rot[:, 2, 1] = 2 * (y * z + x * w)
        rot[:, 2, 2] = 1 - 2 * (x ** 2 + y ** 2)

        return rot
    
  
    def quat_inverse_safe(self, q: torch.Tensor) -> torch.Tensor:
        norm_sq = torch.sum(q * q, dim=-1, keepdim=True)  # (..., 1)
        conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
        return conj / norm_sq
      

    def _reindex_Isaacsim2real(self, vec):

        # joint names real: FR_hip_joint, FR_thigh_joint, FR_calf_joint
        #                   FL_hip_joint, FL_thigh_joint, FL_calf_joint
        #                   RR_hip_joint, RR_thigh_joint, RR_calf_joint
        #                   RL_hip_joint, RL_thigh_joint, RL_calf_joint
        #                   joint1, joint2, joint3, joint4, joint5, joint6, jointGripper

        # joint names sim
        #['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
        # 'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
        # 'joint1', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 
        # 'RR_calf_joint', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper']

 
        return torch.hstack((vec[:, [1, 5, 10, 0, 4, 9, 3, 7, 12, 2, 6, 11, 8]], vec[:, 13:]))
    

    def _reindex_real2Isaacsim(self, vec):
        return torch.hstack((vec[:, [3, 0, 9, 6, 4, 1, 10, 7, 12, 5, 2, 11, 8]], vec[:, 13:]))


    def _step_contact_targets(self):
        # if self.cfg.env.observe_gait_commands:
        # print("====observe_gait_commands is true===")
        frequencies = 2.0
        phases = 0.5
        offsets = 0
        bounds = 0
        self.gait_indices = torch.remainder(self.gait_indices + 0.02 * frequencies, 1.0)

        walking_mask0 = torch.abs(self._low_commands[:, 0]) > 0.1
        walking_mask1 = torch.abs(self._low_commands[:, 1]) > 0.1
        walking_mask2 = torch.abs(self._low_commands[:, 2]) > 0.1
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2

        self.gait_indices[~walking_mask] = 0  # reset gait indices for non-walking commands

        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])


    def log_vel_tracking_result(self):
        """
        save the objects velocity tracking results for each environment separately.
        """
        np.set_printoptions(precision=4, suppress=True)
        self._update_selected_objects_state()

      
        if  self.iter_time > 3800:
            print("Data collection for this environment is already full, skipping logging.")
            return
        
        obj_vel = torch.cat([
            self.object_lin_vel_b[:, :2],    # (vx, vy)
            self.object_ang_vel_b[:, 2:3]    # omega
        ], dim=-1).cpu().numpy()  # Shape: (num_envs, 3)
        
        obj_commands = self._commands[:, :3].cpu().numpy()  # Shape: (num_envs, 3)
        
        
        # 
        if not hasattr(self, 'log_obj_vel_per_env'):
            self.log_obj_vel_per_env = [[] for _ in range(self.num_envs)]
        
        # 
        for env_idx in range(self.num_envs):
            current_data = [
                obj_vel[env_idx, 0],        # vx
                obj_vel[env_idx, 1],        # vy
                obj_vel[env_idx, 2],        # omega
                obj_commands[env_idx, 0],   # cmd_vx
                obj_commands[env_idx, 1],   # cmd_vy
                obj_commands[env_idx, 2],   # cmd_omega
            ]
            
            self.log_obj_vel_per_env[env_idx].append(current_data)
        
 
        print("Time:", len(self.log_obj_vel_per_env[env_idx]))

        for env_idx in range(self.num_envs):
            if  self.iter_time >= 3800:
                import pandas as pd
                
                df = pd.DataFrame(self.log_obj_vel_per_env[env_idx], columns=[
                    'vx', 'vy', 'omega', 'cmd_vx', 'cmd_vy', 'cmd_omega', 
                ])
                
                # filename = f'log_vel_NoId_3_data_env_{env_idx}.csv'
                filename = f'./vel_log/{self.vel_log_file}_{env_idx}.csv'

                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")
                
                self.log_obj_vel_per_env[env_idx] = []


    def log_joint_effort(self):

        np.set_printoptions(precision=4, suppress=True)
        self._update_selected_objects_state()

        if not hasattr(self, 'joint_effort_collection_done'):
            self.joint_effort_collection_done = False

        if  self.joint_effort_collection_done == True:
            print("Data collection for this environment is already full, skipping logging.")
            return
        print("iter_time:", self.iter_time)
        
        joint_efforts = self.robot_view_sim.get_measured_joint_efforts()  # (num_envs, 19)

        # joint_forces = self.robot_view_sim.get_measured_joint_forces()  # (num_envs, 19)
        # joint_force = joint_forces[:, joint_indices+1] # link 1 ~ 6 + gripper

        joint_efforts_arm = joint_efforts[:, self.arm_indices_with_gripper] # (num_envs, 7)
        # print("joint_efforts_arm:", joint_efforts_arm)
        
        if not hasattr(self, 'log_joint_effort_per_env'):
            self.log_joint_effort_per_env = [[] for _ in range(self.num_envs)]
        
        for env_idx in range(self.num_envs):
            # current_data = [
            #     joint_efforts_arm[env_idx, 0],        # vx
            #     joint_efforts_arm[env_idx, 1],        # vy
            #     joint_efforts_arm[env_idx, 2],        # omega
            #     joint_efforts_arm[env_idx, 3],   # cmd_vx
            #     joint_efforts_arm[env_idx, 4],   # cmd_vy
            #     joint_efforts_arm[env_idx, 5],   # cmd_omega
            #     joint_efforts_arm[env_idx, 6],   # cmd_omega
            # ]

            current_data = current_data = joint_efforts_arm[env_idx, :].cpu().numpy().tolist()
            
            self.log_joint_effort_per_env[env_idx].append(current_data)
        
            print("joint_effort Time:", len(self.log_joint_effort_per_env[env_idx]))

        for env_idx in range(self.num_envs):
            if  self.iter_time >= 3800:
                import pandas as pd
                
                df = pd.DataFrame(self.log_joint_effort_per_env[env_idx], columns=[
                    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper',
                ])
                # filename = f'log_vel_NoId_3_data_env_{env_idx}.csv'
                filename = f'./joint_effort_log/{self.vel_log_file}_{env_idx}_direct.csv'

                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")
                
                self.log_joint_effort_per_env[env_idx] = []
                self.joint_effort_collection_done = True



    def log_joint_position(self):

        np.set_printoptions(precision=4, suppress=True)
        self._update_selected_objects_state()

        if not hasattr(self, 'joint_position_collection_done'):
            self.joint_position_collection_done = False

        if  self.joint_position_collection_done == True:
            print("Data collection for this environment is already full, skipping logging.")
            return
        print("iter_time:", self.iter_time)
        
        joint_position1 = self.processed_actions_with_fixed_joint[:, 8].cpu().numpy().tolist() 
        joint_position2 = self.processed_actions_with_fixed_joint[:, 13:18].cpu().numpy().tolist() 

        if not hasattr(self, 'log_joint_position_per_env'):
            self.log_joint_position_per_env = [[] for _ in range(self.num_envs)]
        
        for env_idx in range(self.num_envs):

            current_data = [joint_position1[env_idx]] + joint_position2[env_idx]
            
            self.log_joint_position_per_env[env_idx].append(current_data)
        
            print("joint position Time:", len(self.log_joint_position_per_env[env_idx]))

        for env_idx in range(self.num_envs):
            if  self.iter_time >= 2000:
                import pandas as pd
                
                df = pd.DataFrame(self.log_joint_position_per_env[env_idx], columns=[
                    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
                ])

                # filename = f'log_vel_NoId_3_data_env_{env_idx}.csv'
                filename = f'./joint_log/{self.vel_log_file}_{env_idx}.csv'

                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")
                
                self.log_joint_position_per_env[env_idx] = []
                self.joint_position_collection_done = True


  

    def _load_low_level_model(self, num_priv=5 + 1 + 12, stochastic=False):
        low_level_kwargs = {
            "continue_from_last_std": True,
            "init_std": [[0.8, 1.0, 1.0] * 4 + [1.0] * 6],
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": 'elu', # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
            "output_tanh": False,
            "leg_control_head_hidden_dims": [128, 128],
            "arm_control_head_hidden_dims": [128, 128],
            "priv_encoder_dims": [64, 20],
            "num_leg_actions": 12,
            "num_arm_actions": 6,
            "adaptive_arm_gains": False,
            "adaptive_arm_gains_scale": 10.0
        }
        num_actions = 18
        self.num_priv = num_priv
        self.num_proprio = 71
        self.history_len = 10
        low_actor_critic: ActorCriticLow = ActorCriticLow(  self.num_proprio,
                                                            self.num_proprio,
                                                            num_actions,
                                                            **low_level_kwargs,
                                                            num_priv=self.num_priv,
                                                            num_hist=self.history_len,
                                                            num_prop=self.num_proprio,
                                                            )
        policy_path = self.policy_path
        loaded_dict = torch.load(policy_path, map_location=self.device)
        low_actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        low_actor_critic = low_actor_critic.to(self.device)
        low_actor_critic.eval()
        print("Low level pretrained policy loaded!")
        if not stochastic:
            return low_actor_critic.act_inference
        else:
            return low_actor_critic.act
        

 
        

    

        



