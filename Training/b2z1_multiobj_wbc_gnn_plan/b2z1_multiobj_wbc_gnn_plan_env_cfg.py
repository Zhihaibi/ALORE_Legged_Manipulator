# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import os
import yaml
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg

import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

import gymnasium as gym
import numpy as np


##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_B2Z1Float_CFG, UNITREE_B2Z1_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# from .randomization_utils import custom_randomize_chair_material, custom_randomize_chair_mass

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_angle_axis
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG



@configclass
class EventCfg:
    """Configuration for randomization."""
    object1_material_randomization = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset", # startup, or reset
        params={
            "asset_cfg": SceneEntityCfg("object1", body_names=".*"),
            # "static_friction_range": (0.01, 0.41),
            # "dynamic_friction_range": (0.02, 0.4),
            "static_friction_range": (0.11, 0.61),
            "dynamic_friction_range": (0.1, 0.6),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 64,
        },
    )

    object2_material_randomization = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset", # startup, or reset
        params={
            "asset_cfg": SceneEntityCfg("object2", body_names=".*"),
            "static_friction_range": (0.01, 0.41),
            "dynamic_friction_range": (0.02, 0.4),
            # "static_friction_range": (0.11, 0.61),
            # "dynamic_friction_range": (0.1, 0.6),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 64,
        },
    )

    object3_material_randomization = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset", # startup, or reset
        params={
            "asset_cfg": SceneEntityCfg("object3", body_names=".*"),
            # "static_friction_range": (0.01, 0.21),
            # "dynamic_friction_range": (0.02, 0.2),
            "static_friction_range": (0.01, 0.41),
            "dynamic_friction_range": (0.02, 0.4),
            "restitution_range": (0.0, 0.01),
            "num_buckets": 64,
        },
    )
# 
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.7, 0.7),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


    object1_mass_randomization = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset", # startup, or reset
        params={
            "asset_cfg": SceneEntityCfg("object1", body_names=".*"),
            # "mass_distribution_params": (10.0/14., 15.0/14.), # 14 nums of rigid bodies
            "mass_distribution_params": (15.0, 15.0),
            "operation": "abs",
        },
    )

    object2_mass_randomization = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset", # startup, or reset
        params={
            "asset_cfg": SceneEntityCfg("object2", body_names=".*"),
            # "mass_distribution_params": (10.0/14., 15.0/14.), # 14 nums of rigid bodies
            "mass_distribution_params": (10.0, 12.0),
            "operation": "abs",
        },
    )

    object3_mass_randomization = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset", # startup, or reset
        params={
            "asset_cfg": SceneEntityCfg("object3", body_names=".*"),
            "mass_distribution_params": (10.0, 12.0), # 14 nums of rigid bodies
            # "mass_distribution_params": (10.0, 12.0), # 14 nums of rigid bodies
            "operation": "abs",
        },
    )

    robot_add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )



@configclass
class B2Z1MultiObjWBCGNNPLANFlatEnvCfg(DirectRLEnvCfg):
    # Patch: load prefix from yaml and use for all asset paths
    prefix = None
    try:
        cfg_path = os.path.join(os.path.dirname(__file__), '../config.yaml')
        with open(cfg_path, 'r') as f:
            _cfg = yaml.safe_load(f)
            prefix = _cfg['paths']['prefix']
    except Exception:
        prefix = "/home/v1"  # fallback
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale_low_level  = [0.4, 0.4, 0.4, 0.4, 
                               0.45, 0.45, 0.45, 0.45, 2.1, 
                               0.45, 0.45, 0.45, 0.45, 
                               0.6, 0.6, 0, 0, 0]
    
    action_scale = [0.5, 0.5, 0.5, 
                    0.05, 0.05, 0.05, 
                    0.05, 0.05, 0.05]
    
    action_space = 9  # 3 base (x, y, yaw) + 6 joints angles 
    action_clip = [0.6, 0.0, 0.6, 
                   0.05, 0.05, 0.05, 
                   0.05, 0.05, 0.05]
    
    # action_scale = [0.5, 0.5, 0.5, 
    #                 0.1, 0.1, 0.1, 
    #                 0.1, 0.1, 0.1]
    
    # action_space = 9  # 3 base (x, y, yaw) + 6 joints angles 
    # action_clip = [0.6, 0.0, 0.6, 
    #                0.1, 0.1, 0.1, 
    #                0.1, 0.1, 0.1]
    
    observation_space = 770  # (70)*11 = 770
    observation_space_low_level = 799  # 71*11 + 18

    ## asymmetric actor-critic
    state_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(160,),  #  ## TODO: with only physical estimation
        dtype=np.float32 
    )
    # state_space = 0

    history_len = 10
    num_proprio = 70
    num_proprio_low_level = 71
    catelogy_num = 3

    ## TODO: for plan
    task_state = "WAIT_ROBOT_PATH"  # WAIT_ROBOT_PATH, ROBOT_TRACKING, GRASPING, WAIT_OBJECT_PATH, OBJECT_TRACKING, RELEASING
    robot_vel_cmd = [0.0, 0.0, 0.0] 
    object_vel_cmd = [0.0, 0.0, 0.0]  # [vx, vy, omega]
    joint_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # # [joint1, joint2, joint3, joint4, joint5, joint6, gripper]
    object_type = 1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=3, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = UNITREE_B2Z1_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )


    ### movechair start =======================================
    movechair: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/movechair",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(prefix, "IsaacLab/assets/movechair2/model_office_chair_3_v1.usd"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.0),  
            rot = (1, 0., 0., 0),  #  (w, x, y, z)
        ),
        actuators={
            "wheellink": ImplicitActuatorCfg(
                joint_names_expr = [".*down1", ".*down2", ".*down3", ".*down4", ".*down5"],
                damping=0.1,
                friction=0.1,
                stiffness=0.0,  #
            ),
            "wheel": ImplicitActuatorCfg(
                joint_names_expr = [".*down6", ".*down7", ".*down8", ".*down9", ".*down10"],
                damping=0.1,
                friction=0.1,
                stiffness=0.0,  #
            )
        }
    )

    box: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/box",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(prefix, "IsaacLab/assets/box/box_2.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,  
                kinematic_enabled=False,  
                disable_gravity=False, 
                retain_accelerations=True,
                linear_damping=0.0, 
                angular_damping=0.0,  
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,  
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.57, 0.0, 0.2), 
            rot = (1, 0., 0., 0.),  
        ),
    )

    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(prefix, "IsaacLab/assets/table2/table4_2.usd"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, 
                kinematic_enabled=False,  
                disable_gravity=False, 
                retain_accelerations=True, 
                linear_damping=0.0, 
                angular_damping=0.0,  
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, 
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(1.57, 0.0, 0.2),  
            rot = (1, 0., 0., 0.),  
        ),
    )


    ### movechair1 start =======================================
    movechair1_chair: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/movechair1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(prefix, "IsaacLab/assets/movechair2/model_office_chair_3_v1.usd"),
            # usd_path="/home/xyz/IsaacLab/assets/movechair2/model_office_chair_3_v1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.0), 
            rot = (1, 0., 0., 0), 
        ),
        actuators={
            "wheellink": ImplicitActuatorCfg(
                joint_names_expr = [".*down1", ".*down2", ".*down3", ".*down4", ".*down5"],
                damping=0.1,
                friction=0.1,
                stiffness=0.0,  #
            ),
            "wheel": ImplicitActuatorCfg(
                joint_names_expr = [".*down6", ".*down7", ".*down8", ".*down9", ".*down10"],
                damping=0.1,
                friction=0.1,
                stiffness=0.0,  #
            )
        }
    )

    table1_chair: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/table1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(prefix, "IsaacLab/assets/movechair2/model_office_chair_3_v1.usd"),
            # usd_path="/home/xyz/IsaacLab/assets/movechair2/model_office_chair_3_v1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.0), 
            rot = (1, 0., 0., 0), 
        ),
        actuators={
            "wheellink": ImplicitActuatorCfg(
                joint_names_expr = [".*down1", ".*down2", ".*down3", ".*down4", ".*down5"],
                damping=0.1,
                friction=0.1,
                stiffness=0.0,  #
            ),
            "wheel": ImplicitActuatorCfg(
                joint_names_expr = [".*down6", ".*down7", ".*down8", ".*down9", ".*down10"],
                damping=0.1,
                friction=0.1,
                stiffness=0.0,  #
            )
        }
    )

    table2_chair: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/table2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(prefix, "IsaacLab/assets/movechair2/model_office_chair_3_v1.usd"),
            # usd_path="/home/xyz/IsaacLab/assets/movechair2/model_office_chair_3_v1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.0),  
            rot = (1, 0., 0., 0), 
        ),
        actuators={
            "wheellink": ImplicitActuatorCfg(
                joint_names_expr = [".*down1", ".*down2", ".*down3", ".*down4", ".*down5"],
                damping=0.0,
                friction=0.0, 
                stiffness=0.0,  #
            ),
            "wheel": ImplicitActuatorCfg(
                joint_names_expr = [".*down6", ".*down7", ".*down8", ".*down9", ".*down10"],
                damping=0.0,
                friction=0.0,
                stiffness=0.0,  #
            )
        }
    )


    table3_chair: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/table3",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(prefix, "IsaacLab/assets/movechair2/model_office_chair_3_v1.usd"),
            # usd_path="/home/xyz/IsaacLab/assets/movechair2/model_office_chair_3_v1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.0), 
            rot = (1, 0., 0., 0),  
        ),
        actuators={
            "wheellink": ImplicitActuatorCfg(
                joint_names_expr = [".*down1", ".*down2", ".*down3", ".*down4", ".*down5"],
                damping=0.0,
                friction=0.0,
                stiffness=0.0,  #
            ),
            "wheel": ImplicitActuatorCfg(
                joint_names_expr = [".*down6", ".*down7", ".*down8", ".*down9", ".*down10"],
                damping=0.0,
                friction=0.0,
                stiffness=0.0,  #
            )
        }
    )
    ### movechair1 end ======================================


    #### table1 start ==========================================
    table1_table1: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table1",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/box/box_2.usd"),
                # usd_path="/home/xyz/IsaacLab/assets/box/box_2.usd",
                # usd_path="/home/xyz/IsaacLab/assets/table2/table4_2.usd",

                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  
                    kinematic_enabled=False, 
                    disable_gravity=False,  #
                    retain_accelerations=True,  #
                    linear_damping=0.0,  # 
                    angular_damping=0.0,  # 
                    # solver_position_iteration_count=20,  
                    # solver_velocity_iteration_count=20,  
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0), 
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, 
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2),  
                # rot = (0.5, 0.5, 0.5, 0.5), 
                rot = (1, 0., 0., 0.),  # 
            ),
    )

    table2_table1: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table2",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/box/box_2.usd"),
                # usd_path="/home/xyz/IsaacLab/assets/box/box_2.usd",
                # usd_path="/home/xyz/IsaacLab/assets/table2/table4_2.usd",

                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  
                    kinematic_enabled=False,  #
                    disable_gravity=False,  # 
                    retain_accelerations=True,  
                    linear_damping=0.0,  #
                    angular_damping=0.0,  # 
                    # solver_position_iteration_count=20,  #
                    # solver_velocity_iteration_count=20
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0), 
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,  
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2), 
                rot = (1, 0., 0., 0.),  # 
            ),
    )

    movechair1_table1: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/movechair1",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/box/box_2.usd"),

                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  
                    kinematic_enabled=False,  
                    disable_gravity=False, 
                    retain_accelerations=True,  
                    linear_damping=0.0,  
                    angular_damping=0.0,  
                    # solver_position_iteration_count=20,  
                    # solver_velocity_iteration_count=20, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0), 
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, 
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2), 
                rot = (1, 0., 0., 0.),  
            ),
    )


    table3_table1: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table3",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/box/box_2.usd"),
                # usd_path="/home/xyz/IsaacLab/assets/box/box_2.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  
                    kinematic_enabled=False,  
                    disable_gravity=False,  
                    retain_accelerations=True,  
                    linear_damping=0.0, 
                    angular_damping=0.0,  
                    # solver_position_iteration_count=20,  
                    # solver_velocity_iteration_count=20, 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0), 
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, 
                ),
            ),
          
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2),  
                rot = (1, 0., 0., 0.), 
            ),
    )
    #### table1 end ==========================================


    #### table2 start ==========================================
    table1_table2: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table1",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/table2/table4_2.usd"),
                
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  
                    kinematic_enabled=False, 
                    disable_gravity=False,  #
                    retain_accelerations=True,  # 
                    linear_damping=0.0,  # 
                    angular_damping=0.0,  # 
                    # solver_position_iteration_count=20,  # 
                    # solver_velocity_iteration_count=20,  # 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  # 
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,  # 
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2),  #
                rot = (1, 0., 0., 0.),  #
            ),
    )

    table2_table2: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table2",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/table2/table4_2.usd"),
                # usd_path="/home/xyz/IsaacLab/assets/table2/table4_2.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  # 
                    kinematic_enabled=False,  # 
                    disable_gravity=False,  #
                    retain_accelerations=True,  # 
                    linear_damping=0.0,  # 
                    angular_damping=0.0,  # 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,  
                ),
            ),
          
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2),  
                rot = (1, 0., 0., 0.), 
            ),
    )

    movechair1_table2: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/movechair1",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/table2/table4_2.usd"),
                # usd_path="/home/xyz/IsaacLab/assets/table2/table4_2.usd",
                
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  # 
                    kinematic_enabled=False,  #
                    disable_gravity=False,  # 
                    retain_accelerations=True,  # 
                    linear_damping=0.0,  #
                    angular_damping=0.0, 
                    # solver_position_iteration_count=20,  
                    # solver_velocity_iteration_count=20,  # 
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0), 
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,  # 
                ),
            ),
          
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2),  # 
                rot = (1, 0., 0., 0.),  # 
            ),
    )


    table3_table2: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table3",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/table2/table4_2.usd"),
                # usd_path="/home/xyz/IsaacLab/assets/table2/table4_2.usd",
                
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  # 
                    kinematic_enabled=False,  #
                    disable_gravity=False,  #
                    retain_accelerations=True,  # 
                    linear_damping=0.0,  # 
                    angular_damping=0.0,  #
               
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0), 
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True, 
                ),
            ),
          
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2),  # 
                rot = (1, 0., 0., 0.),  # 
            ),
    )
    #### table2 end ==========================================

    ### all start ==========================================
    table1_mix: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table1",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/box/box_2.usd"),
                # usd_path="/home/xyz/IsaacLab/assets/box/box_2.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  
                    kinematic_enabled=False, 
                    disable_gravity=False,  
                    retain_accelerations=True,  
                    linear_damping=0.0,  #
                    angular_damping=0.0,  # 

                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),  # 
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,  # 
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2), 
                rot = (1, 0., 0., 0.),  
            ),
    )

    table2_mix: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/table2",
            spawn=sim_utils.UsdFileCfg(
                usd_path=os.path.join(prefix, "IsaacLab/assets/table2/table4_2.usd"),
                # usd_path="/home/xyz/IsaacLab/assets/table2/table4_2.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,  
                    kinematic_enabled=False,  
                    disable_gravity=False,  # 
                    retain_accelerations=True,  
                    linear_damping=0.0,  #
                    angular_damping=0.0,  
                    # solver_position_iteration_count=20,  #
                    # solver_velocity_iteration_count=20,  #
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=10.0),  
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,  # 
                ),
            ),
          
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(1.57, 0.0, 0.2),  # 
                rot = (1, 0., 0., 0.),  # 
            ),
    )

    movechair1_mix: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/movechair1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(prefix, "IsaacLab/assets/movechair2/model_office_chair_3_v1.usd"),
            # usd_path="/home/xyz/IsaacLab/assets/movechair2/model_office_chair_3_v1.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0.0, 0.0),  # 
            rot = (1, 0., 0., 0),  # 
        ),
        actuators={
            "wheellink": ImplicitActuatorCfg(
                joint_names_expr = [".*down1", ".*down2", ".*down3", ".*down4", ".*down5"],
                damping=0.1,
                friction=0.1,
                stiffness=0.0,  #
            ),
            "wheel": ImplicitActuatorCfg(
                joint_names_expr = [".*down6", ".*down7", ".*down8", ".*down9", ".*down10"],
                damping=0.1,
                friction=0.1,
                stiffness=0.0,  #
            )
        }
    )
   
    ### all end ==========================================

    marker_goal_vel: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )

    marker_cur_vel: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal")
    
    
    # reward scales
    # Object
    lin_vel_exp_reward_scale = 5.0
    yaw_rate_exp_reward_scale = 5.0

    lin_vel_l2_reward_scale = 0.0
    ang_vel_l2_reward_scale = 0.0

    distance_penalty_scale = -10.0  
    lin_vel_penalty_scale = -5.0
    yaw_rate_penalty_scale = -5.0

    alive_reward_scale = 1.
    pos_alignment_reward_scale = 0.0
    yaw_alignment_reward_scale = 10.0

    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    flat_orientation_reward_scale = -10.0

    lin_vel_change_penalty_scale = -2.0
    ang_vel_change_penalty_scale = -2.0

    action_rate_reward_scale = -0.01
    action_rate2_reward_scale = -0.002
    
    # arm joint
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    joint_default_pos_reward_scale = -5.0   # -1.0
    joint_efforts_arm_reward_scale = 1e-5   # 1e-5.0

    # contact between object and robot
    undesired_contact_reward_scale = -5.0




@configclass
class B2Z1MultiObjWBCGNNPLANRoughEnvCfg(B2Z1MultiObjWBCGNNPLANFlatEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0
