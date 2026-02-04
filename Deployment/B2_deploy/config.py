import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.weak_motor = []
            if "weak_motor" in config:
                self.weak_motor = config["weak_motor"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.low_level_policy_path = config["low_level"]["low_level_policy_path"]
            
            self.leg_joint2motor_idx = config["leg_joint2motor_idx"]

            self.kps = config["low_level"]["kps"]
            self.kds = config["low_level"]["kds"]
            self.kps_stand = config["low_level"]["kps_stand"]
            self.kds_stand = config["low_level"]["kds_stand"]
            self.default_angles_low = np.array(config["low_level"]["default_angles"], dtype=np.float32)
            
            self.arm_default_angles_low = np.array(config["low_level"]["arm_default_angles"], dtype=np.float32)
            self.arm_default_angles_tracking = np.array(config["low_level"]["arm_default_angles_tracking"], dtype=np.float32)
            self.arm_default_angles_initial = np.array(config["low_level"]["arm_default_angles_initial"], dtype=np.float32)
            # self.arm_waist_joint2motor_idx = config["arm_waist_joint2motor_idx"]
            # self.arm_waist_kps = config["arm_waist_kps"]
            # self.arm_waist_kds = config["arm_waist_kds"]
            # self.arm_waist_target = np.array(config["arm_waist_target"], dtype=np.float32)

            self.ang_vel_scale_low = config["low_level"]["ang_vel_scale"]
            self.dof_pos_scale_low = config["low_level"]["dof_pos_scale"]
            self.dof_vel_scale_low = config["low_level"]["dof_vel_scale"]
            self.action_scale_low = config["low_level"]["action_scale"]
            self.cmd_scale_low = np.array(config["low_level"]["cmd_scale"], dtype=np.float32)
            # self.max_cmd = np.array(config["max_cmd"], dtype=np.float32)

            self.num_actions_low = config["low_level"]["num_actions"]
            self.num_obs_low = config["low_level"]["num_obs"]
            self.num_proprio_low = config["low_level"]["num_proprio"]
            self.history_len_low = config["low_level"]["history_len"]

            self.joint_limits_min = np.array(config["joint_limits_min"], dtype=np.float32)
            self.joint_limits_max = np.array(config["joint_limits_max"], dtype=np.float32)

            # high level controller
            self.high_level_policy_path = config["high_level"]["high_level_policy_path"]
                    
            self.ang_vel_scale_high = config["high_level"]["ang_vel_scale"]
            self.dof_pos_scale_high = config["high_level"]["dof_pos_scale"]
            self.dof_vel_scale_high = config["high_level"]["dof_vel_scale"]
            self.action_scale_high = config["high_level"]["action_scale"]
            self.action_clip_high = config["high_level"]["action_clip"]
            self.cmd_scale_high = np.array(config["high_level"]["cmd_scale"], dtype=np.float32)

            self.object_height_offset_chair = config["high_level"]["chair"]["object_height_offset"]
            self.arm_default_angles_high_chair = np.array(config["high_level"]["chair"]["arm_default_pose"], dtype=np.float32)
            self.default_angles_high_chair = np.array(config["high_level"]["chair"]["default_angles"], dtype=np.float32)

            self.object_height_offset_table = config["high_level"]["table"]["object_height_offset"]
            self.arm_default_angles_high_table = np.array(config["high_level"]["table"]["arm_default_pose"], dtype=np.float32)
            self.default_angles_high_table = np.array(config["high_level"]["table"]["default_angles"], dtype=np.float32)

            self.object_height_offset_box = config["high_level"]["box"]["object_height_offset"]
            self.arm_default_angles_high_box = np.array(config["high_level"]["box"]["arm_default_pose"], dtype=np.float32)
            self.default_angles_high_box = np.array(config["high_level"]["box"]["default_angles"], dtype=np.float32)

            self.num_actions_high= config["high_level"]["num_actions"]
            self.num_obs_high= config["high_level"]["num_obs"]
            self.num_proprio_high= config["high_level"]["num_proprio"]
            self.history_len_high = config["high_level"]["history_len"]
