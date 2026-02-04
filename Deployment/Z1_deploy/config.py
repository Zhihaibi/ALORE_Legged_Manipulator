import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]
            self.policy_path = config["policy_path"]
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.action_scale = config["action_scale"]
            self.clip_actions = config["clip_actions"]
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.lower_limits_angles = np.array(config["lower_limits_angles"], dtype=np.float32)
            self.upper_limits_angles = np.array(config["upper_limits_angles"], dtype=np.float32)
