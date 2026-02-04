import argparse
import sys
import hydra
import yaml
import os
import rospy
from std_msgs.msg import Float32MultiArray

# local imports
config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

isaaclab_prefix = config['paths']['isaaclab_prefix']

sys.path.append(isaaclab_prefix)
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args  # isort: skip
from isaaclab.app import AppLauncher

# from isaacsim.core.utils.viewports import set_camera_view


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
# if args_cli.video:
args_cli.enable_cameras = True
args_cli.headless = False  # 


# launch omniverse app
# app_launcher = AppLauncher(args_cli)
app_launcher = AppLauncher(
    args_cli,
)

simulation_app = app_launcher.app


"""Rest everything follows."""
import gymnasium as gym
import os
import time
import torch
import rospy
from scipy.spatial.transform import Rotation as R
import numpy as np
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import sys
sys.path.append(os.path.join(isaaclab_prefix, "source"))
# sys.path.append("/home/v1/IsaacLab/source")
from isaaclab_tasks.direct.b2z1_multiobj_wbc_gnn_plan.rsl_rl.on_policy_runner_physic import PhysicOnPolicyRunner


# PLACEHOLDER: Extension template (do not remove this comment)
FILE_PATH = os.path.join(os.path.dirname(__file__), "cfg")

task_state_mapping = {
        0.0: "WAIT_TASK_PLANNING",
        1.0: "WAIT_ROBOT_PATH",
        2.0: "ROBOT_TRACKING",
        3.0: "GRASPING",
        4.0: "WAIT_OBJECT_PATH",
        5.0: "OBJECT_TRACKING",
        6.0: "RELEASING"
}

# @hydra.main(config_path=FILE_PATH, config_name="sim", version_base=None)
# def hydra_main(cfg):
#     pass  # Placeholder function for Hydra initialization


def env_control_callback(msg, env):
    data = msg.data
    env.cfg.robot_vel_cmd = data[0:3]  #  [vx, vy, wz]
    env.cfg.object_vel_cmd = data[3:6]  # [vx, vy, wz]
    env.cfg.joint_cmd = data[6:13]       # Joint commands
    env.cfg.task_state = task_state_mapping.get(data[13], "UNKNOWN")  # Task state
    env.cfg.object_type = data[14]
    # print
    # print(f"Received control data: {data}")


def publish_obs_data(pub, robot_obs, object_obs):
    robot_obs = robot_obs.flatten().cpu().numpy().tolist()
    object_obs = object_obs.flatten().cpu().numpy().tolist()

    data = robot_obs + object_obs
    msg = Float32MultiArray(data=data)
    pub.publish(msg)



# def camera_follow(env):
#     if (env.unwrapped.scene.num_envs == 1):
#         robot_position = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, :3].cpu().numpy()
#         robot_orientation = env.unwrapped.scene["unitree_go2"].data.root_state_w[0, 3:7].cpu().numpy()
#         rotation = R.from_quat([robot_orientation[1], robot_orientation[2], 
#                                 robot_orientation[3], robot_orientation[0]])
#         yaw = rotation.as_euler('zyx')[0]
#         yaw_rotation = R.from_euler('z', yaw).as_matrix()
#         set_camera_view(
#             yaw_rotation.dot(np.asarray([-4.0, 0.0, 5.0])) + robot_position,
#             robot_position
#         )

# def publish_rgb(camera: Camera, freq):
#     # The following code will link the camera's render product and publish the data to the specified topic name.
#     render_product = camera._render_product_path
#     step_size = int(60/freq)
#     topic_name = camera.name+"_rgb"
#     queue_size = 1
#     node_namespace = ""
#     frame_id = camera.prim_path.split("/")[-1] # This matches what the TF tree is publishing.

#     rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
#     writer = rep.writers.get(rv + "ROS1PublishImage")
#     writer.initialize(
#         frameId=frame_id,
#         nodeNamespace=node_namespace,
#         queueSize=queue_size,
#         topicName=topic_name
#     )
#     writer.attach([render_product])
#     return


def main():
    from b2z1.b2z1_env_cfg import B2Z1MultiObjWBCGNNPLANFlatEnvCfg
    from b2z1.b2z1_ppo_cfg import B2Z1MultiObjWBCGNNPLANFlatPPORunnerCfg
    import env.sim_env as sim_env
    import b2z1.b2z1_sensors as b2z1_sensors
    # import ros1.b2z1_ros1_bridge as b2z1_ros1_bridge

    env_cfg = B2Z1MultiObjWBCGNNPLANFlatEnvCfg()
    agent_cfg = B2Z1MultiObjWBCGNNPLANFlatPPORunnerCfg()
    env = gym.make("Isaac-Velocity-Flat-B2Z1MultiObjWBCGNNPLAN-Direct-v0", cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    resume_path = config['paths']['resume_path']  # ours
    fsm_cfg = config['fsm']
    scenario = fsm_cfg['default_scenario']

    ppo_runner = PhysicOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    
    
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # sim_env.add_semantic_label()
    sim_env.create_office1_env(scenario)

    # sensors
    # sm = b2z1_sensors.SensorManager(1)
    # cameras = sm.add_camera(50)

    # ros cameras
    # robot_front_cam = Camera(
    #     prim_path="/World/envs/env_0/Robot/base/front_cam",
    #     name="robot_front_cam",
    #     position=np.array([-4, 0, 3]),
    #     frequency=1,
    #     resolution=(640, 480),
    #     orientation=np.array([0.9659, 0.0, 0.2588, 0.0])
    # )

    # top_camera = Camera(
    #     prim_path="/World/envs/env_0/top_cam",
    #     name="top_camera",
    #     position=np.array([1.5 , -4, 60]),
    #     frequency=1,
    #     resolution=(2560, 2560),
    #     orientation=np.array([0.7071, 0.0, 0.7071, 0.0])
    # )

    # robot_front_cam.initialize()
    # top_camera.initialize()
    # approx_freq = 1
    # publish_rgb(robot_front_cam, approx_freq)
    # publish_rgb(top_camera, approx_freq)

    dt = env.unwrapped.step_dt

    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    if not timeline.is_playing():
        timeline.play()

    # reset environment
    obs, obs_all = env.get_observations()
    robot_obs = obs_all["observations"]["robot"]
    object_obs = obs_all["observations"]["object"]
    critic_obs = obs_all["observations"]["critic"]

    # Initialize ROS
    rospy.init_node("object_rearrangement", anonymous=True)

    # Publisher and Subscriber
    pub_obs = rospy.Publisher("/env_obs", Float32MultiArray, queue_size=10)
    rospy.Timer(rospy.Duration(0.01), publish_obs_data)  # 

    rospy.Subscriber("/env_control_data", Float32MultiArray, env_control_callback, env)

    # simulate environment
    while simulation_app.is_running():
        
        publish_obs_data(pub_obs, robot_obs, object_obs)
        
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs, critic_obs)
            obs, _, _, obs_all = env.step(actions) 

            robot_obs = obs_all["observations"]["robot"]
            object_obs = obs_all["observations"]["object"]
            critic_obs = obs_all["observations"]["critic"]

        sleep_time = dt - (time.time() - start_time)
        total_hz = 1.0 / (time.time() - start_time)
        print("dt", dt)
        print("total_hz", total_hz)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    # env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
