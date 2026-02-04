import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import math
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R, Slerp
import numpy as np
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Int32MultiArray

import numpy as np


class EnvPerception:
    def __init__(self):
        self.object_poses = np.zeros((4, 8))  # assuming 4 objects, each with (x, y, z, yaw)
        self.object_poses[0, :] = [-17,-17,0,0,0,0,0,1] # [qx, qy, qz, qw]
        self.object_poses[1, :] = [-14,-17,0,0,0,0,0,1] # [qx, qy, qz, qw]
        self.object_poses[2, :] = [-12,-17,0,0,0,0,0,1] # [qx, qy, qz, qw]
        self.object_poses[3, :] = [14,-16,0,0,0,0,0,1] # [qx, qy, qz, qw]

        # self.object_poses[0, :] = [6,-5,0,0,0,0,0,1] # [qx, qy, qz, qw]
        # self.object_poses[1, :] = [9,1,0,0,0,0,0,1] # [qx, qy, qz, qw]
        # self.object_poses[2, :] = [9,-6,0,0,0,0,0,1] # [qx, qy, qz, qw]
        # self.object_poses[3, :] = [9,-5,0,0,0,0,0,1] # [qx, qy, qz, qw]

        self.robot_poses = np.zeros((1, 8))  # assuming 1 robot, with (x, y, z, yaw) + (qx, qy, qz, qw)
        
        # subscribe the env perception data from motion capture system
        self.robot_pose_sub = rospy.Subscriber('/odom', Odometry, self.robot_pose_callback)
        
        # self.object_pose_sub = rospy.Subscriber('/object/pose', PoseStamped, self.object_pose_callback)

        # publish env perception data
        self.env_obs_pub = rospy.Publisher('/env_obs', Float32MultiArray, queue_size=1)
        rospy.Timer(rospy.Duration(0.01), self.pub_env_obs) # 100 Hz
    
    def robot_pose_callback(self, msg):
        pose = msg.pose.pose
        x_lidar = pose.position.x
        y_lidar = pose.position.y
        z_lidar = pose.position.z

        quat_lidar = pose.orientation
        quat_lidar_np = np.array([quat_lidar.x,
                                quat_lidar.y,
                                quat_lidar.z,
                                quat_lidar.w])

        r_odom_lidar = R.from_quat(quat_lidar_np)

        p_base_lidar = np.array([-0.37, 0.0, 0.0])


        yaw   = np.pi          # 180 deg
        pitch = -np.deg2rad(30)  # -30 deg, 
        roll  = 0.0

        r_base_lidar = R.from_euler('zyx', [yaw, pitch, roll])  # R_odom^base

  
        r_lidar_base = r_base_lidar.inv()      
        p_lidar_base = -r_lidar_base.apply(p_base_lidar)
        r_odom_base = r_odom_lidar * r_lidar_base
        quat_base_np = r_odom_base.as_quat()  # [qx, qy, qz, qw]

        p_odom_lidar = np.array([x_lidar, y_lidar, z_lidar])
        offset_in_odom = r_odom_base.apply(p_base_lidar)
        p_odom_base = p_odom_lidar - offset_in_odom

        x_base, y_base, z_base = p_odom_base.tolist()


        roll_b, pitch_b, yaw_b = r_odom_base.as_euler('xyz', degrees=False)

        #   robot_poses[0, :4] = [x, y, z, yaw]
        self.robot_poses[0, :4] = [x_base, y_base, z_base, yaw_b]
        #   robot_poses[0, 4:] = [qx, qy, qz, qw]
        self.robot_poses[0, 4:] = quat_base_np.tolist()


    def pub_env_obs(self, event):
        # publish robot and object poses, 8 + 4*4 = 24 in total
        env_obs = Float32MultiArray()
        env_obs.data = np.concatenate((self.robot_poses.flatten(), self.object_poses.flatten())).tolist()
        self.env_obs_pub.publish(env_obs)


if __name__ == "__main__":
    rospy.init_node('EnvPerception', anonymous=True)
    EnvPerceptionDate = EnvPerception()
    rospy.spin()
    
