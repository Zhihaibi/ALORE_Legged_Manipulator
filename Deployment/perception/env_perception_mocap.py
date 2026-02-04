import rospy
from nav_msgs.msg import Path
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
        self.robot_poses = np.zeros((1, 8))  # assuming 1 robot, with (x, y, z, yaw) + (qx, qy, qz, qw)
        
        # subscribe the env perception data from motion capture system
        self.robot_pose_sub = rospy.Subscriber('/vrpn_client_node/robotBase/pose',  PoseStamped, self.robot_pose_callback)
        self.object_pose_sub_1 = rospy.Subscriber('/vrpn_client_node/object1/pose', PoseStamped, self.object1_pose_callback)
        self.object_pose_sub_2 = rospy.Subscriber('/vrpn_client_node/object2/pose', PoseStamped, self.object2_pose_callback)
        self.object_pose_sub_3 = rospy.Subscriber('/vrpn_client_node/object3/pose', PoseStamped, self.object3_pose_callback)
        self.object_pose_sub_4 = rospy.Subscriber('/vrpn_client_node/object4/pose', PoseStamped, self.object4_pose_callback)

        # publish env perception data
        self.env_obs_pub = rospy.Publisher('/env_obs', Float32MultiArray, queue_size=1)
        rospy.Timer(rospy.Duration(0.01), self.pub_env_obs) # 100 Hz
    
    def robot_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        # Convert quaternion to yaw
        quat = msg.pose.orientation
        yaw = np.arctan2(2.0*(quat.w*quat.z + quat.x*quat.y), 1.0 - 2.0*(quat.y*quat.y + quat.z*quat.z))
        # Fill robot_poses as needed
        self.robot_poses[0, :4] = [x, y, z, yaw]
        quat_temp = [quat.x, quat.y, quat.z, quat.w]    

        # Rotate the quaternion by 90 degrees around the robot's local x-axis (motion capture system)
        quat_original = np.array(quat_temp)
        roll_rotation = R.from_euler('x', np.pi/2, degrees=False)
        r_original = R.from_quat(quat_original)
        r_combined = r_original * roll_rotation
        # Get the resulting quaternion [qx, qy, qz, qw]
        quat_rotated = r_combined.as_quat()
        self.robot_poses[0, 4:] = quat_rotated.tolist()  # [qx, qy, qz, qw]
    
    def object1_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        quat = msg.pose.orientation
        yaw = np.arctan2(2.0*(quat.w*quat.z + quat.x*quat.y), 1.0 - 2.0*(quat.y*quat.y + quat.z*quat.z))
        self.object_poses[0, 0:4] = [x, y, z, yaw]
        quat_temp = [quat.x, quat.y, quat.z, quat.w]

        # Rotate the quaternion by 90 degrees around the robot's local x-axis
        quat_original = np.array(quat_temp)
        roll_rotation = R.from_euler('x', np.pi/2, degrees=False)
        r_original = R.from_quat(quat_original)
        r_combined = r_original * roll_rotation
        # Get the resulting quaternion [qx, qy, qz, qw]
        quat_rotated = r_combined.as_quat()
        self.object_poses[0, 4:] = quat_rotated.tolist()  # [qx, qy, qz, qw]
    
    def object2_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        quat = msg.pose.orientation
        
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        quat_array = [quat.x, quat.y, quat.z, quat.w]
        r = R.from_quat(quat_array)
        roll, pitch, yaw = r.as_euler('xyz', degrees=False)  # radians
        
        self.object_poses[1, 0:4] = [x, y, z, yaw]
        quat_temp = [quat.x, quat.y, quat.z, quat.w]

        # Rotate the quaternion by 90 degrees around the robot's local x-axis
        quat_original = np.array(quat_temp)
        roll_rotation = R.from_euler('x', np.pi/2, degrees=False)
        r_original = R.from_quat(quat_original)
        r_combined = r_original * roll_rotation
        # Get the resulting quaternion [qx, qy, qz, qw]
        quat_rotated = r_combined.as_quat()
        self.object_poses[1, 4:] = quat_rotated.tolist()  # [qx, qy, qz, qw]
        
      
    def object3_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        quat = msg.pose.orientation
        yaw = np.arctan2(2.0*(quat.w*quat.z + quat.x*quat.y), 1.0 - 2.0*(quat.y*quat.y + quat.z*quat.z))
        self.object_poses[2, 0:4] = [x, y, z, yaw]
        quat_temp = [quat.x, quat.y, quat.z, quat.w]

        # Rotate the quaternion by 90 degrees around the robot's local x-axis
        quat_original = np.array(quat_temp)
        roll_rotation = R.from_euler('x', np.pi/2, degrees=False)
        r_original = R.from_quat(quat_original)
        r_combined = r_original * roll_rotation
        # Get the resulting quaternion [qx, qy, qz, qw]
        quat_rotated = r_combined.as_quat()
        self.object_poses[2, 4:] = quat_rotated.tolist()  # [qx, qy, qz, qw]

    
    def object4_pose_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        quat = msg.pose.orientation
        yaw = np.arctan2(2.0*(quat.w*quat.z + quat.x*quat.y), 1.0 - 2.0*(quat.y*quat.y + quat.z*quat.z))
        self.object_poses[3, 0:4] = [x, y, z, yaw]
        quat_temp = [quat.x, quat.y, quat.z, quat.w]

        # Rotate the quaternion by 90 degrees around the robot's local x-axis
        quat_original = np.array(quat_temp)
        roll_rotation = R.from_euler('x', np.pi/2, degrees=False)
        r_original = R.from_quat(quat_original)
        r_combined = r_original * roll_rotation
        # Get the resulting quaternion [qx, qy, qz, qw]
        quat_rotated = r_combined.as_quat()
        self.object_poses[3, 4:] = quat_rotated.tolist()  # [qx, qy, qz, qw]


    def pub_env_obs(self, event):
        # publish robot and object poses, 8 + 4*4 = 24 in total
        env_obs = Float32MultiArray()
        env_obs.data = np.concatenate((self.robot_poses.flatten(), self.object_poses.flatten())).tolist()
        self.env_obs_pub.publish(env_obs)


if __name__ == "__main__":
    rospy.init_node('EnvPerception', anonymous=True)
    EnvPerceptionDate = EnvPerception()
    rospy.spin()
    
