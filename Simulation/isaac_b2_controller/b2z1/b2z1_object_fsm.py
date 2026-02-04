import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray
from visualization_msgs.msg import Marker, MarkerArray
import math
import tf
from tf.transformations import quaternion_from_euler
from sensor_msgs.msg import JointState
from scipy.spatial.transform import Rotation as R, Slerp
import pinocchio as pin
import numpy as np
import geometry_msgs.msg
import tf.transformations
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import copy
import yaml
import os


class MovingBotController:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        fsm_cfg = config['fsm']
        prefix = config['paths']['prefix']

        self.task_state = "WAIT_TASK_PLANNING"

        self.task_state_mapping = {
            "WAIT_TASK_PLANNING": 0.0,
            "WAIT_ROBOT_PATH": 1.0,
            "ROBOT_TRACKING": 2.0,
            "GRASPING": 3.0,
            "WAIT_OBJECT_PATH": 4.0,
            "OBJECT_TRACKING": 5.0,
            "RELEASING": 6.0
        }

        self.obj_num = 4

        self.robot_path = []
        self.object_path = []

        # observation
        self.robot_pose = [0, 0, 0, 0]
        self.object_pose = [[0, 0, 0, 0] for _ in range(self.obj_num)] 
        self.object_pose_rviz = [[0, 0, 0, 0] for _ in range(self.obj_num)] 
        self.arm_base_pose = [0,0,0, 0,0,0,0] # x,y,z, quat(w, x, y, z)
        self.object_grasp_pose = [[0, 0, 0, 0, 0, 0, 0] for _ in range(self.obj_num)]  # x,y,z, quat(w, x, y, z)

        self.sim_2_rviz_offset = fsm_cfg['sim_2_rviz_offset'] # x,y

        # vel cmd
        self.robot_vel_cmd = [0.0, 0.0, 0.0]
        self.object_vel_cmd = [0.0, 0.0, 0.0]

        # grasping cmd
        self.joint_cmd = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # [joint1, joint2, joint3, joint4, joint5, joint6, gripper]

        # task planning
        self.object_type = 0  # 0, 1, 2
        self.target_type = 0
        self.object_sequence = list(range(self.obj_num)) 
        self.target_sequence = list(range(self.obj_num)) 
        self.object_sequence = [0,1,2,3]
        self.target_sequence = [0,1,2,3] 
        self.is_task_plan = False
        
        self.object_category = fsm_cfg['default_object_category']
        object_params = fsm_cfg['objects'][self.object_category]
        self.object_grasp_cfg = object_params['grasp_cfg']
        self.object_plan_cfg = object_params['plan_cfg']
        self.object_com_offset = object_params['com_offset']
        self.arm_default_pose = object_params['arm_default_pose']
        self.mesh_source = "file://" + os.path.join(prefix, object_params['mesh_source'])
      
        self.task_num = 0
        self.object_pose_robot = [0, 0, 0, 0, 0, 0, 0]  # x,y,z, quat(x, y, z, w)
        self.robot_path_index = 0
        self.object_path_index = 0

        self.target_poses = [
            [p[0] + self.sim_2_rviz_offset[0], p[1] + self.sim_2_rviz_offset[1], p[2], p[3]]
            for p in object_params['target_poses']
        ]

        # Publisher for combined data
        self.control_data_pub = rospy.Publisher("/env_control_data", Float32MultiArray, queue_size=10)
        rospy.Timer(rospy.Duration(0.1), self.publish_control_data) # 10Hz

        # Subscriber for combined data
        rospy.Subscriber("/env_obs", Float32MultiArray, self.env_obs_callback)

        # subscribe to robot and object paths
        rospy.Subscriber("/visualizer/mincoPoint", Marker, self.path_callback)

        # pub the object and robot pose
        self.rviz_marker_pub = rospy.Publisher("/mesh_marker", MarkerArray, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.publish_object_poses) # 5Hz

        # pub robot joint states
        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=10)

        self.joint_names = ['FL_hip_joint', 'FR_hip_joint', 'RL_hip_joint', 'RR_hip_joint', 
                            'FL_thigh_joint', 'FR_thigh_joint', 'RL_thigh_joint', 'RR_thigh_joint', 
                            'joint1', 'FL_calf_joint', 'FR_calf_joint', 'RL_calf_joint', 'RR_calf_joint', 
                            'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'jointGripper']

        self.joint_positions = [0.1, -0.1, 0.1, -0.1, 
                                0.8,  0.8, 0.8,  0.8,
                                0.0, -1.5, -1.5, -1.5, -1.5,
                                1.48, -0.63, -0.84, 0.0, 1.57, 0.0]        #
        
        rospy.Timer(rospy.Duration(0.1), self.publish_joint_states)  #  joint state
        self.tf_broadcaster = tf.TransformBroadcaster()
        rospy.Timer(rospy.Duration(0.1), self.publish_tf)

        # pub the grasp pose
        self.grasp_pose_pub = rospy.Publisher('/object_grasp_pose_axes', Marker, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.publish_grasp_pose)

        # pub the planner_start pose
        self.planner_start_pub = rospy.Publisher('/planner_start_pose', PoseStamped, queue_size=1)
        self.planner_goal_pub = rospy.Publisher('/planner_goal_pose', PoseStamped, queue_size=1)

        # pub the task state
        self.task_state_pub = rospy.Publisher('/task_plan/poses', PoseArray, queue_size=1)
        self.task_sequence_sub = rospy.Subscriber('/task_plan/results', Int32MultiArray, self.task_state_callback)

        # pub the target poses
        self.target_poses_pub = rospy.Publisher('/target_poses', MarkerArray, queue_size=1)
        rospy.Timer(rospy.Duration(0.1), self.publish_target_sphere)
        
        # Ik for grasping
        z1_urdf_path = os.path.join(prefix, fsm_cfg['urdf_path'])
        self.z1_dyn_model = pin.buildModelFromUrdf(z1_urdf_path) 
        self.z1_dyn_data = self.z1_dyn_model.createData()
        self.ee_frame_id = self.z1_dyn_model.getFrameId("gripperStator", pin.FrameType.BODY)
        self.base_link_id = self.z1_dyn_model.getFrameId("link00", pin.FrameType.BODY)

        self.k = 0.0
        self.d_gripper = 0.0
        self.gripper_ang = -1.5
        self.err_norm = 1000.0
        self.reach_threshold = 0.25
        self.joint_default = [0,  1.48, -0.63, -0.84, 0.0, 1.57, 0]
        self.grasp_time = 0
        self.object_grasp_flag = False
        self.ee_stable_count = 0

        rospy.loginfo("[Controller] Initialized and waiting for robot path...")
       
    
    def task_state_callback(self, msg):
        self.object_sequence[0] = msg.data[0]
        self.object_sequence[1] = msg.data[2]
        self.object_sequence[2] = msg.data[4]
        self.object_sequence[3] = msg.data[6]

        self.target_sequence[0] = msg.data[1]  #  object_sequence
        self.target_sequence[1] = msg.data[3]  #  target_sequence
        self.target_sequence[2] = msg.data[5]  #  target_sequence
        self.target_sequence[3] = msg.data[7]  #  target_sequence

        self.is_task_plan = True


    def env_obs_callback(self, msg):
        data = msg.data
        robot_obs = data[:30]   
        object_obs = data[30:]  

        # update current poses
        if robot_obs is not None:
            self.robot_pose[0] = float(robot_obs[0]) + self.sim_2_rviz_offset[0] # x
            self.robot_pose[1] = float(robot_obs[1]) + self.sim_2_rviz_offset[1] # y
            self.robot_pose[2] = float(robot_obs[2])  # z
            self.robot_pose[3] = float(robot_obs[3])  # yaw

            self.arm_base_pose[0] = float(robot_obs[4]) + self.sim_2_rviz_offset[0]
            self.arm_base_pose[1] = float(robot_obs[5]) + self.sim_2_rviz_offset[1]
            self.arm_base_pose[2:7] = list(robot_obs[6:11])  # [qx, qy, qz, qw]

            self.joint_positions = list(robot_obs[11:]) 

        if object_obs is not None:
            self.object_pose[0][0] = float(object_obs[0]) + self.sim_2_rviz_offset[0] 
            self.object_pose[0][1] = float(object_obs[1]) + self.sim_2_rviz_offset[1]
            self.object_pose[0][2] = float(object_obs[2])
            self.object_pose[0][3] = float(object_obs[3])

            self.object_pose[1][0] = float(object_obs[4]) + self.sim_2_rviz_offset[0] 
            self.object_pose[1][1] = float(object_obs[5]) + self.sim_2_rviz_offset[1] 
            self.object_pose[1][2] = float(object_obs[6])
            self.object_pose[1][3] = float(object_obs[7])

            self.object_pose[2][0] = float(object_obs[8]) + self.sim_2_rviz_offset[0] 
            self.object_pose[2][1] = float(object_obs[9]) + self.sim_2_rviz_offset[1] 
            self.object_pose[2][2] = float(object_obs[10])
            self.object_pose[2][3] = float(object_obs[11])

            self.object_pose[3][0] = float(object_obs[12]) + self.sim_2_rviz_offset[0] 
            self.object_pose[3][1] = float(object_obs[13]) + self.sim_2_rviz_offset[1]
            self.object_pose[3][2] = float(object_obs[14])
            self.object_pose[3][3] = float(object_obs[15])
            self.object_pose_rviz = copy.deepcopy(self.object_pose)

            self.object_com_cal()
            self.object_grasp_pose[0] = self.get_grasp_pose(self.object_pose[0], self.object_grasp_cfg)  # 0.2,  0.9
            self.object_grasp_pose[1] = self.get_grasp_pose(self.object_pose[1], self.object_grasp_cfg)  # 0.2,  0.9
            self.object_grasp_pose[2] = self.get_grasp_pose(self.object_pose[2], self.object_grasp_cfg)  # 0.2,  0.9
            self.object_grasp_pose[3] = self.get_grasp_pose(self.object_pose[3], self.object_grasp_cfg)  # 0.2,  0.9

    
        # rospy.loginfo(f"Received robot_obs: {robot_obs}")
        # rospy.loginfo(f"Received object_obs: {object_obs}")

    # publish target_poses
    def publish_target_sphere(self, event=None):
        # publish sphere according to the target poses
        marker_array = MarkerArray()
        for i in range(self.obj_num):
            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "target_poses"
            marker.id = i + 1
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            start_x = self.target_poses[i][0]
            start_y = self.target_poses[i][1]
            start_z = self.target_poses[i][2]

            quat = quaternion_from_euler(0.0, 0.0, self.target_poses[i][3])  
            direction = tf.transformations.quaternion_matrix(quat)[:3, 0] 
            end_x = start_x + direction[0] * 0.5  # 
            end_y = start_y + direction[1] * 0.5
            end_z = start_z + direction[2] * 0.5

            start_point = geometry_msgs.msg.Point(start_x, start_y, start_z)
            end_point = geometry_msgs.msg.Point(end_x, end_y, end_z)
            marker.points.append(start_point)
            marker.points.append(end_point)

            marker.scale.x = 0.05 
            marker.scale.y = 0.1   
            marker.scale.z = 0.0   

            if i == self.target_type:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.target_poses_pub.publish(marker_array)


    def publish_task_state(self, event=None):
        # publish the chiar poses and the target poses
        pose_array = PoseArray()
        pose_array.header.frame_id = "world"
        pose_array.header.stamp = rospy.Time.now()

        for i in range(self.obj_num):
            pose = PoseStamped()
            pose.pose.position.x = self.object_pose[i][0]
            pose.pose.position.y = self.object_pose[i][1]
            pose.pose.position.z = self.object_pose[i][2]
            quat = quaternion_from_euler(0.0, 0.0, self.object_pose[i][3])
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            pose_array.poses.append(pose.pose)

        for i in range(self.obj_num):
            pose = PoseStamped()
            pose.pose.position.x = self.target_poses[i][0]
            pose.pose.position.y = self.target_poses[i][1]
            pose.pose.position.z = self.target_poses[i][2]
            quat = quaternion_from_euler(0.0, 0.0, self.target_poses[i][3])
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]
            pose_array.poses.append(pose.pose)

        self.task_state_pub.publish(pose_array)


    def publish_planner_start_pose(self, event=None):
        if self.task_state == "WAIT_ROBOT_PATH":
            robot_start_pose = PoseStamped()
            robot_start_pose.header.frame_id = "world"
            robot_start_pose.header.stamp = rospy.Time.now()
            robot_start_pose.pose.position.x = self.robot_pose[0]
            robot_start_pose.pose.position.y = self.robot_pose[1]
            robot_start_pose.pose.position.z = self.robot_pose[2]
            yaw = self.robot_pose[3]
            quat = quaternion_from_euler(0.0, 0.0, yaw)  #
            robot_start_pose.pose.orientation.x = quat[0]
            robot_start_pose.pose.orientation.y = quat[1]
            robot_start_pose.pose.orientation.z = quat[2]
            robot_start_pose.pose.orientation.w = quat[3]
            self.planner_start_pub.publish(robot_start_pose)
        if self.task_state == "WAIT_OBJECT_PATH":
            object_start_pose = PoseStamped()
            object_start_pose.header.frame_id = "world"
            object_start_pose.header.stamp = rospy.Time.now()
            object_start_pose.pose.position.x = self.object_pose[self.object_type][0]
            object_start_pose.pose.position.y = self.object_pose[self.object_type][1]
            object_start_pose.pose.position.z = self.object_pose[self.object_type][2]
            yaw = self.object_pose[self.object_type][3]
            quat = quaternion_from_euler(0.0, 0.0, yaw)  #
            object_start_pose.pose.orientation.x = quat[0]
            object_start_pose.pose.orientation.y = quat[1]
            object_start_pose.pose.orientation.z = quat[2]
            object_start_pose.pose.orientation.w = quat[3]
            self.planner_start_pub.publish(object_start_pose)
    

    def publish_planner_goal_pose(self, event=None):
        planner_goal_pose = self.get_grasp_pose(self.object_pose[self.object_type], self.object_plan_cfg)
        x, y, z, qx, qy, qz, qw = planner_goal_pose

        # publish the goal pose to grasp the object
        if self.task_state == "WAIT_ROBOT_PATH":
            robot_goal_pose = PoseStamped()
            robot_goal_pose.header.frame_id = "world"
            robot_goal_pose.header.stamp = rospy.Time.now()
            robot_goal_pose.pose.position.x = planner_goal_pose[0]
            robot_goal_pose.pose.position.y = planner_goal_pose[1]
            robot_goal_pose.pose.position.z = planner_goal_pose[2]
            robot_goal_pose.pose.orientation.x = planner_goal_pose[3]
            robot_goal_pose.pose.orientation.y = planner_goal_pose[4]
            robot_goal_pose.pose.orientation.z = planner_goal_pose[5]
            robot_goal_pose.pose.orientation.w = planner_goal_pose[6]
            self.planner_goal_pub.publish(robot_goal_pose)
        # publish the target pose to place the object
        if self.task_state == "WAIT_OBJECT_PATH":
            robot_goal_pose = PoseStamped()
            robot_goal_pose.header.frame_id = "world"
            robot_goal_pose.header.stamp = rospy.Time.now()
            robot_goal_pose.pose.position.x = self.target_poses[self.target_type][0]
            robot_goal_pose.pose.position.y = self.target_poses[self.target_type][1]
            robot_goal_pose.pose.position.z = self.target_poses[self.target_type][2]
            yaw = self.target_poses[self.target_type][3]
            quat = quaternion_from_euler(0.0, 0.0, yaw)  # 
            robot_goal_pose.pose.orientation.x = quat[0]
            robot_goal_pose.pose.orientation.y = quat[1]
            robot_goal_pose.pose.orientation.z = quat[2]
            robot_goal_pose.pose.orientation.w = quat[3]
            self.planner_goal_pub.publish(robot_goal_pose)
    

    def publish_control_data(self, event=None):
        data = (
            list(self.robot_vel_cmd) +   # [vx, vy, wz]
            list(self.object_vel_cmd) +  # [vx, vy, wz]
            list(self.joint_cmd) +       # Joint commands
            [self.task_state_mapping.get(self.task_state, -1.0)]+        # Task state
            [self.object_type]
        )
        msg = Float32MultiArray(data=data)
        self.control_data_pub.publish(msg)
        # print(f"[Controller] Published control data: {data}")


    def publish_tf(self, event=None):
        self.tf_broadcaster.sendTransform(
            (self.robot_pose[0], self.robot_pose[1], self.robot_pose[2]),
            quaternion_from_euler(0, 0, self.robot_pose[3]),
            rospy.Time.now(),
            "base",  #
            "world"
        )


    def publish_joint_states(self, event=None):
        joint_state = JointState()
        joint_state.header.stamp = rospy.Time.now()
        joint_state.name = self.joint_names
        joint_state.position = self.joint_positions 
        self.joint_pub.publish(joint_state)


    def publish_object_poses(self, event=None):
        marker_array = MarkerArray()
        # === Object Markers ===
        if self.object_pose_rviz is not None:
            for i, pose in enumerate(self.object_pose_rviz):  

                object_marker = Marker()
                object_marker.header.frame_id = "world"
                object_marker.header.stamp = rospy.Time.now()
                object_marker.ns = "object"
                object_marker.id = i + 1  
                object_marker.type = Marker.MESH_RESOURCE
                object_marker.action = Marker.ADD
                object_marker.mesh_resource = self.mesh_source 

                object_marker.mesh_use_embedded_materials = False

                object_marker.pose.position.x = pose[0] 
                object_marker.pose.position.y = pose[1]
                object_marker.pose.position.z = pose[2]

                quat = quaternion_from_euler(0, 0, pose[3])  
                object_marker.pose.orientation.x = quat[0]
                object_marker.pose.orientation.y = quat[1]
                object_marker.pose.orientation.z = quat[2]
                object_marker.pose.orientation.w = quat[3]

                object_marker.scale.x = 1.0
                object_marker.scale.y = 1.0
                object_marker.scale.z = 1.0

                object_marker.color.r = 0.6
                object_marker.color.g = 0.5
                object_marker.color.b = 0.5
                object_marker.color.a = 0.8

                marker_array.markers.append(object_marker)

        self.rviz_marker_pub.publish(marker_array)


    def publish_grasp_pose(self, event=None):
        x, y, z, qx, qy, qz, qw = self.object_grasp_pose[self.object_type]
        # print("grasp pose", self.object_grasp_pose[self.object_type])
        # print("object pose", self.object_pose[self.object_type])

        rot = tf.transformations.quaternion_matrix([qx, qy, qz, qw])[:3, :3]

        axis_dirs = [
            (np.array([0.2, 0, 0]), [1.0, 0.0, 0.0, 1.0]),   
            (np.array([0, 0.2, 0]), [0.0, 1.0, 0.0, 1.0]),  
            (np.array([0, 0, 0.2]), [0.0, 0.0, 1.0, 1.0]),  
        ]

        for i, (local_dir, color) in enumerate(axis_dirs):
            dir_world = rot.dot(local_dir)

            marker = Marker()
            marker.header.frame_id = "world"
            marker.header.stamp = rospy.Time.now()
            marker.ns = f"object_grasp_pose_axes"  
            marker.id = i  
            marker.type = Marker.ARROW
            marker.action = Marker.ADD

            pt_start = geometry_msgs.msg.Point(x, y, z)
            pt_end = geometry_msgs.msg.Point(x + dir_world[0], y + dir_world[1], z + dir_world[2])
            marker.points.append(pt_start)
            marker.points.append(pt_end)

            marker.scale.x = 0.02  
            marker.scale.y = 0.04  
            marker.scale.z = 0.0   

            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            self.grasp_pose_pub.publish(marker)


    def path_callback(self, msg):
        # rospy.loginfo(f"Message type: {type(msg)}")
        # rospy.loginfo(f"Message content: {msg}")
 
        if self.task_state == "WAIT_ROBOT_PATH":
            self.robot_path = []
            for pt in msg.points:
                self.robot_path.append(pt)
            self.robot_path_index = 0
            if len(self.robot_path) == 0:
                rospy.logwarn("[Controller] Received empty robot path. Waiting for valid path.")
                return
            self.task_state = "ROBOT_TRACKING"
            rospy.loginfo("[Controller] Robot path (Marker) received. Start tracking.")

        if self.task_state == "WAIT_OBJECT_PATH":
            self.object_path = []
            for pt in msg.points:
                self.object_path.append(pt)
            self.object_path_index = 0
            if len(self.object_path) == 0:
                rospy.logwarn("[Controller] Received empty object path. Waiting for valid path.")
                return
            self.task_state = "OBJECT_TRACKING"
            rospy.loginfo("[Controller] Object path (Marker) received. Start tracking.")


    def FSM(self):
        if self.task_state == "WAIT_TASK_PLANNING":
            self.publish_task_state()
            print("[Controller] Waiting for task planning...")
            if self.is_task_plan:
                print("task sequence", self.object_sequence)
                print("target sequence", self.target_sequence)
                rospy.loginfo("[Controller] Task planning complete. Starting first task.")
                self.task_state = "WAIT_ROBOT_PATH"
                self.task_num = 0
                rospy.sleep(3)
            rospy.sleep(0.1)

        elif self.task_state == "WAIT_ROBOT_PATH":
            self.object_type = self.object_sequence[self.task_num]
            self.target_type = self.target_sequence[self.task_num]
            self.publish_planner_start_pose()
            self.publish_planner_goal_pose()
            rospy.loginfo("[Controller] Waiting for robot path...")
            rospy.sleep(0.1)

        elif self.task_state == "ROBOT_TRACKING":
            done = self.robot_tracking_controller()
            print("tracking")
            # done = True
            if done:
                rospy.loginfo("[Controller] Robot reached. Start grasping.")
                rospy.sleep(1)
                self.task_state = "GRASPING"

        elif self.task_state == "GRASPING":
            done = self.object_grasp()
            if done:
                rospy.sleep(2)
                rospy.loginfo("[Controller] Grasp complete. Waiting for object path.")
                self.task_state = "WAIT_OBJECT_PATH"

        elif self.task_state == "WAIT_OBJECT_PATH":
            self.publish_planner_start_pose()
            self.publish_planner_goal_pose()
            rospy.loginfo("[Controller] Waiting for object path...")
            rospy.sleep(0.1)

        elif self.task_state == "OBJECT_TRACKING":
            done = self.object_tracking_controller()
            if done:
                rospy.loginfo("[Controller] Object reached. Releasing.")
                rospy.sleep(1)
                self.task_state = "RELEASING"

        elif self.task_state == "RELEASING":
            if self.object_release():
                rospy.loginfo("[Controller] Release complete. Back to WAIT_ROBOT_PATH.")
                self.task_num = self.task_num + 1
                self.robot_vel_cmd = [-0.5, 0.0, 0.0] 
                rospy.sleep(4)
                if self.task_num >= self.obj_num:
                    rospy.loginfo("[Controller] All tasks completed")
                    self.robot_vel_cmd = [0.0, 0.0, 0.0]
                    return 
                else:
                    self.task_state = "WAIT_ROBOT_PATH"

                
    def robot_tracking_controller(self):
        """
        robot_cur_pose: Tensor of shape (4,) -> [x, y, z, yaw]
        """
        if self.robot_path_index >= len(self.robot_path):
            if len(self.robot_path) > 1:
                last_point = self.robot_path[-1]
                object_point_x = self.object_pose[self.object_type][0]
                object_point_y = self.object_pose[self.object_type][1]

                dx = object_point_x - self.robot_pose[0]
                dy = object_point_y - self.robot_pose[1]
                final_yaw = math.atan2(dy, dx)  
                
                yaw_error = final_yaw - self.robot_pose[3]
                yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi  #
                if abs(yaw_error) > math.radians(5):  #
                    omega = 2.0 * yaw_error  # 
                    omega = max(min(omega, 0.6), -0.6)  # 
                    self.robot_vel_cmd = [0.0, 0.0, omega]  # 
                    print("final yaw_error", yaw_error)
                    return False  # 
            self.robot_vel_cmd = [0.0, 0.0, 0.0]
            return True

        max_vx = 0.5
        max_wz = 0.6
        Kp_yaw = 2.0

        current_x = self.robot_pose[0]
        current_y = self.robot_pose[1]
        current_yaw = self.robot_pose[3]

        target = self.robot_path[self.robot_path_index]
        self.target_pose = target

        dx = target.x - current_x
        dy = target.y - current_y
        dist = math.hypot(dx, dy)

        is_final_point = (self.robot_path_index == len(self.robot_path) - 1)
        if is_final_point:
            self.reach_threshold = 0.15 

        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - current_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi 

        if dist < self.reach_threshold:
            self.robot_path_index += 1
            return False

        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - current_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi  

        if abs(yaw_error) > math.radians(15):
            vx = 0.0
        else:
            vx = max_vx

        omega = Kp_yaw * yaw_error
        omega = max(min(omega, max_wz), -max_wz)

        self.robot_vel_cmd = [vx, 0.0, omega]
        return False


    def object_grasp(self):
        rospy.loginfo("[Grasp] Executing grasp step...")
        # keep the proper grasp distance
        self.grasp_time = self.grasp_time + 1
        robot_2_obj_distance = np.linalg.norm(np.array(self.robot_pose[0:2]) - np.array(self.object_pose[self.object_type][0:2]))
        dist_gap = robot_2_obj_distance - self.object_plan_cfg[0]

        if  self.object_grasp_flag == False:
            self.joint_cmd[:] = self.joint_default[:]
            if dist_gap > 0.02:
                rospy.loginfo(f"[Grasp] Robot too far from object: {np.linalg.norm(robot_2_obj_distance):.2f}m")
                self.robot_vel_cmd = [0.15, 0.0, 0.0]
                return False
            elif dist_gap < -0.02:
                rospy.loginfo(f"[Grasp] Robot too close to object: {np.linalg.norm(robot_2_obj_distance):.2f}m")
                self.robot_vel_cmd = [-0.15, 0.0, 0.0]
                return False
            else:
                self.robot_vel_cmd = [0.0, 0.0, 0.0]
                self.object_grasp_flag = True
                return False
        if self.grasp_time < 100:
            return False


        # Calculate the transformation matrices
        # print(f"[Grasp] Arm base pose: {self.arm_base_pose}")
        # print(f"[Grasp] Object grasp pose: {self.object_grasp_pose[self.object_type]}")

        T_world_base = self.to_se3(self.arm_base_pose)
        T_world_target = self.to_se3(self.object_grasp_pose[self.object_type])
        T_base_target = T_world_base.inverse() * T_world_target
        if self.err_norm < 0.1:
            self.ee_stable_count = self.ee_stable_count + 1
        else:
            self.ee_stable_count = 0
        if self.ee_stable_count > 20 or self.grasp_time > 700:
            self.d_gripper = 0.01
        self.k  = self.k + self.d_gripper
        self.k = min(self.k, 0.25)  
        translation = T_base_target.translation  #

        if self.object_category == "box":
            translation[2] -= self.k / 2.2
        elif self.object_category == "table":
            translation[0] += self.k / 1.5
            translation[2] -= 0.01  

        else:
            translation[0] += self.k  
            translation[2] += 0.03  


        T_base_target.translation = translation
        oMdes = T_base_target

        # Initialize current joint state
        q = np.zeros(7)  # 18 joints in Z1
        q[0] = self.joint_positions.copy()[8] # joint 1
        q[1:7] = self.joint_positions.copy()[13:19]  # joint 2 to joint 6
        q_init = q.copy()

        eps = 1e-3
        IT_MAX = 200
        damp = 1e-12
        DT = 3e-3    

        for _ in range(IT_MAX):
            pin.forwardKinematics(self.z1_dyn_model, self.z1_dyn_data, q)
            pin.updateFramePlacements(self.z1_dyn_model, self.z1_dyn_data)
            iMd = self.z1_dyn_data.oMf[self.ee_frame_id].actInv(oMdes)

            err = pin.log(iMd).vector
            self.err_norm = np.linalg.norm(err)
            # print(f"[Grasp] Iteration error norm: {self.err_norm}")
            
            if np.linalg.norm(err) < eps:
                break
            print(f"[Grasp] Iteration error: {np.linalg.norm(err)}")

            J = pin.computeFrameJacobian(self.z1_dyn_model, self.z1_dyn_data, q, self.ee_frame_id, pin.ReferenceFrame.LOCAL)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
            q = pin.integrate(self.z1_dyn_model, q, v * DT)
        
        # final_frame = self.z1_dyn_data.oMf[self.ee_frame_id]
        # target_frame = oMdes
        self.gripper_ang += self.d_gripper 
        self.gripper_ang = max(min(self.gripper_ang, -0.1), -1.5) 
        q[6] = self.gripper_ang
        
        if self.object_category == "table":
            if self.gripper_ang >= -0.4:
                q[0:6] = self.arm_default_pose[0:6]

        if self.gripper_ang >= -0.1:
            self.d_gripper = 0.01
            self.k = 0.0
            self.gripper_ang = -1.5
            self.err_norm = 1000
            self.ee_stable_count = 0
            self.object_grasp_flag = False
            self.grasp_time = 0
            return True
        self.joint_cmd = q.tolist()

        # print(f"[Grasp] Grasping with joint commands: {self.joint_cmd}")


    def object_tracking_controller(self):
        print("[tracking]object_type:", self.object_type)
        if self.object_path_index >= len(self.object_path):
            if len(self.object_path) > 1:
                last_point = self.object_path[-1]
                second_last_point = self.object_path[-2]
                dx = last_point.x - second_last_point.x
                dy = last_point.y - second_last_point.y
                final_yaw = math.atan2(dy, dx)  
                yaw_error = final_yaw - self.object_pose[self.object_type][3]
                yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi 

        
                if abs(yaw_error) > math.radians(10):  
                    omega = 2.0 * yaw_error  
                    omega = max(min(omega, 0.5), -0.5) 
                    self.object_vel_cmd = [0.0, 0.0, omega]  
                    print("object_final yaw_error", yaw_error)
                    return False 

            self.object_vel_cmd = [0.0, 0.0, 0.0]
            return True
        
        max_vx = 0.5
        max_wz = 0.5
        Kp_yaw = 2.0

        current_x = self.object_pose[self.object_type][0]
        current_y = self.object_pose[self.object_type][1]
        current_yaw = self.object_pose[self.object_type][3]
        current_yaw = (current_yaw + math.pi) % (2 * math.pi) - math.pi

        target = self.object_path[self.object_path_index]
        self.target_pose = target

        dx = target.x - current_x
        dy = target.y - current_y
        dist = math.hypot(dx, dy)

        is_final_point = (self.object_path_index == len(self.object_path) - 1)
        if is_final_point:
            self.reach_threshold = 0.2 

        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - current_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi 

        if dist < self.reach_threshold:
            self.object_path_index += 1
            return False

        target_yaw = math.atan2(dy, dx)
        yaw_error = target_yaw - current_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi 

        if abs(yaw_error) > math.radians(15):
            vx = 0.0
            if abs(yaw_error) > math.radians(120):
                # vx = -max_vx/2.
                print(f"backing")
            print(f"[tracking] object turning [error: {yaw_error}]")
        else:
            vx = max_vx
            print(f"[tracking] object going forward")
            
        omega = Kp_yaw * yaw_error
        omega = max(min(omega, max_wz), -max_wz)

        self.object_vel_cmd = [vx, 0.0, omega]
        return False


    def object_release(self):
        rospy.loginfo("[Release] Executing release action...")
        self.joint_cmd = copy.deepcopy(self.arm_default_pose) 
        self.joint_cmd[-1] = -1.5
        self.k = self.k - self.d_gripper
        self.k = max(self.k, -0.15) 
        self.joint_cmd[1] = self.joint_cmd[1] + self.k
        rospy.sleep(0.1)
        self.robot_vel_cmd = [0.0, 0.0, 0.0]

        if self.k <= -0.15:
            self.k = 0.0
            self.d_gripper = 0.0  
            return True



## ======== Helper Functions ======== ##
    def _distance(self, p1, p2):
        dx = p1.x - p2.x
        dy = p1.y - p2.y
        dz = p1.z - p2.z
        return (dx*dx + dy*dy + dz*dz)**0.5
    
    def to_se3(self, pose):
        """
        :param pose: [x, y, z, qx, qy, qz, qw]
        :return: pin.SE3 对象
        """
        pos = np.array(pose[:3])
        quat = np.array(pose[3:])  # [x, y, z, w]

        rot = R.from_quat(quat).as_matrix()
        return pin.SE3(rot, pos)
    
    def get_grasp_pose(self, pose, cfg):

        dx, dz, pitch = cfg
        x_c, y_c, z_c, yaw_c = pose
        x_g = x_c - dx * np.cos(yaw_c) 
        y_g = y_c - dx * np.sin(yaw_c) 
        z_g = z_c + dz

        pitch = np.deg2rad(pitch)
        roll = 0.0
        qx, qy, qz, qw = tf.transformations.quaternion_from_euler(
            roll, pitch, yaw_c, axes='sxyz'
        )

        return [x_g, y_g, z_g, qx, qy, qz, qw]
    

    def object_com_cal(self):
        dx, dy = self.object_com_offset

        x_c, y_c, z_c, yaw_c = self.object_pose_rviz[0]
        x_global = x_c + dx * np.cos(yaw_c) - dy * np.sin(yaw_c)
        y_global = y_c + dx * np.sin(yaw_c) + dy * np.cos(yaw_c)
        self.object_pose[0] = [x_global, y_global, z_c, yaw_c]

        x_c, y_c, z_c, yaw_c = self.object_pose_rviz[1]
        x_global = x_c + dx * np.cos(yaw_c) - dy * np.sin(yaw_c)
        y_global = y_c + dx * np.sin(yaw_c) + dy * np.cos(yaw_c)
        self.object_pose[1] = [x_global, y_global, z_c, yaw_c]

        x_c, y_c, z_c, yaw_c = self.object_pose_rviz[2]
        x_global = x_c + dx * np.cos(yaw_c) - dy * np.sin(yaw_c)
        y_global = y_c + dx * np.sin(yaw_c) + dy * np.cos(yaw_c)
        self.object_pose[2] = [x_global, y_global, z_c, yaw_c]

        x_c, y_c, z_c, yaw_c = self.object_pose_rviz[3]
        x_global = x_c + dx * np.cos(yaw_c) - dy * np.sin(yaw_c)
        y_global = y_c + dx * np.sin(yaw_c) + dy * np.cos(yaw_c)
        self.object_pose[3] = [x_global, y_global, z_c, yaw_c]

    

if __name__ == "__main__":
    rospy.init_node('moving_bot_controller', anonymous=True)
    controller = MovingBotController()

    rate = rospy.Rate(100)  # 10Hz
    while not rospy.is_shutdown():
        controller.FSM()
        rate.sleep()
    
