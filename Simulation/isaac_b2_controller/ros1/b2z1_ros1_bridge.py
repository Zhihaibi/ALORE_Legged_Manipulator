from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import numpy as np
from cv_bridge import CvBridge
import cv2
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
import subprocess
import time

import carb
import omni
import omni.graph.core as og
import usdrt.Sdf
from isaacsim.core.api import SimulationContext
from isaacsim.core.utils import extensions, stage
from isaacsim.storage.native import get_assets_root_path
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd, UsdGeom


extensions.enable_extension("isaacsim.ros1.bridge")

from omni.isaac.ros2_bridge import read_camera_info


class RobotDataManager(Node):
    def __init__(self, env, lidar_annotators, cameras, cfg):
        super().__init__("robot_data_manager")
        self.cfg = cfg
        self.create_ros_time_graph()
        sim_time_set = False
        while (rclpy.ok() and sim_time_set==False):
            sim_time_set = self.use_sim_time()

        self.env = env
        self.num_envs = env.unwrapped.scene.num_envs
        self.lidar_annotators = lidar_annotators
        self.cameras = cameras
        self.points = []
        
        # ROS2 Broadcaster
        self.broadcaster= TransformBroadcaster(self)
        
        # ROS2 Publisher
        self.odom_pub = []
        self.pose_pub = []
        self.lidar_pub = []
        self.semantic_seg_img_vis_pub = []

        # ROS2 Subscriber
        self.cmd_vel_sub = []
        self.color_img_sub = []
        self.depth_img_sub = []
        self.semantic_seg_img_sub = []

        # ROS2 Timer
        self.lidar_publish_timer = []
        for i in range(self.num_envs):
            if (self.num_envs == 1):
                self.odom_pub.append(
                    self.create_publisher(Odometry, "unitree_go2/odom", 10))
                self.pose_pub.append(
                    self.create_publisher(PoseStamped, "unitree_go2/pose", 10))
                self.lidar_pub.append(
                    self.create_publisher(PointCloud2, "unitree_go2/lidar/point_cloud", 10)
                )
                self.semantic_seg_img_vis_pub.append(
                    self.create_publisher(Image, "unitree_go2/front_cam/semantic_segmentation_image_vis", 10)
                )
                self.cmd_vel_sub.append(
                    self.create_subscription(Twist, "unitree_go2/cmd_vel", 
                    lambda msg: self.cmd_vel_callback(msg, 0), 10)
                )
                self.semantic_seg_img_sub.append(
                    self.create_subscription(Image, "/unitree_go2/front_cam/semantic_segmentation_image", 
                    lambda msg: self.semantic_segmentation_callback(msg, 0), 10)
                )
            else:
                self.odom_pub.append(
                    self.create_publisher(Odometry, f"unitree_go2_{i}/odom", 10))
                self.pose_pub.append(
                    self.create_publisher(PoseStamped, f"unitree_go2_{i}/pose", 10))
                self.lidar_pub.append(
                    self.create_publisher(PointCloud2, f"unitree_go2_{i}/lidar/point_cloud", 10)
                )
                self.semantic_seg_img_vis_pub.append(
                    self.create_publisher(Image, f"unitree_go2_{i}/front_cam/semantic_segmentation_image_vis", 10)
                )
                self.cmd_vel_sub.append(
                    self.create_subscription(Twist, f"unitree_go2_{i}/cmd_vel", 
                    lambda msg, env_idx=i: self.cmd_vel_callback(msg, env_idx), 10)
                )
                self.semantic_seg_img_sub.append(
                    self.create_subscription(Image, f"/unitree_go2_{i}/front_cam/semantic_segmentation_image", 
                    lambda msg, env_idx=i: self.semantic_segmentation_callback(msg, env_idx), 10)
                )
        
        # self.create_timer(0.1, self.pub_ros2_data_callback)
        # self.create_timer(0.1, self.pub_lidar_data_callback)

        # use wall time for lidar and odom pub
        self.odom_pose_freq = 50.0
        self.lidar_freq = 15.0
        self.odom_pose_pub_time = time.time()
        self.lidar_pub_time = time.time() 
        self.create_static_transform()
        self.create_camera_publisher()  

 
    def create_ros_time_graph(self):
        og.Controller.edit(
            {"graph_path": "/ClockGraph", "evaluator_name": "execution"},
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("ReadSimTime", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                    ("PublishClock", "omni.isaac.ros2_bridge.ROS2PublishClock"),
                    ("OnPlayBack", "omni.graph.action.OnPlaybackTick"),
                ],
                og.Controller.Keys.CONNECT: [
                    # Connecting execution of OnImpulseEvent node to PublishClock so it will only publish when an impulse event is triggered
                    ("OnPlayBack.outputs:tick", "PublishClock.inputs:execIn"),
                    # Connecting simulationTime data of ReadSimTime to the clock publisher node
                    ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ],
                og.Controller.Keys.SET_VALUES: [
                    # Assigning topic name to clock publisher
                    ("PublishClock.inputs:topicName", "/clock"),
                ],
            },
        )

    def use_sim_time(self):
        # Define the command as a list
        command = ["ros2", "param", "set", "/robot_data_manager", "use_sim_time", "true"]

        # Run the command in a non-blocking way
        subprocess.Popen(command)  
        return True

    def create_static_transform(self):
        for i in range(self.num_envs):
            # LiDAR
            # Create and publish the transform
            lidar_broadcaster = StaticTransformBroadcaster(self)
            base_lidar_transform = TransformStamped()
            base_lidar_transform.header.stamp = self.get_clock().now().to_msg()
            if (self.num_envs == 1):
                base_lidar_transform.header.frame_id = "unitree_go2/base_link"
                base_lidar_transform.child_frame_id = "unitree_go2/lidar_frame"
            else:
                base_lidar_transform.header.frame_id = f"unitree_go2_{i}/base_link"
                base_lidar_transform.child_frame_id = f"unitree_go2_{i}/lidar_frame"

            # Translation
            base_lidar_transform.transform.translation.x = 0.2
            base_lidar_transform.transform.translation.y = 0.0
            base_lidar_transform.transform.translation.z = 0.2
            
            # Rotation 
            base_lidar_transform.transform.rotation.x = 0.0
            base_lidar_transform.transform.rotation.y = 0.0
            base_lidar_transform.transform.rotation.z = 0.0
            base_lidar_transform.transform.rotation.w = 1.0
            
            # Publish the transform
            lidar_broadcaster.sendTransform(base_lidar_transform)
    
            # -------------------------------------------------------------
            # Camera
            # Create and publish the transform
            camera_broadcaster = StaticTransformBroadcaster(self)
            base_cam_transform = TransformStamped()
            # base_cam_transform.header.stamp = self.get_clock().now().to_msg()
            if (self.num_envs == 1):
                base_cam_transform.header.frame_id = "unitree_go2/base_link"
                base_cam_transform.child_frame_id = "unitree_go2/front_cam"
            else:
                base_cam_transform.header.frame_id = f"unitree_go2_{i}/base_link"
                base_cam_transform.child_frame_id = f"unitree_go2_{i}/front_cam"

            # Translation
            base_cam_transform.transform.translation.x = 0.4
            base_cam_transform.transform.translation.y = 0.0
            base_cam_transform.transform.translation.z = 0.2
            
            # Rotation 
            base_cam_transform.transform.rotation.x = -0.5
            base_cam_transform.transform.rotation.y = 0.5
            base_cam_transform.transform.rotation.z = -0.5
            base_cam_transform.transform.rotation.w = 0.5
            
            # Publish the transform
            camera_broadcaster.sendTransform(base_cam_transform)
    
    def create_camera_publisher(self):
        # self.pub_image_graph()
        if (self.cfg.sensor.enable_camera):
            if (self.cfg.sensor.color_image):
                self.pub_color_image()
            if (self.cfg.sensor.depth_image):
                self.pub_depth_image()
            if (self.cfg.sensor.semantic_segmentation):
                self.pub_semantic_image()
            # self.pub_cam_depth_cloud()
            self.publish_camera_info()
    
    def publish_odom(self, base_pos, base_rot, base_lin_vel_b, base_ang_vel_b, env_idx):
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = "map"
        if (self.num_envs == 1):
            odom_msg.child_frame_id = "base_link"
        else:
            odom_msg.child_frame_id = f"unitree_go2_{env_idx}/base_link"
        odom_msg.pose.pose.position.x = base_pos[0].item()
        odom_msg.pose.pose.position.y = base_pos[1].item()
        odom_msg.pose.pose.position.z = base_pos[2].item()
        odom_msg.pose.pose.orientation.x = base_rot[1].item()
        odom_msg.pose.pose.orientation.y = base_rot[2].item()
        odom_msg.pose.pose.orientation.z = base_rot[3].item()
        odom_msg.pose.pose.orientation.w = base_rot[0].item()
        odom_msg.twist.twist.linear.x = base_lin_vel_b[0].item()
        odom_msg.twist.twist.linear.y = base_lin_vel_b[1].item()
        odom_msg.twist.twist.linear.z = base_lin_vel_b[2].item()
        odom_msg.twist.twist.angular.x = base_ang_vel_b[0].item()
        odom_msg.twist.twist.angular.y = base_ang_vel_b[1].item()
        odom_msg.twist.twist.angular.z = base_ang_vel_b[2].item()
        self.odom_pub[env_idx].publish(odom_msg)

        # transform
        map_base_trans = TransformStamped()
        map_base_trans.header.stamp = self.get_clock().now().to_msg()
        map_base_trans.header.frame_id = "map"
        if (self.num_envs == 1):
            map_base_trans.child_frame_id = "unitree_go2/base_link"
        else:
            map_base_trans.child_frame_id = f"unitree_go2_{env_idx}/base_link"
        map_base_trans.transform.translation.x = base_pos[0].item()
        map_base_trans.transform.translation.y = base_pos[1].item()
        map_base_trans.transform.translation.z = base_pos[2].item()
        map_base_trans.transform.rotation.x = base_rot[1].item()
        map_base_trans.transform.rotation.y = base_rot[2].item()
        map_base_trans.transform.rotation.z = base_rot[3].item()
        map_base_trans.transform.rotation.w = base_rot[0].item()
        self.broadcaster.sendTransform(map_base_trans)
    
    def publish_pose(self, base_pos, base_rot, env_idx):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = base_pos[0].item()
        pose_msg.pose.position.y = base_pos[1].item()
        pose_msg.pose.position.z = base_pos[2].item()
        pose_msg.pose.orientation.x = base_rot[1].item()
        pose_msg.pose.orientation.y = base_rot[2].item()
        pose_msg.pose.orientation.z = base_rot[3].item()
        pose_msg.pose.orientation.w = base_rot[0].item()
        self.pose_pub[env_idx].publish(pose_msg)


    def publish_lidar_data(self, points, env_idx):
        point_cloud = PointCloud2()
        if (self.num_envs == 1):
            point_cloud.header.frame_id = "unitree_go2/lidar_frame"
        else:
            point_cloud.header.frame_id = f"unitree_go2_{env_idx}/lidar_frame"
        point_cloud.header.stamp = self.get_clock().now().to_msg()
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        point_cloud = point_cloud2.create_cloud(point_cloud.header, fields, points)
        self.lidar_pub[env_idx].publish(point_cloud)        

    def pub_ros2_data_callback(self):
        robot_data = self.env.unwrapped.scene["unitree_go2"].data
        for i in range(self.num_envs):
            self.publish_odom(robot_data.root_state_w[i, :3],
                              robot_data.root_state_w[i, 3:7],
                              robot_data.root_lin_vel_b[i],
                              robot_data.root_ang_vel_b[i],
                              i)
            self.publish_pose(robot_data.root_state_w[i, :3],
                              robot_data.root_state_w[i, 3:7], i)

    def pub_lidar_data_callback(self):
        for i in range(self.num_envs):
            self.publish_lidar_data(self.lidar_annotators[i].get_data()["data"].reshape(-1, 3), i)

    def pub_ros2_data(self):
        pub_odom_pose = False
        pub_lidar = False
        dt_odom_pose = time.time() - self.odom_pose_pub_time
        dt_lidar = time.time() - self.lidar_pub_time
        if (dt_odom_pose >= 1./self.odom_pose_freq):
            pub_odom_pose = True
        
        if (dt_lidar >= 1./self.lidar_freq):
            pub_lidar = True

        if (pub_odom_pose):
            self.odom_pose_pub_time = time.time()
            robot_data = self.env.unwrapped.scene["unitree_go2"].data
            for i in range(self.num_envs):
                self.publish_odom(robot_data.root_state_w[i, :3],
                                robot_data.root_state_w[i, 3:7],
                                robot_data.root_lin_vel_b[i],
                                robot_data.root_ang_vel_b[i],
                                i)
                self.publish_pose(robot_data.root_state_w[i, :3],
                                robot_data.root_state_w[i, 3:7], i)
        if (self.cfg.sensor.enable_lidar):
            if (pub_lidar):
                self.lidar_pub_time = time.time()
                for i in range(self.num_envs):
                    self.publish_lidar_data(self.lidar_annotators[i].get_data()["data"].reshape(-1, 3), i)

    def cmd_vel_callback(self, msg, env_idx):
        go2_ctrl.base_vel_cmd_input[env_idx][0] = msg.linear.x
        go2_ctrl.base_vel_cmd_input[env_idx][1] = msg.linear.y
        go2_ctrl.base_vel_cmd_input[env_idx][2] = msg.angular.z
    
    def semantic_segmentation_callback(self, img, env_idx):
        bridge = CvBridge()
        semantic_image = bridge.imgmsg_to_cv2(img, desired_encoding='passthrough')
        semantic_image_normalized = (semantic_image / semantic_image.max() * 255).astype(np.uint8)

        # Apply a predefined colormap
        color_mapped_image = cv2.applyColorMap(semantic_image_normalized, cv2.COLORMAP_JET)
        # Convert to ROS Image
        bridge = CvBridge()
        image_msg = bridge.cv2_to_imgmsg(color_mapped_image, encoding='rgb8')
        self.semantic_seg_img_vis_pub[env_idx].publish(image_msg)


    def pub_image_graph(self):
        for i in range(self.num_envs):
            if (self.num_envs == 1):
                color_topic_name = "unitree_go2/front_cam/color_image"
                depth_topic_name = "unitree_go2/front_cam/depth_image"
                # segmentation_topic_name = "unitree_go2/front_cam/segmentation_image"
                # depth_cloud_topic_name = "unitree_go2/front_cam/depth_cloud"
                frame_id = "unitree_go2/front_cam"                         
            else:
                color_topic_name = f"unitree_go2_{i}/front_cam/color_image"
                depth_topic_name = f"unitree_go2_{i}/front_cam/depth_image"
                # segmentation_topic_name = f"unitree_go2_{i}/front_cam/segmentation_image"
                # depth_cloud_topic_name = f"unitree_go2_{i}/front_cam/depth_cloud"
                frame_id = f"unitree_go2_{i}/front_cam"
            keys = og.Controller.Keys
            og.Controller.edit(
                {
                    "graph_path": "/CameraROS2Graph",
                    "evaluator_name": "execution",
                },
                {
                    keys.CREATE_NODES: [
                        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                        ("IsaacCreateRenderProduct", "omni.isaac.core_nodes.IsaacCreateRenderProduct"),
                        ("ROS2CameraHelperColor", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        ("ROS2CameraHelperDepth", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        # ("ROS2CameraHelperSegmentation", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                        # ("ROS2CameraHelperDepthCloud", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                    ],

                    keys.SET_VALUES: [
                        ("IsaacCreateRenderProduct.inputs:cameraPrim", f"/World/envs/env_{i}/Go2/base/front_cam"),
                        ("IsaacCreateRenderProduct.inputs:enabled", True),
                        ("IsaacCreateRenderProduct.inputs:height", 480),
                        ("IsaacCreateRenderProduct.inputs:width", 640),
                        
                        # color camera
                        ("ROS2CameraHelperColor.inputs:type", "rgb"),
                        ("ROS2CameraHelperColor.inputs:topicName", color_topic_name),
                        ("ROS2CameraHelperColor.inputs:frameId", frame_id),
                        ("ROS2CameraHelperColor.inputs:useSystemTime", False),

                        # depth camera
                        ("ROS2CameraHelperDepth.inputs:type", "depth"),
                        ("ROS2CameraHelperDepth.inputs:topicName", depth_topic_name),
                        ("ROS2CameraHelperDepth.inputs:frameId", frame_id),
                        ("ROS2CameraHelperDepth.inputs:useSystemTime", False),

                        # semantic camera
                        # ("ROS2CameraHelperSegmentation.inputs:type", "semantic_segmentation"),
                        # ("ROS2CameraHelperSegmentation.inputs:enableSemanticLabels", True),
                        # ("ROS2CameraHelperSegmentation.inputs:topicName", segmentation_topic_name),
                        # ("ROS2CameraHelperSegmentation.inputs:frameId", frame_id),
                        # ("ROS2CameraHelperSegmentation.inputs:useSystemTime", False),

                        # depth camera cloud
                        # ("ROS2CameraHelperDepthCloud.inputs:type", "depth_pcl"),
                        # ("ROS2CameraHelperDepthCloud.inputs:topicName", depth_cloud_topic_name),
                        # ("ROS2CameraHelperDepthCloud.inputs:frameId", frame_id),
                        # ("ROS2CameraHelperDepthCloud.inputs:useSystemTime", True),
                    ],

                    keys.CONNECT: [
                        ("OnPlaybackTick.outputs:tick", "IsaacCreateRenderProduct.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperColor.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperColor.inputs:renderProductPath"),
                        ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperDepth.inputs:execIn"),
                        ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperDepth.inputs:renderProductPath"),
                        # ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperSegmentation.inputs:execIn"),
                        # ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperSegmentation.inputs:renderProductPath"),
                        # ("IsaacCreateRenderProduct.outputs:execOut", "ROS2CameraHelperDepthCloud.inputs:execIn"),
                        # ("IsaacCreateRenderProduct.outputs:renderProductPath", "ROS2CameraHelperDepthCloud.inputs:renderProductPath"),
                    ],

                },
            )

    def pub_color_image(self):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/color_image"
                frame_id = "unitree_go2/front_cam"                         
            else:
                topic_name = f"unitree_go2_{i}/front_cam/color_image"
                frame_id = f"unitree_go2_{i}/front_cam"
            node_namespace = ""         
            queue_size = 1

            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
            
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name,
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def pub_depth_image(self):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/depth_image"                
                frame_id = "unitree_go2/front_cam"          
            else:
                topic_name = f"unitree_go2_{i}/front_cam/depth_image"
                frame_id = f"unitree_go2_{i}/front_cam"
            node_namespace = ""
            queue_size = 1

            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.DistanceToImagePlane.name
                                )
            writer = rep.writers.get(rv + "ROS2PublishImage")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def pub_semantic_image(self):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/semantic_segmentation_image"
                label_topic_name = "unitree_go2/front_cam/semantic_segmentation_label"                       
                frame_id = "unitree_go2/front_cam"          
            else:
                topic_name = f"unitree_go2_{i}/front_cam/semantic_segmentation_image"
                label_topic_name = f"unitree_go2_{i}/front_cam/semantic_segmentation_label"                       
                frame_id = f"unitree_go2_{i}/front_cam"
            node_namespace = ""
            queue_size = 1

            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                                    sd.SensorType.SemanticSegmentation.name
                                )
            writer = rep.writers.get("ROS2PublishSemanticSegmentation")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name
            )
            writer.attach([render_product])

            semantic_writer = rep.writers.get(
               "SemanticSegmentationSD" + f"ROS2PublishSemanticLabels"
            )
            semantic_writer.initialize(
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=label_topic_name,
            )
            semantic_writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    def pub_cam_depth_cloud(self):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/depth_cloud"    
                frame_id = "unitree_go2/front_cam"          
            else:
                topic_name = f"unitree_go2_{i}/front_cam/depth_cloud"
                frame_id = f"unitree_go2_{i}/front_cam"
            node_namespace = ""         
            queue_size = 1

            # Note, this pointcloud publisher will simply convert the Depth image to a pointcloud using the Camera intrinsics.
            # This pointcloud generation method does not support semantic labelled objects.
            rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
                sd.SensorType.DistanceToImagePlane.name
            )

            writer = rep.writers.get(rv + "ROS2PublishPointCloud")
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name,
            )
            writer.attach([render_product])

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                rv + "IsaacSimulationGate", render_product
            )

            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)   

    def publish_camera_info(self):
        for i in range(self.num_envs):
            # The following code will link the camera's render product and publish the data to the specified topic name.
            render_product = self.cameras[i]._render_product_path
            step_size = 1
            if (self.num_envs == 1):
                topic_name = "unitree_go2/front_cam/info"
            else:
                topic_name = f"unitree_go2_{i}/front_cam/info"
            queue_size = 1
            node_namespace = ""
            frame_id = self.cameras[i].prim_path.split("/")[-1] # This matches what the TF tree is publishing.

            writer = rep.writers.get("ROS2PublishCameraInfo")
            camera_info = read_camera_info(render_product_path=render_product)
            writer.initialize(
                frameId=frame_id,
                nodeNamespace=node_namespace,
                queueSize=queue_size,
                topicName=topic_name,
                width=camera_info["width"],
                height=camera_info["height"],
                projectionType=camera_info["projectionType"],
                k=camera_info["k"].reshape([1, 9]),
                r=camera_info["r"].reshape([1, 9]),
                p=camera_info["p"].reshape([1, 12]),
                physicalDistortionModel=camera_info["physicalDistortionModel"],
                physicalDistortionCoefficients=camera_info["physicalDistortionCoefficients"],
            )
            writer.attach([render_product])

            gate_path = omni.syntheticdata.SyntheticData._get_node_path(
                "PostProcessDispatch" + "IsaacSimulationGate", render_product
            )

            # Set step input of the Isaac Simulation Gate nodes upstream of ROS publishers to control their execution rate
            og.Controller.attribute(gate_path + ".inputs:step").set(step_size)            