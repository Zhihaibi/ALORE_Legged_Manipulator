#!/usr/bin/env python3
import os
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32, Float32MultiArray
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import message_filters
import torch  
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PILImage
import math

class_names = [
    "chair_0",
    "chair_135",
    "chair_180",
    "chair_225",
    "chair_270",
    "chair_315",
    "chair_45",
    "chair_90",
]

# --------------------------
# Transform
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

class RealSenseYoloNode:
    def __init__(self):
        rospy.init_node('realsense_yolo_node', anonymous=True)

        self.device = '0' if torch.cuda.is_available() else 'cpu'
        rospy.loginfo(f"🚀 YOLO Device: {self.device}")

        # YOLO Model
        self.yolo_model = YOLO('/home/unitree/work2/Whole_Body_Object_Rearrangement/real_experiment/perception/best.pt')

        # ResNet Model
        self.res_model = models.resnet18()
        self.res_model.fc = nn.Linear(self.res_model.fc.in_features, 8)
        self.res_model.load_state_dict(torch.load("/real_experiment/perception/angle_model.pth", map_location="cpu"))
        self.res_model.eval()
        rospy.loginfo("✅ Estimation models loaded.")

        color_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        
        ats = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], 
            queue_size=10, 
            slop=1.0)
        ats.registerCallback(self.synced_callback)

        self.state_sub = rospy.Subscriber(
            '/start_detect_obj',
            Bool,
            self.state_command_callback,
            queue_size=10)

        self.detected_pub = rospy.Publisher(
            '/object_detection', 
            Bool, 
            queue_size=10)

        self.pose_pub = rospy.Publisher(
            '/object_6d_pose', 
            Float32MultiArray, 
            queue_size=10)

        self.bridge = CvBridge()
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=1, text_thickness=1)
        
        self.state_finding = False
        
        self.pose_buffer = []
        self.target_sample_count = 10
        
        self.fx = 607.0
        self.cx = 320.0
        self.intrinsics_received = False
        
        self.camera_info_sub = rospy.Subscriber('/camera/color/camera_info', CameraInfo, self.camera_info_callback)

        self.frame_count = 0
        self.process_interval = 3  # 

        rospy.loginfo("✅ Pose Estimation node started")

    def predict_angle(self, img_input):
        """Predict angle (int) for a single image (path or numpy array)."""
        if isinstance(img_input, str):
            img = PILImage.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            # OpenCV (BGR) -> PIL (RGB)
            img = PILImage.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
        else:
            return 0

        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            outputs = self.res_model(x)
            _, pred = outputs.max(1)

        class_name = class_names[pred.item()]
        angle = int(class_name.split("_")[1])
        return angle

    def camera_info_callback(self, msg):

        if not self.intrinsics_received:
            K = np.array(msg.K).reshape(3, 3)
            self.fx = K[0, 0]
            self.cx = K[0, 2]
            self.intrinsics_received = True
            rospy.loginfo(f"📸 Camera intrinsics received: fx={self.fx:.2f}, cx={self.cx:.2f}")
            self.camera_info_sub.unregister()

    def state_command_callback(self, msg: Bool):
        if msg.data:
            if not self.state_finding:
                self.state_finding = True
                self.pose_buffer = [] 
                rospy.loginfo(f"🔄 State: finding, need {self.target_sample_count} frames")

    def synced_callback(self, color_msg, depth_msg):
        if not self.state_finding:
            return

        self.frame_count += 1
        if self.frame_count % self.process_interval != 0:
            return

        frame = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        
        results = self.yolo_model.predict(frame, imgsz=640, conf=0.5, verbose=False, device=self.device)[0]
        detections = sv.Detections.from_ultralytics(results)

        if len(detections.xyxy) > 0:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

            self.detected_pub.publish(Bool(data=True))

            i = 0
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            
            h, w = depth.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            depth_crop = depth[y1:y2, x1:x2]
            if depth.dtype == np.uint16:
                depth_m = depth_crop.astype(np.float32) / 1000.0
            else:
                depth_m = depth_crop
            
            valid_mask = (depth_m >= 2.0) & (depth_m <= 4.0)
            valid_pixels = depth_m[valid_mask]
            
            avg_dist = 0.0
            if valid_pixels.size > 0:
                avg_dist = np.mean(valid_pixels)
            
            real_offset_x = 0.0
            if avg_dist > 0:
                obj_center_x = (x1 + x2) / 2.0
                img_center_x = self.cx
                pixel_offset = obj_center_x - img_center_x
                real_offset_x = avg_dist * pixel_offset / self.fx
            

            rgb_crop = frame[y1:y2, x1:x2]
            yaw_deg = 0
            if rgb_crop.size > 0:
                try:
                    yaw_deg = self.predict_angle(rgb_crop)
                except Exception as e:
                    rospy.logwarn(f"Angle prediction failed: {e}")
            
            yaw_rad = math.radians(yaw_deg)
            
            current_data = [
                float(avg_dist),          # depth (Z)
                float(real_offset_x),     # x
                float(0.0),               # y
                float(yaw_rad),           # yaw (radians)
                float(0.0),               # qx
                float(0.0),               # qy
                float(0.0),               # qz
                float(1.0)                # qw
            ]
            
            self.pose_buffer.append(current_data)
            rospy.loginfo(f"📥 Collecting sample {len(self.pose_buffer)}/{self.target_sample_count}: Dist={avg_dist:.2f}m, Angle={yaw_deg}°")

            if len(self.pose_buffer) >= self.target_sample_count:
                final_pose = self.pose_buffer[-1]
                
                pose_array = Float32MultiArray()
                pose_array.data = final_pose
                
                self.pose_pub.publish(pose_array)

                self.state_finding = False
                self.pose_buffer = []
                
                rospy.loginfo(f"✅ Final Pose Published: X={final_pose[0]:.2f}m, Y={final_pose[1]:.2f}m, Yaw={math.degrees(final_pose[3]):.2f}°")

        elif len(detections.xyxy) == 0:
            rospy.logwarn("⚠️ No object detected")

def main():
    node = RealSenseYoloNode()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()