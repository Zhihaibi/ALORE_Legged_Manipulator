#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool, Float32MultiArray
import numpy as np
from pupil_apriltags import Detector

bridge = CvBridge()
camera_matrix = None
dist_coeffs = None
start_detect = False   

detector = Detector(families='tagStandard41h12')
TAG_SIZE = 0.057  # m

result_pub = None

def cam_info_cb(msg):
    global camera_matrix, dist_coeffs
    camera_matrix = np.array(msg.K).reshape(3, 3)
    dist_coeffs = np.array(msg.D)

def trigger_cb(msg: Bool):
    global start_detect
    if msg.data:
        rospy.loginfo("Received detect trigger!")
        start_detect = True

def image_cb(img_msg):
    global start_detect, camera_matrix

    if not start_detect:
        return  

    if camera_matrix is None:
        return

    img = bridge.imgmsg_to_cv2(img_msg, "bgr8")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(camera_matrix[0,0], camera_matrix[1,1],
                       camera_matrix[0,2], camera_matrix[1,2]),
        tag_size=TAG_SIZE
    )

    if len(detections) == 0:
        rospy.logwarn("Tag NOT detected! No result published.")
        start_detect = False
        return

    det = detections[0]
    R = det.pose_R
    t = det.pose_t.flatten()


    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2,1], R[2,2])
    else:
        roll = np.arctan2(-R[1,2], R[1,1])

    roll_deg = np.degrees(roll)


    msg = Float32MultiArray()
    msg.data = [float(t[0]), float(t[1]), float(t[2]), float(roll)]
    result_pub.publish(msg)

    rospy.loginfo("Result Published -> x=%.3f y=%.3f z=%.3f roll=%.2f°"
                  % (t[0], t[1], t[2], roll))

    start_detect = False 

if __name__ == '__main__':
    rospy.init_node("apriltag_trigger_detect")

    result_pub = rospy.Publisher("/apriltag_pose_result", Float32MultiArray, queue_size=1)
    rospy.Subscriber("/apriltag_start_detect", Bool, trigger_cb)
    rospy.Subscriber("/camera/color/image_raw", Image, image_cb)
    rospy.Subscriber("/camera/color/camera_info", CameraInfo, cam_info_cb)

    rospy.loginfo("Waiting for /apriltag_start_detect signal...")
    rospy.spin()
