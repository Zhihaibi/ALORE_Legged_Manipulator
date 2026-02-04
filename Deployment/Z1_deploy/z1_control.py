import sys
import rospy
from std_msgs.msg import Float32MultiArray
import time
import os
import json
import numpy as np
import copy
from collections import deque
# z1 manipulator head files
sys.path.append(os.path.abspath("/z1_sdk/lib"))
import unitree_arm_interface
from functools import partial
from common.rotation_helper import transform_to_quat

arm_target_pos =  [0.0,  0.0,  0.0,  0.0, 0.0, 0.0, 0.0]
# arm_target_pos = np.array( [0.0,  1.9,  -0.6,  -1.3, 0.0, 1.57])

position_window = deque(maxlen=3)  
velocity_window = deque(maxlen=3)

hand_pos_window = deque(maxlen=3)  
hand_quat_window = deque(maxlen=3) 

def sub_arm_target_pos(data):
    global arm_target_pos
    arm_target_pos = np.array(data.data)
    print("receive arm_target_pos", arm_target_pos)

def pub_arm_current_state(event, arm, arm_state_pub):
    global position_window, velocity_window

    cur_Pos = arm.lowstate.getQ()  # get current position
    cur_Gripper = arm.lowstate.getGripperQ()  # get gripper angle
    cur_Pos = np.append(cur_Pos, cur_Gripper)  # append gripper angle to position

    cur_Vel = arm.lowstate.getQd()  # get current velocity

    position_window.append(cur_Pos)
    velocity_window.append(cur_Vel)
    avg_Pos = np.mean(position_window, axis=0) 
    avg_Vel = np.mean(velocity_window, axis=0)

    arm_state_msg = Float32MultiArray(data=np.concatenate((avg_Pos, avg_Vel)).tolist())

    arm_state_pub.publish(arm_state_msg)
    # rospy.loginfo(f"Published current arm_state_msg: {arm_state_msg.data}")

def pub_hand_current_state(event, arm, hand_state_pub):
    global hand_pos_window, hand_quat_window
    arm_theta = arm.lowstate.getQ()
    T_forward = armModel.forwardKinematics(arm_theta, 6)
    hand_pos_arm_frame = T_forward[0:3, 3]
    hand_quat_arm_frame = transform_to_quat(T_forward[0:3, 0:3])  #

    # print("hand_pos_arm_frame", hand_pos_arm_frame)

    hand_pos_window.append(hand_pos_arm_frame)
    hand_quat_window.append(hand_quat_arm_frame)

    avg_hand_pos = np.mean(hand_pos_window, axis=0)
    avg_hand_quat = np.mean(hand_quat_window, axis=0)
    avg_hand_quat = avg_hand_quat / np.linalg.norm(avg_hand_quat)

    hand_state_msg = Float32MultiArray(data=np.concatenate((avg_hand_pos, avg_hand_quat)).tolist())
    hand_state_pub.publish(hand_state_msg)
    # rospy.loginfo(f"Published current hand_state_msg: {hand_state_msg.data}")


if __name__ == "__main__":
    # arm setting

    np.set_printoptions(precision=3, suppress=True)

    arm = unitree_arm_interface.ArmInterface(hasGripper=True)
    armModel = arm._ctrlComp.armModel
    arm.setFsmLowcmd()

    lower_limits_angles = [-2.6180,  0.0000, -2.8798, -1.5184, -1.3439, -2.7925, -1.5]
    upper_limits_angles = [2.6180, 2.9671, 0.0000, 1.5184, 1.3439, 2.7925, 0]

    kp = [264., 328., 264., 264., 264., 264., 100.]
    kp = np.array(kp) / 25.6
    kd = [1.5, 3.0, 1.5, 1.5, 1.5, 1.5, 1.0]
    kd = np.array(kd) / 0.0128
    
    # kp = [20.0, 30.0, 30.0, 20.0, 15.0, 10.0, 20.0]
    # kd = [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
    arm.setArmKD(kp, kd)

    duration = 1000
    count = 0

    lastPos = arm.lowstate.getQ() # get current position
    lastVel = arm.lowstate.getQd() # get current velocity

    rospy.init_node('z1_controller', anonymous=True)
    rospy.Subscriber('/arm_target_pos', Float32MultiArray, sub_arm_target_pos, queue_size=10)

    hand_state_pub = rospy.Publisher('hand_current_state', Float32MultiArray, queue_size=10)
    rospy.Timer(rospy.Duration(0.02), partial(pub_hand_current_state, arm=arm, hand_state_pub=hand_state_pub)) # 50Hz

    arm_state_pub = rospy.Publisher('arm_current_state', Float32MultiArray, queue_size=10)
    rospy.Timer(rospy.Duration(0.02), partial(pub_arm_current_state, arm=arm, arm_state_pub=arm_state_pub)) # 50Hz

    targetPos = np.zeros(7)
    arm_target_pos = np.zeros(7)
    # arm_target_pos = np.array([0.0000,  1.4800, -0.6300, -0.8400,  0.0000,  1.5700, 0.0])
    arm_target_pos = np.array([0,  0.60, -0.60, 0.1, 0.0, 0.0, 0])
    # arm_target_pos = np.array([0,  0.70, -0.70, 0.1, 0.0, 0.0, 0])

    
    while not rospy.is_shutdown():
        start_time = rospy.get_time()

        lastPos = arm.lowstate.getQ()
        lastVel = arm.lowstate.getQd()  # get current velocity

        targetPos = copy.deepcopy(arm_target_pos)  

        delta_pos = np.clip(targetPos[0:6] - lastPos, -0.05, 0.05)
        targetPos[0:6] = lastPos + delta_pos

        targetPos = np.clip(targetPos, lower_limits_angles, upper_limits_angles)
        
        duration = 20

        for i in range(0, duration):
            arm.q = lastPos * (1 - i / duration) + targetPos[0:6] * (i / duration)  # set position
            arm.qd = (targetPos[0:6] - lastPos) / (duration * 0.002)  # set velocity
            arm.tau = armModel.inverseDynamics(arm.q, arm.qd, np.zeros(6), np.zeros(6))  # set torque
            # arm.qd = np.zeros(6)
            # arm.tau = np.zeros(6)
            # arm.gripperQ = -1*(i/duration)
            
            arm.setArmCmd(arm.q, arm.qd, arm.tau)
            arm.gripperQ = targetPos[6]
            arm.setGripperCmd(arm.gripperQ, arm.gripperQd, arm.gripperTau)

            arm.sendRecv()  # udp connection
            time.sleep(arm._ctrlComp.dt) # 20 * 0.002s = 0.04s = 25Hz

        control_duration = rospy.get_time() - start_time
        # print("control_duration(HZ):", 1.0/control_duration)
        time.sleep(max(0.04 - control_duration, 0)) # 25Hz 1/ 0.05 = 20Hz
    
    arm.loopOn()
    arm.backToStart()
    arm.loopOff() 

        

        


       
