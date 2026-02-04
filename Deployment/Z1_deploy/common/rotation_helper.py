import numpy as np
from scipy.spatial.transform import Rotation as R


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w


def transform_to_quat(transform_matrix):
    return R.from_matrix(transform_matrix).as_quat()


def euler_from_transform(transform_matrix):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    # print("transform_matrix", transform_matrix)
    quat_angle = R.from_matrix(transform_matrix).as_quat()
    # print("quat_angle", quat_angle)

    x = quat_angle[0]
    y = quat_angle[1]
    z = quat_angle[2]
    w = quat_angle[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.stack([roll_x, pitch_y, yaw_z], axis=-1) # in radians


def euler_from_quat(quat_angle):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    x = quat_angle[0]
    y = quat_angle[1]
    z = quat_angle[2]
    w = quat_angle[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)
    
    return np.stack([roll_x, pitch_y, yaw_z], axis=-1) # in radians


def transform_pose(box_quat, box_trans, arm_quat, arm_trans):
    box_rot = R.from_quat(box_quat).as_matrix()
    arm_rot = R.from_quat(arm_quat).as_matrix()
    
    box_pose = np.eye(4)
    box_pose[:3, :3] = box_rot
    box_pose[:3, 3] = box_trans
    
    arm_pose = np.eye(4)
    arm_pose[:3, :3] = arm_rot
    arm_pose[:3, 3] = arm_trans
    
    box_pose_arm_frame_matrix = np.linalg.inv(arm_pose) @ box_pose
    return box_pose_arm_frame_matrix


def quat_from_euler_xyz(roll, pitch, yaw):
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qx, qy, qz, qw], axis=-1)


def quat_apply(a, b):
    # print(a.shape, b.shape)
    xyz = a[:3]
    t = np.cross(xyz, b) * 2
    result = b + a[3] * t + np.cross(xyz, t)
    return result


def cart2sphere(cart_coords):
    """ Convert cartesian coordinates to spherical coordinates
    Args:
        cart_coords: Cartesian coordinates (x, y, z)
    Returns:
        sphere_coords: Spherical coordinates (l, pitch, yaw)
    """
    sphere_coords = np.zeros_like(cart_coords)
    xy_len = np.linalg.norm(cart_coords[:2], axis=0)
    sphere_coords[0] = np.linalg.norm(cart_coords, axis=0)
    sphere_coords[1] = np.arctan2(cart_coords[2], xy_len)
    sphere_coords[2] = np.arctan2(cart_coords[1], cart_coords[0])
    return sphere_coords

def sphere2cart(sphere_coords):
    """ Convert spherical coordinates to cartesian coordinates
    Args:
        sphere_coords (np.ndarray): Spherical coordinates (l, pitch, yaw)
    Returns:
        cart_coords (np.ndarray): Cartesian coordinates (x, y, z)
    """
    l = sphere_coords[0]
    pitch = sphere_coords[1]
    yaw = sphere_coords[2]
    cart_coords = np.zeros_like(sphere_coords)
    cart_coords[0] = l * np.cos(pitch) * np.cos(yaw)
    cart_coords[1] = l * np.cos(pitch) * np.sin(yaw)
    cart_coords[2] = l * np.sin(pitch)
    return cart_coords

def quat_conjugate(a):
    result = np.concatenate((-a[:3], a[-1:]), axis=0)
    return result

def quat_mul(a, b):
    assert a.shape == b.shape
    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = np.stack([x, y, z, w], axis=0)
    return quat

def orientation_error(desired, current):
    # print("orientation_error", desired.shape, current.shape)
    cc = quat_conjugate(current)
    # print("cc", cc.shape)
    q_r = quat_mul(desired, cc)
    # print("q_r", q_r.shape)
    return q_r[0:3] * np.sign(q_r[3])

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * (np.dot(q_vec, v)) * 2.0
    return a - b + c

def get_body_orientation(quaternion):
    w = quaternion[0]
    x = quaternion[1]
    y = quaternion[2]
    z = quaternion[3]
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = np.arctan2(t0, t1)
    
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1, 1)
    pitch_y = np.arcsin(t2)
    
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = np.arctan2(t3, t4)

    body_orientation_rp = np.stack([roll_x, pitch_y], axis=-1)

    return body_orientation_rp