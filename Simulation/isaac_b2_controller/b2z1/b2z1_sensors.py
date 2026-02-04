import omni
import numpy as np
from pxr import Gf
# import omni.replicator.core as rep
# from omni.isaac.sensor import Camera
from isaaclab.sensors.camera import Camera, CameraCfg
# import isaaclab.core.utils.numpy.rotations as rot_utils
# import isaaclab.utils.math as math_utils
from scipy.spatial.transform import Rotation as R
import isaaclab.sim as sim_utils


class SensorManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs

    # def add_rtx_lidar(self):
    #     lidar_annotators = []
    #     for env_idx in range(self.num_envs):
    #         _, sensor = omni.kit.commands.execute(
    #             "IsaacSensorCreateRtxLidar",
    #             path="/lidar",
    #             parent=f"/World/envs/env_{env_idx}/Robot/base",
    #             config="Hesai_XT32_SD10",
    #             # config="Velodyne_VLS128",
    #             translation=(0.2, 0, 0.2),
    #             orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # Gf.Quatd is w,i,j,k
    #         )

    #         annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
    #         hydra_texture = rep.create.render_product(sensor.GetPath(), [1, 1], name="Isaac")
    #         annotator.attach(hydra_texture.path)
    #         lidar_annotators.append(annotator)
    #     return lidar_annotators


    def add_camera(self):
        cameras = []

        for env_idx in range(self.num_envs):
            from_camera_cfg = CameraCfg(
                prim_path="/World/envs/env_0/Robot/base/front_cam",
                offset=CameraCfg.OffsetCfg(pos=(-0.45, 0, 0.5), rot=(0.7071, 0, 0.7071, 0), convention="ros"),
                height=480,
                width=640,
                data_types=["depth", "rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),

            )
            camera = Camera(from_camera_cfg)
            cameras.append(camera)
        
        # for env_idx in range(self.num_envs):
        #     camera = Camera(
        #         prim_path="/World/envs/env_{env_idx}/Robot/base/front_cam",
        #         translation=np.array([0.45, 0.0, 0.0]),
        #         frequency=freq,
        #         resolution=(640, 480),
        #         # orientation=rot_utils.euler_angles_to_quats(np.array([0, 0, 0]), degrees=True),
        #         orientation = R.from_euler('xyz', [0, 0, 0], degrees=True).as_quat()
        #     )
        #     camera.initialize()
        #     camera.set_focal_length(1.5)
        #     cameras.append(camera)
        
        return cameras