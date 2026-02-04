# -*- coding: utf-8 -*-
import open3d as o3d

obj_file = "office14.obj"
resolution = 0.1  # 目标分辨率：0.01m
occupancy_rate = 0.3

# 读取 .obj 文件（包含网格）
mesh = o3d.io.read_triangle_mesh(obj_file)

# 获取网格的边界框以估算合适的采样点数
bbox = mesh.get_axis_aligned_bounding_box()
bbox_size = bbox.get_extent()
volume = bbox_size[0] * bbox_size[1] * bbox_size[2] * occupancy_rate

# 根据0.01m分辨率估算需要的点数

# 估算点密度：每个0.01m³体积内大约需要1个点
estimated_points = int(volume / (resolution ** 3))
print(f"网格尺寸: {bbox_size}")
print(f"估算采样点数: {estimated_points}")

# 采样点云（先采样较多的点）
pcd = mesh.sample_points_uniformly(number_of_points=max(estimated_points, 100000))

# 使用体素下采样确保0.01m分辨率
print(f"正在进行体素下采样以确保{resolution}m分辨率...")
pcd_downsampled = pcd.voxel_down_sample(voxel_size=resolution)

print(f"原始采样点数: {len(pcd.points)}")
print(f"下采样后点数: {len(pcd_downsampled.points)}")

# 保存为 .pcd 文件
pcd_file = obj_file.replace(".obj", ".pcd")
o3d.io.write_point_cloud(pcd_file, pcd_downsampled)
print(f"点云已保存到 {pcd_file}，分辨率: {resolution}m")
