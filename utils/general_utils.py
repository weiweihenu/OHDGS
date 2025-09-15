#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
import sys
import os
from datetime import datetime
import numpy as np
import random
import cv2
import math
import matplotlib.cm as cm
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter
from numba import jit, cuda, float32


def calculate_normal_from_covariance(covariance):
    """
    根据协方差矩阵计算法向量。
    输入：
        covariance - 协方差矩阵的tensor，格式为 [cxx, cxy, cxz, cyy, cyz, czz]
    输出：
        法向量 (3D tensor)
    """
    cov_matrix = torch.tensor([[covariance[0], covariance[1], covariance[2]],
                                [covariance[1], covariance[3], covariance[4]],
                                [covariance[2], covariance[4], covariance[5]]])
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    normal_vector = eigvecs[:, 0]  # 最小特征值对应的特征向量为法向量
    return normal_vector

def calculate_normal_from_neighbors(points):
    """
    根据邻域点计算法向量。
    输入：
        points - 邻域点的tensor数组，形状为 (k, 3)
    输出：
        法向量 (3D tensor)
    """
    mean_point = points.mean(dim=0)  # 计算质心
    centered_points = points - mean_point
    cov_matrix = centered_points.T @ centered_points / points.size(0)
    eigvals, eigvecs = torch.linalg.eigh(cov_matrix)
    normal_vector = eigvecs[:, 0]  # 最小特征值对应的特征向量为法向量
    return normal_vector

def normal_densify(input_xyz, covariances=None, radius=0.1, num_points=1):
    """
    封装函数：对输入点云进行法向量计算并执行致密化。
    输入：
        input_xyz - 输入点云的tensor (n, 3)
        covariances - 每个点的协方差矩阵tensor (n, 6)，如果为None则使用邻域点计算法向量
        radius - 正交平面上的采样半径
        num_points - 每个中心点生成的新点数量
    输出：
        output_xyz - 致密化后生成的新的xyz坐标tensor
    """
    def generate_local_coordinate_system(normal_vector):
        """
        根据法向量生成局部正交坐标系。
        输入：normal_vector - 法向量（3D向量）
        输出：局部坐标系的基向量 (u, v, n)
        """
        normal_vector = normal_vector / torch.norm(normal_vector)  # 单位化法向量
        if torch.abs(normal_vector[0]) < 0.99:
            tangent = torch.tensor([1.0, 0.0, 0.0], device=normal_vector.device)
        else:
            tangent = torch.tensor([0.0, 1.0, 0.0], device=normal_vector.device)
        u = torch.cross(normal_vector, tangent)
        u = u / torch.norm(u)
        v = torch.cross(normal_vector, u)
        return u, v, normal_vector

    input_xyz = input_xyz.detach().cpu()
    normals = []
    for i in range(input_xyz.size(0)):
        if covariances is not None:
            normal = calculate_normal_from_covariance(covariances[i])
        else:
            neighbors = input_xyz[torch.randperm(input_xyz.size(0))[:10]]  # 随机选取10个邻域点
            normal = calculate_normal_from_neighbors(neighbors)
        normals.append(normal)

    normals = torch.stack(normals)

    # 开始致密化
    new_xyz = []
    for center, normal_vector in zip(input_xyz, normals):
        u, v, _ = generate_local_coordinate_system(normal_vector)  # 生成局部坐标系
        for _ in range(num_points):
            theta = torch.rand(1) * 2 * torch.pi  # 随机角度
            r = torch.rand(1) * radius            # 随机半径
            offset = r * (torch.cos(theta) * u + torch.sin(theta) * v)  # 偏移向量
            new_center = center + offset  # 计算新的高斯中心点
            new_xyz.append(new_center)

    return torch.stack(new_xyz), normals  # 返回新的xyz坐标和法向量


def interpolate_properties(new_xyz, original_xyz, scaling, rotation, opacity, features_dc, features_rest):
    """
    Interpolates the attributes for new positions based on nearest neighbors in the original dataset.

    Parameters:
        new_xyz (torch.Tensor): New positions (N, 3)
        original_xyz (torch.Tensor): Original positions (M, 3)
        scaling (torch.Tensor): Original scaling (M, 3)
        rotation (torch.Tensor): Original rotation (M, 4)
        opacity (torch.Tensor): Original opacity (M, 1)
        features_dc (torch.Tensor): Original features_dc (M, 1, 3)
        features_rest (torch.Tensor): Original features_rest (M, 15, 3)

    Returns:
        dict: Interpolated attributes for new_xyz
    """
    tree = cKDTree(original_xyz.cpu().detach().numpy())
    distances, indices = tree.query(new_xyz, k=3)  # Use 3 nearest neighbors for interpolation

    # Convert distances and indices back to torch.Tensor
    distances = torch.from_numpy(distances)
    indices = torch.from_numpy(indices)

    # Weights based on distances (inverse distance weighting)
    weights = 1 / (distances + 1e-8)
    weights /= torch.sum(weights, dim=1, keepdim=True)
    weights = weights.to(scaling.device)

    # Interpolate each attribute
    interp_scaling = torch.sum(weights[:, :, None] * scaling[indices], dim=1)
    interp_rotation = torch.sum(weights[:, :, None] * rotation[indices], dim=1)
    interp_opacity = torch.sum(weights[:, :, None] * opacity[indices], dim=1)
    interp_features_dc = torch.sum(weights[:, :, None, None] * features_dc[indices], dim=1)
    interp_features_rest = torch.sum(weights[:, :, None, None] * features_rest[indices], dim=1)

    return {
        "scaling": interp_scaling,
        "rotation": interp_rotation,
        "opacity": interp_opacity,
        "features_dc": interp_features_dc,
        "features_rest": interp_features_rest,
    }

def z_score_and_min_max_normalize(depth_map, new_min=0, new_max=1):
    """
    对深度图进行 Z-Score 标准化和 Min-Max 归一化的组合。

    参数：
        depth_map (np.ndarray): 输入的深度图数组
        new_min (float): 目标范围的最小值
        new_max (float): 目标范围的最大值

    返回：
        np.ndarray: 归一化后的深度图
    """
    if not isinstance(depth_map, np.ndarray):
        raise ValueError("输入必须是 np.ndarray 类型")

    # 计算 Z-Score 标准化
    mean = np.mean(depth_map)
    std = np.std(depth_map)
    if std == 0:
        raise ValueError("深度图的标准差为 0，无法进行标准化")
    z_score_tensor = (depth_map - mean) / std

    # 计算 Min-Max 归一化
    min_val = np.min(z_score_tensor)
    max_val = np.max(z_score_tensor)
    if max_val == min_val:
        raise ValueError("标准化后的数组最小值等于最大值，无法进行 Min-Max 归一化")
    normalized_tensor = (z_score_tensor - min_val) / (max_val - min_val) * (new_max - new_min) + new_min

    return normalized_tensor

def fov_to_intrinsics(fov_x, fov_y, width, height, radians=True):
    if radians:
        fov_x_rad = fov_x
        fov_y_rad = fov_y
    else:
        # Convert degrees to radians
        fov_x_rad = np.radians(fov_x)
        fov_y_rad = np.radians(fov_y)

    # Compute focal lengths
    fx = width / (2 * np.tan(fov_x_rad / 2))
    fy = height / (2 * np.tan(fov_y_rad / 2))

    # Principal point (assume center of the image)
    cx = width / 2
    cy = height / 2

    # Construct intrinsic matrix
    K = np.array([
        [fx,  0,  cx],
        [ 0, fy,  cy],
        [ 0,  0,   1]
    ])

    return K

def pixel_to_3d_point(u, v, depth, K, T_world_to_camera):
    # Step 1: Compute normalized camera coordinates
    K_inv = np.linalg.inv(K)
    pixel_homogeneous = np.array([u, v, 1])
    p_camera = np.dot(K_inv, pixel_homogeneous)  # Normalized ray direction in camera space

    # Step 2: Scale by depth to get 3D point in camera coordinates
    P_camera = p_camera * depth

    # Step 3: Transform to world coordinates
    T_camera_to_world = np.linalg.inv(T_world_to_camera)
    P_camera_homogeneous = np.append(P_camera, 1)  # Make it homogeneous
    P_world_homogeneous = np.dot(T_camera_to_world, P_camera_homogeneous)

    # Return 3D point in world coordinates
    return P_world_homogeneous[:3]

def generate_3d_points(original_image, rendered_image, original_depth, rendered_depth,
                        K, T_world_to_camera, color_threshold=0.5, depth_threshold=0.5):
    _, height, width = original_image.shape
    generated_points = []
    local_color_differences = []
    local_depth_differences = []

    original_image = original_image.cpu().numpy()
    rendered_image = rendered_image.cpu().detach().numpy()
    rendered_depth = rendered_depth.cpu().detach().numpy()
    original_depth = original_depth.cpu().detach().numpy()
    T_world_to_camera = T_world_to_camera.cpu().numpy()

    rendered_depth_norm = z_score_and_min_max_normalize(rendered_depth)
    original_depth_norm = z_score_and_min_max_normalize(original_depth)

    # Compute local color difference
    color_diff_matrix = np.linalg.norm(original_image - rendered_image, axis=0)
    local_color_diff = uniform_filter(color_diff_matrix, size=5)

    # Compute local depth difference
    depth_diff_matrix = np.abs(rendered_depth_norm - original_depth_norm)
    local_depth_diff = uniform_filter(depth_diff_matrix, size=5)

    for v in range(height):
        for u in range(width):
            # Calculate color and depth differences
            color_diff = np.linalg.norm(original_image[:, v, u] - rendered_image[:, v, u])
            depth_diff = abs(rendered_depth_norm[v, u] - original_depth_norm[v, u])

            # Check if the differences exceed thresholds
            if color_diff > color_threshold and depth_diff > depth_threshold:
                # Generate 3D point at the original depth value
                depth = rendered_depth[v, u]
                point_3d = pixel_to_3d_point(u, v, depth, K, T_world_to_camera)
                generated_points.append(point_3d)
                local_color_differences.append(local_color_diff[v, u])
                local_depth_differences.append(local_depth_diff[v, u])

    return np.array(generated_points), np.array(local_color_differences), np.array(local_depth_differences)

def save_metrics_to_txt(metrics, file_path):
    with open(file_path, 'w') as f:
        # 写入标题行
        f.write("NAME\tITER\tL1\tPSNR\tSSIM\tLPIPS\tNUMBER\n")
        
        # 写入每个epoch的指标
        for name, epoch, loss, psnr, ssim, lpips, number in zip(metrics['NAME'], metrics['ITER'], metrics['L1'], metrics['PSNR'], metrics['SSIM'], metrics['LPIPS'], metrics['NUMBER']):
            f.write(f"{name}\t{epoch}\t{loss}\t{psnr}\t{ssim}\t{lpips}\t{number}\n")

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    seed = 1
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(torch.device("cuda:0"))
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weighted_percentile(x, w, ps, assume_sorted=False):
    """Compute the weighted percentile(s) of a single vector."""
    x = x.reshape([-1])
    w = w.reshape([-1])
    if not assume_sorted:
        sortidx = np.argsort(x)
        x, w = x[sortidx], w[sortidx]
    acc_w = np.cumsum(w)
    return np.interp(np.array(ps) * (acc_w[-1] / 100), acc_w, x)


def vis_depth(depth):
    """Visualize the depth map with colormap.
       Rescales the values so that depth_min and depth_max map to 0 and 1,
       respectively.
    """
    percentile = 99
    eps = 1e-10

    lo_auto, hi_auto = weighted_percentile(
        depth, np.ones_like(depth), [50 - percentile / 2, 50 + percentile / 2])
    lo = None or (lo_auto - eps)
    hi = None or (hi_auto + eps)
    curve_fn = lambda x: 1/x + eps

    depth, lo, hi = [curve_fn(x) for x in [depth, lo, hi]]
    depth = np.nan_to_num(
            np.clip((depth - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1))
    colorized = cm.get_cmap('turbo')(depth)[:, :, :3]

    return np.uint8(colorized[..., ::-1] * 255)


def chamfer_dist(array1, array2):
    dist = torch.norm(array1[None] - array2[:, None], 2, dim=-1)
    return dist.min(1)[0]