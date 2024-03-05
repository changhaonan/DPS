"""Miscellaneous utilities."""

from __future__ import annotations
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import copy
from scipy.spatial.transform import Rotation as R
import open3d as o3d


def position_encoding(x: torch.Tensor, min_timescale: float = 1.0, max_timescale: float = 1.0e2):
    """Sinusoidal Position Encoding."""
    length = x.shape[-1]
    channels = x.shape[-2]
    position = torch.arange(length, dtype=torch.float32, device=x.device)
    num_timescales = channels // 2
    log_timescale_increment = np.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1)
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32, device=x.device) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    signal = signal.unsqueeze(0).repeat(x.shape[:-1] + (1,))
    return signal


# -----------------------------------------------------------------------------
# RL Utils
# -----------------------------------------------------------------------------


def draw_V_map(observations: np.ndarray, V_f: nn.Module, sample_ratio: float = 0.01, output_path: str = None):
    obs = torch.from_numpy(observations).float().to("cuda")
    V = V_f(obs).detach().cpu().numpy()
    num_samples = int(obs.shape[0] * sample_ratio)
    obs_samples = obs[np.random.choice(obs.shape[0], num_samples, replace=False)]
    value = V_f(obs_samples).detach().cpu().numpy()
    obs_samples = obs_samples.detach().cpu().numpy()
    plt.clf()
    plt.scatter(obs_samples[:, 0], obs_samples[:, 1], s=0.01, c=value)
    plt.colorbar()
    plt.title("V")
    # set axis equal
    plt.gca().set_aspect("equal", adjustable="box")
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def draw_Q_map(
    observations: np.ndarray,
    Q_f: nn.Module,
    action: np.ndarray,
    sample_ratio: float = 0.01,
    output_path: str = None,
):
    obs = torch.from_numpy(observations).float().to("cuda")
    num_samples = int(obs.shape[0] * sample_ratio)
    obs_samples = obs[np.random.choice(obs.shape[0], num_samples, replace=False)]
    action_samples = torch.from_numpy(np.tile(action, (num_samples, 1))).float().to("cuda")
    Q = Q_f(torch.cat([obs_samples, action_samples], axis=1)).detach().cpu().numpy()
    obs_samples = obs_samples.detach().cpu().numpy()
    plt.clf()
    plt.scatter(obs_samples[:, 0], obs_samples[:, 1], s=0.01, c=Q)
    plt.colorbar()
    plt.title(f"Q under: {action}")
    # set axis equal
    plt.gca().set_aspect("equal", adjustable="box")
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def draw_A_map(
    observations: np.ndarray,
    Q_f1: nn.Module,
    Q_f2: nn.Module,
    V_f: nn.Module,
    action: np.ndarray,
    sample_ratio: float = 0.01,
    output_path: str = None,
):
    obs = torch.from_numpy(observations).float().to("cuda")
    num_samples = int(obs.shape[0] * sample_ratio)
    obs_samples = obs[np.random.choice(obs.shape[0], num_samples, replace=False)]
    action_samples = torch.from_numpy(np.tile(action, (num_samples, 1))).float().to("cuda")
    Q1 = Q_f1(torch.cat([obs_samples, action_samples], axis=1)).detach().cpu().numpy()
    Q2 = Q_f2(torch.cat([obs_samples, action_samples], axis=1)).detach().cpu().numpy()
    Q = np.minimum(Q1, Q2)
    V = V_f(obs_samples).detach().cpu().numpy()
    A = Q - V
    obs_samples = obs_samples.detach().cpu().numpy()
    plt.clf()
    plt.scatter(obs_samples[:, 0], obs_samples[:, 1], s=0.01, c=A)
    plt.colorbar()
    plt.title(f"A under: {action}")
    # set axis equal
    plt.gca().set_aspect("equal", adjustable="box")
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


# -----------------------------------------------------------------------------
# Visualization Utils
# -----------------------------------------------------------------------------


def plot_hist_scatter(
    data,
    title: str = "hist and scatter plot visualization of data",
    fig_name: str = "dataset.png",
    save_path: str = "output",
):
    """
    Args:
        data: (batch_size, state_dim)
    """
    plt.title(title)
    fig, axes = plt.subplots(nrows=data.shape[1], ncols=data.shape[1], figsize=(20, 20))  # Adjust the figsize as needed
    # We'll create an aggregated version of data to compute statistics across all data
    all_data = np.stack(data, axis=0)
    # Compute max and min for the data for consistent axis limits
    data_min = all_data.min(axis=0)
    data_max = all_data.max(axis=0)
    # Loop over all pairs of features (including pairs with themselves)
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            # If diagonal, plot a histogram
            if i == j:
                axes[i, j].hist(data[:, i], bins=30, color="gray", alpha=0.7)
                axes[i, j].set_xlim([-1.0, 1.0])  # Set x-axis range
                axes[i, j].set_ylim([-1.0, 1.0])  # Set y-axis range
                axes[i, j].set_yticklabels([])
                axes[i, j].set_xticklabels([])
            # If off-diagonal, plot a scatter plot
            else:
                axes[i, j].scatter(data[:, j], data[:, i], alpha=0.5, s=5, color="blue")
                axes[i, j].set_xlim([-1.0, 1.0])  # Set x-axis range
                axes[i, j].set_ylim([-1.0, 1.0])  # Set y-axis range
                axes[i, j].set_yticklabels([])
                axes[i, j].set_xticklabels([])

    # Set tighter layout
    plt.tight_layout()
    # Save the figure if desired
    plt.savefig(f"{save_path}/{fig_name}")


def resize_image(image, image_size):
    """Resize image to image_size."""
    image_resize = cv2.resize(image, (image_size, image_size))
    return image_resize


# -----------------------------------------------------------------------------
# RL Utils
# -----------------------------------------------------------------------------


def generate_video(frames, video_name="video.mp4"):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"mp4v"), 1, (width, height))

    # Loop to add images to video
    for frame in frames:
        video.write(frame)

    # Release the video writer
    video.release()
    cv2.destroyAllWindows()

    return video


####################### Transform #######################


def quat_to_mat(quat_pose):
    if len(quat_pose.shape) > 1:
        quat_pose = quat_pose.squeeze()
    if quat_pose.shape[0] == 7:
        quat = quat_pose[3:]
        if np.linalg.norm(quat) < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat = quat / (np.linalg.norm(quat) + 1e-8)
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(quat).as_matrix()
        mat[:3, 3] = quat_pose[:3]
        return mat
    elif quat_pose.shape[0] == 4:
        quat = quat_pose
        quat = quat / np.linalg.norm(quat)
        mat = np.eye(3)
        mat = R.from_quat(quat).as_matrix()
        return mat
    else:
        raise NotImplementedError


def mat_to_quat(mat_pose):
    if mat_pose.shape == (4, 4):
        quat = R.from_matrix(mat_pose[:3, :3]).as_quat()
        quat_pose = np.zeros(7)
        quat_pose[:3] = mat_pose[:3, 3]
        quat_pose[3:] = quat
        return quat_pose
    elif mat_pose.shape == (3, 3):
        quat = R.from_matrix(mat_pose).as_quat()
        return quat
    else:
        raise NotImplementedError


def mat_to_rotvec(mat_pose):
    if mat_pose.shape == (4, 4):
        rotvec = R.from_matrix(mat_pose[:3, :3]).as_rotvec()
        return rotvec
    elif mat_pose.shape == (3, 3):
        rotvec = R.from_matrix(mat_pose).as_rotvec()
        return rotvec
    else:
        raise NotImplementedError


def rotvec_to_mat(rotvec_pose):
    if rotvec_pose.shape == (3,):
        mat_pose = R.from_rotvec(rotvec_pose).as_matrix()
        return mat_pose
    else:
        raise NotImplementedError


def quat_to_rotvec(quat_pose):
    if quat_pose.shape == (7,):
        quat = quat_pose[3:]
        if np.linalg.norm(quat) < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat = quat / np.linalg.norm(quat)
        rotvec = R.from_quat(quat).as_rotvec()
        return rotvec
    elif quat_pose.shape == (4,):
        quat = quat_pose
        if np.linalg.norm(quat) < 1e-8:
            quat = np.array([1.0, 0.0, 0.0, 0.0])
        quat = quat / np.linalg.norm(quat)
        rotvec = R.from_quat(quat).as_rotvec()
        return rotvec
    else:
        raise NotImplementedError


def pose9d_to_mat(pose9d, rot_axis="xy"):
    """
    Args:
        pose9d: (7,)
        rot_axis: "xy", "yz", "zx"
    """
    # Normalize the pose
    pose9d = perform_gram_schmidt_transform(pose9d)
    # Convert to matrix
    trans = pose9d[:3]
    mat = np.eye(4, dtype=np.float32)
    if rot_axis == "xy":
        mat[:3, 0] = pose9d[3:6]
        mat[:3, 1] = pose9d[6:9]
        mat[:3, 2] = np.cross(mat[:3, 0], mat[:3, 1])
    elif rot_axis == "yz":
        mat[:3, 1] = pose9d[3:6]
        mat[:3, 2] = pose9d[6:9]
        mat[:3, 0] = np.cross(mat[:3, 1], mat[:3, 2])
    elif rot_axis == "zx":
        mat[:3, 2] = pose9d[3:6]
        mat[:3, 0] = pose9d[6:9]
        mat[:3, 1] = np.cross(mat[:3, 2], mat[:3, 0])
    else:
        raise NotImplementedError
    mat[:3, 3] = trans
    return mat


def mat_to_pose9d(mat, rot_axis="xy"):
    pose9d = np.zeros((9,), dtype=np.float32)
    pose9d[:3] = mat[:3, 3]
    if rot_axis == "xy":
        pose9d[3:6] = mat[:3, 0]
        pose9d[6:9] = mat[:3, 1]
    elif rot_axis == "yz":
        pose9d[3:6] = mat[:3, 1]
        pose9d[6:9] = mat[:3, 2]
    elif rot_axis == "zx":
        pose9d[3:6] = mat[:3, 2]
        pose9d[6:9] = mat[:3, 0]
    else:
        raise NotImplementedError
    return pose9d


# -----------------------------------------------------------------------------
# 3D Utils
# -----------------------------------------------------------------------------


def get_o3d_pointcloud(color, depth, intrinsic, flip_x: bool = False, flip_y: bool = False):
    """Get Open3D pointcloud from perspective depth image and color image.

    Args:
      color: HxWx3 uint8 array of RGB images.
      depth: HxW float array of perspective depth in meters.
      intrinsics: 3x3 float array of camera intrinsics matrix.

    Returns:
      points: HxWx3 float array of 3D points in camera coordinates.
    """

    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    if flip_x:
        px = width - 1 - px
    if flip_y:
        py = height - 1 - py
    px = (px - intrinsic[0, 2]) * (depth / intrinsic[0, 0])
    py = (py - intrinsic[1, 2]) * (depth / intrinsic[1, 1])
    # Stack the coordinates and reshape
    points = np.float32([px, py, depth]).transpose(1, 2, 0).reshape(-1, 3)

    # Assuming color image is in the format height x width x 3 (RGB)
    # Reshape color image to align with points
    colors = color.reshape(-1, 3)

    pcolors = np.hstack((points, colors))
    pcolors = pcolors[pcolors[:, 0] != 0.0, :]
    if pcolors.shape[0] == 0:
        return None, 0

    points = pcolors[:, :3]
    colors = pcolors[:, 3:]

    tpcd = o3d.geometry.PointCloud()
    tpcd.points = o3d.utility.Vector3dVector(points)
    tpcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Estimate normals
    tpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    # Optional: Orient the normals to be consistent
    tpcd.orient_normals_consistent_tangent_plane(k=50)

    # return pcd_with_color
    return tpcd, points.shape[0]


def farthest_point_sampling_with_color(pcd, n_points):
    """
    Perform Farthest Point Sampling on a point cloud with color information.

    :param pcd: numpy array of shape (N, 6), where N is the number of points in the point cloud.
                The first three columns are spatial coordinates and the last three are color information.
    :param n_points: number of points to sample.
    :return: sampled point cloud of shape (n_points, 6).
    """
    # farthest_points = np.zeros((n_points, 6))
    farthest_points = np.zeros((n_points, 9))
    # Initialize an array to store the shortest distance of each point to any selected point
    shortest_distances = np.full(len(pcd), np.inf)
    # Randomly choose the first point and update the distances
    first_index = np.random.randint(len(pcd))
    farthest_points[0] = pcd[first_index]
    shortest_distances = np.linalg.norm(pcd[:, :3] - farthest_points[0, :3], axis=1)

    for i in range(1, n_points):
        # Select the point that is farthest to any point in the farthest_points set
        farthest_index = np.argmax(shortest_distances)
        farthest_points[i] = pcd[farthest_index]
        # Update shortest_distances array
        distances_to_new_point = np.linalg.norm(pcd[:, :3] - farthest_points[i, :3], axis=1)
        shortest_distances = np.minimum(shortest_distances, distances_to_new_point)

    return farthest_points


def perform_gram_schmidt_transform(transform):
    """Perform Gram-Schmidt normalization on transform matrix or 9d pose."""
    assert transform.shape == (9,), "Input transform should be a 9d pose."
    transform = copy.deepcopy(transform)
    translation = transform[:3]
    v1 = transform[3:6]
    v1_normalized = v1 / np.linalg.norm(v1)
    v2 = transform[6:9]
    v2_orthogonal = v2 - np.dot(v2, v1_normalized) * v1_normalized
    v2_normalized = v2_orthogonal / np.linalg.norm(v2_orthogonal)
    return np.concatenate((translation, v1_normalized, v2_normalized))


def visualize_pcd(
    coordinate: np.ndarray,
    normal: np.ndarray | None = None,
    color: np.ndarray | None = None,
    pose: np.ndarray | None = None,
):
    """Visualize the given point cloud using open3d."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coordinate)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    if pose is not None:
        pcd.transform(pose)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, origin])


def visualize_pcd_list(
    coordinate_list: list[np.ndarray],
    normal_list: list[np.ndarray] | None = None,
    color_list: list[np.ndarray] | None = None,
    pose_list: list[np.ndarray] | None = None,
    point_show_normal: bool = False,
):
    """Visualize a list of point cloud"""
    vis_list = []
    for i in range(len(coordinate_list)):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coordinate_list[i])
        if normal_list is not None and normal_list[i].shape[-1] > 0:
            pcd.normals = o3d.utility.Vector3dVector(normal_list[i])
        if color_list is not None and color_list[i].shape[-1] > 0:
            pcd.colors = o3d.utility.Vector3dVector(color_list[i])
        if pose_list is not None:
            pcd.transform(pose_list[i])
        vis_list.append(pcd)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    vis_list.append(origin)
    o3d.visualization.draw_geometries(vis_list, point_show_normal=point_show_normal)
