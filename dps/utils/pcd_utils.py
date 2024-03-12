"""Utility function for 3D operations.
"""

from __future__ import annotations
import open3d as o3d
import copy
import torch
import numpy as np


def check_pcd_pyramid(pcd: o3d.PointCloud, grid_sizes: list[int]):
    """Visualize the pcd pyramid."""
    pcd_list = [pcd]
    current_pcd = pcd
    print(f"Pcd size: {len(current_pcd.points)}")
    for grid_size in grid_sizes:
        pcd_down = current_pcd.voxel_down_sample(voxel_size=grid_size)
        pcd_list.append(pcd_down)
        print(f"Pcd size: {len(pcd_down.points)}")
        o3d.visualization.draw_geometries([pcd_down])
        current_pcd = pcd_down
    return pcd_list


def visualize_tensor_pcd(pcd: torch.Tensor):
    """Visualize the tensor pcd."""
    pcd = pcd.cpu().numpy()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    o3d.visualization.draw_geometries([pcd])


def normalize_pcd(pcd_anchor, pcd_target, do_scaling: bool = True):
    # Normalize to unit cube
    pcd_center = (pcd_anchor.get_max_bound() + pcd_anchor.get_min_bound()) / 2
    pcd_anchor = pcd_anchor.translate(-pcd_center)
    scale_xyz = pcd_anchor.get_max_bound() - pcd_anchor.get_min_bound()
    scale_xyz = np.max(scale_xyz)
    if not do_scaling:
        scale_xyz = 1.0
    pcd_anchor = pcd_anchor.scale(1 / scale_xyz, center=np.array([0, 0, 0]))

    # Normalize the child point clouds
    pcd_target = pcd_target.translate(-pcd_center)
    normalize_pcd_target = pcd_target.scale(1 / scale_xyz, center=np.array([0, 0, 0]))
    return pcd_anchor, normalize_pcd_target, pcd_center, scale_xyz


def check_collision(pcd_anchor: np.ndarray, pcd_target: np.ndarray, threshold=0.01):
    """Check if there existing collision between two point clouds."""
    dists = np.linalg.norm(pcd_anchor[:, None, :] - pcd_target[None, :, :], axis=-1)
    min_dists = np.min(dists, axis=1)
    return np.any(min_dists < threshold)


def visualize_point_pyramid(pos: np.ndarray | torch.Tensor, normal: np.ndarray | torch.Tensor | None, cluster_indices: list[np.ndarray] | list[torch.Tensor]):
    """Visualize the point pyramid."""
    if isinstance(pos, torch.Tensor):
        pos = pos.cpu().numpy()
    if isinstance(normal, torch.Tensor):
        normal = normal.cpu().numpy()
    if isinstance(cluster_indices[0], torch.Tensor):
        cluster_indices = [c.cpu().numpy() for c in cluster_indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos)
    if normal is not None:
        pcd.normals = o3d.utility.Vector3dVector(normal)
    num_clusters = [np.unique(c).size for c in cluster_indices]
    num_color = max(num_clusters)
    color = np.random.rand(num_color, 3)
    cum_cluster_index = cluster_indices[0]
    for i, cluster_index in enumerate(cluster_indices):
        cluster_index_mask = cluster_index != 0
        cluster_index[cluster_index_mask] = cluster_index[cluster_index_mask] - np.min(cluster_index[cluster_index_mask])
        # Map the cluster index
        if i != 0:
            for j in range(len(cluster_index)):
                if cluster_index[j] != 0:
                    cum_cluster_index[cum_cluster_index == j] = cluster_index[j]
        pcd.colors = o3d.utility.Vector3dVector(color[cum_cluster_index])
        o3d.visualization.draw_geometries([pcd])


def create_box_point(center, extent, R, step=0.01):
    """
    Create a dense point cloud for a rotated 3D bounding box.

    Parameters:
    - center: Tuple or list (Cx, Cy, Cz) representing the center of the bounding box.
    - extent: Tuple or list (L, W, H) representing the length, width, and height of the box.
    - R: 3x3 numpy array representing the rotation matrix of the bounding box.
    - density: Number of points to generate per unit length along each dimension.

    Returns:
    - points: A numpy array of points representing the dense point cloud.
    """
    L, W, H = extent
    Cx, Cy, Cz = center

    # Calculate the number of points to generate based on the density
    num_points_L = np.ceil(L / step).astype(int)
    num_points_W = np.ceil(W / step).astype(int)
    num_points_H = np.ceil(H / step).astype(int)

    # Generate grid points
    x = np.linspace(-L / 2, L / 2, num_points_L)
    y = np.linspace(-W / 2, W / 2, num_points_W)
    z = np.linspace(-H / 2, H / 2, num_points_H)

    # Points on each face (ignore the ends to avoid duplicating corner points)
    faces = []
    # Top and bottom
    for zz in (-H / 2, H / 2):
        xx, yy = np.meshgrid(x, y)
        zz = np.ones_like(xx) * zz
        faces.append(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T)
    # Left and right
    for yy in (-W / 2, W / 2):
        xx, zz = np.meshgrid(x, z)
        yy = np.ones_like(xx) * yy
        faces.append(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T)
    # Front and back
    for xx in (-L / 2, L / 2):
        yy, zz = np.meshgrid(y, z)
        xx = np.ones_like(yy) * xx
        faces.append(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T)

    # Combine all faces and apply rotation and translation
    all_points = np.vstack(faces)
    rotated_points = (R @ all_points.T).T  # Apply rotation
    translated_points = rotated_points + np.array(center)  # Translate to center

    return translated_points


def complete_shape(coord: np.ndarray, padding: float = 0.1, strategy: str = "bbox", vis: bool = False, **kwargs):
    """Complete a shape completion of point cloud. The completion will have a padding around the original shape."""
    if strategy == "bbox":
        # Compute minimal bounding box
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        bbox = pcd.get_minimal_oriented_bounding_box()
        step = kwargs.get("step", 0.01)
        box_extent = np.array(bbox.extent) * (1 + padding)
        box_points = create_box_point(bbox.center, box_extent, bbox.R, step=step)
        box_pcd = o3d.geometry.PointCloud()
        box_pcd.points = o3d.utility.Vector3dVector(box_points)
        # compute normal
        box_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # flip normals so that they are pointing outwards w.r.t. the center
        normals = np.asarray(box_pcd.normals)
        center = np.mean(box_points, axis=0)
        to_center = center - box_points
        dot = np.sum(normals * to_center, axis=-1)
        flip_mask = dot > 0
        normals[flip_mask] *= -1

        if vis:
            # transparence
            box_pcd.paint_uniform_color([0.9, 0.1, 0.1])
            o3d.visualization.draw_geometries([pcd, box_pcd])
        return np.asarray(box_pcd.points), normals
    else:
        raise ValueError(f"Invalid strategy: {strategy}")


def compute_corr_radius(pcd1: np.ndarray | torch.Tensor, pcd2: np.ndarray | torch.Tensor, radius: float = 0.1):
    """Compute the correspondence matrix between two point clouds with radius
    Correspondence matrix is a binary matrix, where each row represents the correspondence of a point in pcd1 to pcd2; nearest & within the radius is 1, otherwise 0.
    """
    if isinstance(pcd1, torch.Tensor):
        pcd1 = pcd1.cpu().numpy()
    if isinstance(pcd2, torch.Tensor):
        pcd2 = pcd2.cpu().numpy()

    dist_matrix = np.linalg.norm(pcd1[:, None, :3] - pcd2[None, :, :3], axis=-1)
    corr_1to2 = np.where(dist_matrix <= radius)
    return np.stack([corr_1to2[0], corr_1to2[1]], axis=-1)


def compute_batch_corr_radius(pcd1, pcd2, normal1=None, normal2=None, radius=0.1, dot_threshold=-0.9):
    """Vectorized computation of the correspondence matrix for batches of point clouds.
    Note: This function returns the distance matrices for each batch, from which correspondences can be derived.
    """
    is_tensor = isinstance(pcd1, torch.Tensor)
    if isinstance(pcd1, torch.Tensor):
        pcd1 = pcd1.cpu().numpy()
    if isinstance(pcd2, torch.Tensor):
        pcd2 = pcd2.cpu().numpy()
    if isinstance(normal1, torch.Tensor):
        normal1 = normal1.cpu().numpy()
    if isinstance(normal2, torch.Tensor):
        normal2 = normal2.cpu().numpy()
    # Calculate the squared distances between all pairs (for each batch)
    # Expanding dimensions to support broadcasting
    dist_sq = np.sum((pcd1[:, :, np.newaxis, :] - pcd2[:, np.newaxis, :, :]) ** 2, axis=-1)

    if (normal1 is not None) and (normal2 is not None):
        normal_dot = np.sum(normal1[:, :, np.newaxis, :] * normal2[:, np.newaxis, :, :], axis=-1)
    else:
        normal_dot = np.ones((pcd1.shape[0], pcd1.shape[1], pcd2.shape[1])) * dot_threshold
    # Convert squared distances to a binary correspondence matrix within the specified radius
    # This step essentially checks if distances are within the square of the radius to avoid sqrt computation
    if dot_threshold > 0:
        # Filter out correspondences with normals in opposite directions
        corr_matrix = np.logical_and(dist_sq <= radius**2, normal_dot >= dot_threshold).astype(np.float32)
    else:
        # Filter out correspondences with normals in the same direction
        corr_matrix = np.logical_and(dist_sq <= radius**2, normal_dot <= dot_threshold).astype(np.float32)

    if is_tensor:
        corr_matrix = torch.from_numpy(corr_matrix)
    return corr_matrix
