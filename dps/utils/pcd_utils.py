"""Utility function for 3D operations.
"""

from __future__ import annotations
import open3d as o3d
import copy
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt


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
        # Add center point & normal using z-axis
        box_points = np.concatenate([box_points, [center]])
        normals = np.concatenate([normals, [[0, 0, 1]]])
        if vis:
            # transparence
            box_pcd.paint_uniform_color([0.9, 0.1, 0.1])
            o3d.visualization.draw_geometries([pcd, box_pcd])
        return box_points, normals
    elif strategy == "axis_aligned_bbox":
        # Compute axis-aligned bounding box
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord)
        bbox = pcd.get_axis_aligned_bounding_box()
        step = kwargs.get("step", 0.01)
        box_extent = bbox.get_max_bound() - bbox.get_min_bound()
        box_extent = box_extent * (1 + padding)
        box_points = create_box_point(bbox.get_center(), box_extent, np.eye(3), step=step)
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
        # Add center point & normal using z-axis
        box_points = np.concatenate([box_points, [center]])
        normals = np.concatenate([normals, [[0, 0, 1]]])
        if vis:
            # transparence
            box_pcd.paint_uniform_color([0.9, 0.1, 0.1])
            o3d.visualization.draw_geometries([pcd, box_pcd])
        return box_points, normals
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


def arun(p, q):
    """Compute the optimal rotation matrix using Arun's method."""
    # Compute the centroids
    p_centroid = np.mean(p, axis=0)
    q_centroid = np.mean(q, axis=0)
    # Center the point clouds
    p_centered = p - p_centroid
    q_centered = q - q_centroid
    # Compute the covariance matrix
    H = np.dot(p_centered.T, q_centered)
    # Compute the SVD
    U, _, Vt = np.linalg.svd(H)
    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)
    t = q_centroid - np.dot(R, p_centroid)
    return R, t


def icp_pose_refine(coord1, coord2, normal1, normal2, pose_init, max_iter, vis: bool = False, **kwargs):
    """Refine realtive pose between two point clouds using ICP. 1 is anchor, 2 is target."""
    # Do fartherst point sampling
    sample_size = kwargs.get("sample_size", 10000)
    coord1_idx = np.random.choice(len(coord1), min(sample_size, coord1.shape[0]), replace=False)
    coord1 = coord1[coord1_idx]
    normal1 = normal1[coord1_idx]
    coord2_idx = np.random.choice(len(coord2), min(sample_size, coord2.shape[0]), replace=False)
    coord2 = coord2[coord2_idx]
    normal2 = normal2[coord2_idx]

    coord2_g = (pose_init[:3, :3] @ coord2.T + pose_init[:3, 3][:, None]).T
    coord2_g_normal = (pose_init[:3, :3] @ normal2.T).T

    if vis:
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(coord1)
        pcd1.normals = o3d.utility.Vector3dVector(normal1)
        pcd1.paint_uniform_color([0.1, 0.1, 0.7])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(coord2_g)
        pcd2.normals = o3d.utility.Vector3dVector(coord2_g_normal)
        pcd2.paint_uniform_color([0.7, 0.1, 0.1])
        o3d.visualization.draw_geometries([pcd1, pcd2])

    for i in range(max_iter):
        # Compute the correspondence matrix
        corr_matrix = compute_batch_corr_radius(
            coord1[None, ...], coord2_g[None, ...], normal1[None, ...], coord2_g_normal[None, ...], radius=kwargs.get("radius", 0.1), dot_threshold=kwargs.get("dot_threshold", -0.9)
        )[0]
        conf_idx = np.where(corr_matrix > 0)
        # Compute the transformation matrix
        coord1_select = coord1[conf_idx[0]]
        coord2_select = coord2_g[conf_idx[1]]
        R, t = arun(coord1_select, coord2_select)
        # Apply the transformation matrix
        coord2_g = (R @ coord2_g.T + t[:, None]).T
        coord2_g_normal = (R @ coord2_g_normal.T).T
        # Update the pose
        pose_init[:3, :3] = R @ pose_init[:3, :3]
        pose_init[:3, 3] = R @ pose_init[:3, 3] + t
        if vis:
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(coord1)
            pcd1.normals = o3d.utility.Vector3dVector(normal1)
            pcd1.paint_uniform_color([0.1, 0.1, 0.7])
            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(coord2_g)
            pcd2.normals = o3d.utility.Vector3dVector(coord2_g_normal)
            pcd2.paint_uniform_color([0.7, 0.1, 0.1])
            o3d.visualization.draw_geometries([pcd1, pcd2])
    return pose_init


def estimate_collision(coord1, coord2):
    """Estimate the collision status between two objects. pcd1 must be convex, but pcd2 can be concave."""
    # Compute the minimal bounding box for pcd1
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(coord1)
    bbox1 = pcd1.get_minimal_oriented_bounding_box()
    # Rotate pcd2 to the same frame as pcd1
    R = bbox1.R
    t = bbox1.center
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = t
    pose_inv = np.linalg.inv(pose)
    coord2 = (pose_inv[:3, :3] @ coord2.T).T + pose_inv[:3, 3]
    pcd1.transform(pose_inv)
    # Create an axis-aligned bounding box for pcd1
    bbox1_o3d = pcd1.get_axis_aligned_bounding_box()

    # R = bbox1.R.T
    # coord2 = (R @ coord2.T).T
    # Count the number of points in pcd2 that are inside the bounding box of pcd1
    bbox1_min = bbox1_o3d.get_min_bound()
    bbox1_max = bbox1_o3d.get_max_bound()
    inside_indices = np.where(
        (coord2[:, 0] >= bbox1_min[0])
        & (coord2[:, 0] <= bbox1_max[0])
        & (coord2[:, 1] >= bbox1_min[1])
        & (coord2[:, 1] <= bbox1_max[1])
        & (coord2[:, 2] >= bbox1_min[2])
        & (coord2[:, 2] <= bbox1_max[2])
    )[0]
    # # Visualize the result
    # pcd1.paint_uniform_color([0.1, 0.1, 0.7])
    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(coord2)
    # colors = np.zeros_like(coord2).astype(np.float64)
    # colors[inside_indices] = [0.7, 0.1, 0.1]
    # pcd2.colors = o3d.utility.Vector3dVector(colors)
    # origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # bbox1_o3d.color = (1, 0, 0)
    # o3d.visualization.draw_geometries([pcd1, pcd2, origin, bbox1_o3d])
    return inside_indices.shape[0]


def screen_shot_pcd(pcd, **kwargs):
    """Take a screenshot of the point cloud."""
    show_axis = kwargs.get("show_axis", False)
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=270, height=270, visible=False)
    viewer.add_geometry(pcd)
    if show_axis:
        viewer.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    # Control visualization
    opt = viewer.get_render_option()
    opt.point_size = 7  # Adjust this value as needed
    # Render image
    image = viewer.capture_screen_float_buffer(do_render=True)
    viewer.destroy_window()
    image = (np.asarray(image) * 255.0).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def visualize_corr(coord1, coord2, corr, **kwargs):
    """Visualize the correspondence between two point clouds.
    Args:
    - coord1: The first point cloud.
    - coord2: The second point cloud.
    - corr: The correspondence list between the two point clouds.
    """
    pcd = o3d.geometry.PointCloud()
    combined_coord = np.concatenate([coord1, coord2], axis=0)
    combined_color = np.zeros((combined_coord.shape[0], 3))
    combined_color[: coord1.shape[0], 0] = 1
    combined_color[coord1.shape[0] :, 2] = 1
    pcd.points = o3d.utility.Vector3dVector(combined_coord)
    pcd.colors = o3d.utility.Vector3dVector(combined_color)
    lines = np.concatenate([corr[:, 0:1], corr[:, 1:2] + coord1.shape[0]], axis=1)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(combined_coord)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color([1, 0, 1])
    o3d.visualization.draw_geometries([pcd, line_set])


def visualize_point_scalar(coord, scalar, **kwargs):
    """Visualize the point cloud with scalar values."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coord)
    colors = plt.get_cmap(kwargs.get("cmap", "viridis"))(scalar)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
