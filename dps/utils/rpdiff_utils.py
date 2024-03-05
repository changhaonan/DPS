"""Utility function for rpdiff evaluation"""

import dps.utils.misc_utils as utils
import pickle
import os
import numpy as np
import open3d as o3d
from dps.utils.pcd_utils import normalize_pcd, check_collision, complete_shape
from torch.utils.data import Dataset


# Utility functions
def preprocess_input_rpdiff(target_coord, anchor_coord, target_normal=None, anchor_normal=None, **kwargs):
    """Preprocess data for eval on RPDiff"""
    # Build o3d object
    target_pcd_o3d = o3d.geometry.PointCloud()
    target_pcd_o3d.points = o3d.utility.Vector3dVector(target_coord)
    target_pcd_o3d.normals = o3d.utility.Vector3dVector(target_normal)
    target_pcd_o3d.paint_uniform_color([0, 0, 1])
    anchor_pcd_o3d = o3d.geometry.PointCloud()
    anchor_pcd_o3d.points = o3d.utility.Vector3dVector(anchor_coord)
    anchor_pcd_o3d.normals = o3d.utility.Vector3dVector(anchor_normal)
    anchor_pcd_o3d.paint_uniform_color([1, 0, 0])

    # Estimate the pose of fixed coord using a rotating bbox
    anchor_pcd_bbox = anchor_pcd_o3d.get_minimal_oriented_bounding_box()
    anchor_pcd_bbox.color = [0, 1, 0]
    anchor_R = anchor_pcd_bbox.R
    anchor_t = (np.max(anchor_coord, axis=0) + np.min(anchor_coord, axis=0)) / 2
    anchor_extent = anchor_pcd_bbox.extent
    # Play around axis
    anchor_R_z = np.array([0, 0, 1])
    # Remove the axis that is parallel to the z-axis
    anchor_R_z_dot = anchor_R_z @ anchor_R
    anchor_R_z_idx = np.argmax(np.abs(anchor_R_z_dot))
    anchor_R_axis = np.delete(anchor_R, anchor_R_z_idx, axis=1)
    anchor_R_extent = np.delete(anchor_extent, anchor_R_z_idx)
    # The one with shorter extent is the x-axis
    anchor_R_x_idx = np.argmin(anchor_R_extent)
    anchor_R_x = anchor_R_axis[:, anchor_R_x_idx]
    # The other one is the y-axis
    anchor_R_y = np.cross(anchor_R_z, anchor_R_x)
    anchor_R = np.column_stack([anchor_R_x, anchor_R_y, anchor_R_z])
    anchor_pose = np.eye(4)
    anchor_pose[:3, 3] = anchor_t
    anchor_pose[:3, :3] = anchor_R
    target_t = (np.max(target_coord, axis=0) + np.min(target_coord, axis=0)) / 2
    target_pose = np.eye(4)
    target_pose[:3, :3] = anchor_R
    target_pose[:3, 3] = target_t

    # Shift the target coord to the origin
    anchor_pcd_o3d.transform(np.linalg.inv(anchor_pose))
    target_pcd_o3d.transform(np.linalg.inv(target_pose))

    # Normalize pcd
    do_normalize = kwargs.get("do_normalize", False)
    if do_normalize:
        target_pcd_o3d, anchor_pcd_o3d, _, scale_xyz = normalize_pcd(target_pcd_o3d, anchor_pcd_o3d)
    else:
        scale_xyz = 1.0

    # Downsample the point cloud
    grid_size = kwargs.get("grid_size", 0.01)
    anchor_pcd_o3d = anchor_pcd_o3d.voxel_down_sample(voxel_size=grid_size)
    target_pcd_o3d = target_pcd_o3d.voxel_down_sample(voxel_size=grid_size)
    # Compute normal
    if target_normal is None:
        target_pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    if anchor_normal is None:
        anchor_pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Build the input
    target_coord = np.array(target_pcd_o3d.points).astype(np.float32)
    target_normal = np.array(target_pcd_o3d.normals).astype(np.float32)
    anchor_coord = np.array(anchor_pcd_o3d.points).astype(np.float32)
    anchor_normal = np.array(anchor_pcd_o3d.normals).astype(np.float32)

    data = {
        "target_coord": target_coord,
        "anchor_coord": anchor_coord,
        "anchor_pose": np.linalg.inv(anchor_pose),
        "target_pose": np.linalg.inv(target_pose),
        "scale_xyz": scale_xyz,
    }
    return data


def load_superpoint_rpdiff(data: dict, superpoint_path: str, f_keys: list = ["planarity", "linearity", "verticality", "scattering"]):
    """Load superpoint file from file for RPDiff"""
    superpoint_data = np.load(superpoint_path, allow_pickle=True)["data"]
    c_superpoint_data = superpoint_data["child"]
    p_superpoint_data = superpoint_data["parent"]

    # Assemble data
    anchor_coord = p_superpoint_data["pos"]
    anchor_normal = p_superpoint_data["normal"]
    anchor_feat = []
    for f_key in f_keys:
        anchor_feat.append(p_superpoint_data[f_key])
    anchor_feat = np.concatenate(anchor_feat, axis=-1)
    anchor_super_index = np.vstack(p_superpoint_data["super_index"]).T
    target_coord = c_superpoint_data["pos"]
    target_normal = c_superpoint_data["normal"]
    target_feat = []
    for f_key in f_keys:
        target_feat.append(c_superpoint_data[f_key])
    target_feat = np.concatenate(target_feat, axis=-1)
    target_super_index = np.vstack(c_superpoint_data["super_index"]).T
    tmorp_data = {
        "target_coord": target_coord,
        "target_normal": target_normal,
        "target_feat": target_feat,
        "target_super_index": target_super_index,
        "anchor_coord": anchor_coord,
        "anchor_normal": anchor_normal,
        "anchor_feat": anchor_feat,
        "anchor_super_index": anchor_super_index,
    }
    return tmorp_data


def pose_recover_rpdiff(pred_pose: np.ndarray, crop_center: np.ndarray, data: dict):
    """Recover the pose back to the original coordinate system for RPDiff"""
    T_shift = np.eye(4)
    T_shift[:3, 3] = crop_center
    anchor_pose = data["anchor_pose"]
    target_pose = data["target_pose"]
    scale_xyz = data["scale_xyz"]
    T_scale = np.eye(4)
    T_scale[:3, :3] = np.diag([1.0 / scale_xyz, 1.0 / scale_xyz, 1.0 / scale_xyz])
    recover_pose = np.linalg.inv(T_scale @ anchor_pose) @ T_shift @ pred_pose @ T_scale @ target_pose
    return recover_pose


def post_filter_rpdiff(pred_pose: np.ndarray, samples: list, collision_threshold: float = 0.01):
    """Post-process for rpdiff; filter out collision results"""
    collison_scores = []
    for i in range(pred_pose.shape[0]):
        anchor_coord = samples[i]["anchor_coord"]
        target_coord = samples[i]["target_coord"]
        pred_pose_i = pred_pose[i]

        # Transform
        target_coord = (pred_pose_i[:3, :3] @ target_coord.T).T + pred_pose_i[:3, 3]

        # Check collision
        collision = check_collision(anchor_coord, target_coord, threshold=collision_threshold)
        collison_scores.append(collision)
    collison_scores = np.array(collison_scores)
    return collison_scores


################################## Dataset ##################################
class RpdiffWrapper:
    """RpdiffWrapper; directly loading from raw point cloud data"""

    def __init__(self) -> None:
        super().__init__()

    def process_data(self, target_coord, anchor_coord, target_normal=None, anchor_normal=None, **kwargs):
        """Convert raw data into format for evaluation"""
        # First do reorientation
        reorient_data = preprocess_input_rpdiff(target_coord, anchor_coord, target_normal, anchor_normal, **kwargs)
        # Save data to file
        temp_dir = kwargs.get("temp_dir", "temp")
        pickle.dump(reorient_data, open(temp_dir + "/reorient_data.pkl", "wb"))
        # Call superpoint script
        superpoint_script = kwargs.get("superpoint_script", "superpoint")
        os.system(f"python {superpoint_script} --input {temp_dir}/reorient_data.pkl --output {temp_dir}/superpoint_data.npz")
        # Load superpoint data
        superpoint_path = os.path.join(temp_dir, "superpoint_data.npz")
        superpoint_data = load_superpoint_rpdiff(reorient_data, superpoint_path)
        return superpoint_data
