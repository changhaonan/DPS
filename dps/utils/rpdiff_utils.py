"""Utility function for rpdiff evaluation; Converting raw file to batch file for evaluation."""

import dps.utils.misc_utils as utils
import pickle
import os
import numpy as np
import open3d as o3d
from dps.utils.pcd_utils import normalize_pcd, check_collision, complete_shape
from torch.utils.data import Dataset
from dps.data.superpoint_tool import *


def parse_child_parent(arr):
    pcd_dict = arr[()]
    parent_val = pcd_dict["parent"]
    child_val = pcd_dict["child"]
    return parent_val, child_val


def read_rpdiff_data(data_file=None, target_coord=None, anchor_coord=None, target_normal=None, anchor_normal=None, **kwargs):
    if data_file is not None:
        raw_data = np.load(data_file, allow_pickle=True)
        parent_pcd_s, child_pcd_s = parse_child_parent(raw_data["multi_obj_start_pcd"])
        parent_normal_s, child_normal_s = parse_child_parent(raw_data["normals"])
        parent_color_s, child_color_s = parse_child_parent(raw_data["colors"])
        data = {
            "target_coord": child_pcd_s,
            "target_normal": child_normal_s,
            "target_color": child_color_s,
            "anchor_coord": parent_pcd_s,
            "anchor_normal": parent_normal_s,
            "anchor_color": parent_color_s,
        }
    else:
        data = {
            "target_coord": target_coord,
            "target_normal": target_normal,
            "anchor_coord": anchor_coord,
            "anchor_normal": anchor_normal,
        }
    vis = kwargs.get("vis", False)
    if vis:
        # Check data
        target_pcd_o3d = o3d.geometry.PointCloud()
        target_pcd_o3d.points = o3d.utility.Vector3dVector(data["target_coord"])
        target_pcd_o3d.normals = o3d.utility.Vector3dVector(data["target_normal"])
        target_pcd_o3d.paint_uniform_color([0, 0, 1])
        anchor_pcd_o3d = o3d.geometry.PointCloud()
        anchor_pcd_o3d.points = o3d.utility.Vector3dVector(data["anchor_coord"])
        anchor_pcd_o3d.normals = o3d.utility.Vector3dVector(data["anchor_normal"])
        anchor_pcd_o3d.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([target_pcd_o3d, anchor_pcd_o3d])
    return data


def reorient_pcd(target_coord, anchor_coord, target_normal=None, anchor_normal=None, **kwargs):
    """Reorient the object"""
    table_center = kwargs.get("table_center", [0, 0, 0])
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
    # x-axis should point to the table center, filp if not
    if anchor_R_x @ (table_center - anchor_t) < 0:
        anchor_R_x = -anchor_R_x
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
        "target_normal": target_normal,
        "anchor_normal": anchor_normal,
        "anchor_pose": anchor_pose,
        "target_pose": target_pose,
    }
    return data


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
class RpdiffHelper:
    """RpdiffHelper; directly loading from raw point cloud data"""

    def __init__(self, scale=1.0, downsample_voxel_size=0.02, batch_size: int = 8, target_padding=0.2, superpoint_cfg: list = []) -> None:
        self.scale = scale
        self.downsample_voxel_size = downsample_voxel_size
        self.batch_size = batch_size
        cfg = init_config(overrides=superpoint_cfg)
        self.target_padding = target_padding
        # Instantiate the datamodule
        datamodule = hydra.utils.instantiate(cfg.datamodule)
        # Initialize SuperPointTool
        self.spt = SuperPointTool(pre_transform=datamodule.pre_transform)

    def process_data(self, target_coord, anchor_coord, target_normal, anchor_normal, **kwargs):
        """Convert raw data into format for evaluation"""
        f_keys = ["planarity", "linearity", "verticality", "scattering"]
        # First do reorientation; reorientation can make the training more efficient
        reorient_data = reorient_pcd(target_coord, anchor_coord, target_normal, anchor_normal, **kwargs)
        # Downsample the point cloud
        target_coord, target_color, target_normal = downsample_points(reorient_data["target_coord"], None, reorient_data["target_normal"], self.downsample_voxel_size)
        anchor_coord, anchor_color, anchor_normal = downsample_points(reorient_data["anchor_coord"], None, reorient_data["anchor_normal"], self.downsample_voxel_size)
        # Gen superpoint
        anchor_superpoint = self.spt.gen_superpoint(anchor_coord, anchor_color, anchor_normal, scale=self.scale, vis=False)
        # visualize_superpoint(anchor_superpoint)
        # Complete shape
        target_coord, target_normal = complete_shape(target_coord, padding=self.target_padding, strategy="bbox", vis=False)
        target_feat = np.zeros((target_coord.shape[0], len(f_keys)), dtype=np.float32)

        anchor_feat = []
        for key in f_keys:
            anchor_feat.append(anchor_superpoint[key])
        anchor_feat = np.concatenate(anchor_feat, axis=-1)
        data = {
            "target_coord": target_coord,
            "target_normal": target_normal,
            "target_feat": target_feat,
            "anchor_coord": anchor_superpoint["pos"],
            "anchor_normal": anchor_superpoint["normal"],
            "anchor_feat": anchor_feat,
            "anchor_super_index": np.vstack(anchor_superpoint["super_index"]).T,
            "anchor_pose": reorient_data["anchor_pose"][None, ...],
            "target_pose": reorient_data["target_pose"][None, ...],
        }
        return self.convert_to_batch(data, self.batch_size)

    def convert_to_batch(self, data: dict, batch_size: int = 1):
        """Convert the data into batch format"""
        target = {
            "target_coord": [],
            "target_normal": [],
            "target_feat": [],
            "anchor_coord": [],
            "anchor_normal": [],
            "anchor_feat": [],
            "target_batch_index": [],
            "anchor_batch_index": [],
            "anchor_super_index": [],
            "anchor_pose": [],
            "target_pose": [],
        }
        for idx in range(batch_size):
            target["target_coord"].append(data["target_coord"].astype(np.float32))
            target["target_normal"].append(data["target_normal"].astype(np.float32))
            target["target_feat"].append(data["target_feat"].astype(np.float32))
            target["anchor_coord"].append(data["anchor_coord"].astype(np.float32))
            target["anchor_normal"].append(data["anchor_normal"].astype(np.float32))
            target["anchor_feat"].append(data["anchor_feat"].astype(np.float32))
            target["target_batch_index"].append(np.full([len(data["target_coord"])], fill_value=idx, dtype=np.int64))
            target["anchor_batch_index"].append(np.full([len(data["anchor_coord"])], fill_value=idx, dtype=np.int64))
            target["anchor_super_index"].append(data["anchor_super_index"].astype(np.int64))
            target["anchor_pose"].append(data["anchor_pose"].astype(np.float32))
            target["target_pose"].append(data["target_pose"].astype(np.float32))
        return {k: torch.from_numpy(np.concatenate(v)) for k, v in target.items()}
