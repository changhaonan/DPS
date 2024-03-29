"""Generate superpoint RPdiff data for Rpdiff."""

import os
import sys
import random

# Add the project's files to the python path
# file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(file_path, "external", "superpoint_transformer")
sys.path.append(file_path)

# Necessary for advanced config parsing with hydra and omegaconf
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)

import hydra
from src.utils import init_config
import torch
from src.visualization import show
from src.datasets.s3dis import CLASS_NAMES, CLASS_COLORS
from src.datasets.s3dis import S3DIS_NUM_CLASSES as NUM_CLASSES
from src.transforms import *

from tqdm import tqdm
import logging
import open3d as o3d
import pickle
import numpy as np
from src.data import NAG, Data
from scipy.spatial.transform import Rotation as R
from detectron2.config import LazyConfig

log = logging.getLogger(__name__)


def visualize_superpoint(superpoint_data):
    pos = superpoint_data["pos"]
    normal = superpoint_data["normal"]
    super_indexes = superpoint_data["super_index"]
    num_color = np.max(super_indexes[0]) + 1
    # Generate random color
    color = np.random.rand(num_color, 3)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    for i, super_index in enumerate(super_indexes):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)
        # pcd.normals = o3d.utility.Vector3dVector(normal)
        pcd.colors = o3d.utility.Vector3dVector(color[super_index])
        o3d.visualization.draw_geometries([pcd, origin])


def has_outlier(points, range: float = 3.0):
    """Check if there are outliers in the point cloud."""
    mean = np.mean(points, axis=0)
    # Check if there exists a point that is outside the range
    dist = np.linalg.norm(points - mean, axis=1)
    return np.any(dist > range)


def pose7d_to_mat(pose7d):
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R.from_quat(pose7d[3:]).as_matrix()
    pose_mat[:3, 3] = pose7d[:3]
    return pose_mat


def check_pcd(points, colors, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])


def downsample_points(points, colors, normals, voxel_size: float = 0.03):
    pcd = o3d.t.geometry.PointCloud()
    pcd.point["positions"] = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    if colors is not None:
        pcd.point["colors"] = o3d.core.Tensor(colors, dtype=o3d.core.Dtype.UInt8)
    if normals is not None:
        pcd.point["normals"] = o3d.core.Tensor(normals, dtype=o3d.core.Dtype.Float32)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points = pcd.point["positions"].numpy()
    if colors is not None:
        colors = pcd.point["colors"].numpy()
    else:
        colors = np.zeros_like(points).astype(np.uint8)
    if normals is not None:
        normals = pcd.point["normals"].numpy()
    return points, colors, normals


class SuperPointTool:
    def __init__(self, pre_transform=None, **kwargs):
        self.pre_transform = pre_transform
        self.kwargs = kwargs

    def wrap_data(self, points: np.ndarray, colors: np.ndarray, normals: np.ndarray, **kwargs):
        """Wrap data into a PyTorch tensor."""
        # label
        pos = torch.from_numpy(points).to(torch.float32)
        # Compute the intensity from the RGB values
        cloud_colors = colors / 255.0
        intensity = cloud_colors[:, 0] * 0.299 + cloud_colors[:, 1] * 0.587 + cloud_colors[:, 2] * 0.114
        intensity = torch.from_numpy(intensity).unsqueeze(1).to(torch.float32)
        rgb = torch.from_numpy(cloud_colors).to(torch.float32)
        normals = torch.from_numpy(normals).to(torch.float32)
        custom_dict = {}
        for k, v in kwargs.items():
            custom_dict[f"raw_{k}"] = torch.from_numpy(v).to(torch.float32)
        data = Data(pos=pos, intensity=intensity, rgb=rgb, raw_normal=normals, **custom_dict)
        return data

    def preprocess(self, data: Data):
        # Apply pre_transform
        if self.pre_transform is not None:
            nag = self.pre_transform(data)
        else:
            nag = NAG([data])
        return nag

    def gen_superpoint(self, points: np.ndarray, colors: np.ndarray, normals: np.ndarray, scale: float = 1.0, vis: bool = False, **kwargs):
        """Generate superpoint data from input points and colors."""
        points = points * scale  # Scale the points
        # Move points to the origin
        points_center = np.mean(points, axis=0)
        points -= points_center
        data = self.wrap_data(points, colors, normals, **kwargs)
        nag = self.preprocess(data)

        # Construct superpoint data for parent object
        pos = (nag[0].pos.detach().cpu().numpy() + points_center) / scale
        color = nag[0].rgb.detach().cpu().numpy()
        normal = nag[0].raw_normal.detach().cpu().numpy()
        # normal = nag[0].normal.detach().cpu().numpy()
        planarity = nag[0].planarity.detach().cpu().numpy()
        linearity = nag[0].linearity.detach().cpu().numpy()
        verticality = nag[0].verticality.detach().cpu().numpy()
        scattering = nag[0].scattering.detach().cpu().numpy()
        custom_dict = {}
        for k, v in kwargs.items():
            custom_dict[k] = nag[0][f"raw_{k}"].detach().cpu().numpy()
        super_index_list = []
        for i in range(nag.num_levels - 1):
            _super_index = nag[i].super_index.detach().cpu().numpy()
            # Remap from last super_index to current super_index
            if i == 0:
                super_index = _super_index
            else:
                super_index = np.zeros_like(super_index_list[-1])
                for j in range(len(super_index_list[-1])):
                    super_index[j] = _super_index[super_index_list[-1][j]]
            super_index_list.append(super_index)
        super_point_data = {
            "pos": pos,
            "color": color,
            "normal": normal,
            "planarity": planarity,
            "linearity": linearity,
            "verticality": verticality,
            "scattering": scattering,
            "super_index": super_index_list,
            **custom_dict,
        }
        if vis:
            # Show the superpoint data
            show(nag, class_names=CLASS_NAMES, ignore=NUM_CLASSES, class_colors=CLASS_COLORS, max_points=100000)
        return super_point_data


def parse_child_parent(arr):
    pcd_dict = arr[()]
    parent_val = pcd_dict["parent"]
    child_val = pcd_dict["child"]
    return parent_val, child_val


if __name__ == "__main__":
    # Parse task cfg
    task_name = "book_in_bookshelf"
    root_path = os.path.dirname((os.path.abspath(__file__)))
    task_cfg_file = os.path.join(root_path, "config", f"pose_transformer_rpdiff_{task_name}.py")
    task_cfg = LazyConfig.load(task_cfg_file)
    scale = task_cfg.PREPROCESS.TARGET_RESCALE
    downsample_voxel_size = task_cfg.PREPROCESS.GRID_SIZE
    # Parse the configs using hydra
    cfg = init_config(
        overrides=[
            "experiment=semantic/scannet.yaml",
            "datamodule.voxel=0.03",
            "datamodule.pcp_regularization=[0.01, 0.1]",
            "datamodule.pcp_spatial_weight=[0.1, 0.1]",
            "datamodule.pcp_cutoff=[10, 10]",
            "datamodule.graph_gap=[0.2, 0.5]",
            "datamodule.graph_chunk=[1e6, 1e5]",
            "+net.nano=True",
        ]
    )
    # Instantiate the datamodule
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    # Initialize SuperPointTool
    spt = SuperPointTool(pre_transform=datamodule.pre_transform)
    # Load rpdiff data
    data_path_dict = {
        "stack_can_in_cabinet": "/home/harvey/Project/dps/dps/external/rpdiff/data/task_demos/can_in_cabinet_stack/task_name_stack_can_in_cabinet",
        "book_in_bookshelf": "/home/harvey/Data/rpdiff_V3/book_in_bookshelf",
        "mug_on_rack_multi": "/home/harvey/Project/dps/dps/external/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos/task_name_mug_on_rack_multi",
    }

    data_dir = data_path_dict[task_name]
    data_file_list = os.listdir(data_dir)
    data_file_list = [f for f in data_file_list if f.endswith(".npz")]
    export_dir = os.path.join(data_dir, "superpoint_data")
    os.makedirs(export_dir, exist_ok=True)
    super_point_dict = {}
    failed_lists = []
    for data_file in tqdm(data_file_list, desc="Processing RPdiff data"):
        # try:
        data = np.load(os.path.join(data_dir, data_file), allow_pickle=True)
        parent_pcd_s, child_pcd_s = parse_child_parent(data["multi_obj_start_pcd"])
        parent_normal_s, child_normal_s = parse_child_parent(data["normals"])
        parent_color_s, child_color_s = parse_child_parent(data["colors"])
        # Downsample the point cloud
        parent_pcd_s, _, parent_normal_s = downsample_points(parent_pcd_s, None, parent_normal_s, downsample_voxel_size)
        child_pcd_s, _, child_normal_s = downsample_points(child_pcd_s, None, child_normal_s, downsample_voxel_size)

        # check_pcd(parent_pcd_s, parent_color_s, parent_normal_s)
        parent_pose_s, child_pose_s = parse_child_parent(data["multi_obj_start_obj_pose"])
        parent_pose_f, child_pose_f = parse_child_parent(data["multi_obj_final_obj_pose"])
        parent_pose_s_inv = np.linalg.inv(pose7d_to_mat(parent_pose_s[0]))
        child_pose = parent_pose_s_inv @ pose7d_to_mat(child_pose_f[0]) @ np.linalg.inv(pose7d_to_mat(child_pose_s[0]))
        parent_pcd_s = np.dot(parent_pcd_s, parent_pose_s_inv[:3, :3].T) + parent_pose_s_inv[:3, 3]
        child_pcd_s = np.dot(child_pcd_s, child_pose[:3, :3].T) + child_pose[:3, 3]
        parent_normal_s = np.dot(parent_normal_s, parent_pose_s_inv[:3, :3].T)
        child_normal_s = np.dot(child_normal_s, child_pose[:3, :3].T)
        # Sanity check
        if has_outlier(parent_pcd_s) or has_outlier(child_pcd_s):
            raise ValueError("Outliers detected in the point cloud")
        # Process parent_pcd_s
        p_points = np.array(parent_pcd_s)
        c_points = np.array(child_pcd_s)
        p_colors = np.zeros_like(p_points).astype(np.uint8)
        c_colors = np.zeros_like(c_points).astype(np.uint8)
        p_normals = np.array(parent_normal_s)
        c_normals = np.array(child_normal_s)

        # # Check
        # p_pcd = o3d.geometry.PointCloud()
        # p_pcd.points = o3d.utility.Vector3dVector(p_points)
        # c_pcd = o3d.geometry.PointCloud()
        # c_pcd.points = o3d.utility.Vector3dVector(c_points)
        # c_pcd.paint_uniform_color([0, 0, 1])
        # o3d.visualization.draw_geometries([p_pcd, c_pcd])

        p_super_point_data = spt.gen_superpoint(p_points, p_colors, p_normals, scale=scale, vis=True)
        c_super_point_data = spt.gen_superpoint(c_points, c_colors, c_normals, scale=scale, vis=False)

        check_pcd(p_super_point_data["pos"], p_super_point_data["color"], p_super_point_data["normal"])

        super_point_dict[data_file] = {
            "parent": p_super_point_data,
            "child": c_super_point_data,
        }
        # except Exception as e:
        #     # [DEBUG]
        #     # data = np.load(os.path.join(data_dir, data_file), allow_pickle=True)
        #     # parent_pcd_s, child_pcd_s = parse_child_parent(data["multi_obj_start_pcd"])
        #     # child_pcd_o3d = o3d.geometry.PointCloud()
        #     # child_pcd_o3d.points = o3d.utility.Vector3dVector(child_pcd_s)
        #     # o3d.visualization.draw_geometries([child_pcd_o3d])
        #     # parent_pcd_o3d = o3d.geometry.PointCloud()
        #     # parent_pcd_o3d.points = o3d.utility.Vector3dVector(parent_pcd_s)
        #     # o3d.visualization.draw_geometries([parent_pcd_o3d])
        #     log.error(f"Error processing {data_file}: {e}")
        #     super_point_dict[data_file] = []
        #     failed_lists.append(data_file)
        #     continue

    # Save the super_point_dict
    with open(os.path.join(export_dir, "superpoint_dict.pkl"), "wb") as f:
        pickle.dump(super_point_dict, f)

    # Do train, val, test split
    valid_data_file_list = [f for f in data_file_list if f not in failed_lists]
    random.shuffle(valid_data_file_list)
    train_data_file_list = valid_data_file_list[: int(0.7 * len(valid_data_file_list))]
    val_data_file_list = valid_data_file_list[int(0.7 * len(valid_data_file_list)) : int(0.85 * len(valid_data_file_list))]
    test_data_file_list = valid_data_file_list[int(0.85 * len(valid_data_file_list)) :]

    # Save in train_split.txt, train_val_split.txt, test_split.txt
    split_info_dir = os.path.join(data_dir, "split_info")
    os.makedirs(split_info_dir, exist_ok=True)
    with open(os.path.join(split_info_dir, "train_split.txt"), "w") as f:
        f.write("\n".join(train_data_file_list))
    with open(os.path.join(split_info_dir, "train_val_split.txt"), "w") as f:
        f.write("\n".join(val_data_file_list))
    with open(os.path.join(split_info_dir, "test_split.txt"), "w") as f:
        f.write("\n".join(test_data_file_list))
