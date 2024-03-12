from dps.data.superpoint_tool import SuperPointTool, has_outlier, downsample_points, visualize_superpoint, parse_child_parent, pose7d_to_mat
import os
import sys
from tqdm import tqdm
import numpy as np
import pickle
import open3d as o3d
import random
import logging

# Add the project's files to the python path
# file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # for .py script
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(file_path, "external", "superpoint_transformer")
sys.path.append(file_path)

import hydra
from src.utils import init_config

log = logging.getLogger(__name__)
from detectron2.config import LazyConfig

if __name__ == "__main__":
    # Parse task cfg
    task_name = "book_in_bookshelf"
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_cfg_file = os.path.join(root_path, "config", f"pose_transformer_rpdiff_{task_name}.py")
    task_cfg = LazyConfig.load(task_cfg_file)
    scale = task_cfg.PREPROCESS.TARGET_RESCALE
    downsample_voxel_size = task_cfg.PREPROCESS.GRID_SIZE
    # Parse the configs using hydra
    cfg = init_config(
        overrides=[
            "experiment=semantic/scannet.yaml",
            "datamodule.voxel=0.02",
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
        "stack_can_in_cabinet": "/home/harvey/Project/VIL2/vil2/external/rpdiff/data/task_demos/can_in_cabinet_stack/task_name_stack_can_in_cabinet",
        "book_in_bookshelf": "/home/harvey/Data/rpdiff_V3/book_in_bookshelf",
    }
    task_name = "book_in_bookshelf"
    data_dir = data_path_dict[task_name]
    data_file_list = os.listdir(data_dir)
    data_file_list = [f for f in data_file_list if f.endswith(".npz")]
    export_dir = os.path.join(data_dir, "superpoint_data")
    os.makedirs(export_dir, exist_ok=True)
    super_point_dict = {}
    failed_lists = []
    for data_file in tqdm(data_file_list, desc="Processing RPdiff data"):
        try:
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

            p_super_point_data = spt.gen_superpoint(p_points, p_colors, p_normals, scale=scale, vis=False)
            c_super_point_data = spt.gen_superpoint(c_points, c_colors, c_normals, scale=scale, vis=False)

            visualize_superpoint(p_super_point_data)
            super_point_dict[data_file] = {
                "parent": p_super_point_data,
                "child": c_super_point_data,
            }
        except Exception as e:
            # [DEBUG]
            # data = np.load(os.path.join(data_dir, data_file), allow_pickle=True)
            # parent_pcd_s, child_pcd_s = parse_child_parent(data["multi_obj_start_pcd"])
            # child_pcd_o3d = o3d.geometry.PointCloud()
            # child_pcd_o3d.points = o3d.utility.Vector3dVector(child_pcd_s)
            # o3d.visualization.draw_geometries([child_pcd_o3d])
            # parent_pcd_o3d = o3d.geometry.PointCloud()
            # parent_pcd_o3d.points = o3d.utility.Vector3dVector(parent_pcd_s)
            # o3d.visualization.draw_geometries([parent_pcd_o3d])
            log.error(f"Error processing {data_file}: {e}")
            super_point_dict[data_file] = []
            failed_lists.append(data_file)
            continue

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
