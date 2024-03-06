"""Generate superpoint for anet3d dataset."""

from dps.data.superpoint_tool import SuperPointTool, has_outlier, downsample_points, visualize_superpoint, check_pcd
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
    # Load anet3d data
    data_root = "/home/harvey/Data"
    dataset_name = "Anet3D"
    dataset_type = "full_shape"
    splits = ["train", "val"]  # "train", "val"
    export_dir = os.path.join(os.path.join(data_root), dataset_name, "superpoint_data")
    os.makedirs(export_dir, exist_ok=True)
    for split in splits:
        data_file = os.path.join(data_root, dataset_name, f"{dataset_type}_{split}_data.pkl")
        super_point_dict = {}
        data_list = pickle.load(open(data_file, "rb"))
        # Infos
        affordances = list(data_list[0]["full_shape"]["label"].keys())
        affordances = sorted(affordances)
        valid_data_idx = 0
        all_data_idx = 0
        for data in tqdm(data_list, desc="Processing Anet3d data"):
            try:
                pcd = data["full_shape"]["coordinate"]
                affordance_dict = data["full_shape"]["label"]
                label = np.zeros([pcd.shape[0], len(affordances)], dtype=np.float32)
                for i, affordance in enumerate(affordances):
                    if affordance in affordance_dict:
                        label[:, i] = affordance_dict[affordance].squeeze()
                # Compute normals using open3d
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
                pcd_o3d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
                normal = np.array(pcd_o3d.normals)
                # Downsample the point cloud
                pcd, _, normal = downsample_points(pcd, None, normal, downsample_voxel_size)

                # check_pcd(pcd, None, normal)
                # Process parent_pcd_s
                points = np.array(pcd)
                colors = np.zeros_like(points).astype(np.uint8)
                normals = np.array(normal)

                superpoint_data = spt.gen_superpoint(points, colors, normals, scale=scale, vis=False, label=label)
                # visualize_superpoint(superpoint_data)
                # check_pcd(superpoint_data["pos"], superpoint_data["color"], superpoint_data["normal"])
                superpoint_data["semantic"] = data["semantic class"]
                super_point_dict[valid_data_idx] = superpoint_data
                valid_data_idx += 1
                all_data_idx += 1
            except Exception as e:
                log.error(f"Error processing {all_data_idx}: {e}")
                all_data_idx += 1
                continue

        # Save the super_point_dict
        with open(os.path.join(export_dir, f"superpoint_dict_{split}.pkl"), "wb") as f:
            pickle.dump(super_point_dict, f)
