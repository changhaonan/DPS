"""
Compute normal & voxel downsample & Sync the data-format
"""

import json
import os
import numpy as np
import h5py
import open3d as o3d
import dps.utils.misc_utils as utils
import torch
import pickle
from tqdm import tqdm
import copy
from detectron2.config import LazyConfig
import argparse
from scipy.spatial.transform import Rotation as R


def compute_nearby_pcd(pcd1: np.ndarray, pcd2: np.ndarray, radius: float = 0.1):
    """
    Compute the nearby points between two point clouds with radius;
    pcd1 is the target point cloud, pcd2 is the anchor point cloud.
    Return a list of nearby indices and distances.
    """
    dist_matrix = np.linalg.norm(pcd1[:, None, :3] - pcd2[None, :, :3], axis=-1)
    nearby_idx = np.where(dist_matrix <= radius)
    nearby_data = {}
    for i, j in zip(nearby_idx[0], nearby_idx[1]):
        nearby_data[i] = np.min(dist_matrix[i, :])
    return list(nearby_data.keys()), list(nearby_data.values())


def compute_corr_radius(pcd1: np.ndarray, pcd2: np.ndarray, radius: float = 0.1):
    """Compute the correspondence matrix between two point clouds with radius
    Correspondence matrix is a binary matrix, where each row represents the correspondence of a point in pcd1 to pcd2; nearest & within the radius is 1, otherwise 0.
    """
    dist_matrix = np.linalg.norm(pcd1[:, None, :3] - pcd2[None, :, :3], axis=-1)
    corr_1to2 = np.where(dist_matrix <= radius)
    return np.stack([corr_1to2[0], corr_1to2[1]], axis=-1)


def pose7d_to_mat(pose7d):
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R.from_quat(pose7d[3:]).as_matrix()
    pose_mat[:3, 3] = pose7d[:3]
    return pose_mat


def read_hdf5(file_name):
    """Read HDF5 file and return data."""
    with h5py.File(file_name, "r") as file:
        return np.array(file["colors"]), np.array(file["depth"])


def read_scene_hdf5(anchor_hdf5, target_hdf5, intrinsic_file):
    with open(intrinsic_file, "r") as f:
        intrinsic_json = json.load(f)
    intrinsic = np.array(
        [
            [-intrinsic_json["fx"], 0.0, intrinsic_json["cx"]],
            [0.0, intrinsic_json["fy"], intrinsic_json["cy"]],
            [0.0, 0.0, 1.0],
        ]
    )
    target_color, target_depth = read_hdf5(target_hdf5)
    anchor_color, anchor_depth = read_hdf5(anchor_hdf5)
    # Filter depth
    target_depth = target_depth.astype(np.float32)
    target_depth[target_depth > 1000.0] = 0.0
    target_depth = -target_depth
    anchor_depth = anchor_depth.astype(np.float32)
    anchor_depth[anchor_depth > 1000.0] = 0.0
    anchor_depth = -anchor_depth
    return target_color, target_depth, anchor_color, anchor_depth, intrinsic


def normalize_pcd(pcd_anchor, pcd_target, do_scaling: bool = True):
    # Normalize to unit cube
    pcd_center = (pcd_anchor.get_max_bound() + pcd_anchor.get_min_bound()) / 2
    pcd_anchor = pcd_anchor.translate(-pcd_center)
    scale_xyz = np.linalg.norm(pcd_anchor.get_max_bound() - pcd_anchor.get_min_bound())  # Diagonal length
    if not do_scaling:
        scale_xyz = 1.0
    pcd_anchor = pcd_anchor.scale(1 / scale_xyz, center=np.array([0, 0, 0]))

    # Normalize the child point clouds
    pcd_target = pcd_target.translate(-pcd_center)
    normalize_pcd_target = pcd_target.scale(1 / scale_xyz, center=np.array([0, 0, 0]))
    return pcd_anchor, normalize_pcd_target, pcd_center, scale_xyz


def visualize_pcd_with_open3d(
    pcd_with_color1,
    pcd_with_color2,
    transform1: np.ndarray = None,
    shift_transform: np.ndarray = None,
    camera_pose=None,
):
    pcd1 = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()

    pcd1.points = o3d.utility.Vector3dVector(pcd_with_color1[:, :3])
    pcd2.points = o3d.utility.Vector3dVector(pcd_with_color2[:, :3])

    if pcd_with_color1.shape[1] >= 6:
        pcd1.normals = o3d.utility.Vector3dVector(pcd_with_color1[:, 3:6])
        pcd2.normals = o3d.utility.Vector3dVector(pcd_with_color2[:, 3:6])

    if pcd_with_color1.shape[1] >= 9:
        color_scale = 1.0 if pcd_with_color1[:, 6:9].max() <= 1.0 else 255.0
        pcd1.colors = o3d.utility.Vector3dVector(pcd_with_color1[:, 6:9] / color_scale)
        pcd2.colors = o3d.utility.Vector3dVector(pcd_with_color2[:, 6:9] / color_scale)
    else:
        pcd1.paint_uniform_color([0.0, 0.651, 0.929])
        pcd2.paint_uniform_color([1.0, 0.706, 0.0])

    if shift_transform is not None:
        pcd1.transform(shift_transform)
        pcd2.transform(shift_transform)

    if transform1 is not None:
        pcd1.transform(transform1)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    if camera_pose is not None:
        camera_origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
        camera_origin.transform(camera_pose)
        o3d.visualization.draw_geometries([pcd1, pcd2, origin, camera_origin])
    else:
        o3d.visualization.draw_geometries([pcd1, pcd2, origin])


def build_dataset_real(data_path, cfg, data_id: int = 0, vis: bool = False, filter_key: str = None):
    """Build the dataset from the real data"""
    data_file_list = os.listdir(data_path)
    data_file_list = [f for f in data_file_list if f.endswith(".pkl")]
    # Filter by key
    if filter_key is not None:
        data_file_list = [f for f in data_file_list if filter_key in f]
    dtset = []
    pcd_size = cfg.MODEL.PCD_SIZE
    rot_axis = cfg.MODEL.ROT_AXIS
    for data_file in tqdm(data_file_list, desc="Processing data"):
        # Load data
        pcd_dict = pickle.load(open(os.path.join(data_path, data_file), "rb"))
        data_len = len(pcd_dict["object_0"])
        for i in tqdm(range(data_len), desc="Processing frames", leave=False):
            # Assemble data
            target_pcd_arr = pcd_dict["object_0"][i]
            anchor_pcd_arr = pcd_dict["object_1"][i]
            target_label = pcd_dict["object_0_semantic"][i]
            anchor_label = pcd_dict["object_1_semantic"][i]
            if target_pcd_arr.shape[0] < pcd_size or anchor_pcd_arr.shape[0] < pcd_size:
                continue

            data_id = data_id
            # Shift all points to the origin
            target_pcd_center = np.mean(target_pcd_arr[:, :3], axis=0)
            anchor_pcd_center = np.mean(anchor_pcd_arr[:, :3], axis=0)
            target_pcd_arr[:, :3] -= target_pcd_center
            anchor_pcd_arr[:, :3] -= anchor_pcd_center
            shift_transform = np.eye(4, dtype=np.float32)
            shift_transform[:3, 3] = target_pcd_center - anchor_pcd_center
            pose_9d = utils.mat_to_pose9d(shift_transform, rot_axis=rot_axis)
            tmorp_data = {
                "target": target_pcd_arr,
                "fixed": anchor_pcd_arr,
                "target_label": target_label,
                "anchor_label": anchor_label,
                "transform": shift_transform,
                "9dpose": pose_9d,
                "cam_pose": np.eye(4, dtype=np.float32),
                "data_id": data_id,
            }
            # Visualize & Check
            if vis:
                visualize_pcd_with_open3d(tmorp_data["target"], tmorp_data["fixed"], np.eye(4, dtype=np.float32), camera_pose=tmorp_data["cam_pose"])
                visualize_pcd_with_open3d(tmorp_data["target"], tmorp_data["fixed"], tmorp_data["transform"], camera_pose=tmorp_data["cam_pose"])
            dtset.append(tmorp_data)
    print("Len of dtset:", len(dtset))
    print(f"Saving dataset to {os.path.join(root_dir, 'test_data', 'data_real')}...")
    # Save the dtset into a .pkl file
    if filter_key is not None:
        with open(os.path.join(root_dir, "test_data", "data_real", f"diffusion_dataset_{data_id}_{cfg.MODEL.PCD_SIZE}_{cfg.MODEL.DATASET_CONFIG}_{filter_key}.pkl"), "wb") as f:
            pickle.dump(dtset, f)
    else:
        with open(os.path.join(root_dir, "test_data", "data_real", f"diffusion_dataset_{data_id}_{cfg.MODEL.PCD_SIZE}_{cfg.MODEL.DATASET_CONFIG}.pkl"), "wb") as f:
            pickle.dump(dtset, f)
    print("Done!")


def build_dataset_rpdiff(data_dir, cfg, task_name: str, vis: bool = False, do_scaling: bool = True):
    """Build the dataset from the rpdiff data"""
    data_file_list = os.listdir(data_dir)
    data_file_list = [f for f in data_file_list if f.endswith(".npz")]
    grid_size = cfg.PREPROCESS.GRID_SIZE
    rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
    num_point_lower_bound = cfg.PREPROCESS.NUM_POINT_LOW_BOUND
    # Split info
    split_dict = {}
    train_split_info = os.path.join(data_dir, "split_info", "train_split.txt")
    val_split_info = os.path.join(data_dir, "split_info", "train_val_split.txt")
    test_split_info = os.path.join(data_dir, "split_info", "test_split.txt")
    with open(train_split_info, "r") as f:
        train_split_list = f.readlines()
        train_split_info = [x.split("\n")[0] for x in train_split_list]
    with open(val_split_info, "r") as f:
        val_split_list = f.readlines()
        val_split_info = [x.split("\n")[0] for x in val_split_list]
    with open(test_split_info, "r") as f:
        test_split_list = f.readlines()
        test_split_info = [x.split("\n")[0] for x in test_split_list]
    split_dict["train"] = train_split_info
    split_dict["val"] = val_split_info
    split_dict["test"] = test_split_info

    # Read data
    def parse_child_parent(arr):
        pcd_dict = arr[()]
        parent_val = pcd_dict["parent"]
        child_val = pcd_dict["child"]
        return parent_val, child_val

    # Load superpoint info
    superpoint_info_file = os.path.join(data_dir, "superpoint_data", "superpoint_dict.pkl")
    superpoint_info = pickle.load(open(superpoint_info_file, "rb"))

    data_dict = {}
    data_dict["train"] = []
    data_dict["val"] = []
    data_dict["test"] = []
    for data_file in tqdm(data_file_list, desc="Processing data"):
        super_indexes = superpoint_info[data_file]
        if len(super_indexes) == 0:
            print("No superpoint found")
            continue

        data = np.load(os.path.join(data_dir, data_file), allow_pickle=True)
        parent_pcd_s, child_pcd_s = parse_child_parent(data["multi_obj_start_pcd"])
        parent_normal_s, child_normal_s = parse_child_parent(data["normals"])
        parent_pose_s, child_pose_s = parse_child_parent(data["multi_obj_start_obj_pose"])
        _, child_pose_f = parse_child_parent(data["multi_obj_final_obj_pose"])

        if task_name == "stack_can_in_cabinet":
            parent_pose_s = [parent_pose_s]
            child_pose_f = [child_pose_f]
            child_pose_s = [child_pose_s]

        for i in range(len(parent_pose_s)):
            # Transform pose to matrix
            parent_mat_s = pose7d_to_mat(parent_pose_s[i])
            child_mat_s = pose7d_to_mat(child_pose_s[0])
            child_mat_f = pose7d_to_mat(child_pose_f[0])

            if child_pcd_s.shape[0] <= num_point_lower_bound or parent_pcd_s.shape[0] <= num_point_lower_bound:
                print(f"Target pcd has {child_pcd_s.shape[0]} points, fixed pcd has {parent_pcd_s.shape[0]} points")
                continue

            # Filter outliers
            if np.sum(np.linalg.norm(parent_pcd_s, axis=1) >= 3.0) > 1 or np.sum(np.linalg.norm(child_pcd_s, axis=1) >= 3.0) > 1:
                print("Outliers found")
                continue

            # Use superpoint as color; superpoint are only provided for the anchor object
            superpoint_layers = {}
            for _i, super_index in enumerate(super_indexes):
                superpoint_layers[f"superpoint_{_i}"] = super_index
                pass
            # Rescale the target pcd in case that there are not enough points after voxel downsampling
            # Shift all points to the origin
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(child_pcd_s)
            target_pcd.normals = o3d.utility.Vector3dVector(child_normal_s)
            anchor_pcd = o3d.geometry.PointCloud()
            anchor_pcd.points = o3d.utility.Vector3dVector(parent_pcd_s)
            anchor_pcd.normals = o3d.utility.Vector3dVector(parent_normal_s)
            anchor_pcd.transform(np.linalg.inv(parent_mat_s))

            # Sample & Compute normal
            target_pcd.transform(np.linalg.inv(child_mat_s)).transform(child_mat_f).transform(np.linalg.inv(parent_mat_s))

            # anchor_pcd, target_pcd, _, __ = normalize_pcd(anchor_pcd, target_pcd, do_scaling=do_scaling)
            target_pcd, anchor_pcd, _, __ = normalize_pcd(target_pcd, anchor_pcd, do_scaling=do_scaling)  # Normalize to target

            # Compute normal
            target_pcd_center = (target_pcd.get_max_bound() + target_pcd.get_min_bound()) / 2
            target_pcd.translate(-target_pcd_center)

            target_pcd = target_pcd.voxel_down_sample(grid_size)
            anchor_pcd = anchor_pcd.voxel_down_sample(grid_size)
            target_pcd_arr = np.hstack((np.array(target_pcd.points), np.array(target_pcd.normals)))
            anchor_pcd_arr = np.hstack((np.array(anchor_pcd.points), np.array(anchor_pcd.normals)))

            # Move target to center
            target_transform = np.eye(4, dtype=np.float32)
            target_transform[:3, 3] = target_pcd_center

            if vis or target_pcd_arr.shape[0] < (num_point_lower_bound / 2) or anchor_pcd_arr.shape[0] < num_point_lower_bound:
                visualize_pcd_with_open3d(target_pcd_arr, anchor_pcd_arr, np.eye(4, dtype=np.float32))
                visualize_pcd_with_open3d(target_pcd_arr, anchor_pcd_arr, target_transform)
                print(f"Target pcd has {target_pcd_arr.shape[0]} points, fixed pcd has {anchor_pcd_arr.shape[0]} points")
                continue

            # DEBUG: sanity check
            if np.max(np.abs(target_pcd_arr[:, :3])) == 0 or np.max(np.abs(anchor_pcd_arr[:, :3])) == 0:
                print("Zero pcd found")
                # Check raw pcd
                vis_list = []
                target_pcd = o3d.geometry.PointCloud()
                target_pcd.points = o3d.utility.Vector3dVector(child_pcd_s)
                vis_list.append(target_pcd)
                anchor_pcd = o3d.geometry.PointCloud()
                anchor_pcd.points = o3d.utility.Vector3dVector(parent_pcd_s)
                vis_list.append(anchor_pcd)
                o3d.visualization.draw_geometries(vis_list)
                continue

            tmorp_data = {
                "target": target_pcd_arr,
                "fixed": anchor_pcd_arr,
                "target_label": np.array([0]),
                "anchor_label": np.array([1]),
                "super_index": super_indexes,
                "9dpose": utils.mat_to_pose9d(target_transform, rot_axis=rot_axis),
            }
            for split, split_list in split_dict.items():
                if data_file in split_list:
                    data_dict[split].append(tmorp_data)
                    break

    print("Len of dtset:", len(data_dict["train"]), len(data_dict["val"]), len(data_dict["test"]))
    # Save the dtset into a .pkl file
    export_dir = os.path.join(root_dir, "test_data", "data_rpdiff", task_name)
    os.makedirs(export_dir, exist_ok=True)
    for split, split_list in split_dict.items():
        print(f"Saving dataset to {os.path.join(root_dir, 'test_data', 'data_rpdiff')}...")
        with open(os.path.join(export_dir, f"diffusion_dataset_{cfg.MODEL.PCD_SIZE}_{cfg.MODEL.DATASET_CONFIG}_{split}.pkl"), "wb") as f:
            pickle.dump(data_dict[split], f)


def build_dataset_superpoint(data_dir, cfg, task_name: str, vis: bool = False, f_keys: list = ["planarity", "linearity", "verticality", "scattering"]):
    """Build the dataset from the superpoint data"""
    # Split info
    split_dict = {}
    train_split_info = os.path.join(data_dir, "split_info", "train_split.txt")
    val_split_info = os.path.join(data_dir, "split_info", "train_val_split.txt")
    test_split_info = os.path.join(data_dir, "split_info", "test_split.txt")
    with open(train_split_info, "r") as f:
        train_split_list = f.readlines()
        train_split_info = [x.split("\n")[0] for x in train_split_list]
    with open(val_split_info, "r") as f:
        val_split_list = f.readlines()
        val_split_info = [x.split("\n")[0] for x in val_split_list]
    with open(test_split_info, "r") as f:
        test_split_list = f.readlines()
        test_split_info = [x.split("\n")[0] for x in test_split_list]
    split_dict["train"] = train_split_info
    split_dict["val"] = val_split_info
    split_dict["test"] = test_split_info

    superpoint_file = os.path.join(data_dir, "superpoint_data", "superpoint_dict.pkl")
    superpoint_dict = pickle.load(open(superpoint_file, "rb"))

    data_dict = {}
    data_dict["train"] = []
    data_dict["val"] = []
    data_dict["test"] = []
    for data_file, superpoint_data in tqdm(superpoint_dict.items()):
        if data_file not in train_split_info and data_file not in val_split_info and data_file not in test_split_info:
            continue
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
        # Voxel Downsample
        # # FIXME: Downsample method still need to improve as super_index is also averaged
        # grid_size = cfg.PREPROCESS.GRID_SIZE
        # anchor_pcd = o3d.t.geometry.PointCloud()
        # anchor_pcd.point["positions"] = o3d.core.Tensor(anchor_coord, dtype=o3d.core.Dtype.Float32)
        # anchor_pcd.point["normals"] = o3d.core.Tensor(anchor_normal, dtype=o3d.core.Dtype.Float32)
        # anchor_pcd.point["features"] = o3d.core.Tensor(anchor_feat, dtype=o3d.core.Dtype.Float32)
        # anchor_pcd.point["super_index"] = o3d.core.Tensor(anchor_super_index, dtype=o3d.core.Dtype.Int32)
        # anchor_pcd = anchor_pcd.voxel_down_sample(voxel_size=grid_size)
        # anchor_coord = anchor_pcd.point["positions"].numpy()
        # anchor_normal = anchor_pcd.point["normals"].numpy()
        # anchor_feat = anchor_pcd.point["features"].numpy()
        # anchor_super_index = anchor_pcd.point["super_index"].numpy().astype(np.int64)

        # target_pcd = o3d.t.geometry.PointCloud()
        # target_pcd.point["positions"] = o3d.core.Tensor(target_coord, dtype=o3d.core.Dtype.Float32)
        # target_pcd.point["normals"] = o3d.core.Tensor(target_normal, dtype=o3d.core.Dtype.Float32)
        # target_pcd.point["features"] = o3d.core.Tensor(target_feat, dtype=o3d.core.Dtype.Float32)
        # target_pcd.point["super_index"] = o3d.core.Tensor(target_super_index, dtype=o3d.core.Dtype.Int32)
        # target_pcd = target_pcd.voxel_down_sample(voxel_size=grid_size)
        # target_coord = target_pcd.point["positions"].numpy()
        # target_normal = target_pcd.point["normals"].numpy()
        # target_feat = target_pcd.point["features"].numpy()
        # target_super_index = target_pcd.point["super_index"].numpy().astype(np.int64)

        # Compute nearby label
        target_label = -np.ones((target_coord.shape[0],), dtype=np.float32)  # -1: not nearby, 1: nearby
        nearyby_radius = cfg.PREPROCESS.NEARBY_RADIUS
        use_soft_label = cfg.PREPROCESS.USE_SOFT_LABEL
        anchor_nearby_indices, anchor_nearby_distances = compute_nearby_pcd(anchor_coord, target_coord, radius=nearyby_radius)
        anchor_label = -np.ones((anchor_coord.shape[0],), dtype=np.float32)  # -1: not nearby, 1: nearby
        if not use_soft_label:
            anchor_label[anchor_nearby_indices] = 1.0
        else:
            anchor_label[anchor_nearby_indices] = 2.0 * np.exp(-np.array(anchor_nearby_distances) / nearyby_radius) - 1.0
        if vis:
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_coord)
            target_pcd.normals = o3d.utility.Vector3dVector(target_normal)
            anchor_pcd = o3d.geometry.PointCloud()
            anchor_pcd.normals = o3d.utility.Vector3dVector(anchor_normal)
            anchor_pcd.points = o3d.utility.Vector3dVector(anchor_coord)
            # Visualize overall anchor
            anchor_color = np.zeros((anchor_coord.shape[0], 3))
            anchor_color[anchor_nearby_indices, :] = [1, 0, 0]
            anchor_pcd.normals = o3d.utility.Vector3dVector(anchor_normal)
            anchor_pcd.colors = o3d.utility.Vector3dVector(anchor_color)
            o3d.visualization.draw_geometries([target_pcd])
            # Visualize by superpoint average
            superpoint_layer0 = anchor_super_index[:, 0]
            num_superpoint = np.max(superpoint_layer0) + 1
            superpoint_list = []
            for _i in range(num_superpoint):
                print(f"Superpoint {_i} has {np.sum(superpoint_layer0 == _i)} points")
                superpoint_indices = np.where(superpoint_layer0 == _i)[0]
                if len(superpoint_indices) == 0:
                    continue
                superpoint_pcd = o3d.geometry.PointCloud()
                superpoint_pcd.points = o3d.utility.Vector3dVector(anchor_coord[superpoint_indices])
                # superpoint_color = np.mean(anchor_color[superpoint_indices], axis=0)
                # superpoint_pcd.paint_uniform_color(superpoint_color)
                random_color = np.random.uniform(0, 1, size=(3,))
                superpoint_pcd.paint_uniform_color(random_color)
                superpoint_list.append(superpoint_pcd)
            o3d.visualization.draw_geometries(superpoint_list)
        # Compute correspondence matrix
        corr = compute_corr_radius(anchor_coord, target_coord, radius=nearyby_radius)

        tmorp_data = {
            "target_coord": target_coord,
            "target_normal": target_normal,
            "target_feat": target_feat,
            "target_super_index": target_super_index,
            "target_label": target_label,
            "anchor_coord": anchor_coord,
            "anchor_normal": anchor_normal,
            "anchor_feat": anchor_feat,
            "anchor_super_index": anchor_super_index,
            "anchor_label": anchor_label,
            "corr": corr,
        }
        for split, split_list in split_dict.items():
            if data_file in split_list:
                data_dict[split].append(tmorp_data)
                break
    print("Len of dtset:", len(data_dict["train"]), len(data_dict["val"]), len(data_dict["test"]))
    # Save the dtset into a .pkl file
    export_dir = os.path.join(root_dir, "test_data", "data_superpoint", task_name)
    os.makedirs(export_dir, exist_ok=True)
    for split, split_list in split_dict.items():
        print(f"Saving dataset to {os.path.join(root_dir, 'test_data', 'data_superpoint')}...")
        with open(os.path.join(export_dir, f"diffusion_dataset_{cfg.MODEL.PCD_SIZE}_{cfg.MODEL.DATASET_CONFIG}_{split}.pkl"), "wb") as f:
            pickle.dump(data_dict[split], f)


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="book_in_bookshelf", help="stack_can_in_cabinet, book_in_bookshelf, mug_on_rack_multi")
    parser.add_argument("--data_type", type=str, default="superpoint", help="real, rpdiff, superpoint")
    parser.add_argument("--filter_key", type=str, default=None)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    # Prepare path
    data_path_dict = {
        "stack_can_in_cabinet": "/home/harvey/Project/dps/dps/external/rpdiff/data/task_demos/can_in_cabinet_stack/task_name_stack_can_in_cabinet",
        "book_in_bookshelf": "/home/harvey/Data/rpdiff_V3/book_in_bookshelf",
        "mug_on_rack_multi": "/home/harvey/Project/dps/dps/external/rpdiff/data/task_demos/mug_on_rack_multi_large_proc_gen_demos/task_name_mug_on_rack_multi",
    }
    task_name = args.task_name
    data_dir = data_path_dict[task_name]
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_file = os.path.join(root_dir, "config", f"pose_transformer_rpdiff_{task_name}.py")
    cfg = LazyConfig.load(cfg_file)
    cfg.PREPROCESS.USE_SOFT_LABEL = False
    filter_key = args.filter_key
    vis = args.vis
    do_scaling = True
    vis = False

    dtset = []
    if args.data_type == "real":
        data_path = os.path.join(root_dir, "test_data", "data_real", f"{0:06d}")
        build_dataset_real(data_path, cfg, data_id=0, vis=vis, filter_key=filter_key)
    elif args.data_type == "rpdiff":
        build_dataset_rpdiff(data_dir, cfg, task_name=task_name, vis=vis, do_scaling=do_scaling)
    elif args.data_type == "superpoint":
        build_dataset_superpoint(data_dir, cfg, task_name=task_name, vis=vis)
    else:
        raise NotImplementedError
