"""Dataset definition for point cloud like data.
Code refered from Mask3D: https://github.com/JonasSchult/Mask3D/blob/11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5/datasets/semseg.py#L486
"""

from __future__ import annotations
from typing import Optional
from torch.utils.data import Dataset
import numpy as np
import scipy
import random

# import volumentations as V
import yaml
from pathlib import Path
import pickle
from copy import deepcopy
from random import random, sample, uniform
import dps.utils.misc_utils as utils
import dps.utils.pcd_utils as pcd_utils
from scipy.spatial.transform import Rotation as R
from yaml import load
from box import Box
import copy
import open3d as o3d
from sklearn.neighbors import NearestNeighbors


class PcdPairDataset(Dataset):
    """Dataset definition for point cloud like data."""

    def __init__(
        self,
        data_file_list: list[str],
        dataset_name: str,
        indices: list = None,
        color_mean_std: str = "color_mean_std.yaml",
        add_colors: bool = False,
        add_normals: bool = False,
        volume_augmentations_path: Optional[str] = None,
        image_augmentations_path: Optional[str] = None,
        is_elastic_distortion: bool = False,
        elastic_distortion_granularity: float = 1.0,
        elastic_distortion_magnitude: float = 1.0,
        is_random_distortion: bool = False,
        random_distortion_rate: float = 0.2,
        random_distortion_mag: float = 0.01,
        crop_pcd: bool = False,
        crop_size: float = 0.2,
        crop_noise: float = 0.1,
        crop_strategy: str = "knn",
        random_crop_prob: float = 0.5,
        rot_noise_level: float = 0.1,
        trans_noise_level: float = 0.1,
        rot_axis: str = "xy",
        corr_radius: float = 0.1,
        use_shape_complete: bool = True,
        **kwargs,
    ):
        # Set parameters
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.is_elastic_distortion = is_elastic_distortion
        self.elastic_distortion_granularity = elastic_distortion_granularity
        self.elastic_distortion_magnitude = elastic_distortion_magnitude
        self.is_random_distortion = is_random_distortion
        self.random_distortion_rate = random_distortion_rate
        self.random_distortion_mag = random_distortion_mag
        self.crop_pcd = crop_pcd
        self.crop_size = crop_size
        self.crop_noise = crop_noise
        self.crop_strategy = crop_strategy
        self.knn_k = kwargs.get("knn_k", 20)
        self.random_crop_prob = random_crop_prob
        self.rot_noise_level = rot_noise_level
        self.trans_noise_level = trans_noise_level
        if volume_augmentations_path is not None:
            self.vas = Box(yaml.load(open(volume_augmentations_path, "r"), Loader=yaml.FullLoader))
        else:
            self.vas = None
        self.rot_axis = rot_axis
        self.corr_radius = corr_radius
        self.use_shape_complete = use_shape_complete
        # Load data
        data_list = []
        for data_file in data_file_list:
            raw_data = pickle.load(open(data_file, "rb"))  # A list of (coordinates, color, normals, labels, pose)
            if indices is not None:
                data_list += [raw_data[i] for i in indices]
            else:
                data_list += raw_data
        self._data = data_list
        self.mode = "train"  # train, val, test

    def set_mode(self, mode: str):
        self.mode = mode

    def augment_pcd_instance(self, coordinate, normal, color, label, pose, disable_rot: bool = False, noise_scale: float = 1.0):
        """Augment a single point cloud instance."""
        if self.is_elastic_distortion:
            coordinate = elastic_distortion(coordinate, self.elastic_distortion_granularity, self.elastic_distortion_magnitude)
        if self.is_random_distortion:
            coordinate, color, normal, label = random_around_points(coordinate, color, normal, label, rate=self.random_distortion_rate, noise_level=self.random_distortion_mag)

        if self.vas is not None:
            if "rotation" in self.vas.keys() and not disable_rot:
                max_angle = np.pi * self.rot_noise_level * noise_scale
                if random() < self.vas.rotation.prob:
                    angle = np.random.uniform(-max_angle, max_angle)
                    coordinate, normal, color, pose = rotate_around_axis(coordinate=coordinate, normal=normal, pose=pose, color=color, axis=np.random.rand(3), angle=angle, center_point=None)
            if "translation" in self.vas.keys():
                trans_noise_level = noise_scale * self.trans_noise_level
                if random() < self.vas.translation.prob:
                    random_offset = np.random.rand(1, 3)
                    random_offset[0, 0] = np.random.uniform(self.vas.translation.min_x * trans_noise_level, self.vas.translation.max_x * trans_noise_level, size=(1,))
                    random_offset[0, 1] = np.random.uniform(self.vas.translation.min_y * trans_noise_level, self.vas.translation.max_y * trans_noise_level, size=(1,))
                    random_offset[0, 2] = np.random.uniform(self.vas.translation.min_z * trans_noise_level, self.vas.translation.max_z * trans_noise_level, size=(1,))
                    coordinate, normal, color, pose = random_translation(coordinate=coordinate, normal=normal, pose=pose, color=color, offset_type="given", offset=random_offset)
            if "segment_drop" in self.vas.keys():
                if random() < self.vas.segment_drop.prob:
                    coordinate, normal, color, label = random_segment_drop(coordinate=coordinate, normal=normal, color=color, label=label)

        return coordinate, normal, color, label, pose

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        idx = idx % len(self._data)
        # Parse data from the dataset & Convert to float32
        target_coord = self._data[idx]["target_coord"].astype(np.float32)
        target_feat = self._data[idx]["target_feat"].astype(np.float32)
        target_super_index = self._data[idx]["target_super_index"]
        target_label = self._data[idx]["target_label"]
        anchor_coord = self._data[idx]["anchor_coord"].astype(np.float32)
        anchor_feat = self._data[idx]["anchor_feat"].astype(np.float32)
        anchor_super_index = self._data[idx]["anchor_super_index"]
        anchor_label = self._data[idx]["anchor_label"]
        target_pose = self._data[idx].get("pose", np.eye(4))
        anchor_pose = np.eye(4)
        if self.add_normals:
            target_normal = self._data[idx]["target_normal"].astype(np.float32)
            anchor_normal = self._data[idx]["anchor_normal"].astype(np.float32)
        else:
            target_normal = None
            anchor_normal = None
        if self.add_colors:
            target_color = self._data[idx]["target_color"].astype(np.float32)
            anchor_color = self._data[idx]["anchor_color"].astype(np.float32)
        else:
            target_color = None
            anchor_color = None
        if self.use_shape_complete:
            # Compute convex hull of target object
            target_coord, target_normal = pcd_utils.complete_shape(target_coord, strategy="bbox", vis=False)
            target_feat = np.zeros((target_coord.shape[0], target_feat.shape[1]), dtype=np.float32)

        # Augment data
        if self.mode == "train" or self.mode == "val":
            noise_scale = 1.0
            target_coord, target_normal, target_color, target_label, target_pose = self.augment_pcd_instance(
                target_coord, target_normal, target_color, target_label, target_pose, noise_scale=noise_scale
            )
            anchor_coord, anchor_normal, anchor_color, anchor_label, anchor_pose = self.augment_pcd_instance(
                anchor_coord, anchor_normal, anchor_color, anchor_label, anchor_pose, disable_rot=True, noise_scale=noise_scale
            )
        else:
            noise_scale = 1.0

        # Crop pcd to focus around the goal
        is_valid_crop = True
        if self.crop_pcd:
            # Crop around the target pose
            target_coord_goal = (target_pose[:3, :3] @ target_coord.T).T + target_pose[:3, 3]
            crop_indicator = random()
            x_min, y_min, z_min = anchor_coord.min(axis=0)
            x_max, y_max, z_max = anchor_coord.max(axis=0)
            # Apply crop around the target pose
            max_crop_attempts = 10
            for i in range(max_crop_attempts):
                anchor_nearby = anchor_coord[anchor_label >= 0.0]
                anchor_nearby_size = (np.max(anchor_nearby, axis=0) - np.min(anchor_nearby, axis=0)) / 2.0
                if crop_indicator > self.random_crop_prob:
                    crop_center = anchor_nearby.mean(axis=0) + (np.random.rand(3) * 2 * self.crop_noise - self.crop_noise)
                    crop_size = (0.7 + 0.6 * np.random.rand(1)) * anchor_nearby_size
                    # crop_size = anchor_nearby_size
                    is_valid_crop = True
                else:
                    crop_center = np.random.rand(3) * (np.array([x_max, y_max, z_max]) - np.array([x_min, y_min, z_min])) + np.array([x_min, y_min, z_min])
                    if np.linalg.norm(crop_center - target_pose[:3, 3]) < self.crop_noise * np.sqrt(3):
                        is_valid_crop = True
                    else:
                        is_valid_crop = False
                    crop_size = (0.4 + 0.8 * np.random.rand(1)) * anchor_nearby_size
                anchor_indices = PcdPairDataset.crop(pcd=anchor_coord, crop_center=crop_center, crop_strategy=self.crop_strategy, crop_size=crop_size, knn_k=self.knn_k, ref_points=target_coord_goal)
                if anchor_indices.sum() > 20:  # Make sure there are at least 20 points in the crop
                    break
                if i == max_crop_attempts - 1:
                    print("Warning: Failed to find a crop")
                    anchor_indices = np.arange(len(anchor_coord))

            anchor_coord = anchor_coord[anchor_indices]
            if self.add_normals:
                anchor_normal = anchor_normal[anchor_indices]
            if self.add_colors:
                anchor_color = anchor_color[anchor_indices]
            anchor_label = anchor_label[anchor_indices]
            anchor_super_index = anchor_super_index[anchor_indices] - anchor_super_index[anchor_indices].min()
            anchor_feat = anchor_feat[anchor_indices]

        # Move anchor & origin to center
        anchor_shift = np.eye(4)
        anchor_center = anchor_coord.mean(axis=0)
        anchor_shift[:3, 3] = anchor_center
        anchor_pose = anchor_pose @ anchor_shift
        anchor_coord -= anchor_center
        target_shift = np.eye(4)
        target_center = target_coord.mean(axis=0)
        target_shift[:3, 3] = target_center
        target_pose = target_pose @ target_shift
        target_coord -= target_center

        # Apply translation disturbance after cropping
        if self.mode == "train" or self.mode == "val":
            x_min, y_min, z_min = target_coord.min(axis=0)
            x_max, y_max, z_max = target_coord.max(axis=0)
            x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min
            anchor_disturbance = (np.random.rand(3) * 2 - 1.0) * self.trans_noise_level * np.array([x_range, y_range, z_range])
            target_disturbance = (np.random.rand(3) * 2 - 1.0) * self.trans_noise_level * np.array([x_range, y_range, z_range])
            anchor_shift = np.eye(4)
            anchor_shift[:3, 3] = -anchor_disturbance
            anchor_pose = anchor_pose @ anchor_shift
            target_shift = np.eye(4)
            target_shift[:3, 3] = -target_disturbance
            target_pose = target_pose @ target_shift
            anchor_coord += anchor_disturbance
            target_coord += target_disturbance

        target_pose = utils.mat_to_pose9d(np.linalg.inv(anchor_pose) @ target_pose, rot_axis=self.rot_axis)
        anchor_pose = utils.mat_to_pose9d(anchor_pose, rot_axis=self.rot_axis)

        # Concat feat
        target_feat_list = []
        anchor_feat_list = []
        if self.add_colors:
            target_feat_list.append(target_color)
            anchor_feat_list.append(anchor_color)
        target_feat_list.append(target_feat)
        anchor_feat_list.append(anchor_feat)
        target_feat = np.concatenate(target_feat_list, axis=1)
        anchor_feat = np.concatenate(anchor_feat_list, axis=1)

        # DEBUG: sanity check
        if anchor_coord.shape[0] == 0 or np.max(np.abs(anchor_coord)) == 0:
            print("Fixed coord is zero")
            # After crop this becomes empty
            vis_list = []
            anchor_pcd = o3d.geometry.PointCloud()
            anchor_pcd.points = o3d.utility.Vector3dVector(anchor_coord)
            vis_list.append(anchor_pcd)
            # Add a sphere to visualize the crop center
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.crop_size, resolution=20)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color([0.1, 0.1, 0.7])
            sphere.translate(crop_center)
            vis_list.append(sphere)
            o3d.visualization.draw_geometries(vis_list)

        if target_coord.shape[0] == 0 or np.max(np.abs(target_coord)) == 0:
            print("Target coord is zero")

        # Compute corr
        target_pose_mat = utils.pose9d_to_mat(target_pose, rot_axis=self.rot_axis)
        target_coord_goal = (target_pose_mat[:3, :3] @ target_coord.T).T + target_pose_mat[:3, 3]
        corr = pcd_utils.compute_corr_radius(target_coord_goal, anchor_coord, radius=self.corr_radius)

        # # # # DEBUG: check corr
        # pcd = o3d.geometry.PointCloud()
        # target_coord_goal = target_coord
        # combined_coord = np.concatenate([target_coord_goal, anchor_coord], axis=0)
        # combined_color = np.zeros((combined_coord.shape[0], 3))
        # combined_color[: target_coord_goal.shape[0], 0] = 1
        # combined_color[target_coord_goal.shape[0] :, 2] = 1
        # pcd.points = o3d.utility.Vector3dVector(combined_coord)
        # pcd.colors = o3d.utility.Vector3dVector(combined_color)
        # lines = np.concatenate([corr[:, 0:1], corr[:, 1:2] + target_coord_goal.shape[0]], axis=1)
        # line_set = o3d.geometry.LineSet()
        # line_set.points = o3d.utility.Vector3dVector(combined_coord)
        # line_set.lines = o3d.utility.Vector2iVector(lines)
        # line_set.paint_uniform_color([1, 0, 1])
        # # line sphere of corr_radius
        # sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.corr_radius, resolution=20)
        # o3d.visualization.draw_geometries([pcd, line_set, sphere])
        # Return
        return {
            "target_coord": target_coord.astype(np.float32),
            "target_normal": target_normal.astype(np.float32) if target_normal is not None else None,
            "target_feat": target_feat.astype(np.float32),
            "target_label": target_label,
            "target_super_index": target_super_index,
            "target_pose": target_pose.astype(np.float32),
            "anchor_coord": anchor_coord.astype(np.float32),
            "anchor_normal": anchor_normal.astype(np.float32) if anchor_normal is not None else None,
            "anchor_feat": anchor_feat.astype(np.float32),
            "anchor_label": anchor_label,
            "anchor_super_index": anchor_super_index,
            "corr": corr,
            "anchor_pose": anchor_pose.astype(np.float32),
            "is_valid_crop": np.array([is_valid_crop]).astype(np.int64),
        }

    @staticmethod
    def _load_yaml(filepath):
        with open(filepath) as f:
            # file = yaml.load(f, Loader=Loader)
            file = yaml.load(f)
        return file

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data

    @staticmethod
    def crop(pcd, crop_center, crop_strategy, **kwargs):
        """Crop point cloud to a given size around a given center."""
        if crop_strategy == "bbox":
            crop_size = kwargs.get("crop_size", 0.2)
            if isinstance(crop_size, (int, float)):
                crop_size = np.array([crop_size, crop_size, crop_size])
            x_min, y_min, z_min = crop_center - crop_size
            x_max, y_max, z_max = crop_center + crop_size
            return crop_bbox(pcd, x_min, y_min, z_min, x_max, y_max, z_max)
        elif crop_strategy == "radius":
            crop_size = kwargs.get("crop_size", 0.2)
            return crop_radius(pcd, crop_center, crop_size)
        elif crop_strategy == "knn":
            knn_k = kwargs.get("knn_k", 20)
            ref_points = kwargs.get("ref_points", pcd)
            return crop_knn(pcd, ref_points, crop_center, k=knn_k)
        elif crop_strategy == "knn_bbox":
            # Get knn first, and then crop the bbox
            knn_k = kwargs.get("knn_k", 20)
            ref_points = kwargs.get("ref_points", pcd)
            knn_indices = crop_knn(pcd, ref_points, crop_center, k=knn_k)
            knn_pcd = pcd[knn_indices]
            x_min, y_min, z_min = knn_pcd.min(axis=0)
            x_max, y_max, z_max = knn_pcd.max(axis=0)
            return crop_bbox(pcd, x_min, y_min, z_min, x_max, y_max, z_max)
        elif crop_strategy == "knn_bbox_max":
            # Get knn first, and then crop the max bbox
            knn_k = kwargs.get("knn_k", 20)
            ref_points = kwargs.get("ref_points", pcd)
            knn_indices = crop_knn(pcd, ref_points, crop_center, k=knn_k)
            knn_pcd = pcd[knn_indices]
            x_min, y_min, z_min = knn_pcd.min(axis=0)
            x_max, y_max, z_max = knn_pcd.max(axis=0)
            bbox_size = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
            x_min, y_min, z_min = crop_center - bbox_size / 2
            x_max, y_max, z_max = crop_center + bbox_size / 2
            return crop_bbox(pcd, x_min, y_min, z_min, x_max, y_max, z_max)
        else:
            raise ValueError("Invalid crop strategy")


def elastic_distortion(pointcloud, granularity, magnitude):
    """Apply elastic distortion on sparse coordinate space.

    pointcloud: numpy array of (number of points, at least 3 spatial dims)
    granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
    magnitude: noise multiplier
    """
    blurx = np.ones((3, 1, 1, 1)).astype("float32") / 3
    blury = np.ones((1, 3, 1, 1)).astype("float32") / 3
    blurz = np.ones((1, 1, 3, 1)).astype("float32") / 3
    coords = pointcloud[:, :3]
    coords_min = coords.min(0)

    # Create Gaussian noise tensor of the size given by granularity.
    noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
    noise = np.random.randn(*noise_dim, 3).astype(np.float32)

    # Smoothing.
    for _ in range(2):
        noise = scipy.ndimage.filters.convolve(noise, blurx, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blury, mode="constant", cval=0)
        noise = scipy.ndimage.filters.convolve(noise, blurz, mode="constant", cval=0)

    # Trilinear interpolate noise filters for each spatial dimensions.
    ax = [np.linspace(d_min, d_max, d) for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity * (noise_dim - 2), noise_dim)]
    interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=0, fill_value=0)
    pointcloud[:, :3] = coords + interp(coords) * magnitude
    return pointcloud


def crop_bbox(points, x_min, y_min, z_min, x_max, y_max, z_max):
    if x_max <= x_min or y_max <= y_min or z_max <= z_min:
        raise ValueError(
            "We should have x_min < x_max and y_min < y_max and z_min < z_max. But we got"
            " (x_min = {x_min}, y_min = {y_min}, z_min = {z_min},"
            " x_max = {x_max}, y_max = {y_max}, z_max = {z_max})".format(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, z_min=z_min, z_max=z_max)
        )
    inds = np.all([(points[:, 0] >= x_min), (points[:, 0] < x_max), (points[:, 1] >= y_min), (points[:, 1] < y_max), (points[:, 2] >= z_min), (points[:, 2] < z_max)], axis=0)
    return inds


def crop_radius(points, center, radius):
    inds = np.linalg.norm(points - center, axis=1) < radius
    return inds


def crop_knn(points, ref_points, crop_center, k=20):
    if points.shape[0] < k:
        raise ValueError("The number of points should be larger than k")
    points_shifted = points - crop_center
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(points_shifted)
    indices = neigh.kneighbors(ref_points, return_distance=False)
    return indices.flatten()


def random_around_points(coordinates, color, normals, labels, rate=0.2, noise_level=0.01, ignore_label=255):
    # coordinate
    coord_indexes = sample(list(range(len(coordinates))), k=int(len(coordinates) * rate))
    coordinate_noises = np.random.rand(len(coord_indexes), 3) * 2 - 1
    coordinates[coord_indexes] += coordinate_noises * noise_level

    # normals
    if normals is not None:
        normal_noises = np.random.rand(len(coord_indexes), 3) * 2 - 1
        normals[coord_indexes] += normal_noises * noise_level
    return coordinates, color, normals, labels


def random_on_pose(pose, noise_level=0.01):
    tran_noise = (np.random.rand(3) * 2 - 1) * noise_level
    rot_vector = np.random.rand(3) * 2 - 1
    rot_vector = rot_vector / np.linalg.norm(rot_vector)
    rot_angle = (np.random.rand(1) * 2 - 1) * np.pi / 2 * noise_level
    pose[:3, 3] += tran_noise
    pose[:3, :3] = R.from_rotvec(rot_vector * rot_angle).as_matrix() @ pose[:3, :3]
    return pose


def rotate_around_axis(coordinate, normal, color, pose, axis, angle, center_point=None):
    axis = axis / np.linalg.norm(axis)
    rotation_matrix = R.from_rotvec(axis * angle).as_matrix()

    transformed_points = (rotation_matrix @ coordinate.T).T
    if normal is not None:
        transformed_normals = (rotation_matrix @ normal.T).T
    else:
        transformed_normals = None
    pose_transform = np.eye(4)
    pose_transform[:3, :3] = np.linalg.inv(rotation_matrix)
    pose = pose @ pose_transform
    return transformed_points, transformed_normals, color, pose


def random_translation(coordinate, normal, color, pose, offset_type: str = "given", offset=None):
    """
    Return the translated coordinates, normals and the pose
    """
    if offset_type == "center":
        offset = coordinate.mean(axis=0).astype(coordinate.dtype)
        offset = -offset
    else:
        assert offset is not None

    transformed_points = coordinate + offset
    transformed_normals = normal

    pose_transform = np.eye(4)
    pose_transform[:3, 3] = -offset
    pose = pose @ pose_transform
    return transformed_points, transformed_normals, color, pose


def random_segment_drop(coordinate, normal=None, color=None, label=None):
    # Boundary
    x_min, x_max = coordinate[:, 0].min(), coordinate[:, 0].max()
    y_min, y_max = coordinate[:, 1].min(), coordinate[:, 1].max()
    z_min, z_max = coordinate[:, 2].min(), coordinate[:, 2].max()
    drop_center = np.array([uniform(x_min, x_max), uniform(y_min, y_max), uniform(z_min, z_max)])

    total_points = len(coordinate)
    half_points = total_points * 0.5
    radius = uniform(0.2, 0.4) * min(x_max - x_min, y_max - y_min, z_max - z_min)
    distances = np.linalg.norm(coordinate - drop_center, axis=1)
    # Count points inside the sphere
    inside_count = np.sum(distances < radius)
    if inside_count <= half_points:
        mask = distances >= radius  # keep mask
        if mask.sum() == 0:
            return coordinate, normal, color, label
        if mask.sum() < half_points:
            mask = distances < radius
        new_points = coordinate[mask]
        # Duplicate points to make up for the dropped points
        # print("Dropped points: ", total_points - len(new_points))
        num_points_to_add = total_points - len(new_points)
        indices_to_duplicate = np.random.choice(len(new_points), num_points_to_add)
        duplicated_points = new_points[indices_to_duplicate]
        coordinate = np.concatenate((new_points, duplicated_points))
        if normal is not None:
            new_normals = normal[mask]
            duplicated_normals = new_normals[indices_to_duplicate]
            normal = np.concatenate((new_normals, duplicated_normals))
        if color is not None:
            new_colors = color[mask]
            duplicated_colors = new_colors[indices_to_duplicate]
            color = np.concatenate((new_colors, duplicated_colors))
        if label is not None:
            new_labels = label[mask]
            duplicated_labels = new_labels[indices_to_duplicate]
            label = np.concatenate((new_labels, duplicated_labels))
    return coordinate, normal, color, label


if __name__ == "__main__":
    import os
    from detectron2.config import LazyConfig

    dataset_name = "data_rdiff"
    split = "test"
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    task_name = "book_in_bookshelf"  # "stack_can_in_cabinet, book_in_bookshelf, mug_on_rack_multi"
    cfg_file = os.path.join(root_path, "config", f"pose_transformer_rpdiff_{task_name}.py")
    cfg = LazyConfig.load(cfg_file)

    # Test data loader
    pcd_size = cfg.MODEL.PCD_SIZE
    is_elastic_distortion = cfg.DATALOADER.AUGMENTATION.IS_ELASTIC_DISTORTION
    elastic_distortion_granularity = cfg.DATALOADER.AUGMENTATION.ELASTIC_DISTORTION_GRANULARITY
    elastic_distortion_magnitude = cfg.DATALOADER.AUGMENTATION.ELASTIC_DISTORTION_MAGNITUDE
    is_random_distortion = cfg.DATALOADER.AUGMENTATION.IS_RANDOM_DISTORTION
    random_distortion_rate = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_RATE
    random_distortion_mag = cfg.DATALOADER.AUGMENTATION.RANDOM_DISTORTION_MAG
    volume_augmentation_file = cfg.DATALOADER.AUGMENTATION.VOLUME_AUGMENTATION_FILE
    crop_pcd = cfg.DATALOADER.AUGMENTATION.CROP_PCD
    crop_size = cfg.DATALOADER.AUGMENTATION.CROP_SIZE
    crop_strategy = cfg.DATALOADER.AUGMENTATION.CROP_STRATEGY
    random_crop_prob = cfg.DATALOADER.AUGMENTATION.RANDOM_CROP_PROB
    crop_noise = cfg.DATALOADER.AUGMENTATION.CROP_NOISE
    rot_noise_level = cfg.DATALOADER.AUGMENTATION.ROT_NOISE_LEVEL
    trans_noise_level = cfg.DATALOADER.AUGMENTATION.TRANS_NOISE_LEVEL
    rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
    knn_k = cfg.DATALOADER.AUGMENTATION.KNN_K
    add_normals = cfg.DATALOADER.ADD_NORMALS
    add_colors = cfg.DATALOADER.ADD_COLORS
    corr_radius = cfg.DATALOADER.CORR_RADIUS

    # Load dataset & data loader
    if cfg.ENV.GOAL_TYPE == "multimodal":
        dataset_folder = "data_multimodal"
    elif "real" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "data_real"
    elif "struct" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "data_struct"
    elif "rpdiff" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "data_rpdiff"
    elif "superpoint" in cfg.ENV.GOAL_TYPE:
        dataset_folder = "data_superpoint"
    else:
        dataset_folder = "data_faster"

    # Get different split
    splits = ["train", "val", "test"]
    data_file_dict = {}
    for split in splits:
        data_file_dict[split] = os.path.join(root_path, "test_data", dataset_folder, task_name, f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_{split}.pkl")
    print("Data loaded from: ", data_file_dict)

    # Override config
    crop_pcd = True
    add_normals = True
    volume_augmentations_path = os.path.join(root_path, "config", volume_augmentation_file) if volume_augmentation_file is not None else None
    dataset = PcdPairDataset(
        data_file_list=[data_file_dict["train"]],
        dataset_name="data",
        add_colors=add_colors,
        add_normals=add_normals,
        is_elastic_distortion=is_elastic_distortion,
        elastic_distortion_granularity=elastic_distortion_granularity,
        elastic_distortion_magnitude=elastic_distortion_magnitude,
        is_random_distortion=is_random_distortion,
        random_distortion_rate=random_distortion_rate,
        random_distortion_mag=random_distortion_mag,
        volume_augmentations_path=volume_augmentations_path,
        crop_pcd=crop_pcd,
        crop_size=crop_size,
        crop_noise=crop_noise,
        crop_strategy=crop_strategy,
        random_crop_prob=random_crop_prob,
        rot_noise_level=rot_noise_level,
        trans_noise_level=trans_noise_level,
        rot_axis=rot_axis,
        knn_k=knn_k,
        corr_radius=corr_radius,
    )
    dataset.set_mode("train")

    # Test data augmentation
    for i in range(20):
        # random_idx = np.random.randint(0, len(dataset))
        random_idx = i
        data = dataset[random_idx]
        target_coord = data["target_coord"]
        anchor_coord = data["anchor_coord"]
        anchor_normal = data["anchor_normal"]
        target_pose = data["target_pose"]
        target_feat = data["target_feat"]
        target_normal = data["target_normal"]
        anchor_feat = data["anchor_feat"]
        anchor_label = data["anchor_label"]
        target_label = data["target_label"]
        is_valid_crop = data["is_valid_crop"]
        print(f"Number of target points: {len(target_coord)}, Number of fixed points: {len(anchor_coord)}, Crop valid: {is_valid_crop}")

        target_color = np.zeros_like(target_coord)
        # target_color[np.where(target_label == 1)[0], 0] = 1
        target_color[:, 0] = 1.0
        # anchor_color = anchor_label[:, None] * np.array([[0, 1, 0]]) + (1 - anchor_label[:, None]) * np.array([[1, 0, 0]])
        anchor_color = np.zeros_like(anchor_coord)
        anchor_color[:, 2] = 1.0

        target_pose_mat = utils.pose9d_to_mat(target_pose, rot_axis=cfg.DATALOADER.AUGMENTATION.ROT_AXIS)

        utils.visualize_pcd_list(
            coordinate_list=[target_coord, anchor_coord],
            normal_list=[target_normal, anchor_normal],
            color_list=[target_color, anchor_color],
            pose_list=[np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
        )
        utils.visualize_pcd_list(
            coordinate_list=[target_coord, anchor_coord],
            normal_list=[target_normal, anchor_normal],
            color_list=[target_color, anchor_color],
            pose_list=[target_pose_mat, np.eye(4, dtype=np.float32)],
        )
