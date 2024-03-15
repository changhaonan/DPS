"""Preprocess the real world data"""

import open3d as o3d
import pickle
import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def process_anchor(anchor_pcd, z_max=0.08, z_min=0.01):
    # Remove points from anchor_pcd, z-value < z_min
    # anchor_pcd = anchor_pcd.select_by_index(np.where(np.array(anchor_pcd.points)[:, 2] > z_min)[0])
    # Remove points from anchor_pcd, z-value > z_max
    anchor_pcd = anchor_pcd.select_by_index(np.where(np.array(anchor_pcd.points)[:, 2] < z_max)[0])
    # Remove noises
    anchor_pcd = anchor_pcd.voxel_down_sample(voxel_size=0.005)
    anchor_pcd.remove_radius_outlier(nb_points=20, radius=0.01)
    anchor_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # Return bbox
    return anchor_pcd, anchor_pcd.get_minimal_oriented_bounding_box()


def process_target(target_pcd, z_max=0.2, z_min=0.01):
    # Remove points from target_pcd, z-value < z_min
    target_pcd = target_pcd.select_by_index(np.where(np.array(target_pcd.points)[:, 2] > z_min)[0])
    # Remove points from target_pcd, z-value > z_max
    target_pcd = target_pcd.select_by_index(np.where(np.array(target_pcd.points)[:, 2] < z_max)[0])
    # Remove noises
    target_pcd = target_pcd.voxel_down_sample(voxel_size=0.005)
    target_pcd.remove_radius_outlier(nb_points=30, radius=0.01)
    target_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    # Do dbscan
    labels = np.array(target_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
    # Remove points from target_pcd, label == -1
    target_pcd = target_pcd.select_by_index(np.where(labels != -1)[0])

    # Only keep the largest cluster
    labels = np.array(target_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
    cluster_size = np.bincount(labels)
    target_pcd = target_pcd.select_by_index(np.where(labels == np.argmax(cluster_size))[0])

    # Return bbox
    return target_pcd, target_pcd.get_minimal_oriented_bounding_box()


if __name__ == "__main__":
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--task_name", type=str, default="cup_to_holder", help="stack_can_in_cabinet, book_in_bookshelf, cup_to_holder")
    args = argparser.parse_args()

    # Parse task cfg
    task_name = args.task_name
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load rpdiff data
    data_path_dict = {
        "cup_to_holder": "/home/harvey/Data/rpdiff_V3/cup_to_holder",
        "lid_to_cup": "/home/harvey/Data/rpdiff_V3/lid_to_cup",
    }

    data_dir = os.path.join(data_path_dict[task_name], "dataset")
    data_file_list = os.listdir(data_dir)

    data_idx = 0
    for data_file in tqdm(data_file_list, desc="Processing Real data"):
        data = pickle.load(open(os.path.join(data_dir, data_file), "rb"))
        dps_data = {}
        for stage in ["start", "end"]:
            whole_coord = data[stage]["points"]
            whole_normal = data[stage]["normals"]
            whole_label = data[stage]["labels"]
            whole_color = data[stage]["colors"] / 255.0
            # Table
            table_coord = whole_coord[whole_label == 0]
            table_normal = whole_normal[whole_label == 0]
            table_color = whole_color[whole_label == 0]
            table_pcd = o3d.geometry.PointCloud()
            table_pcd.points = o3d.utility.Vector3dVector(table_coord)
            table_pcd.normals = o3d.utility.Vector3dVector(table_normal)
            table_pcd.colors = o3d.utility.Vector3dVector(table_color)
            # Using ransac to fit table plane
            plane_model, inliers = table_pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
            table_pcd = table_pcd.select_by_index(inliers)

            # Anchor
            anchor_coord = whole_coord[whole_label == 1]
            anchor_normal = whole_normal[whole_label == 1]
            anchor_color = whole_color[whole_label == 1]
            # Target
            target_coord = whole_coord[whole_label == 2]
            target_normal = whole_normal[whole_label == 2]
            target_color = whole_color[whole_label == 2]

            # Visualize
            anchor_pcd = o3d.geometry.PointCloud()
            anchor_pcd.points = o3d.utility.Vector3dVector(anchor_coord)
            anchor_pcd.normals = o3d.utility.Vector3dVector(anchor_normal)
            anchor_pcd.colors = o3d.utility.Vector3dVector(anchor_color)
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_coord)
            target_pcd.normals = o3d.utility.Vector3dVector(target_normal)
            target_pcd.colors = o3d.utility.Vector3dVector(target_color)

            # Compute a minimum bounding box
            bbox = table_pcd.get_minimal_oriented_bounding_box()
            bbox.color = (1, 0, 0)
            # Compute axis
            bbox_R = bbox.R
            # The axis which is closest to the table normals is the table z-axis
            table_normal = np.mean(np.array(table_normal), axis=0)
            z_axis = None
            product = -np.inf
            for axis in bbox_R:
                if abs(np.dot(axis, table_normal)) > product:
                    z_axis = axis
                    product = abs(np.dot(axis, table_normal))
            # Flip the z-axis to align with the table normal
            if np.dot(z_axis, table_normal) < 0:
                z_axis = -z_axis
            # The axis which is closest to the x-axis is the table x-axis
            x_axis = None
            for axis in bbox_R:
                if abs(axis[0]) > 0.9:
                    x_axis = axis
                    break
            y_axis = np.cross(z_axis, x_axis)
            center = bbox.get_center()
            pose = np.eye(4)
            pose[:3, :3] = np.array([x_axis, y_axis, z_axis])
            pose[:3, 3] = center
            # Reorient everything
            pose_inv = np.linalg.inv(pose)
            anchor_pcd.transform(pose_inv)
            target_pcd.transform(pose_inv)
            table_pcd.transform(pose_inv)

            # Process anchor
            anchor_pcd, anchor_bbox = process_anchor(anchor_pcd)
            target_pcd, target_bbox = process_target(target_pcd)
            anchor_bbox.color = (0, 1, 0)
            target_bbox.color = (0, 0, 1)
            # o3d.visualization.draw_geometries([target_pcd])
            # o3d.visualization.draw_geometries([table_pcd, anchor_pcd, target_pcd, origin])
            dps_data[stage] = {
                "anchor_coord": np.array(anchor_pcd.points),
                "anchor_normal": np.array(anchor_pcd.normals),
                "anchor_color": np.array(anchor_pcd.colors),
                "target_coord": np.array(target_pcd.points),
                "target_normal": np.array(target_pcd.normals),
                "target_color": np.array(target_pcd.colors),
            }

        # Compare target_s and target_g
        target_pcd_s = o3d.geometry.PointCloud()
        target_pcd_s.points = o3d.utility.Vector3dVector(dps_data["start"]["target_coord"])
        target_pcd_s.normals = o3d.utility.Vector3dVector(dps_data["start"]["target_normal"])
        target_pcd_s.colors = o3d.utility.Vector3dVector(dps_data["start"]["target_color"])
        target_pcd_g = o3d.geometry.PointCloud()
        target_pcd_g.points = o3d.utility.Vector3dVector(dps_data["end"]["target_coord"])
        target_pcd_g.normals = o3d.utility.Vector3dVector(dps_data["end"]["target_normal"])
        target_pcd_g.colors = o3d.utility.Vector3dVector(dps_data["end"]["target_color"])
        # Compute relative transformation using icp
        trans_init = np.eye(4)
        trans_init[:3, 3] = np.mean(dps_data["end"]["target_coord"], axis=0) - np.mean(dps_data["start"]["target_coord"], axis=0)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            target_pcd_s, target_pcd_g, 0.05, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        pose = np.asarray(reg_p2p.transformation).copy()
        # Reorient everything
        # pose_inv = np.linalg.inv(pose)
        # target_pcd_s.transform(pose_inv)
        target_pcd_s.transform(pose)
        anchor_pcd_g = o3d.geometry.PointCloud()
        anchor_pcd_g.points = o3d.utility.Vector3dVector(dps_data["end"]["anchor_coord"])
        anchor_pcd_g.normals = o3d.utility.Vector3dVector(dps_data["end"]["anchor_normal"])
        anchor_pcd_g.colors = o3d.utility.Vector3dVector(dps_data["end"]["anchor_color"])
        # o3d.visualization.draw_geometries([target_pcd_s, anchor_pcd_g])

        # Save dps_data, matching rpdiff pipeline; so we can use the same pipeline to train the model
        rpdiff_data = {}
        rpdiff_data["multi_obj_start_pcd"] = {
            "parent": dps_data["end"]["anchor_coord"],
            "child": dps_data["start"]["target_coord"],
        }
        rpdiff_data["normals"] = {
            "parent": dps_data["end"]["anchor_normal"],
            "child": dps_data["start"]["target_normal"],
        }
        rpdiff_data["colors"] = {
            "parent": dps_data["end"]["anchor_color"],
            "child": dps_data["start"]["target_color"],
        }
        rpdiff_data["multi_obj_start_obj_pose"] = {
            "parent": [np.array([0, 0, 0, 0.0, 0, 0, 1])],
            "child": [np.array([0, 0, 0, 0.0, 0, 0, 1])],
        }
        rpdiff_data["multi_obj_final_obj_pose"] = {
            "parent": [np.array([0, 0, 0, 0.0, 0, 0, 1])],
            "child": [np.concatenate([pose[:3, 3], R.from_matrix(pose[:3, :3]).as_quat()])],
        }

        # Save rpdiff_data into npz
        data_idx += 1
        np.savez_compressed(os.path.join(data_path_dict[task_name], f"{task_name}_{data_idx}.npz"), **rpdiff_data)
