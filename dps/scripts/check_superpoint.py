"""Check & Visualize superpoint data"""

"""Check superpoint data"""

import os
import pickle
import numpy as np
import open3d as o3d


def visualize_superpoint(superpoint_data):
    pos = superpoint_data["pos"]
    normal = superpoint_data["normal"]
    super_indexes = superpoint_data["super_index"]
    num_color = np.max(super_indexes[0]) + 1
    # Generate random color
    color = np.random.rand(num_color, 3)
    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    pcd_list = []
    for i, super_index in enumerate(super_indexes):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pos)
        # pcd.normals = o3d.utility.Vector3dVector(normal)
        pcd.colors = o3d.utility.Vector3dVector(color[super_index])
        o3d.visualization.draw_geometries([pcd, origin])
        pcd_list.append(pcd)
    return pcd_list


def visualize_rpdiff(superpoint_path):
    # Check superpoint data
    superpoint_path = os.path.join(data_dir, "superpoint_data")
    superpoint_file = os.path.join(superpoint_path, "superpoint_dict.pkl")
    superpoint_dict = pickle.load(open(superpoint_file, "rb"))

    for key, superpoint_data in superpoint_dict.items():
        c_superpoint_data = superpoint_data["child"]
        p_superpoint_data = superpoint_data["parent"]

        c_pcd_list = visualize_superpoint(c_superpoint_data)
        p_pcd_list = visualize_superpoint(p_superpoint_data)
        o3d.visualization.draw_geometries(c_pcd_list + p_pcd_list)

def visualize_anet3d(superpoint_path, split="train"):
    # Check superpoint data
    superpoint_path = os.path.join(data_dir, "superpoint_data")
    superpoint_file = os.path.join(superpoint_path, f"superpoint_dict_{split}.pkl")
    superpoint_dict = pickle.load(open(superpoint_file, "rb"))

    for key, superpoint_data in superpoint_dict.items():
        visualize_superpoint(superpoint_data)


if __name__ == "__main__":
    data_path_dict = {
        "can_in_cabinet": "/home/harvey/Data/rpdiff_V3/can_in_cabinet",
        "book_in_bookshelf": "/home/harvey/Data/rpdiff_V3/book_in_bookshelf",
        "cup_to_holder": "/home/harvey/Data/rpdiff_V3/cup_to_holder",
        "lid_to_cup": "/home/harvey/Data/rpdiff_V3/lid_to_cup",
        "anet3d": "/home/harvey/Data/Anet3D",
    }

    task_name = "cup_to_holder"  # "book_in_bookshelf", "can_in_cabinet" "anet3d"
    data_dir = data_path_dict[task_name]
    if task_name in ["can_in_cabinet", "book_in_bookshelf", "cup_to_holder"]:
        visualize_rpdiff(data_dir)
    elif task_name == "anet3d":
        visualize_anet3d(data_dir)
