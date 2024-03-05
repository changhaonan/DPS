"""Dataset for loading affordance net 3d dataset.
"""

from __future__ import annotations
from torch.utils.data import Dataset
import os
import pickle
import dps.utils.misc_utils as utils
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as R


class ANet3DDataset(Dataset):
    """Dataset definition for point cloud like data."""

    def __init__(
        self,
        data_file: str,
        dataset_mode: str = "train",
        add_colors: bool = False,
        add_normals: bool = False,
        **kwargs,
    ):
        # Set parameters
        self.add_colors = add_colors
        self.add_normals = add_normals
        self.dataset_mode = dataset_mode

        # Load data
        data_list = pickle.load(open(data_file, "rb"))  # Only a list of (coordinates)
        self.affordances = list(data_list[0]["full_shape"]["label"].keys())
        sorted(self.affordances)
        self.semantic_classes = set()
        for obj in data_list:
            self.semantic_classes.add(obj["semantic class"])
        self.semantic_classes = list(self.semantic_classes)
        sorted(self.semantic_classes)
        self._data = data_list
        self.mode = "train"  # train, val, test

    def set_mode(self, mode: str):
        self.mode = mode

    def parse_pcd_data(self, batch_idx):
        """Parse data from the dataset."""
        pcd = self._data[batch_idx]["full_shape"]["coordinate"]
        # expanded_data = np.zeros((data["full_shape"]["coordinate"].shape[0], 3 + len(self.affordances)))
        # for affordance in self.affordances:
        #     aff_points = np.asarray(np.where(data["full_shape"]["label"][affordance] != 0)[0])
        affordance_dict = self._data[batch_idx]["full_shape"]["label"]
        return pcd, affordance_dict, self._data[batch_idx]["semantic class"]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        idx = idx % len(self._data)
        # Parse data from the dataset & Convert to float32
        pcd, affordance_dict, semantic_class = self.parse_pcd_data(idx)
        return pcd, affordance_dict, semantic_class

    @property
    def data(self):
        """database file containing information about preproscessed dataset"""
        return self._data


if __name__ == "__main__":

    dataset_name = "anet3d"
    dataset_type = "full_shape"
    split = "train"  # "train", "val"
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(root_path, "test_data", f"{dataset_name}", f"{dataset_type}_{split}_data.pkl")

    dataset = ANet3DDataset(
        data_file=data_file,
        add_colors=False,
        add_normals=False,
    )
    dataset.set_mode("train")

    # Test data augmentation
    for i in range(20):
        # random_idx = np.random.randint(0, len(dataset))
        random_idx = i
        pcd, affordance_dict, semantic_class = dataset[random_idx]
        utils.visualize_pcd_list(
            coordinate_list=[pcd],
            # color_list=[target_color, anchor_color],
            # pose_list=[np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32)],
        )
