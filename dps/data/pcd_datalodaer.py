import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PcdPairCollator:
    def __call__(self, samples):
        target = {
            "target_coord": [],
            "target_normal": [],
            "target_feat": [],
            "anchor_coord": [],
            "anchor_normal": [],
            "anchor_feat": [],
            "target_pose": [],
            "target_batch_index": [],
            "anchor_batch_index": [],
            "target_super_index": [],
            "anchor_super_index": [],
            "target_label": [],
            "anchor_label": [],
            "is_valid_crop": [],
            "corr": [],
            "corr_batch_index": [],
        }
        num_anchor_cluster = 0
        num_target_cluster = 0
        for sample_id, item in enumerate(samples):
            target["target_coord"].append(item["target_coord"])  # (N, 3)
            if item["target_normal"] is not None:
                target["target_normal"].append(item["target_normal"])
            target["target_feat"].append(item["target_feat"])  # (N, 3)
            target["target_batch_index"].append(np.full([len(item["target_coord"])], fill_value=sample_id))  # (N,)
            target["anchor_coord"].append(item["anchor_coord"])  # (M, 3)
            if item["anchor_normal"] is not None:
                target["anchor_normal"].append(item["anchor_normal"])
            target["anchor_feat"].append(item["anchor_feat"])  # (M, 3)
            target["anchor_batch_index"].append(np.full([len(item["anchor_coord"])], fill_value=sample_id))  # (M,)
            target["target_pose"].append(item["target_pose"][None, :])  #
            target["target_super_index"].append(item["target_super_index"] + num_target_cluster)
            target["anchor_super_index"].append(item["anchor_super_index"] + num_anchor_cluster)
            target["target_label"].append(item["target_label"])
            target["anchor_label"].append(item["anchor_label"])
            target["is_valid_crop"].append(item["is_valid_crop"])
            # Corr
            target["corr"].append(item["corr"])
            target["corr_batch_index"].append(np.full([len(item["corr"])], fill_value=sample_id))
            # Update num_cluster
            num_anchor_cluster = num_anchor_cluster + np.max(item["anchor_super_index"]) + 1
            num_target_cluster = num_target_cluster + np.max(item["target_super_index"]) + 1
        return {k: torch.from_numpy(np.concatenate(v)) for k, v in target.items()}


if __name__ == "__main__":
    import os
    from dps.data.pcd_dataset import PcdPairDataset

    dataset_name = "data_rdiff"
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    split = "test"
    # Test data loader
    dataset = PcdPairDataset(
        data_file_list=[f"{root_dir}/test_data/{dataset_name}/diffusion_dataset_0_2048_s25000-c1-r0.5_{split}.pkl"],
        dataset_name="data",
        add_colors=True,
        add_normals=True,
        is_elastic_distortion=False,
        is_random_distortion=True,
        volume_augmentations_path=f"{root_dir}/config/va_rotation.yaml",
        noise_level=0.5,
        crop_pcd=False,
        crop_size=0.4,
    )
    dataset.set_mode("train")

    # Test data loader
    collate_fn = PcdPairCollator()
    for d in DataLoader(dataset, collate_fn=collate_fn, batch_size=4):
        print(d["target_batch_index"].shape)
        print(d["anchor_batch_index"].shape)
        print(d["target_coord"].shape)
        print(d["target_feat"].shape)
        print(d["anchor_coord"].shape)
        print(d["anchor_feat"].shape)
        print(d["target_pose"].shape)
        break
