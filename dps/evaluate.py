"""Evaluate rpdiff pipeline."""

import os
import open3d as o3d
import torch
import numpy as np
import pickle
import argparse
import dps.utils.misc_utils as utils
from dps.data.pcd_dataset import PcdPairDataset
from dps.data.pcd_datalodaer import PcdPairCollator
from dps.model.network.rigpose_transformer import RigPoseTransformer
from dps.model.network.pcd_seg_noise_net import PcdSegNoiseNet
from dps.model.pcdd_seg_model import PCDDModel
from dps.model.rgt_model import RGTModel
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split
from dps.utils.dps_utils import build_rpdiff_dataset
from dps.utils.rpdiff_utils import RpdiffHelper, read_rpdiff_data
import random


class DPSEvaluator:
    """Diffusion point cloud segmentation and action evaluation."""

    def __init__(self, root_path, seg_cfg, act_cfg, device) -> None:
        self.device = device
        self.seg_cfg = seg_cfg
        self.act_cfg = act_cfg
        self.batch_size = min(seg_cfg.DATALOADER.BATCH_SIZE, act_cfg.DATALOADER.BATCH_SIZE)
        # Build segmentation model
        net_name = seg_cfg.MODEL.NOISE_NET.NAME
        net_init_args = seg_cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
        seg_net = PcdSegNoiseNet(**net_init_args)
        self.seg_model = PCDDModel(seg_cfg, seg_net)
        seg_net_name = seg_cfg.MODEL.NOISE_NET.NAME
        save_dir = os.path.join(root_path, "test_data", seg_cfg.ENV.TASK_NAME, "checkpoints", seg_net_name)
        save_path = os.path.join(save_dir, self.seg_model.experiment_name())
        checkpoint_path = f"{save_path}/checkpoints"
        checkpoints = os.listdir(checkpoint_path)
        sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
        checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
        self.seg_model.load(checkpoint_file)
        self.seg_model.lpcd_noise_net.to(device)

        # Build action model
        act_net_name = act_cfg.MODEL.NOISE_NET.NAME
        net_init_args = act_cfg.MODEL.NOISE_NET.INIT_ARGS[act_net_name]
        act_net = RigPoseTransformer(**net_init_args)
        self.act_model = RGTModel(act_cfg, act_net)
        seg_net_name = act_cfg.MODEL.NOISE_NET.NAME
        save_dir = os.path.join(root_path, "test_data", seg_cfg.ENV.TASK_NAME, "checkpoints", seg_net_name)
        save_path = os.path.join(save_dir, self.act_model.experiment_name())
        checkpoint_path = f"{save_path}/checkpoints"
        checkpoints = os.listdir(checkpoint_path)
        sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
        checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
        self.act_model.load(checkpoint_file)
        self.act_model.lpose_transformer.to(device)

    def process(self, batch, check_batch_idx: int = 1, vis: bool = False, **kwargs):
        crop_strategy = kwargs.get("crop_strategy", "bbox")
        max_try = kwargs.get("max_try", 3)
        # Perform segmentation
        for i in range(max_try):
            pred_anchor_label, anchor_coord, anchor_normal, anchor_feat = self.seg_model.predict(batch=batch, check_batch_idx=check_batch_idx, vis=vis, batch_size=self.batch_size)
            seg_list = self.seg_model.seg_and_rank(anchor_coord, pred_anchor_label, normal=anchor_normal, feat=anchor_feat, crop_strategy=crop_strategy)
            if len(seg_list) > 0:
                break
            else:
                print(f"Retry segmentation {i}...")

        # DEBUG: visualize the segmentation result
        if vis:
            anchor_pcd = o3d.geometry.PointCloud()
            anchor_pcd.points = o3d.utility.Vector3dVector(anchor_coord[check_batch_idx, ...])
            anchor_pcd.normals = o3d.utility.Vector3dVector(anchor_normal[check_batch_idx, ...])
            anchor_pcd.paint_uniform_color([0, 1, 0])
            for seg in seg_list:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(seg["coord"])
                pcd.normals = o3d.utility.Vector3dVector(seg["normal"])
                pcd.paint_uniform_color([1, 0, 0])
                origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                o3d.visualization.draw_geometries([anchor_pcd, pcd, origin])

        # Prepare data
        target_batch_idx = batch["target_batch_index"]
        target_coord = batch["target_coord"][target_batch_idx == check_batch_idx].numpy()
        target_normal = batch["target_normal"][target_batch_idx == check_batch_idx].numpy()
        target_feat = batch["target_feat"][target_batch_idx == check_batch_idx].numpy()
        # Cropped coord
        anchor_coord = seg_list[0]["coord"]
        anchor_normal = seg_list[0]["normal"]
        anchor_feat = seg_list[0]["feat"]
        # Full coord
        anchor_batch_idx = batch["anchor_batch_index"]
        anchor_coord_full = batch["anchor_coord"][anchor_batch_idx == check_batch_idx].numpy()
        anchor_normal_full = batch["anchor_normal"][anchor_batch_idx == check_batch_idx].numpy()
        anchor_feat_full = batch["anchor_feat"][anchor_batch_idx == check_batch_idx].numpy()

        # Move anchor to center
        anchor_center = np.mean(anchor_coord, axis=0)
        anchor_coord -= anchor_center
        for k in range(3):
            print(f"Iteration {k}...")
            if k != 0:
                do_icp = True
                prev_R = pred_R
                prev_t = pred_t
            else:
                prev_R = np.eye(3)[None, :, :].astype(np.float32)
                prev_t = np.zeros((1, 3)).astype(np.float32)
                do_icp = False
            conf_matrix, gt_corr, (pred_R, pred_t) = self.act_model.predict(
                target_coord=target_coord,
                target_normal=target_normal,
                target_feat=target_feat,
                anchor_coord=anchor_coord,
                anchor_normal=anchor_normal,
                anchor_feat=anchor_feat,
                prev_R=prev_R,
                prev_t=prev_t,
                vis=vis,
                do_icp=do_icp,
            )

        if vis:
            # Visualize everything together
            target_pcd = o3d.geometry.PointCloud()
            target_pcd.points = o3d.utility.Vector3dVector(target_coord)
            target_pcd.normals = o3d.utility.Vector3dVector(target_normal)
            target_pcd.paint_uniform_color([1, 0, 0])
            pose = np.eye(4)
            pose[:3, :3] = pred_R
            pose[:3, 3] = pred_t
            target_pcd.transform(pose)
            target_pcd.translate(anchor_center)

            anchor_pcd = o3d.geometry.PointCloud()
            anchor_pcd.points = o3d.utility.Vector3dVector(anchor_coord_full)
            anchor_pcd.normals = o3d.utility.Vector3dVector(anchor_normal_full)
            anchor_pcd.paint_uniform_color([0, 1, 0])
            o3d.visualization.draw_geometries([target_pcd, anchor_pcd])

        # Recover pose if there exists pose before
        if "anchor_pose" in batch:
            anchor_pose = batch["anchor_pose"][check_batch_idx, ...].detach().cpu().numpy()
        else:
            anchor_pose = np.eye(4)
        if "target_pose" in batch:
            target_pose = batch["target_pose"][check_batch_idx, ...].detach().cpu().numpy()
        else:
            target_pose = np.eye(4)
        pred_pose = np.eye(4)
        pred_pose[:3, :3] = pred_R
        pred_pose[:3, 3] = pred_t
        T_shift = np.eye(4)
        T_shift[:3, 3] = anchor_center
        pred_pose = anchor_pose @ T_shift @ pred_pose @ np.linalg.inv(target_pose)
        return pred_pose


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--random_index", type=int, default=0)
    argparser.add_argument("--task_name", type=str, default="book_in_bookshelf", help="stack_can_in_cabinet, book_in_bookshelf, mug_on_rack_multi")
    args = argparser.parse_args()
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load config
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    task_name = args.task_name
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", f"pose_transformer_rpdiff_{task_name}.py")
    act_cfg = LazyConfig.load(cfg_file)
    seg_cfg = LazyConfig.load(cfg_file)
    # Overriding config
    act_cfg.MODEL.NOISE_NET.NAME = "RGTModel"
    act_cfg.DATALOADER.AUGMENTATION.CROP_PCD = True
    act_cfg.DATALOADER.BATCH_SIZE = 32
    seg_cfg.MODEL.NOISE_NET.NAME = "PCDSAMNOISENET"
    seg_cfg.DATALOADER.AUGMENTATION.CROP_PCD = False
    seg_cfg.DATALOADER.BATCH_SIZE = 2

    # Load dataset & data loader
    train_dataset, val_dataset, test_dataset = build_rpdiff_dataset(root_path, seg_cfg)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=seg_cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=act_cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=seg_cfg.DATALOADER.BATCH_SIZE, shuffle=False, num_workers=act_cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())

    # Raw data
    use_raw_data = True
    raw_data_dir = "/home/harvey/Data/rpdiff_V3"
    raw_data_dir = os.path.join(raw_data_dir, task_name)
    raw_data_file_list = os.listdir(raw_data_dir)
    raw_data_file_list = [os.path.join(raw_data_dir, f) for f in raw_data_file_list]
    rpdiff_helper = RpdiffHelper(
        downsample_voxel_size=seg_cfg.PREPROCESS.GRID_SIZE, scale=seg_cfg.PREPROCESS.TARGET_RESCALE, batch_size=seg_cfg.DATALOADER.BATCH_SIZE, superpoint_cfg=seg_cfg.DATALOADER.SUPER_POINT
    )

    # Build evaluator
    evaluator = DPSEvaluator(root_path, seg_cfg, act_cfg, device)

    # Testing raw material
    for i in range(40):
        if not use_raw_data:
            batch = next(iter(val_data_loader))
        else:
            data_file = next(iter(raw_data_file_list))
            data = read_rpdiff_data(data_file)
            batch = rpdiff_helper.process_data(target_coord=data["target_coord"], target_normal=data["target_normal"], anchor_coord=data["anchor_coord"], anchor_normal=data["anchor_normal"])
        if i < 2:
            continue

        # Perform segmentation
        check_batch_idx = 1
        pred_pose = evaluator.process(batch, check_batch_idx=check_batch_idx, vis=True)
