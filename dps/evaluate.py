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
from dps.model.pcdc_seg_model import PCDCModel
from dps.model.rgt_model import RGTModel
from detectron2.config import LazyConfig
from torch.utils.data.dataset import random_split
from dps.utils.dps_utils import build_rpdiff_dataset
from dps.utils.rpdiff_utils import RpdiffHelper, read_rpdiff_data
from dps.model.network.geometric import to_dense_batch, to_flat_batch
from dps.utils.pcd_utils import estimate_collision
import random
import copy


class DPSEvaluator:
    """Diffusion point cloud segmentation and action evaluation."""

    def __init__(self, root_path, seg_cfg, act_cfg, device, seg_type="diffusion") -> None:
        self.device = device
        self.seg_cfg = seg_cfg
        self.act_cfg = act_cfg
        self.batch_size = min(seg_cfg.DATALOADER.BATCH_SIZE, act_cfg.DATALOADER.BATCH_SIZE)
        self.seg_prob_thresh = seg_cfg.MODEL.SEG_PROB_THRESH
        self.seg_num_thresh = seg_cfg.MODEL.SEG_NUM_THRESH
        # HACK: use simple policy: directly going to seg center
        self.use_simple_policy = True

        # Build segmentation model
        net_name = seg_cfg.MODEL.NOISE_NET.NAME
        net_init_args = seg_cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
        seg_net = PcdSegNoiseNet(**net_init_args)
        self.seg_type = seg_type
        if seg_type == "diffusion":
            self.seg_model = PCDDModel(seg_cfg, seg_net)
        elif seg_type == "classification":
            self.seg_model = PCDCModel(seg_cfg, seg_net)
        else:
            raise ValueError(f"Unsupported seg_type: {seg_type}")
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
        max_try = kwargs.get("max_try", 20)
        act_batch_size = 8 if self.seg_type == "diffusion" else 1  # For classification, we can only process one at a time
        # Perform segmentation
        seg_list = []
        for _i in range(max_try):
            pred_anchor_label, anchor_coord, anchor_normal, anchor_feat = self.seg_model.predict(batch=batch, check_batch_idx=check_batch_idx, vis=vis, batch_size=self.batch_size)
            seg_list += self.seg_model.seg_and_rank(
                anchor_coord, pred_anchor_label, normal=anchor_normal, feat=anchor_feat, crop_strategy=crop_strategy, prob_thresh=self.seg_prob_thresh, num_thresh=self.seg_num_thresh
            )
            if len(seg_list) >= act_batch_size:
                break
            else:
                print(f"Retry segmentation {_i}...")
        if len(seg_list) < act_batch_size and len(seg_list) > 0:
            print(f"Warning: only {len(seg_list)} segments are found.")
            # Fill with last segment
            seg_list += [seg_list[-1]] * (act_batch_size - len(seg_list))
        elif len(seg_list) == 0:
            print(f"Warning: no segment is found.")
            pred_pose_list = [np.eye(4)] * act_batch_size
            seg_size_list = [0] * act_batch_size
            return pred_pose_list, seg_size_list
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
        data = {"target_coord": target_coord, "target_normal": target_normal, "target_feat": target_feat}
        act_batch = self.convert_to_batch(data, seg_list, batch_size=act_batch_size)
        anchor_batch_idx = batch["anchor_batch_index"]
        anchor_coord_full = batch["anchor_coord"][anchor_batch_idx == check_batch_idx].numpy()
        anchor_normal_full = batch["anchor_normal"][anchor_batch_idx == check_batch_idx].numpy()
        # Shift anchor to center
        anchor_batch_idx = act_batch["anchor_batch_index"]
        anchor_coord = act_batch["anchor_coord"]
        batch_anchor_coord, anchor_mask = to_dense_batch(anchor_coord, anchor_batch_idx)
        batch_anchor_center = batch_anchor_coord.mean(dim=1, keepdim=True)
        anchor_coord = batch_anchor_coord - batch_anchor_center
        act_batch["anchor_coord"] = to_flat_batch(anchor_coord, anchor_mask)[0]
        if not self.use_simple_policy:
            for k in range(3):
                print(f"Iteration {k}...")
                if k != 0:
                    do_icp = True
                    prev_R = pred_R
                    prev_t = pred_t
                else:
                    prev_R = np.eye(3)[None, :, :].astype(np.float32).repeat(act_batch_size, axis=0)
                    prev_t = np.zeros((1, 3)).astype(np.float32).repeat(act_batch_size, axis=0)
                    do_icp = False
                act_batch["prev_R"] = torch.from_numpy(prev_R).to(self.device)
                act_batch["prev_t"] = torch.from_numpy(prev_t).to(self.device)
                conf_matrix, gt_corr, (pred_R, pred_t) = self.act_model.predict(batch=act_batch, vis=vis, do_icp=do_icp)
        else:
            pred_R = np.eye(3)[None, :, :].astype(np.float32).repeat(act_batch_size, axis=0)
            # Move target to origin
            pred_t = -np.mean(target_coord, axis=0)[None, :].astype(np.float32).repeat(act_batch_size, axis=0)
        # Recover pose if there exists pose before
        if "anchor_pose" in batch:
            anchor_pose = batch["anchor_pose"][check_batch_idx, ...].detach().cpu().numpy()
        else:
            anchor_pose = np.eye(4)
        if "target_pose" in batch:
            target_pose = batch["target_pose"][check_batch_idx, ...].detach().cpu().numpy()
        else:
            target_pose = np.eye(4)
        pred_pose_list = []
        seg_size_list = []
        for _i in range(act_batch_size):
            pred_pose = np.eye(4)
            pred_pose[:3, :3] = pred_R[_i, ...]
            pred_pose[:3, 3] = pred_t[_i, ...]
            T_shift = np.eye(4)
            T_shift[:3, 3] = batch_anchor_center[_i, ...].detach().cpu().numpy()
            pred_pose = anchor_pose @ T_shift @ pred_pose @ np.linalg.inv(target_pose)
            pred_pose_list.append(pred_pose)
            seg_size_list.append(seg_list[_i]["coord"].shape[0])
        return pred_pose_list, seg_size_list

    def convert_to_batch(self, data: dict, seg_list: list, batch_size: int = 1):
        target = {
            "target_coord": [],
            "target_normal": [],
            "target_feat": [],
            "anchor_coord": [],
            "anchor_normal": [],
            "anchor_feat": [],
            "target_batch_index": [],
            "anchor_batch_index": [],
        }
        for idx in range(min(len(seg_list), batch_size)):
            seg = seg_list[idx]
            target["target_coord"].append(data["target_coord"])
            target["target_normal"].append(data["target_normal"])
            target["target_feat"].append(data["target_feat"])
            target["anchor_coord"].append(seg["coord"])
            target["anchor_normal"].append(seg["normal"])
            target["anchor_feat"].append(seg["feat"])
            target["target_batch_index"].append(np.full([len(data["target_coord"])], fill_value=idx, dtype=np.int64))
            target["anchor_batch_index"].append(np.full([len(seg["coord"])], fill_value=idx, dtype=np.int64))
        return {k: torch.from_numpy(np.concatenate(v)) for k, v in target.items()}


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=1)
    argparser.add_argument("--random_index", type=int, default=1)
    argparser.add_argument("--task_name", type=str, default="book_in_bookshelf", help="can_in_cabinet, book_in_bookshelf, mug_on_rack_multi")
    argparser.add_argument("--seg_type", type=str, default="diffusion", help="diffusion, classification")
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
    seg_type = args.seg_type
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
    seg_cfg.DATALOADER.BATCH_SIZE = 16

    # Load dataset & data loader
    train_dataset, val_dataset, test_dataset = build_rpdiff_dataset(root_path, seg_cfg)
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=seg_cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=act_cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=seg_cfg.DATALOADER.BATCH_SIZE, shuffle=False, num_workers=act_cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())

    # Raw data
    use_raw_data = True
    raw_data_dir = "/home/harvey/Data/rpdiff_V3"
    raw_data_dir = os.path.join(raw_data_dir, task_name)
    # raw_data_dir = "/home/harvey/Project/DPS/dps/external/rpdiff/eval_data/eval_data/can_on_cabinet_nosc/seed_0/failed"  # Focus on failed casess
    raw_data_file_list = os.listdir(raw_data_dir)
    raw_data_file_list = [os.path.join(raw_data_dir, f) for f in raw_data_file_list]
    rpdiff_helper = RpdiffHelper(
        downsample_voxel_size=seg_cfg.PREPROCESS.GRID_SIZE,
        target_scale=seg_cfg.PREPROCESS.TARGET_RESCALE,
        anchor_scale=seg_cfg.PREPROCESS.ANCHOR_RESCALE,
        batch_size=seg_cfg.DATALOADER.BATCH_SIZE,
        superpoint_cfg=seg_cfg.DATALOADER.SUPER_POINT,
        complete_strategy=seg_cfg.DATALOADER.COMPLETE_STRATEGY,
    )

    # Build evaluator
    evaluator = DPSEvaluator(root_path, seg_cfg, act_cfg, device, seg_type=seg_type)
    do_vis = True
    # Testing raw material
    for i in range(40):
        if not use_raw_data:
            batch = next(iter(val_data_loader))
            # Remove target pose from batch
            batch.pop("target_pose", None)
        else:
            data_file = next(iter(raw_data_file_list))
            data = read_rpdiff_data(data_file)
            # Visualize row data
            anchor_pcd = o3d.geometry.PointCloud()
            anchor_pcd.points = o3d.utility.Vector3dVector(data["anchor_coord"])
            anchor_pcd.normals = o3d.utility.Vector3dVector(data["anchor_normal"])
            anchor_pcd.paint_uniform_color([234.0 / 255.0, 182.0 / 255.0, 118.0 / 255.0])
            o3d.visualization.draw_geometries([anchor_pcd])
            batch = rpdiff_helper.process_data(
                target_coord=data["target_coord"], target_normal=data["target_normal"], anchor_coord=data["anchor_coord"], anchor_normal=data["anchor_normal"], vis=do_vis
            )
        if i < 2:
            continue

        # Perform segmentation
        check_batch_idx = 1
        pred_pose_list, seg_size_list = evaluator.process(batch, check_batch_idx=check_batch_idx, vis=do_vis)

        # Visualize on whole
        target_pcd = o3d.geometry.PointCloud()
        target_pcd.points = o3d.utility.Vector3dVector(data["target_coord"])
        target_pcd.normals = o3d.utility.Vector3dVector(data["target_normal"])
        target_pcd.paint_uniform_color([1, 0, 0])

        anchor_pcd = o3d.geometry.PointCloud()
        anchor_pcd.points = o3d.utility.Vector3dVector(data["anchor_coord"])
        anchor_pcd.normals = o3d.utility.Vector3dVector(data["anchor_normal"])
        anchor_pcd.paint_uniform_color([0, 1, 0])

        vis_list = [anchor_pcd]
        target_pcd_list = []
        max_num_collision = 0
        for pred_pose in pred_pose_list:
            target_pcd_copy = copy.deepcopy(target_pcd)
            target_pcd_copy.transform(pred_pose)
            # Estimate collision status
            num_collision = estimate_collision(np.asarray(target_pcd_copy.points), np.asarray(anchor_pcd.points))
            target_pcd_list.append((target_pcd_copy, num_collision))
            max_num_collision = max(max_num_collision, num_collision)

        # Sort by num_collision
        target_pcd_list = sorted(target_pcd_list, key=lambda x: x[1])
        for target_pcd, num_collision in target_pcd_list:
            target_pcd.paint_uniform_color([0, 0, 1.0 / (max_num_collision + 1) * num_collision])
            vis_list.append(target_pcd)
        o3d.visualization.draw_geometries(vis_list)
