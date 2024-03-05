"""Diffusion model for generating segmentation using PCD diffusion net"""

from __future__ import annotations
from typing import Any
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import math
import os
import time
import torch
import torch.nn.functional as F
import lightning as L
from dps.model.network.geometric import batch2offset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from dps.model.network.pcd_seg_noise_net import PcdSegNoiseNet
from dps.model.network.geometric import batch2offset, offset2batch, to_dense_batch, to_flat_batch
from box import Box
import yaml
import torch_scatter
import dps.utils.misc_utils as utils
from torch.optim.lr_scheduler import LambdaLR
import open3d as o3d
import cv2
import wandb
from scipy.spatial.transform import Rotation as R


class LPcdSegDiffusion(L.LightningModule):
    """Lignthing module for PCD diffusion model; This uses superpoint."""

    def __init__(self, pcd_noise_net: PcdSegNoiseNet, cfg: Box, **kwargs: Any) -> None:
        super().__init__()
        self.pcd_noise_net = pcd_noise_net
        self.lr = cfg.TRAIN.LR
        self.start_time = time.time()
        self.diffusion_process = cfg.MODEL.DIFFUSION_PROCESS
        self.num_diffusion_iters = cfg.MODEL.NUM_DIFFUSION_ITERS
        self.warm_up_step = cfg.TRAIN.WARM_UP_STEP
        self.rot_axis = cfg.DATALOADER.AUGMENTATION.ROT_AXIS
        self.superpoint_layer = cfg.MODEL.SUPERPOINT_LAYER
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            # the choise of beta schedule has big impact on performance
            # we found squared cosine works the best
            beta_schedule="squaredcos_cap_v2",
            # clip output to [-1,1] to improve stability
            clip_sample=True,
            # our network predicts noise (instead of denoised action)
            prediction_type="epsilon",
        )
        # Logging
        self.sample_batch = None
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        # Init
        self.pcd_noise_net.initialize_weights()

    def criterion(self, batch):
        if self.sample_batch is None:
            self.sample_batch = batch
        # Assemble input
        coord = batch["anchor_coord"]
        normal = batch["anchor_normal"]
        feat = batch["anchor_feat"]
        label = batch["anchor_label"]
        batch_idx = batch["anchor_batch_index"]
        offset = batch2offset(batch_idx)
        points = [coord, feat, offset]
        label = batch["anchor_label"]
        super_index = batch["anchor_super_index"][:, self.superpoint_layer]  # Use selected layer
        # Do one-time encoding
        enc_points = self.pcd_noise_net.encode_pcd(points, attrs={"normal": normal})
        # Scatter to super_index
        coord, feat, offset = enc_points
        super_label = torch_scatter.scatter(label, super_index, dim=0, reduce="mean").unsqueeze(1)
        super_coord = torch_scatter.scatter(coord, super_index, dim=0, reduce="mean")
        super_feat = torch_scatter.scatter(feat, super_index, dim=0, reduce="mean")
        super_batch_index = torch_scatter.scatter(batch_idx, super_index, dim=0, reduce="mean")
        super_offset = batch2offset(super_batch_index)

        super_points = [super_coord, super_feat, super_offset]
        batch_label, label_mask = to_dense_batch(super_label, super_batch_index)
        if self.diffusion_process == "ddpm":
            # sample noise to add to actions
            noise = torch.randn(batch_label.shape, device=self.device)
            # sample a diffusion iteration for each data point
            timesteps = torch.randint(low=0, high=self.noise_scheduler.config.num_train_timesteps, size=(batch_label.shape[0], 1), device=self.device).long()
            # add noisy actions
            noisy_label = self.noise_scheduler.add_noise(batch_label, noise, timesteps)
            noisy_label, _ = to_flat_batch(noisy_label, label_mask)
            # predict noise residual
            noise_pred = self.pcd_noise_net(noisy_label, super_points, timesteps)
            # Compute loss with mask
            diff_loss = F.mse_loss(noise_pred[label_mask], noise[label_mask])
            loss = diff_loss
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")
        return loss

    def training_step(self, batch, batch_idx):
        """Training step; DDPM loss"""
        loss = self.criterion(batch)
        # log
        self.log("train_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.criterion(batch)
        # log
        self.log("val_loss", loss, sync_dist=True, batch_size=self.batch_size)
        elapsed_time = (time.time() - self.start_time) / 3600
        self.log("train_runtime(hrs)", elapsed_time, sync_dist=True, batch_size=self.batch_size)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Validate by running full diffusion inference."""
        if (self.current_epoch + 1) % 1 == 0:
            batch = self.sample_batch
            batch_idx = batch["anchor_batch_index"]
            label = batch["anchor_label"]
            super_index = batch["anchor_super_index"][:, self.superpoint_layer]  # Use the first super index
            super_label = torch_scatter.scatter(label, super_index, dim=0, reduce="mean").unsqueeze(1)
            super_batch_index = torch_scatter.scatter(batch_idx, super_index, dim=0, reduce="mean")
            batch_super_label, super_mask = to_dense_batch(super_label, super_batch_index)
            batch_label = to_dense_batch(label, batch_idx)[0]
            pred_label = self.forward(batch)
            # Visualize results
            anchor_coord = batch["anchor_coord"]
            batch_anchor_coord = to_dense_batch(anchor_coord, batch_idx)[0]
            batch_pred_label = to_dense_batch(pred_label, batch_idx)[0]
            for i in range(min(batch_super_label.shape[0], 4)):
                pred_label_i = pred_label[i]
                anchord_coord_i = batch_anchor_coord[i]
                pred_label_i = batch_pred_label[i]
                gt_label_i = batch_label[i][:, None]
                pred_image = self.view_result(anchord_coord_i, pred_label_i)
                gt_image = self.view_result(anchord_coord_i, gt_label_i)
                # Concatenate image
                concat_image = np.concatenate([gt_image, pred_image], axis=1)
                # Log image in wandb
                wandb_logger = self.logger.experiment
                wandb_logger.log({f"val_label_image": [wandb.Image(concat_image, caption=f"val_label_image_{i}")]})

    def forward(self, batch):
        """Inference for PCD diffusion model."""
        # Assemble input
        coord = batch["anchor_coord"]
        normal = batch["anchor_normal"]
        feat = batch["anchor_feat"]
        batch_idx = batch["anchor_batch_index"]
        offset = batch2offset(batch_idx)
        points = [coord, feat, offset]
        super_index = batch["anchor_super_index"][:, self.superpoint_layer]  # Use selected layer
        # Do one-time encoding
        enc_points = self.pcd_noise_net.encode_pcd(points, attrs={"normal": normal})
        # Scatter to super_index
        coord, feat, offset = enc_points
        super_coord = torch_scatter.scatter(coord, super_index, dim=0, reduce="mean")
        super_feat = torch_scatter.scatter(feat, super_index, dim=0, reduce="mean")
        super_batch_index = torch_scatter.scatter(batch_idx, super_index, dim=0, reduce="mean")
        super_offset = batch2offset(super_batch_index)
        super_points = [super_coord, super_feat, super_offset]

        # Batchify
        batch_coord, super_index_mask = to_dense_batch(super_coord, super_batch_index)
        if self.diffusion_process == "ddpm":
            # initialize action from Guassian noise
            noisy_label = torch.randn((batch_coord.shape[0], batch_coord.shape[1], 1), device=self.device)
            for k in self.noise_scheduler.timesteps:
                timesteps = torch.tensor([k], device=self.device).to(torch.long).repeat(batch_coord.shape[0])
                noisy_label, _ = to_flat_batch(noisy_label, super_index_mask)
                # predict noise residual
                noise_pred = self.pcd_noise_net(noisy_label, super_points, timesteps)
                noisy_label, _ = to_dense_batch(noisy_label, super_batch_index)
                # inverse diffusion step (remove noise)
                noisy_label = self.noise_scheduler.step(model_output=noise_pred, timestep=k, sample=noisy_label).prev_sample
        else:
            raise ValueError(f"Diffusion process {self.diffusion_process} not supported.")

        # Assign to cluster
        noisy_label, _ = to_flat_batch(noisy_label, super_index_mask)
        pred_label = noisy_label[super_index]
        return pred_label

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def lr_foo(epoch):
            if epoch < self.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.warm_up_step - epoch)
            else:
                lr_scale = 0.95**epoch
            return lr_scale

        scheduler = LambdaLR(optimizer, lr_lambda=lr_foo)
        return [optimizer], [scheduler]

    def view_result(self, batch_coord: torch.Tensor, pred_label: torch.Tensor, vis_3d: bool = False):
        """Render label image."""
        pred_label = pred_label.detach().cpu().numpy()
        batch_coord = batch_coord.detach().cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(batch_coord)
        color = (1 + pred_label) * np.array([0.0, 0.5, 0.0]) + (1 - pred_label) * np.array([0.5, 0.0, 0.0])
        color = np.clip(color, 0, 1)
        pcd.colors = o3d.utility.Vector3dVector(color)
        # Rotate the object for better view: rotate around y-axis for 90 degree & z-axis for -90 degree
        rot = R.from_euler("z", -90, degrees=True) * R.from_euler("y", -90, degrees=True)
        pcd.rotate(rot.as_matrix(), center=(0, 0, 0))
        if vis_3d:
            o3d.visualization.draw_geometries([pcd])
            return None
        else:
            # Offscreen rendering
            viewer = o3d.visualization.Visualizer()
            viewer.create_window(width=540, height=540, visible=False)
            viewer.add_geometry(pcd)
            # Render image
            image = viewer.capture_screen_float_buffer(do_render=True)
            viewer.destroy_window()
            return np.asarray(image)


class PCDDModel:
    """PCD diffusion model"""

    def __init__(self, cfg, pcd_noise_net: PcdSegNoiseNet) -> None:
        self.cfg = cfg
        # parameters
        self.logger_project = cfg.LOGGER.PROJECT
        # build model
        self.pcd_noise_net = pcd_noise_net
        self.lpcd_noise_net = LPcdSegDiffusion(pcd_noise_net, cfg).to(torch.float32)

    def train(self, num_epochs: int, train_data_loader, val_data_loader, save_path: str):
        # Checkpoint callback
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", dirpath=os.path.join(save_path, "checkpoints"), filename="Dmorp_model-{epoch:02d}-{val_loss:.4f}", save_top_k=3, mode="min")
        # Trainer
        # If not mac, using ddp_find_unused_parameters_true
        strategy = "ddp_find_unused_parameters_true" if os.uname().sysname != "Darwin" else "auto"
        accelerator = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
        trainer = L.Trainer(
            max_epochs=num_epochs,
            logger=WandbLogger(name=self.experiment_name(), project=self.logger_project, save_dir=os.path.join(save_path, "logs")),
            callbacks=[checkpoint_callback],
            strategy=strategy,
            log_every_n_steps=5,
            accelerator=accelerator,
        )
        trainer.fit(self.lpcd_noise_net, train_dataloaders=train_data_loader, val_dataloaders=val_data_loader)

    def test(self, test_data_loader, save_path: str):
        strategy = "ddp_find_unused_parameters_true" if os.uname().sysname != "Darwin" else "auto"
        os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
        trainer = L.Trainer(logger=WandbLogger(name=self.experiment_name(), project=self.logger_project, save_dir=os.path.join(save_path, "logs")), strategy=strategy)
        checkpoint_path = f"{save_path}/checkpoints"
        # Select the best checkpoint
        checkpoints = os.listdir(checkpoint_path)
        sorted_checkpoints = sorted(checkpoints, key=lambda x: float(x.split("=")[-1].split(".ckpt")[0]))
        checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
        checkpoint_file = os.path.join(checkpoint_path, sorted_checkpoints[0])
        trainer.test(self.lpcd_noise_net.__class__.load_from_checkpoint(checkpoint_file, pcd_noise_net=self.pcd_noise_net, cfg=self.cfg), dataloaders=test_data_loader)

    def predict(
        self, anchor_coord: np.ndarray | None = None, anchor_normal: np.ndarray | None = None, anchor_feat: np.ndarray | None = None, super_index: np.ndarray | None = None, batch=None, **kwargs
    ):
        self.lpcd_noise_net.eval()
        # Assemble batch
        if batch is None:
            batch_size = kwargs.get("batch_size", 8)
            anchor_coord_list = []
            anchor_normal_list = []
            anchor_feat_list = []
            anchor_super_index_list = []
            anchor_batch_idx_list = []
            num_anchor_cluster = 0
            for i in range(batch_size):
                anchor_coord_list.append(anchor_coord)
                anchor_normal_list.append(anchor_normal)
                anchor_feat_list.append(anchor_feat)
                anchor_super_index_list.append(super_index + num_anchor_cluster)
                anchor_batch_idx_list.append(np.array([i] * anchor_coord.shape[0], dtype=np.int64))
                num_anchor_cluster = num_anchor_cluster + np.max(super_index) + 1
            batch = {
                "anchor_coord": np.concatenate(anchor_coord_list, axis=0),
                "anchor_normal": np.concatenate(anchor_normal_list, axis=0),
                "anchor_feat": np.concatenate(anchor_feat_list, axis=0),
                "anchor_super_index": np.concatenate(anchor_super_index_list, axis=0),
                "anchor_batch_index": np.concatenate(anchor_batch_idx_list, axis=0),
            }
            # Put to torch
            for key in batch.keys():
                batch[key] = torch.from_numpy(batch[key])
        else:
            check_batch_idx = kwargs.get("check_batch_idx", 0)
            batch_size = kwargs.get("batch_size", 8)
            anchor_coord = batch["anchor_coord"]
            anchor_normal = batch["anchor_normal"]
            anchor_feat = batch["anchor_feat"]
            anchor_batch_index = batch["anchor_batch_index"]
            anchor_super_index_i = batch["anchor_super_index"][anchor_batch_index == check_batch_idx]
            anchor_super_index_i[:, 0] = anchor_super_index_i[:, 0] - torch.min(anchor_super_index_i[:, 0])
            anchor_super_index_i[:, 1] = anchor_super_index_i[:, 1] - torch.min(anchor_super_index_i[:, 1])
            anchor_coord_i = anchor_coord[anchor_batch_index == check_batch_idx]
            anchor_normal_i = anchor_normal[anchor_batch_index == check_batch_idx]
            anchor_feat_i = anchor_feat[anchor_batch_index == check_batch_idx]
            anchor_coord_list = []
            anchor_normal_list = []
            anchor_super_index_list = []
            anchor_feat_list = []
            anchor_batch_idx_list = []
            num_anchor_cluster = 0
            for i in range(batch_size):
                anchor_coord_list.append(anchor_coord_i)
                anchor_normal_list.append(anchor_normal_i)
                anchor_super_index_list.append(anchor_super_index_i + num_anchor_cluster)
                anchor_feat_list.append(anchor_feat_i)
                anchor_batch_idx_list.append(torch.tensor([i] * anchor_coord_i.shape[0], dtype=torch.int64))
                num_anchor_cluster = num_anchor_cluster + torch.max(anchor_super_index_i) + 1
            batch = {
                "anchor_coord": torch.cat(anchor_coord_list, dim=0),
                "anchor_normal": torch.cat(anchor_normal_list, dim=0),
                "anchor_super_index": torch.cat(anchor_super_index_list, dim=0),
                "anchor_feat": torch.cat(anchor_feat_list, dim=0),
                "anchor_batch_index": torch.cat(anchor_batch_idx_list, dim=0),
            }
        # Put to device
        for key in batch.keys():
            batch[key] = batch[key].to(self.lpcd_noise_net.device)
        pred_label = self.lpcd_noise_net(batch)
        batch_idx = batch["anchor_batch_index"]
        pred_label = to_dense_batch(pred_label, batch_idx)[0]  # to dense

        anchor_coord = batch["anchor_coord"]
        anchor_batch_coord = to_dense_batch(anchor_coord, batch_idx)[0]
        anchor_normal = batch["anchor_normal"]
        anchor_batch_normal = to_dense_batch(anchor_normal, batch_idx)[0]
        anchor_feat = batch["anchor_feat"]
        anchor_batch_feat = to_dense_batch(anchor_feat, batch_idx)[0]
        vis = kwargs.get("vis", False)
        if vis:
            for i in range(min(pred_label.shape[0], 4)):
                pred_label_i = pred_label[i]
                anchord_coord_i = anchor_batch_coord[i]
                pred_label_i = pred_label[i]
                self.lpcd_noise_net.view_result(anchord_coord_i, pred_label_i, vis_3d=True)

        return pred_label.detach().cpu().numpy(), anchor_batch_coord.detach().cpu().numpy(), anchor_batch_normal.detach().cpu().numpy(), anchor_batch_feat.detach().cpu().numpy()

    def load(self, checkpoint_path: str) -> None:
        print(f"Loading checkpoint from {checkpoint_path}")
        self.lpcd_noise_net.load_state_dict(torch.load(checkpoint_path)["state_dict"])

    def save(self, save_path: str) -> None:
        torch.save(self.lpcd_noise_net.state_dict(), save_path)

    def experiment_name(self):
        noise_net_name = self.cfg.MODEL.NOISE_NET.NAME
        init_args = self.cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name]
        crop_strategy = self.cfg.DATALOADER.AUGMENTATION.CROP_STRATEGY
        return f"PCDD_model_{crop_strategy}"

    ############################### UTILS ################################
    def seg_and_rank(self, coord, prob, crop_strategy="none", **kwargs):
        """Segment point and rank by probability"""
        prob_thresh = 0.5
        num_thresh = 100
        seg_list = []
        for i in range(coord.shape[0]):
            prob_i = prob[i]
            coord_i = coord[i]
            coord_i_crop = coord_i[(prob_i >= prob_thresh).squeeze()]
            if coord_i_crop.shape[0] < num_thresh:
                continue
            avg_prob = np.mean(prob_i[(prob_i >= prob_thresh).squeeze()])
            if crop_strategy == "none":
                seg_list.append({"coord": coord_i_crop, "prob": avg_prob})
                for k, v in kwargs.items():
                    v_i = v[i]
                    v_i_crop = v_i[(prob_i >= prob_thresh).squeeze()]
                    seg_list[-1][k] = v_i_crop
            elif crop_strategy == "bbox":
                # Compute bbox bounding crop
                min_coord = np.min(coord_i_crop, axis=0)
                max_coord = np.max(coord_i_crop, axis=0)
                # Crop by bbox
                crop_index = np.all((coord_i >= min_coord) & (coord_i <= max_coord), axis=1)
                coord_i_crop_bbox = coord_i[crop_index]
                seg_list.append({"coord": coord_i_crop_bbox, "prob": avg_prob})
                for k, v in kwargs.items():
                    v_i = v[i]
                    v_i_crop = v_i[crop_index]
                    seg_list[-1][k] = v_i_crop

        # Sort by prob
        seg_list = sorted(seg_list, key=lambda x: x["prob"], reverse=True)
        return seg_list
