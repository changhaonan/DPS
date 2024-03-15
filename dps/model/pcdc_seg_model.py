"""Classification model for generating segmentation using PCD diffusion net"""

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
from dps.model.network.geometric import batch2offset, offset2batch, to_dense_batch, to_flat_batch, voxel_grid, offset2mask
from dps.utils.pcd_utils import visualize_point_pyramid
from torch_scatter import segment_csr
from box import Box
import yaml
import torch_scatter
import dps.utils.misc_utils as utils
from torch.optim.lr_scheduler import LambdaLR
import open3d as o3d
import cv2
import wandb
from scipy.spatial.transform import Rotation as R
from dps.model.pcdd_seg_model import LPcdSegDiffusion, PCDDModel


class LPcdSegClassfication(LPcdSegDiffusion):
    """Lignthing module for PCD diffusion model; This uses superpoint. But use classification loss"""

    def __init__(self, pcd_noise_net: PcdSegNoiseNet, cfg: Box, **kwargs: Any) -> None:
        super().__init__(pcd_noise_net, cfg, **kwargs)

    def criterion(self, batch):
        """Override criterion to use classification loss."""
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
        if self.use_voxel_superpoint:
            super_index = self.voxel_grid_as_super_index(points)
        else:
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

        # Zero-query
        label_query = torch.zeros(super_coord.shape[0], 2, device=super_coord.device)  # 2 class: 0, 1
        zero_t = torch.Tensor([0]).to(torch.long).to(super_label.device)
        pred_label = self.pcd_noise_net(label_query, super_points, zero_t)
        # Compute binary classification loss
        batch_label = torch.where(batch_label > 0, torch.ones_like(batch_label), torch.zeros_like(batch_label))
        batch_label = torch.concatenate([1 - batch_label, batch_label], dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred_label[label_mask], batch_label[label_mask])
        return loss

    def forward(self, batch):
        """Ovrride inference for PCD classfication model."""
        # Assemble input
        coord = batch["anchor_coord"]
        normal = batch["anchor_normal"]
        feat = batch["anchor_feat"]
        batch_idx = batch["anchor_batch_index"]
        offset = batch2offset(batch_idx)
        points = [coord, feat, offset]
        if self.use_voxel_superpoint:
            super_index = self.voxel_grid_as_super_index(points)
        else:
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
        super_index_mask = offset2mask(super_offset)

        # Zero-query
        label_query = torch.zeros(super_coord.shape[0], 2, device=super_coord.device)
        # Batchify
        zero_t = torch.Tensor([0]).to(torch.long).to(super_coord.device)
        pred_label = self.pcd_noise_net(label_query, super_points, zero_t)
        # Perform max pooling
        pred_label = torch.argmax(pred_label, dim=-1, keepdim=True)
        pred_label = to_flat_batch(pred_label, super_index_mask)[0]
        pred_label = pred_label[super_index]
        return pred_label


class PCDCModel(PCDDModel):
    """PCD classification model"""

    def __init__(self, cfg, pcd_noise_net: PcdSegNoiseNet) -> None:
        super().__init__(cfg, pcd_noise_net=None)
        self.lpcd_noise_net = LPcdSegClassfication(pcd_noise_net, cfg)

    def experiment_name(self):
        noise_net_name = self.cfg.MODEL.NOISE_NET.NAME
        init_args = self.cfg.MODEL.NOISE_NET.INIT_ARGS[noise_net_name]
        crop_strategy = self.cfg.DATALOADER.AUGMENTATION.CROP_STRATEGY
        return f"PCDC_model_{crop_strategy}"
