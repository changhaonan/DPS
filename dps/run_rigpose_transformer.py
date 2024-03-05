"""Run pose transformer."""

import numpy as np
import os
import torch
import pickle
import argparse
from dps.data.pcd_dataset import PcdPairDataset
from dps.data.pcd_datalodaer import PcdPairCollator
from dps.model.network.rigpose_transformer import RigPoseTransformer
from dps.model.rgt_model import RGTModel
from detectron2.config import LazyConfig
from dps.utils.dps_utils import build_rpdiff_dataset
import random


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--seed", type=int, default=0)
    argparser.add_argument("--random_index", type=int, default=0)
    argparser.add_argument("--task_name", type=str, default="book_in_bookshelf", help="stack_can_in_cabinet, book_in_bookshelf")
    args = argparser.parse_args()
    # Set seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load config
    task_name = args.task_name
    root_path = os.path.dirname((os.path.abspath(__file__)))
    cfg_file = os.path.join(root_path, "config", f"pose_transformer_rpdiff_{task_name}.py")
    cfg = LazyConfig.load(cfg_file)
    # Overriding config
    cfg.MODEL.NOISE_NET.NAME = "RGTModel"
    cfg.DATALOADER.AUGMENTATION.CROP_PCD = True
    cfg.DATALOADER.BATCH_SIZE = 8
    # Load dataset & data loader
    train_dataset, val_dataset, test_dataset = build_rpdiff_dataset(root_path, cfg)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=True, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.DATALOADER.BATCH_SIZE, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=PcdPairCollator())

    # Build model
    net_name = cfg.MODEL.NOISE_NET.NAME
    net_init_args = cfg.MODEL.NOISE_NET.INIT_ARGS[net_name]
    pose_transformer = RigPoseTransformer(**net_init_args)
    rgt_model = RGTModel(cfg, pose_transformer)

    model_name = rgt_model.experiment_name()
    noise_net_name = cfg.MODEL.NOISE_NET.NAME
    save_dir = os.path.join(root_path, "test_data", task_name, "checkpoints", noise_net_name)
    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_dir, exist_ok=True)

    rgt_model.train(num_epochs=cfg.TRAIN.NUM_EPOCHS, train_data_loader=train_data_loader, val_data_loader=val_data_loader, save_path=save_path)
