import os
from detectron2.config import LazyConfig
from dps.data.pcd_dataset import PcdPairDataset


def build_rpdiff_dataset(root_path, cfg):
    task_name = cfg.ENV.TASK_NAME
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
        data_file_dict[split] = os.path.join(
            root_path,
            "test_data",
            dataset_folder,
            task_name,
            f"diffusion_dataset_{pcd_size}_{cfg.MODEL.DATASET_CONFIG}_{split}.pkl",
        )
    print("Data loaded from: ", data_file_dict)
    volume_augmentations_path = os.path.join(root_path, "config", volume_augmentation_file) if volume_augmentation_file is not None else None
    train_dataset = PcdPairDataset(
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
    val_dataset = PcdPairDataset(
        data_file_list=[data_file_dict["val"]],
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
    test_dataset = PcdPairDataset(
        data_file_list=[data_file_dict["test"]],
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
    return train_dataset, val_dataset, test_dataset
