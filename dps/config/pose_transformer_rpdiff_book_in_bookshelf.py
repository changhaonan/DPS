# Most simplified version of obj_dp experiment
ENV = dict(
    TASK_NAME="book_in_bookshelf",
    NUM_OBJ=8,
    NUM_STRUCTURE=4,
    SEMANTIC_FEAT_DIM=10,
    SEMANTIC_FEAT_TYPE="one_hot",  # random, clip, one_hot
    GOAL_TYPE="superpoint",  # rpdiff, superpoint
)
PREPROCESS = dict(GRID_SIZE=0.02, TARGET_RESCALE=3.0, NUM_POINT_LOW_BOUND=40, NUM_POINT_HIGH_BOUND=400, NEARBY_RADIUS=0.03, USE_SOFT_LABEL=True)

DATALOADER = dict(
    BATCH_SIZE=4,
    NUM_WORKERS=0,  # Set to 0 if using ilab
    ADD_NORMALS=True,
    ADD_COLORS=False,
    CORR_RADIUS=0.05,
    AUGMENTATION=dict(
        IS_ELASTIC_DISTORTION=False,
        ELASTIC_DISTORTION_GRANULARITY=1.0,
        ELASTIC_DISTORTION_MAGNITUDE=1.0,
        IS_RANDOM_DISTORTION=False,
        RANDOM_DISTORTION_RATE=0.2,
        RANDOM_DISTORTION_MAG=0.1,
        VOLUME_AUGMENTATION_FILE="va_rotation.yaml",  # None
        RANDOM_SEGMENT_DROP_RATE=0.15,
        CROP_PCD=False,
        CROP_SIZE=0.75,
        CROP_NOISE=0.00,
        CROP_STRATEGY="bbox",  # bbox, radius, knn, knn_bbox, knn_bbox_max
        RANDOM_CROP_PROB=0.0,
        ROT_NOISE_LEVEL=0.8,
        TRANS_NOISE_LEVEL=0.00,
        ROT_AXIS="yz",
        KNN_K=20,
        NORMALIZE_COORD=True,
    ),
)
TRAIN = dict(
    NUM_EPOCHS=10000,
    WARM_UP_STEP=0,
    LR=1e-4,
    GRADIENT_CLIP_VAL=1.0,
)
MODEL = dict(
    DIFFUSION_PROCESS="ddpm",
    NUM_DIFFUSION_ITERS=100,
    NOISE_NET=dict(
        NAME="PCDSEGNOISENET",
        INIT_ARGS=dict(
            RPTModel=dict(
                grid_sizes=[0.1, 0.3],
                depths=[2, 3, 3],
                dec_depths=[1, 1],
                hidden_dims=[128, 256, 256],  # 1+ dim are the same
                n_heads=[8, 16, 16],
                ks=[16, 24, 32],
                in_dim=10,
                fusion_projection_dim=512,
            ),
            RGTModel=dict(
                grid_sizes=[0.1, 0.3],
                depths=[2, 3, 3],
                dec_depths=[1, 1],
                hidden_dims=[128, 256, 256],  # 1+ dim are the same
                n_heads=[8, 16, 16],
                ks=[16, 24, 32],
                in_dim=10,
                fusion_projection_dim=512,
            ),
            PCDSEGNOISENET=dict(
                condition_strategy="FiLM",  # FiLM, cross_attn, concat
                condition_pooling="max",
                grid_sizes=[0.15, 0.3],
                depths=[2, 3, 3],
                dec_depths=[1, 1],
                hidden_dims=[128, 256, 256],
                n_heads=[8, 16, 16],
                ks=[8, 8, 16],
                in_dim=10,
                hidden_dim_denoise=256,
                n_heads_denoise=8,
                num_denoise_layers=3,
            ),
            PCDSAMNOISENET=dict(
                grid_sizes=[0.1, 0.2],
                depths=[2, 3, 3],
                dec_depths=[1, 1],
                hidden_dims=[128, 256, 256],  # 1+ dim are the same
                n_heads=[8, 16, 16],
                ks=[8, 8, 16],
                in_dim=10,
                hidden_dim_denoise=256,
                num_sam_blocks=2,
                num_dit_blocks=2,
                seg_knn=8,
            ),
        ),
    ),
    SUPERPOINT_LAYER=0,
    PCD_SIZE=2048,
    TRAIN_SPLIT=0.7,
    VAL_SPLIT=0.2,
    TEST_SPLIT=0.1,
    DATASET_CONFIG="s25000-c1-r0.5",  # "s1000-c200-r0.5",  # "s300-c20-r0.5", #"s500-c20-r0.5" #"s1000-c1-r0.5", # "s250-c40-r2", # "s100-c20-r2",
    NUM_GRID=5,
    SAMPLE_SIZE=64,
    SAMPLE_STRATEGY="grid",  # random, grid
    VALID_CROP_THRESHOLD=0.5,
)
LOGGER = dict(
    PROJECT="tns",
)
CUDA_DEVICE = "cuda"
