import torch

class Config:
    # Data
    DATA_DIR = "data"
    AUTO_CAPTION_FILE = "posescript_auto_100k.json"
    HUMAN_CAPTION_FILE = "posescript_human_6293.json"
    TRAIN_IDS_FILE = "train_ids_100k.json"
    VAL_IDS_FILE = "val_ids_100k.json"
    POSE_MAPPING_FILE = "ids_2_dataset_sequence_and_frame_index_100k.json"
    NUM_JOINTS = 52  # SMPL has 52 joints
    POSE_DIM = 3     # x, y, z coordinates

    # Model
    TEXT_ENCODER_NAME = "openai/clip-vit-large-patch14"
    UNET_IN_CHANNELS = 3
    UNET_OUT_CHANNELS = 3
    UNET_CROSS_ATTENTION_DIM = 768  # CLIP's hidden size
    SCHEDULER_NUM_TIMESTEPS = 1000

    # Training
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-5
    EPOCHS = 10
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "checkpoints"

    # Logging
    LOG_INTERVAL = 100  # Steps between loss logs

    # Flow Matching settings
    FM_TIME_STEPS = 100  # Fewer steps than diffusion (faster!)
    FM_SIGMA = 0.01      # Noise scale for initial distribution
