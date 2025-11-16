import json
import os
import numpy as np
from torch.utils.data import Dataset
from config import Config

class PoseScriptDataset(Dataset):
    def __init__(self, split="train", use_human_annotations=False):
        self.config = Config()
        self.split = split
        self.data_dir = self.config.DATA_DIR
        
        # Load captions
        caption_file = self.config.HUMAN_CAPTION_FILE if use_human_annotations else self.config.AUTO_CAPTION_FILE
        with open(os.path.join(self.data_dir, caption_file), "r") as f:
            self.captions = json.load(f)
        
        # Load split IDs
        split_file = self.config.TRAIN_IDS_FILE if split == "train" else self.config.VAL_IDS_FILE
        with open(os.path.join(self.data_dir, split_file), "r") as f:
            self.ids = json.load(f)
        
        # Load pose mappings (to link IDs to AMASS poses)
        with open(os.path.join(self.data_dir, self.config.POSE_MAPPING_FILE), "r") as f:
            self.pose_mappings = json.load(f)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pose_id = self.ids[idx]
        caption = self.captions[pose_id]["caption"]
        
        # In practice: Load actual 3D pose from AMASS using self.pose_mappings[pose_id]
        # For this example, we generate dummy 3D pose data (replace with real AMASS loading)
        pose = np.random.randn(self.config.NUM_JOINTS, self.config.POSE_DIM).astype(np.float32)
        
        return {
            "pose_id": pose_id,
            "caption": caption,
            "pose": pose
        }
