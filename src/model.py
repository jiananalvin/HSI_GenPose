import torch
from diffusers import UNet3D, FlowMatchingScheduler  # New scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from config import Config

class TextConditionedPoseFlowMatching:  # Renamed for clarity
    def __init__(self):
        self.config = Config()
        self.device = self.config.DEVICE

        # Text encoder (same as before)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.TEXT_ENCODER_NAME)
        self.text_encoder = CLIPTextModel.from_pretrained(self.config.TEXT_ENCODER_NAME).to(self.device)

        # UNet (same architecture works for FM)
        self.unet = UNet3D(
            sample_size=self.config.NUM_JOINTS,
            in_channels=self.config.UNET_IN_CHANNELS,
            out_channels=self.config.UNET_OUT_CHANNELS,
            cross_attention_dim=self.config.UNET_CROSS_ATTENTION_DIM,
            num_layers_per_block=2,
            block_out_channels=(64, 128, 256),
            time_embedding_dim=128
        ).to(self.device)

        # Flow Matching scheduler (replaces DDPMScheduler)
        self.scheduler = FlowMatchingScheduler(
            num_train_timesteps=self.config.FM_TIME_STEPS,
            sigma=self.config.FM_SIGMA  # Initial noise scale
        )

    def encode_text(self, texts):
        """Convert text descriptions to embeddings"""
        inputs = self.tokenizer(
            texts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            embeddings = self.text_encoder(** inputs)[0]
        return embeddings

    def save_checkpoint(self, path):
        """Save model weights"""
        torch.save({
            "unet": self.unet.state_dict(),
            "text_encoder": self.text_encoder.state_dict()
        }, path)

    def load_checkpoint(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.unet.load_state_dict(checkpoint["unet"])
        self.text_encoder.load_state_dict(checkpoint["text_encoder"])
