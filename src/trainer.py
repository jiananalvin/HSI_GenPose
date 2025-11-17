import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import Config

class Trainer:
    def __init__(self, model, train_dataset, val_dataset):
        self.config = Config()
        self.model = model
        self.device = self.config.DEVICE
        
        # Data loaders (same as before)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=4
        )
        
        # Optimizer (same as before)
        self.optimizer = torch.optim.AdamW(
            self.model.unet.parameters(),
            lr=self.config.LEARNING_RATE
        )
        
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)

    def train_epoch(self, epoch):
        self.model.unet.train()
        total_loss = 0.0
        
        for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}")):
            poses = torch.tensor(batch["pose"]).to(self.device)  # (batch, 52, 3)
            captions = batch["caption"]
            text_embeddings = self.model.encode_text(captions)
            
            # 1. Sample random time steps t ∈ [0, 1)
            t = torch.rand(poses.shape[0], device=self.device)  # (batch,)
            
            # 2. Generate initial noise (x0) from Gaussian distribution
            x0 = torch.randn_like(poses) * self.config.FM_SIGMA  # (batch, 52, 3)
            
            # 3. Compute intermediate state xt = (1-t)*x0 + t*poses (linear interpolation)
            xt = (1 - t.view(-1, 1, 1)) * x0 + t.view(-1, 1, 1) * poses  # (batch, 52, 3)
            
            # 4. Compute target velocity: v_target = poses - x0 (desired transformation direction)
            v_target = poses - x0  # (batch, 52, 3)
            
            # 5. Model predicts velocity v_pred from xt, t, and text
            v_pred = self.model.unet(
                xt,
                t,  # Flow Matching uses continuous time (0→1) instead of discrete steps
                encoder_hidden_states=text_embeddings
            ).sample  # (batch, 52, 3)
            
            # 6. Loss: MSE between predicted and target velocity
            loss = torch.mean((v_pred - v_target) **2)
            total_loss += loss.item()
            
            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (step + 1) % self.config.LOG_INTERVAL == 0:
                print(f"Step {step+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.unet.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                poses = torch.tensor(batch["pose"]).to(self.device)
                captions = batch["caption"]
                text_embeddings = self.model.encode_text(captions)
                
                noise = torch.randn_like(poses).to(self.device)
                timesteps = torch.randint(
                    0, self.config.SCHEDULER_NUM_TIMESTEPS,
                    (poses.shape[0],), device=self.device
                ).long()
                
                noisy_poses = self.model.scheduler.add_noise(poses, noise, timesteps)
                noise_pred = self.model.unet(
                    noisy_poses,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                loss = torch.mean((noise_pred - noise)** 2)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)

    def train(self):
        for epoch in range(1, self.config.EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            print(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n")
            
            checkpoint_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                f"fm_epoch_{epoch}.pth"
            )
            self.model.save_checkpoint(checkpoint_path)
            print(f"Saved Flow Matching checkpoint to {checkpoint_path}")
