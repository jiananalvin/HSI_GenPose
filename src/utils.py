import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from config import Config

def visualize_pose(pose, title="3D Human Pose"):
    """Visualize 3D pose (shape: [52, 3])"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot joints
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='blue', s=50)
    
    # Connect joints (SMPL skeleton connections)
    connections = [
        (0, 1), (1, 2), (2, 3),  # Spine
        (3, 4), (4, 5), (5, 6),  # Neck/head
        (3, 7), (7, 8), (8, 9), (9, 10),  # Left arm
        (3, 11), (11, 12), (12, 13), (13, 14),  # Right arm
        (0, 15), (15, 16), (16, 17), (17, 18),  # Left leg
        (0, 19), (19, 20), (20, 21), (21, 22)   # Right leg
    ]
    for (i, j) in connections:
        ax.plot(
            [pose[i, 0], pose[j, 0]],
            [pose[i, 1], pose[j, 1]],
            [pose[i, 2], pose[j, 2]],
            'r-'
        )
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()

def generate_pose(model, text):
    """Generate 3D pose from text using trained model"""
    model.unet.eval()
    config = Config()
    
    # Encode text
    text_embeddings = model.encode_text([text])
    
    # Initialize with random noise
    pose = torch.randn(1, config.NUM_JOINTS, config.POSE_DIM).to(config.DEVICE)
    
    # Denoise step-by-step
    model.scheduler.set_timesteps(config.SCHEDULER_NUM_TIMESTEPS)
    with torch.no_grad():
        for t in model.scheduler.timesteps:
            noise_pred = model.unet(
                pose,
                torch.tensor([t]).to(config.DEVICE),
                encoder_hidden_states=text_embeddings
            ).sample
            pose = model.scheduler.step(noise_pred, t, pose).prev_sample
    
    return pose.cpu().numpy().squeeze()
