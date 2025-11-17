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
    """Generate 3D pose using Flow Matching"""
    model.unet.eval()
    config = Config()
    
    text_embeddings = model.encode_text([text])
    
    # Start with initial noise (x0)
    x = torch.randn(1, config.NUM_JOINTS, config.POSE_DIM).to(config.DEVICE) * config.FM_SIGMA
    
    # Iterate through time steps (0 → 1)
    num_steps = config.FM_TIME_STEPS
    dt = 1.0 / num_steps  # Time step size
    
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.tensor([i / num_steps], device=config.DEVICE)  # Current time (0 to 1)
            
            # Predict velocity at current time
            v = model.unet(x, t, encoder_hidden_states=text_embeddings).sample
            
            # Update x using the velocity (Euler integration)
            x = x + v * dt
    
    return x.cpu().numpy().squeeze()

