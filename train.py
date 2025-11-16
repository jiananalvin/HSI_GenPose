from src.dataset import PoseScriptDataset
from src.model import TextConditionedPoseDiffusion
from src.trainer import Trainer
from src.utils import visualize_pose, generate_pose

def main():
    # Initialize dataset
    print("Loading datasets...")
    train_dataset = PoseScriptDataset(split="train", use_human_annotations=False)
    val_dataset = PoseScriptDataset(split="val", use_human_annotations=False)
    
    # Initialize model
    print("Initializing model...")
    model = TextConditionedPoseDiffusion()
    
    # Initialize trainer and start training
    print("Starting training...")
    trainer = Trainer(model, train_dataset, val_dataset)
    trainer.train()
    
    # Example generation after training
    print("Generating example pose...")
    generated_pose = generate_pose(model, "a person standing with arms raised")
    visualize_pose(generated_pose, title="Generated Pose: Arms Raised")

if __name__ == "__main__":
    main()
