from src.dataset import PoseScriptDataset
from src.model import TextConditionedPoseFlowMatching  # Updated model
from src.trainer import Trainer
from src.utils import visualize_pose, generate_pose

def main():
    print("Loading datasets...")
    train_dataset = PoseScriptDataset(split="train", use_human_annotations=False)
    val_dataset = PoseScriptDataset(split="val", use_human_annotations=False)
    
    print("Initializing Flow Matching model...")
    model = TextConditionedPoseFlowMatching()  # Updated
    
    print("Starting Flow Matching training...")
    trainer = Trainer(model, train_dataset, val_dataset)
    trainer.train()
    
    # Generate example
    generated_pose = generate_pose(model, "a person standing with arms raised")
    visualize_pose(generated_pose, title="Flow Matching: Arms Raised")

if __name__ == "__main__":
    main()
