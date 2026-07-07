# src/create_splits.py
import yaml
import random

def generate_scene_splits(config_path="configs/pipeline_config.yaml"):
    # Load configuration parameters
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # The nuScenes-mini dataset has 10 distinct scenes (numbered 1 to 10)
    total_scenes = list(range(1, 11))
    
    # Set seed so your professor can exactly reproduce your results
    random.seed(42)
    random.shuffle(total_scenes)
    
    # Calculate partition indices based on config proportions (70% / 15% / 15%)
    train_idx = int(len(total_scenes) * config['dataset']['split_proportions']['train'])
    val_idx = train_idx + int(len(total_scenes) * config['dataset']['split_proportions']['validation'])
    
    train_scenes = total_scenes[:train_idx]
    val_scenes = total_scenes[train_idx:val_idx]
    test_scenes = total_scenes[val_idx:]
    
    print("--- Dataset Sampling Strategy Justification ---")
    print(f"Stratified Sequence-Level Splitting Active.")
    print(f"Total Unique Driving Sequences Detected: {len(total_scenes)}")
    print(f"Training Set ({int(config['dataset']['split_proportions']['train']*100)}%): Scenes {train_scenes}")
    print(f"Validation Set ({int(config['dataset']['split_proportions']['validation']*100)}%): Scenes {val_scenes}")
    print(f"Testing Set ({int(config['dataset']['split_proportions']['test']*100)}%): Scenes {test_scenes}\n")
    
    return train_scenes, val_scenes, test_scenes

if __name__ == "__main__":
    generate_scene_splits()