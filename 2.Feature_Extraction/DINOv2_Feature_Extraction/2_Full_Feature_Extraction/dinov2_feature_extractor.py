import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from tqdm import tqdm

class DINOv2FeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base').to(self.device)
        self.model.eval()
    
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            return features[0]
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def extract_features_for_interval(self, interval_dir):
        """Extract features for all frames in an interval and return both average and sequence"""
        features_list = []
        
        # Get all frame files in the interval directory
        frame_files = sorted([f for f in os.listdir(interval_dir) if f.endswith('.png')])
        
        for frame_file in frame_files:
            frame_path = os.path.join(interval_dir, frame_file)
            features = self.extract_features(frame_path)
            if features is not None:
                features_list.append(features)
        
        if not features_list:
            return None, None
            
        # Return both the sequence and the average
        return np.array(features_list), np.mean(features_list, axis=0)

def process_images_and_metadata():
    """Process images and append features to matching metadata rows"""
    # Create output directory
    output_dir = "Processed Data/DinoV2 Features Extracted"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_csv("Processed Data/Expert Majority/all_labels.csv")
    
    # Initialize feature extractor
    print("Initializing feature extractor...")
    extractor = DINOv2FeatureExtractor()
    
    # Create a list to store per-second features
    per_second_features = []
    
    # Process each interval and append features to matching metadata row
    print("Processing images and extracting features...")
    for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        # Use proper metadata columns instead of parsing ID
        participant_id = row['Participant ID']  # e.g., C1
        game_type = row['Game']                 # e.g., LeapBall or CatchAPet
        
        # Convert game name to abbreviation for directory path
        game_abbr = "CP" if "catchapet" in game_type.lower() else "LP"
        second = int(row['Second'])
        
        # Construct interval directory path
        interval_dir = os.path.join("Processed Data/Frames", 
                                  f"{participant_id}_{game_abbr}_frames",
                                  f"{participant_id}_{game_abbr}_{second}")
        
        if os.path.exists(interval_dir):
            # Extract both sequence and average features
            features_sequence, features_avg = extractor.extract_features_for_interval(interval_dir)
            
            if features_avg is not None:
                # Add averaged features to the metadata row
                for i, feat in enumerate(features_avg):
                    metadata_df.at[idx, f'feat_{i}'] = feat
                
                # Store per-second features with metadata
                for frame_idx, frame_features in enumerate(features_sequence):
                    per_second_row = {
                        'ID': row['ID'],
                        'Participant ID': row['Participant ID'],
                        'Name': row['Name'],
                        'Condition': row['Condition'],
                        'Game': row['Game'],
                        'Minute': row['Minute'],
                        'Second': row['Second'],
                        'Stress/No Stress': row['Stress/No Stress'],
                        'Reaction/No Reaction': row['Reaction/No Reaction'],
                        'Frame_Index': frame_idx,
                        **{f'feat_{i}': feat for i, feat in enumerate(frame_features)}
                    }
                    per_second_features.append(per_second_row)
    
    # Drop rows with missing values
    print("Dropping rows with missing values...")
    metadata_df = metadata_df.dropna()
    print(f"Remaining rows after dropping missing values: {len(metadata_df)}")
    
    # Save the averaged features
    output_path = os.path.join(output_dir, "all_features_per_interval.csv")
    metadata_df.to_csv(output_path, index=False)
    print(f"Averaged features saved to: {output_path}")
    
    # Save the per-second features
    per_second_df = pd.DataFrame(per_second_features)
    per_second_path = os.path.join(output_dir, "all_features_per_second.csv")
    per_second_df.to_csv(per_second_path, index=False)
    print(f"Per-second features saved to: {per_second_path}")
    
    # Print summary statistics
    print(f"\n=== Feature Extraction Summary ===")
    print(f"Total intervals processed: {len(metadata_df)}")
    print(f"Total frames processed: {len(per_second_df)}")
    print(f"Feature dimensions: {metadata_df.filter(like='feat_').shape[1]}")
    
    # Print condition distribution
    print(f"\nCondition distribution:")
    condition_counts = metadata_df['Condition'].value_counts()
    for condition, count in condition_counts.items():
        print(f"  {condition}: {count} intervals")
    
    # Print game distribution
    print(f"\nGame distribution:")
    game_counts = metadata_df['Game'].value_counts()
    for game, count in game_counts.items():
        print(f"  {game}: {count} intervals")

if __name__ == "__main__":
    process_images_and_metadata() 