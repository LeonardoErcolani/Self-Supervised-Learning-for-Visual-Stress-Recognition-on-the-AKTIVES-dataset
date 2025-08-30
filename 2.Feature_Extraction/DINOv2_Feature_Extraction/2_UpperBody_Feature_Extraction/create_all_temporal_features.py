import pandas as pd
import sys
import os

# Add parent directory to path to import temporal_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from temporal_utils import (
    create_single_interval_features,
    create_triple_interval_features,
    create_overlapping_interval_features,
    print_dataset_statistics
)

def create_upperbody_dinov2_temporal_features():
    """
    Create all temporal features for upper body DINOv2 features using shared utilities.
    This script creates single, triple, and overlapping interval features.
    """
    print("=== Creating Upper Body DINOv2 Temporal Features ===\n")
    
    # Define paths for upper body DINOv2 features
    per_second_path = "Processed Data/DinoV2 UpperBody Features Extracted/all_upperbody_features_per_second.csv"
    interval_path = "Processed Data/DinoV2 UpperBody Features Extracted/all_upperbody_features_per_interval.csv"
    output_dir = "Processed Data/DinoV2 UpperBody Features Extracted"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input files exist
    if not os.path.exists(per_second_path):
        print(f"❌ Error: Per-second file not found: {per_second_path}")
        return
    
    if not os.path.exists(interval_path):
        print(f"❌ Error: Interval file not found: {interval_path}")
        return
    
    print("Loading upper body DINOv2 data...")
    per_second_df = pd.read_csv(per_second_path)
    interval_df = pd.read_csv(interval_path)
    
    print(f"Loaded {len(per_second_df)} per-second records")
    print(f"Loaded {len(interval_df)} interval records\n")
    
    # Create single interval features
    print("1. Creating Single Interval Features...")
    single_output = os.path.join(output_dir, "all_upperbody_features_at_sec_10.csv")
    single_df = create_single_interval_features(
        per_second_df, interval_df, "upperbody_feat_", single_output
    )
    print_dataset_statistics(single_df, "Upper Body DINOv2 Single Interval")
    
    # Create triple interval features
    print("\n2. Creating Triple Interval Features...")
    triple_output = os.path.join(output_dir, "all_upperbody_features_triple_intervals.csv")
    triple_df = create_triple_interval_features(
        per_second_df, interval_df, "upperbody_feat_", triple_output
    )
    print_dataset_statistics(triple_df, "Upper Body DINOv2 Triple Intervals")
    
    # Create overlapping interval features
    print("\n3. Creating Overlapping Interval Features...")
    overlapping_output = os.path.join(output_dir, "all_upperbody_features_overlapping_intervals.csv")
    overlapping_df = create_overlapping_interval_features(
        per_second_df, interval_df, "upperbody_feat_", overlapping_output
    )
    print_dataset_statistics(overlapping_df, "Upper Body DINOv2 Overlapping Intervals")
    
    print("\n✅ All upper body DINOv2 temporal features created successfully!")
    print(f"Files saved in: {output_dir}")
    
    return {
        'single': single_df,
        'triple': triple_df,
        'overlapping': overlapping_df
    }

if __name__ == "__main__":
    create_upperbody_dinov2_temporal_features()
