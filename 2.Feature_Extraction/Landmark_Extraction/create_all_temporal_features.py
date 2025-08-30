import pandas as pd
import sys
import os
import numpy as np
from tqdm import tqdm

# Add parent directory to path to import temporal_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from temporal_utils import print_dataset_statistics

def create_landmark_single_interval_features(per_second_df, interval_df, output_path):
    """
    Create single-interval features for landmarks by selecting the 10th frame (Frame_Index==9) if it exists,
    otherwise select the last available frame. This matches the original single interval logic.
    """
    print("Creating single interval features for landmarks...")
    
    # Get column structure - landmarks don't have a prefix, so identify by excluding metadata
    original_columns = interval_df.columns.tolist()
    metadata_cols = ['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction']
    feature_cols = [col for col in per_second_df.columns if col not in metadata_cols and col != 'Frame_Index']
    
    print(f"Processing {len(feature_cols)} landmark feature columns")
    print(f"Processing {len(metadata_cols)} metadata columns")
    
    selected_rows = []
    
    # Use the original interval ordering to preserve row order
    for _, interval_row in interval_df.iterrows():
        interval_id = interval_row['ID']
        
        # Get the per-second data for this interval
        group = per_second_df[per_second_df['ID'] == interval_id]
        if len(group) == 0:
            continue
            
        group = group.sort_values('Frame_Index')
        
        # Select the 10th frame (Frame_Index==9) if it exists, otherwise select the last available frame
        if (group['Frame_Index'] == 9).any():
            selected_row = group[group['Frame_Index'] == 9].iloc[0]
        else:
            selected_row = group.iloc[-1]
        
        selected_rows.append(selected_row)
    
    # Create result DataFrame
    result_df = pd.DataFrame(selected_rows)
    result_df.reset_index(drop=True, inplace=True)
    
    # Save the result
    result_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    print(f"Total landmark single interval samples: {len(result_df)}")
    
    return result_df

def create_landmark_triple_interval_features(per_second_df, interval_df, output_path):
    """
    Create three intervals within each 10-second period for landmark features.
    This will generate 3 samples per 10-second interval:
    - Interval 1: Frame_Index 0-2 (first third) - "early"
    - Interval 2: Frame_Index 3-6 (second third) - "middle" 
    - Interval 3: Frame_Index 7-9 (third third) - "late"
    """
    print("Creating triple intervals for landmark features...")
    
    # Get column structure - landmarks don't have a prefix, so identify by excluding metadata
    original_columns = interval_df.columns.tolist()
    metadata_cols = ['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction']
    feature_cols = [col for col in per_second_df.columns if col not in metadata_cols and col != 'Frame_Index']
    
    print(f"Original dataset has {len(metadata_cols)} metadata columns and {len(feature_cols)} feature columns")
    
    # Create a mapping from ID to all metadata and labels
    label_mapping = {}
    for _, row in interval_df.iterrows():
        id_key = row['ID']
        label_mapping[id_key] = {col: row[col] for col in metadata_cols if col != 'ID'}
    
    # Group by ID to process each 10-second interval
    grouped = per_second_df.groupby('ID')
    all_intervals = []
    
    for id_name, group in tqdm(grouped, desc="Processing landmark triple intervals"):
        if id_name not in label_mapping:
            print(f"Warning: No label found for {id_name}")
            continue
            
        # Sort by frame index (which represents the second within the 10-second interval)
        group = group.sort_values('Frame_Index')
        
        # Get the metadata info for this ID
        metadata_info = label_mapping[id_name]
        
        # Create three non-overlapping intervals based on Frame_Index
        intervals = [
            (0, 2, "early"),    # Frame_Index 0-2 (seconds 0-3)
            (3, 6, "middle"),   # Frame_Index 3-6 (seconds 3-7)
            (7, 9, "late")      # Frame_Index 7-9 (seconds 7-10)
        ]
        
        for start_frame, end_frame, interval_name in intervals:
            # Filter frames within this interval using Frame_Index
            interval_frames = group[(group['Frame_Index'] >= start_frame) & (group['Frame_Index'] <= end_frame)]
            
            if len(interval_frames) == 0:
                print(f"Warning: No frames found for {id_name} interval {start_frame}-{end_frame}")
                continue
            
            # Calculate mean of features for this interval
            interval_features = interval_frames[feature_cols].mean()
            
            # Create new row with EXACT same structure as original
            new_row = {}
            
            # Add ID with interval suffix
            new_row['ID'] = f"{id_name}_{interval_name}"
            
            # Copy all original metadata fields exactly
            for col in metadata_cols:
                if col != 'ID':
                    new_row[col] = metadata_info[col]
            
            # Add interval-specific metadata (additional fields)
            new_row['Interval_Type'] = interval_name
            new_row['Original_ID'] = id_name
            new_row['Start_Frame'] = start_frame
            new_row['End_Frame'] = end_frame
            new_row['Num_Frames_Used'] = len(interval_frames)
            
            # Add the averaged features in the same order as original
            for feat_col in feature_cols:
                new_row[feat_col] = interval_features[feat_col]
            
            all_intervals.append(new_row)
    
    # Create DataFrame with columns in the same order as original + new interval fields
    result_df = pd.DataFrame(all_intervals)
    
    # Reorder columns to match original structure + interval fields
    new_metadata_cols = ['ID'] + [col for col in metadata_cols if col != 'ID'] + ['Interval_Type', 'Original_ID', 'Start_Frame', 'End_Frame', 'Num_Frames_Used']
    final_column_order = new_metadata_cols + feature_cols
    
    # Ensure all columns exist and reorder
    result_df = result_df.reindex(columns=final_column_order)
    
    print(f"Created {len(result_df)} triple intervals from {len(interval_df)} original intervals")
    print(f"Data augmentation factor: {len(result_df) / len(interval_df):.1f}x")
    
    # Save the result
    result_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    return result_df

def create_landmark_overlapping_interval_features(per_second_df, interval_df, output_path):
    """
    Create overlapping intervals using sliding window approach for landmark features.
    This creates three overlapping 5-second windows from each 10-second interval.
    """
    print("Creating overlapping intervals for landmark features...")
    
    # Get column structure - landmarks don't have a prefix, so identify by excluding metadata
    original_columns = interval_df.columns.tolist()
    metadata_cols = ['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction']
    feature_cols = [col for col in per_second_df.columns if col not in metadata_cols and col != 'Frame_Index']
    
    print(f"Original dataset has {len(metadata_cols)} metadata columns and {len(feature_cols)} feature columns")
    
    # Create a mapping from ID to all metadata and labels
    label_mapping = {}
    for _, row in interval_df.iterrows():
        id_key = row['ID']
        label_mapping[id_key] = {col: row[col] for col in metadata_cols if col != 'ID'}
    
    # Group by ID to process each 10-second interval
    grouped = per_second_df.groupby('ID')
    all_intervals = []
    
    for id_name, group in tqdm(grouped, desc="Processing landmark overlapping intervals"):
        if id_name not in label_mapping:
            print(f"Warning: No label found for {id_name}")
            continue
            
        # Sort by frame index
        group = group.sort_values('Frame_Index')
        
        # Get the metadata info for this ID
        metadata_info = label_mapping[id_name]
        
        # Create three overlapping 5-second windows
        intervals = [
            (0, 5, "early"),    # Frame_Index 0-4 (5 frames)
            (3, 8, "middle"),   # Frame_Index 3-7 (5 frames)
            (5, 10, "late")     # Frame_Index 5-9 (5 frames)
        ]
        
        for start_frame, end_frame, interval_name in intervals:
            # Filter frames within this interval
            interval_frames = group[(group['Frame_Index'] >= start_frame) & (group['Frame_Index'] < end_frame)]
            
            if len(interval_frames) == 0:
                print(f"Warning: No frames found for {id_name} interval {start_frame}-{end_frame}")
                continue
            
            # Calculate mean of features for this interval
            interval_features = interval_frames[feature_cols].mean()
            
            # Create new row
            new_row = {}
            new_row['ID'] = f"{id_name}_{interval_name}"
            
            # Copy all original metadata fields
            for col in metadata_cols:
                if col != 'ID':
                    new_row[col] = metadata_info[col]
            
            # Add interval-specific metadata
            new_row['Interval_Type'] = interval_name
            new_row['Original_ID'] = id_name
            new_row['Start_Frame'] = start_frame
            new_row['End_Frame'] = end_frame
            new_row['Num_Frames_Used'] = len(interval_frames)
            
            # Add the averaged features
            for feat_col in feature_cols:
                new_row[feat_col] = interval_features[feat_col]
            
            all_intervals.append(new_row)
    
    # Create DataFrame
    result_df = pd.DataFrame(all_intervals)
    
    # Reorder columns
    new_metadata_cols = ['ID'] + [col for col in metadata_cols if col != 'ID'] + ['Interval_Type', 'Original_ID', 'Start_Frame', 'End_Frame', 'Num_Frames_Used']
    final_column_order = new_metadata_cols + feature_cols
    result_df = result_df.reindex(columns=final_column_order)
    
    print(f"Created {len(result_df)} overlapping intervals from {len(interval_df)} original intervals")
    print(f"Data augmentation factor: {len(result_df) / len(interval_df):.1f}x")
    
    # Save the result
    result_df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    
    return result_df

def create_landmark_temporal_features():
    """
    Create all temporal features for landmark extraction using custom functions.
    This script creates single, triple, and overlapping interval features.
    """
    print("=== Creating Landmark Temporal Features ===\n")
    
    # Define paths for landmark features
    per_second_path = "Processed Data/Expression_Pose_Analysis/all_features_per_second.csv"
    interval_path = "Processed Data/Expression_Pose_Analysis/all_features_per_interval.csv"
    output_dir = "Processed Data/Expression_Pose_Analysis"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input files exist
    if not os.path.exists(per_second_path):
        print(f"❌ Error: Per-second file not found: {per_second_path}")
        return
    
    if not os.path.exists(interval_path):
        print(f"❌ Error: Interval file not found: {interval_path}")
        return
    
    print("Loading landmark data...")
    per_second_df = pd.read_csv(per_second_path)
    interval_df = pd.read_csv(interval_path)
    
    print(f"Loaded {len(per_second_df)} per-second records")
    print(f"Loaded {len(interval_df)} interval records\n")
    
    # Create single interval features
    print("1. Creating Single Interval Features...")
    single_output = os.path.join(output_dir, "all_features_at_sec_10.csv")
    single_df = create_landmark_single_interval_features(
        per_second_df, interval_df, single_output
    )
    print_dataset_statistics(single_df, "Landmark Single Interval")
    
    # Create triple interval features
    print("\n2. Creating Triple Interval Features...")
    triple_output = os.path.join(output_dir, "all_features_triple_intervals.csv")
    triple_df = create_landmark_triple_interval_features(
        per_second_df, interval_df, triple_output
    )
    print_dataset_statistics(triple_df, "Landmark Triple Intervals")
    
    # Create overlapping interval features
    print("\n3. Creating Overlapping Interval Features...")
    overlapping_output = os.path.join(output_dir, "all_features_overlapping_intervals.csv")
    overlapping_df = create_landmark_overlapping_interval_features(
        per_second_df, interval_df, overlapping_output
    )
    print_dataset_statistics(overlapping_df, "Landmark Overlapping Intervals")
    
    print("\n✅ All landmark temporal features created successfully!")
    print(f"Files saved in: {output_dir}")
    
    return {
        'single': single_df,
        'triple': triple_df,
        'overlapping': overlapping_df
    }

if __name__ == "__main__":
    create_landmark_temporal_features()
