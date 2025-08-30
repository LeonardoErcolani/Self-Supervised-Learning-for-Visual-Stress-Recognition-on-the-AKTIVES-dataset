import pandas as pd
import numpy as np
from tqdm import tqdm
import os

def create_single_interval_features(per_second_df, interval_df, feature_prefix, output_path):
    """
    Create single-interval features by selecting the 10th frame (Frame_Index==9) if it exists,
    otherwise select the last available frame. This matches the original single interval logic.
    
    Args:
        per_second_df: DataFrame with per-second features
        interval_df: DataFrame with interval labels and metadata
        feature_prefix: Prefix for feature columns (e.g., 'face_feat_', 'feat_', 'upperbody_feat_')
        output_path: Path to save the output CSV
    
    Returns:
        DataFrame with single interval features
    """
    print(f"Creating single interval features for {feature_prefix}...")
    
    # Get column structure
    original_columns = interval_df.columns.tolist()
    feature_cols = [col for col in original_columns if col.startswith(feature_prefix)]
    metadata_cols = [col for col in original_columns if not col.startswith(feature_prefix)]
    
    print(f"Processing {len(feature_cols)} feature columns")
    print(f"Processing {len(metadata_cols)} metadata columns")
    
    # Create mapping from ID to metadata and labels
    label_mapping = {}
    for _, row in interval_df.iterrows():
        id_key = row['ID']
        label_mapping[id_key] = {col: row[col] for col in metadata_cols if col != 'ID'}
    
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
    print(f"Total single interval samples: {len(result_df)}")
    
    return result_df

def create_triple_interval_features(per_second_df, interval_df, feature_prefix, output_path):
    """
    Create three intervals within each 10-second period.
    This will generate 3 samples per 10-second interval:
    - Interval 1: Frame_Index 0-2 (first third) - "early"
    - Interval 2: Frame_Index 3-6 (second third) - "middle" 
    - Interval 3: Frame_Index 7-9 (third third) - "late"
    
    Args:
        per_second_df: DataFrame with per-second features
        interval_df: DataFrame with interval labels and metadata
        feature_prefix: Prefix for feature columns
        output_path: Path to save the output CSV
    
    Returns:
        DataFrame with triple interval features
    """
    print(f"Creating triple intervals for {feature_prefix}...")
    
    # Get the exact column structure from the original interval data
    original_columns = interval_df.columns.tolist()
    feature_cols = [col for col in original_columns if col.startswith(feature_prefix)]
    metadata_cols = [col for col in original_columns if not col.startswith(feature_prefix)]
    
    print(f"Original dataset has {len(metadata_cols)} metadata columns and {len(feature_cols)} feature columns")
    
    # Create a mapping from ID to all metadata and labels
    label_mapping = {}
    for _, row in interval_df.iterrows():
        id_key = row['ID']
        label_mapping[id_key] = {col: row[col] for col in metadata_cols if col != 'ID'}
    
    # Group by ID to process each 10-second interval
    grouped = per_second_df.groupby('ID')
    all_intervals = []
    
    for id_name, group in tqdm(grouped, desc=f"Processing {feature_prefix} triple intervals"):
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

def create_overlapping_interval_features(per_second_df, interval_df, feature_prefix, output_path):
    """
    Create overlapping intervals using sliding window approach.
    This creates three overlapping 5-second windows from each 10-second interval.
    
    Args:
        per_second_df: DataFrame with per-second features
        interval_df: DataFrame with interval labels and metadata
        feature_prefix: Prefix for feature columns
        output_path: Path to save the output CSV
    
    Returns:
        DataFrame with overlapping interval features
    """
    print(f"Creating overlapping intervals for {feature_prefix}...")
    
    # Get the exact column structure from the original interval data
    original_columns = interval_df.columns.tolist()
    feature_cols = [col for col in original_columns if col.startswith(feature_prefix)]
    metadata_cols = [col for col in original_columns if not col.startswith(feature_prefix)]
    
    print(f"Original dataset has {len(metadata_cols)} metadata columns and {len(feature_cols)} feature columns")
    
    # Create a mapping from ID to all metadata and labels
    label_mapping = {}
    for _, row in interval_df.iterrows():
        id_key = row['ID']
        label_mapping[id_key] = {col: row[col] for col in metadata_cols if col != 'ID'}
    
    # Group by ID to process each 10-second interval
    grouped = per_second_df.groupby('ID')
    all_intervals = []
    
    for id_name, group in tqdm(grouped, desc=f"Processing {feature_prefix} overlapping intervals"):
        if id_name not in label_mapping:
            print(f"Warning: No label found for {id_name}")
            continue
            
        # Sort by frame index
        group = group.sort_values('Frame_Index')
        
        # Get the metadata info for this ID
        metadata_info = label_mapping[id_name]
        
        # Create three overlapping 5-second windows
        intervals = [
            (0, 5, "early"),    #Frame_Index 0-5 (seconds 0-5)
            (3, 8, "middle"),   #Frame_Index 3-8 (seconds 3-8)
            (5, 10, "late")     #Frame_Index 5-10 (seconds 5-10)
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

def print_dataset_statistics(result_df, feature_type):
    """Print comprehensive dataset statistics"""
    print(f"\n=== {feature_type.upper()} Dataset Statistics ===")
    print(f"Total samples: {len(result_df)}")
    
    if 'Participant ID' in result_df.columns:
        print(f"Unique participants: {result_df['Participant ID'].nunique()}")
    
    if 'Stress/No Stress' in result_df.columns:
        print(f"Stress distribution:")
        print(result_df['Stress/No Stress'].value_counts())
    
    if 'Reaction/No Reaction' in result_df.columns:
        print(f"Reaction distribution:")
        print(result_df['Reaction/No Reaction'].value_counts())
    
    if 'Condition' in result_df.columns:
        print(f"Condition distribution:")
        print(result_df['Condition'].value_counts())
    
    if 'Interval_Type' in result_df.columns:
        print(f"Interval types:")
        print(result_df['Interval_Type'].value_counts())
    
    # Check for missing values in critical columns
    print(f"\nMissing value check:")
    critical_cols = ['Participant ID', 'Stress/No Stress']
    for col in critical_cols:
        if col in result_df.columns:
            missing_count = result_df[col].isna().sum()
            if missing_count > 0:
                print(f"WARNING: {missing_count} missing values in {col}")
            else:
                print(f"✓ No missing values in {col}")
    
    if 'Interval_Type' in result_df.columns:
        missing_count = result_df['Interval_Type'].isna().sum()
        if missing_count > 0:
            print(f"WARNING: {missing_count} missing values in Interval_Type")
        else:
            print(f"✓ No missing values in Interval_Type")
