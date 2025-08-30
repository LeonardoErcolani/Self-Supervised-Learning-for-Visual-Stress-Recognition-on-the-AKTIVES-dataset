#!/usr/bin/env python3
"""
Unified Experiment Runner - Core Training Engine

⚠️  IMPORTANT: DO NOT RUN THIS FILE DIRECTLY!
    Use 'python train.py' to run experiments instead.

This module provides utility functions for the training pipeline:
- run_experiment(): Execute a single experiment
- create_comprehensive_summary(): Generate summary visualizations
- Data processing and evaluation utilities

The train.py file orchestrates the entire pipeline using these functions.
"""

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import torch
import torch.nn as nn

# Configuration
np.random.seed(42)
torch.manual_seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return torch.mean(focal_loss)

def calculate_focal_loss(y_true, y_pred_proba):
    """Calculate focal loss using PyTorch implementation"""
    y_true_tensor = torch.FloatTensor(y_true.to_numpy() if hasattr(y_true, 'to_numpy') else np.array(y_true))
    y_pred_tensor = torch.FloatTensor(y_pred_proba.to_numpy() if hasattr(y_pred_proba, 'to_numpy') else np.array(y_pred_proba))
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss_fn(y_pred_tensor, y_true_tensor)
    return loss.item()



# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def load_and_preprocess_data(input_csv, feature_type="DINOv2"):
    """Load and preprocess data based on feature type"""
    logger.info(f"Loading {feature_type} data from {input_csv}")
    
    # Check if file exists
    if not os.path.exists(input_csv):
        logger.error(f"File not found: {input_csv}")
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    # Load data
    data = pd.read_csv(input_csv)
    
    # Handle feature type specific preprocessing
    if feature_type == "Expression_Pose":
        # Exclude problematic hand features for expression data
        excluded_features = ['left_hand_visible', 'right_hand_visible', 'left_hand_y', 'right_hand_y']
        found_excluded = [col for col in excluded_features if col in data.columns]
        if found_excluded:
            logger.info(f"Excluding {len(found_excluded)} problematic hand features: {found_excluded}")
            data = data.drop(columns=found_excluded)
        
        # Clean values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        inf_count = np.isinf(data[numeric_cols]).sum().sum()
        if inf_count > 0:
            logger.info(f"Found {inf_count} infinite values, replacing with NaN")
            data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        large_threshold = 1e6
        large_count = (np.abs(data[numeric_cols]) > large_threshold).sum().sum()
        if large_count > 0:
            logger.info(f"Found {large_count} extremely large values (>1e6), replacing with NaN")
            data[numeric_cols] = data[numeric_cols].where(np.abs(data[numeric_cols]) <= large_threshold, np.nan)
    
    # Clean data
    data = data.dropna()
    data = data.reset_index(drop=True)
    
    logger.info(f"Original dataset size: {len(data)}")
    
    # Remove baseline periods to avoid noise from calm periods
    baseline_periods = [10, 20, 30, 460, 470, 480]
    baseline_removal = data[data['Second'].isin(baseline_periods)]
    
    logger.info(f"Removing baseline periods:")
    logger.info(f"  Baseline periods (sec 10,20,30,460,470,480): {len(baseline_removal)} samples")
    
    data = data[~data['Second'].isin(baseline_periods)]
    data = data.reset_index(drop=True)
    
    logger.info(f"Dataset size after baseline removal: {len(data)}")
    logger.info(f"Conditions: {data['Condition'].value_counts().to_dict()}")
    
    # Show time range
    time_range = data['Second'].agg(['min', 'max'])
    logger.info(f"Time range: {time_range['min']} - {time_range['max']} seconds")
    
    return data

def load_and_preprocess_combined_data(dinov2_csv, second_csv, second_feature_type):
    """Load and preprocess combined DINOv2 + second feature data"""
    logger.info(f"Loading combined DINOv2 + {second_feature_type} data")
    
    # Load both datasets
    dinov2_data = load_and_preprocess_data(dinov2_csv, "DINOv2")
    second_data = load_and_preprocess_data(second_csv, second_feature_type)
    
    return dinov2_data, second_data

def get_feature_columns(data, feature_type):
    """Get feature columns based on feature type"""
    if feature_type == "DINOv2":
        feature_cols = [col for col in data.columns if col.startswith('feat_')]
    elif feature_type == "Face DINOv2":
        feature_cols = [col for col in data.columns if col.startswith('face_feat_')]
    elif feature_type == "Upper Body DINOv2":
        feature_cols = [col for col in data.columns if col.startswith('upperbody_feat_')]
    elif feature_type == "Emotions":
        feature_cols = [col for col in data.columns if col.startswith('prob_')]
    elif feature_type == "Expression_Pose":
        # Exclude metadata columns
        metadata_cols = ['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 
                        'Stress/No Stress', 'Reaction/No Reaction']
        if 'Interval_Type' in data.columns:
            metadata_cols.extend(['Interval_Type', 'Original_ID', 'Start_Frame', 'End_Frame', 'Num_Frames_Used'])
        
        # Get all columns that are not metadata
        feature_cols = [col for col in data.columns if col not in metadata_cols]
        
        # Additional filtering to ensure only numeric columns
        numeric_cols = data[feature_cols].select_dtypes(include=[np.number]).columns
        feature_cols = list(numeric_cols)
        
        logger.info(f"Expression_Pose: Found {len(feature_cols)} numeric feature columns")
        logger.info(f"Sample features: {feature_cols[:5]}")
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")
    
    # Validate that we found features
    if not feature_cols:
        raise ValueError(f"No features found for {feature_type}. Available columns: {list(data.columns)}")
    
    # Ensure all feature columns are numeric
    non_numeric = []
    for col in feature_cols:
        if not np.issubdtype(data[col].dtype, np.number):
            non_numeric.append(col)
    
    if non_numeric:
        logger.warning(f"Found non-numeric columns in {feature_type} features: {non_numeric}")
        logger.warning(f"These will be excluded from training")
        feature_cols = [col for col in feature_cols if col not in non_numeric]
    
    logger.info(f"{feature_type}: Using {len(feature_cols)} feature columns")
    return feature_cols

def create_participant_based_split(data, feature_type, test_size=0.1, random_state=42):
    """Create train/test split with deterministic proportion-based participant selection"""
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Get feature columns
    feature_cols = get_feature_columns(data, feature_type)
    X = data[feature_cols].values
    y = (data['Stress/No Stress'] == 'Stress').astype(int)
    participant_ids = data['Participant ID']
    
    # Analyze participants
    unique_participants = participant_ids.unique()
    participant_info = []
    
    for participant in unique_participants:
        participant_mask = participant_ids == participant
        participant_data = data[participant_mask]
        participant_condition = participant_data['Condition'].iloc[0]
        participant_stress_labels = y[participant_mask]
        stress_ratio = participant_stress_labels.mean()
        
        participant_info.append({
            'participant': participant,
            'condition': participant_condition,
            'dominant_stress': 'Stress' if stress_ratio > 0.5 else 'No_Stress',
            'stress_ratio': stress_ratio,
            'total_samples': len(participant_stress_labels),
            'stress_samples': participant_stress_labels.sum(),
            'no_stress_samples': len(participant_stress_labels) - participant_stress_labels.sum()
        })
    
    participant_df = pd.DataFrame(participant_info)
    participant_df['stratify_label'] = participant_df['condition'] + '_' + participant_df['dominant_stress']
    
    logger.info(f"Total participants: {len(unique_participants)}")
    
    # Calculate current proportions
    condition_proportions = participant_df['condition'].value_counts(normalize=True).to_dict()
    stress_proportions = participant_df['dominant_stress'].value_counts(normalize=True).to_dict()
    combined_proportions = participant_df['stratify_label'].value_counts(normalize=True).to_dict()
    
    logger.info(f"Condition proportions: {condition_proportions}")
    logger.info(f"Stress proportions: {stress_proportions}")
    logger.info(f"Combined proportions: {combined_proportions}")
    
    # Calculate target test set composition
    n_test_participants = max(1, int(len(unique_participants) * test_size))
    logger.info(f"Target test participants: {n_test_participants} ({test_size*100:.0f}% of {len(unique_participants)})")
    
    # Calculate how many participants needed for each stratify_label in test set
    target_test_composition = {}
    for label, proportion in combined_proportions.items():
        target_count = max(1, round(n_test_participants * proportion))
        target_test_composition[label] = target_count
    
    # Adjust if total exceeds target (due to rounding)
    total_target = sum(target_test_composition.values())
    if total_target > n_test_participants:
        sorted_labels = sorted(target_test_composition.keys(), 
                             key=lambda x: target_test_composition[x], reverse=True)
        excess = total_target - n_test_participants
        for label in sorted_labels:
            if excess <= 0:
                break
            if target_test_composition[label] > 1:
                target_test_composition[label] -= 1
                excess -= 1
    
    logger.info(f"Target test composition: {target_test_composition}")
    
    # Select participants for test set based on target composition
    test_participants = []
    remaining_participants = participant_df.copy()
    
    for stratify_label, target_count in target_test_composition.items():
        candidates = remaining_participants[remaining_participants['stratify_label'] == stratify_label]
        
        if len(candidates) == 0:
            logger.warning(f"No candidates available for {stratify_label}")
            continue
            
        n_to_select = min(target_count, len(candidates))
        
        if n_to_select > 0:
            selected_indices = np.random.choice(candidates.index, size=n_to_select, replace=False)
            selected_participants = candidates.loc[selected_indices, 'participant'].tolist()
            test_participants.extend(selected_participants)
            remaining_participants = remaining_participants.drop(selected_indices)
            
            logger.info(f"Selected {n_to_select} participants for {stratify_label}")
    
    # All remaining participants go to train set
    train_participants = remaining_participants['participant'].tolist()
    
    # Create train/test masks
    train_mask = participant_ids.isin(train_participants)
    test_mask = participant_ids.isin(test_participants)
    
    # Verify no overlap
    train_participant_set = set(participant_ids[train_mask])
    test_participant_set = set(participant_ids[test_mask])
    overlap = train_participant_set.intersection(test_participant_set)
    
    if overlap:
        raise ValueError(f"Participant overlap detected: {overlap}")
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    logger.info(f"Final split: {len(train_participant_set)} train participants, {len(test_participant_set)} test participants")
    logger.info(f"Train: {len(train_participant_set)} participants, {len(X_train)} samples")
    logger.info(f"Test: {len(test_participant_set)} participants, {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def create_participant_based_split_combined(dinov2_data, second_data, second_feature_type, test_size=0.1, random_state=42):
    """Create train/test split for combined features using participant-based splitting"""
    
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # Merge datasets on ID
    logger.info(f"Merging datasets on ID...")
    
    # Merge with suffixes
    combined_data = pd.merge(dinov2_data, second_data, on='ID', how='inner', 
                            suffixes=('_dinov2', '_second'))
    
    logger.info(f"Original DINOv2 samples: {len(dinov2_data)}")
    logger.info(f"Original {second_feature_type} samples: {len(second_data)}")
    logger.info(f"Combined samples after merge: {len(combined_data)}")
    
    # Check for any missing data
    missing_dinov2 = len(dinov2_data) - len(combined_data)
    missing_second = len(second_data) - len(combined_data)
    
    if missing_dinov2 > 0:
        logger.warning(f"⚠️  {missing_dinov2} DINOv2 samples lost during merge")
    if missing_second > 0:
        logger.warning(f"⚠️  {missing_second} {second_feature_type} samples lost during merge")
    
    logger.info("✅ Datasets successfully merged")
    
    # Debug: Show available columns after merge
    logger.info(f"Available columns after merge: {list(combined_data.columns)}")
    
    # Get feature columns
    dinov2_cols = get_feature_columns(combined_data, "DINOv2")
    second_cols = get_feature_columns(combined_data, second_feature_type)
    
    # Combine features
    X_dinov2 = combined_data[dinov2_cols].values
    X_second = combined_data[second_cols].values
    
    # Use stress labels from merged data
    y = (combined_data['Stress/No Stress_dinov2'] == 'Stress').astype(int)
    participant_ids = combined_data['Participant ID_dinov2']
    
    # Create participant-based split (same logic as single feature experiments)
    logger.info("Creating participant-based train/test split...")
    
    # Get unique participants and their conditions
    unique_participants = participant_ids.unique()
    participant_info = []
    
    for participant in unique_participants:
        participant_mask = participant_ids == participant
        participant_stress_labels = y[participant_mask]
        
        # Find the condition column (could be Condition_dinov2 or Condition_second)
        condition_col = None
        if 'Condition_dinov2' in combined_data.columns:
            condition_col = 'Condition_dinov2'
        elif 'Condition_second' in combined_data.columns:
            condition_col = 'Condition_second'
        elif 'Condition' in combined_data.columns:
            condition_col = 'Condition'
        else:
            # Try to find any column that contains 'Condition'
            condition_cols = [col for col in combined_data.columns if 'Condition' in col]
            if condition_cols:
                condition_col = condition_cols[0]
            else:
                logger.warning(f"No condition column found for participant {participant}")
                continue
        
        participant_condition = combined_data.loc[combined_data['Participant ID_dinov2'] == participant, condition_col].iloc[0]
        
        # Calculate stress ratio for this participant
        stress_ratio = participant_stress_labels.mean()
        
        participant_info.append({
            'participant': participant,
            'condition': participant_condition,
            'dominant_stress': 'Stress' if stress_ratio > 0.5 else 'No_Stress',
            'stress_ratio': stress_ratio,
            'total_samples': len(participant_stress_labels),
            'stress_samples': participant_stress_labels.sum(),
            'no_stress_samples': len(participant_stress_labels) - participant_stress_labels.sum()
        })
    
    if not participant_info:
        raise ValueError("No participant information could be extracted. Check if condition columns exist.")
    
    participant_df = pd.DataFrame(participant_info)
    participant_df['stratify_label'] = participant_df['condition'] + '_' + participant_df['dominant_stress']
    
    logger.info(f"Total participants: {len(unique_participants)}")
    logger.info(f"Participant info extracted: {len(participant_info)} participants")
    
    # Calculate current proportions
    condition_proportions = participant_df['condition'].value_counts(normalize=True).to_dict()
    stress_proportions = participant_df['dominant_stress'].value_counts(normalize=True).to_dict()
    combined_proportions = participant_df['stratify_label'].value_counts(normalize=True).to_dict()
    
    logger.info(f"Condition proportions: {condition_proportions}")
    logger.info(f"Stress proportions: {stress_proportions}")
    logger.info(f"Combined proportions: {combined_proportions}")
    
    # Calculate target test set composition
    n_test_participants = max(1, int(len(unique_participants) * test_size))
    logger.info(f"Target test participants: {n_test_participants} ({test_size*100:.0f}% of {len(unique_participants)})")
    
    # Calculate how many participants needed for each stratify_label in test set
    target_test_composition = {}
    for label, proportion in combined_proportions.items():
        target_count = max(1, round(n_test_participants * proportion))
        target_test_composition[label] = target_count
    
    # Adjust if total exceeds target (due to rounding)
    total_target = sum(target_test_composition.values())
    if total_target > n_test_participants:
        sorted_labels = sorted(target_test_composition.keys(), 
                             key=lambda x: target_test_composition[x], reverse=True)
        excess = total_target - n_test_participants
        for label in sorted_labels:
            if excess <= 0:
                break
            if target_test_composition[label] > 1:
                target_test_composition[label] -= 1
                excess -= 1
    
    logger.info(f"Target test composition: {target_test_composition}")
    
    # Select participants for test set based on target composition
    test_participants = []
    remaining_participants = participant_df.copy()
    
    for stratify_label, target_count in target_test_composition.items():
        candidates = remaining_participants[remaining_participants['stratify_label'] == stratify_label]
        
        if len(candidates) == 0:
            logger.warning(f"No candidates available for {stratify_label}")
            continue
            
        n_to_select = min(target_count, len(candidates))
        
        if n_to_select > 0:
            selected_indices = np.random.choice(candidates.index, size=n_to_select, replace=False)
            selected_participants = candidates.loc[selected_indices, 'participant'].tolist()
            test_participants.extend(selected_participants)
            remaining_participants = remaining_participants.drop(selected_indices)
            
            logger.info(f"Selected {n_to_select} participants for {stratify_label}")
    
    # All remaining participants go to train set
    train_participants = remaining_participants['participant'].tolist()
    
    # Create train/test masks
    train_mask = participant_ids.isin(train_participants)
    test_mask = participant_ids.isin(test_participants)
    
    # Verify no overlap
    train_participant_set = set(participant_ids[train_mask])
    test_participant_set = set(participant_ids[test_mask])
    overlap = train_participant_set.intersection(test_participant_set)
    
    if overlap:
        raise ValueError(f"Participant overlap detected: {overlap}")
    
    # Split the data
    X_dinov2_train, X_dinov2_test = X_dinov2[train_mask], X_dinov2[test_mask]
    X_second_train, X_second_test = X_second[train_mask], X_second[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    logger.info(f"Final split: {len(train_participant_set)} train participants, {len(test_participant_set)} test participants")
    logger.info(f"Train: {len(train_participant_set)} participants, {len(X_dinov2_train)} samples")
    logger.info(f"Test: {len(test_participant_set)} participants, {len(X_dinov2_test)} samples")
    
    return X_dinov2_train, X_dinov2_test, X_second_train, X_second_test, y_train, y_test

# ============================================================================
# MODEL TRAINING AND EVALUATION
# ============================================================================

def get_models(class_weights=None):
    """Get all models with consistent hyperparameters"""
    if class_weights is None:
        class_weights = {0: 1.0, 1: 1.0}
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(
            random_state=42, n_estimators=300, class_weight=class_weights,
            max_depth=15, min_samples_split=3, min_samples_leaf=2
        ),
        'XGBoost': XGBClassifier(
            random_state=42, n_estimators=300, scale_pos_weight=class_weights[1]/class_weights[0],
            max_depth=8, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8
        ),
        'XGBoost(Regularized)': XGBClassifier(
            random_state=42, n_estimators=500, scale_pos_weight=class_weights[1]/class_weights[0],
            max_depth=6, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
            reg_alpha=0.1, reg_lambda=1.0, min_child_weight=3, gamma=0.1
        )
    }
    
    return models

def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test, results_dir, 
                           experiment_name, all_reports_dir=None, feature_info="", data_augmentation="SMOTE"):
    """Train and evaluate a single model"""
    
    logger.info(f"Training {name}...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create pipeline with SMOTE
    pipeline = ImbPipeline([
        ('resampler', SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', model)
    ])
    
    # Train model
    pipeline.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_prob = pipeline.predict_proba(X_test_scaled)[:, 1]
    # Use model-specific optimal threshold from thesis results
    optimal_thresholds = {
        "Logistic Regression": 0.453,
        "SVM": 0.108,
        "Random Forest": 0.492,
        "XGBoost": 0.211,
        "XGBoost(Regularized)": 0.072
    }
    threshold = optimal_thresholds.get(name, 0.5)  # Default to 0.5 if model not found
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Focal Loss': calculate_focal_loss(y_test, y_prob),
        'Optimal Threshold': threshold,
        'Features': feature_info,
        'Total Features': X_train.shape[1],
        'Data Augmentation': data_augmentation,
        'Experiment': experiment_name
    }
    
    logger.info(f"{name} - F1: {metrics['F1-Score']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stress', 'Stress'],
                yticklabels=['No Stress', 'Stress'])
    plt.title(f'Confusion Matrix - {name} ({experiment_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}_{experiment_name.lower().replace(' ', '_')}.png"))
    plt.close()
    
    # Save detailed classification report
    if all_reports_dir:
        report = classification_report(y_test, y_pred, target_names=['No Stress', 'Stress'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Add metadata
        report_df['Model'] = name
        report_df['Experiment'] = experiment_name
        report_df['Optimal_Threshold'] = threshold
        
        # Save individual report
        report_filename = f"classification_report_{name.lower().replace(' ', '_')}_{experiment_name.lower().replace(' ', '_')}.png"
        report_df.to_csv(os.path.join(all_reports_dir, report_filename), index=True)
        
        logger.info(f"Classification report saved: {report_filename}")
    
    return metrics

def train_and_evaluate_combined_model(name, model, X_dinov2_train, X_dinov2_test, X_second_train, X_second_test,
                                    y_train, y_test, results_dir, experiment_name, all_reports_dir=None, 
                                    second_feature_type="", data_augmentation="SMOTE"):
    """Train and evaluate a model with combined features"""
    
    logger.info(f"Training {name} with combined features...")
    
    # Combine features
    X_train_combined = np.hstack([X_dinov2_train, X_second_train])
    X_test_combined = np.hstack([X_dinov2_test, X_second_test])
    
    logger.info(f"Combined feature dimensions:")
    logger.info(f"  DINOv2 features: {X_dinov2_train.shape[1]}")
    logger.info(f"  {second_feature_type} features: {X_second_train.shape[1]}")
    logger.info(f"  Total combined features: {X_train_combined.shape[1]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)
    
    # Create pipeline with SMOTE
    pipeline = ImbPipeline([
        ('resampler', SMOTE(random_state=42, k_neighbors=5)),
        ('classifier', model)
    ])
    
    # Train model
    pipeline.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_prob = pipeline.predict_proba(X_test_scaled)[:, 1]
    # Use model-specific optimal threshold
    optimal_thresholds = {
        "Logistic Regression": 0.453,
        "SVM": 0.108,
        "Random Forest": 0.492,
        "XGBoost": 0.211,
        "XGBoost(Regularized)": 0.072
    }
    threshold = optimal_thresholds.get(name, 0.5)  # Default to 0.5 if model not found
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {        
        'Features': f"{X_dinov2_train.shape[1]} DINOv2 + {X_second_train.shape[1]} {second_feature_type}",
        'Data Augmentation': data_augmentation,
        'Experiment': experiment_name,
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Focal Loss': calculate_focal_loss(y_test, y_prob),
        'Optimal Threshold': threshold,
        'DINOv2 Features': X_dinov2_train.shape[1],
        f'{second_feature_type} Features': X_second_train.shape[1],
        'Total Features': X_train_combined.shape[1],

    }
    
    logger.info(f"{name} - F1: {metrics['F1-Score']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stress', 'Stress'],
                yticklabels=['No Stress', 'Stress'])
    plt.title(f'Confusion Matrix - {name} ({experiment_name})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}_{experiment_name.lower().replace(' ', '_')}.png"))
    plt.close()
    
    # Save detailed classification report
    if all_reports_dir:
        report = classification_report(y_test, y_pred, target_names=['No Stress', 'Stress'], output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Add metadata
        report_df['Model'] = name
        report_df['Experiment'] = experiment_name
        report_df['Optimal_Threshold'] = threshold
        
        # Save individual report
        report_filename = f"classification_report_{name.lower().replace(' ', '_')}_{experiment_name.lower().replace(' ', '_')}.csv"
        report_df.to_csv(os.path.join(all_reports_dir, report_filename), index=True)
        
        logger.info(f"Classification report saved: {report_filename}")
    
    return metrics

# ============================================================================
# EXPERIMENT RUNNING
# ============================================================================

def run_single_feature_experiment(experiment_config, results_dir=None, all_reports_dir=None):
    """Run a complete experiment with single feature type"""
    
    experiment_name = experiment_config['name']
    input_csv = experiment_config['input_csv']
    feature_type = experiment_config['feature_type']
    use_pca = experiment_config.get('use_pca', False)
    temporal_strategy = experiment_config.get('temporal_strategy', 'Unknown')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING SINGLE FEATURE EXPERIMENT: {experiment_name}")
    logger.info(f"Feature Type: {feature_type}")
    logger.info(f"Temporal Strategy: {temporal_strategy}")
    logger.info(f"PCA: {use_pca}")
    logger.info(f"{'='*60}")
    
    # Create results directory for this experiment
    if results_dir is None:
        results_dir = f"Results/{feature_type.replace(' ', '_')}_Results/{experiment_name}"
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    data = load_and_preprocess_data(input_csv, feature_type)
    
    # Create participant-based split
    X_train, X_test, y_train, y_test = create_participant_based_split(data, feature_type)
    
    # Apply PCA if requested
    if use_pca:
        n_components = min(10, X_train.shape[1])
        logger.info(f"Applying PCA to reduce features from {X_train.shape[1]} to {n_components}")
        pca = PCA(n_components=n_components, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        
        feature_info = f"{feature_type}({X_train.shape[1]}) -> PCA({n_components})"
    else:
        feature_info = f"{feature_type}({X_train.shape[1]})"
    
    # Calculate class weights
    n_samples = len(y_train)
    n_classes = 2
    class_weights = {
        0: n_samples / (n_classes * np.sum(y_train == 0)),
        1: n_samples / (n_classes * np.sum(y_train == 1))
    }
    
    logger.info(f"Class weights: {class_weights}")
    
    # Get models
    models = get_models(class_weights)
    
    # Train and evaluate models
    results = []
    for name, model in models.items():
        metrics = train_and_evaluate_model(name, model, X_train, X_test, 
                                         y_train, y_test, results_dir, experiment_name, 
                                         all_reports_dir, feature_info, temporal_strategy)
        results.append(metrics)
    
    # Save results for this experiment
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(results_dir, f"{experiment_name.lower().replace(' ', '_')}_results.csv"), index=False)
    
    # Print summary for this experiment
    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS SUMMARY - {experiment_name}")
    logger.info(f"{'='*50}")
    for _, row in results_df.iterrows():
        logger.info(f"{row['Model']}: F1={row['F1-Score']:.4f}, Acc={row['Accuracy']:.4f}")
    
    return results

def run_combined_feature_experiment(experiment_config, results_dir=None, all_reports_dir=None):
    """Run a complete experiment with combined features"""
    
    experiment_name = experiment_config['name']
    dinov2_csv = experiment_config['dinov2_csv']
    second_csv = experiment_config['second_csv']
    second_feature_type = experiment_config['second_feature_type']
    use_pca = experiment_config.get('use_pca', False)
    temporal_strategy = experiment_config.get('temporal_strategy', 'Unknown')
    
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING COMBINED FEATURE EXPERIMENT: {experiment_name}")
    logger.info(f"Combining: DINOv2 + {second_feature_type}")
    logger.info(f"Temporal Strategy: {temporal_strategy}")
    logger.info(f"PCA: {use_pca}")
    logger.info(f"{'='*60}")
    
    # Create results directory for this experiment
    if results_dir is None:
        results_dir = f"Results/Combined_DINOv2_{second_feature_type.replace(' ', '_')}_Results/{experiment_name}"
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess combined data
    dinov2_data, second_data = load_and_preprocess_combined_data(dinov2_csv, second_csv, second_feature_type)
    
    # Create participant-based split
    X_dinov2_train, X_dinov2_test, X_second_train, X_second_test, y_train, y_test = create_participant_based_split_combined(
        dinov2_data, second_data, second_feature_type
    )
    
    # Apply PCA if requested
    if use_pca:
        logger.info("Applying PCA to reduce combined features to 10 components...")
        X_train_combined = np.hstack([X_dinov2_train, X_second_train])
        X_test_combined = np.hstack([X_dinov2_test, X_second_test])
        
        # Ensure data is numeric before PCA
        if not np.issubdtype(X_train_combined.dtype, np.number):
            logger.error(f"Non-numeric data detected before PCA. Data type: {X_train_combined.dtype}")
            logger.error(f"Sample data: {X_train_combined[:5, :5]}")
            raise ValueError("Data must be numeric for PCA transformation")
        
        pca = PCA(n_components=10, random_state=42)
        X_train_pca = pca.fit_transform(X_train_combined)
        X_test_pca = pca.transform(X_test_combined)
        
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        logger.info(f"Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
        
        # Update feature arrays
        X_dinov2_train, X_dinov2_test = X_train_pca, X_test_pca
        X_second_train, X_second_test = np.zeros((X_train_pca.shape[0], 0)), np.zeros((X_test_pca.shape[0], 0))
    
    # Calculate class weights
    n_samples = len(y_train)
    n_classes = 2
    class_weights = {
        0: n_samples / (n_classes * np.sum(y_train == 0)),
        1: n_samples / (n_classes * np.sum(y_train == 1))
    }
    
    logger.info(f"Class weights: {class_weights}")
    
    # Get models
    models = get_models(class_weights)
    
    # Train and evaluate each model
    results = []
    for name, model in models.items():
        try:
            result = train_and_evaluate_combined_model(
                name=name,
                model=model,
                X_dinov2_train=X_dinov2_train,
                X_dinov2_test=X_dinov2_test,
                X_second_train=X_second_train,
                X_second_test=X_second_test,
                y_train=y_train,
                y_test=y_test,
                results_dir=results_dir,
                experiment_name=experiment_name,
                all_reports_dir=all_reports_dir,
                second_feature_type=second_feature_type,
                data_augmentation=temporal_strategy
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error training {name}: {e}")
            continue
    
    # Save results for this experiment
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(results_dir, f"{experiment_name.lower().replace(' ', '_')}_results.csv"), index=False)
        
        # Print summary for this experiment
        logger.info(f"\n{'='*50}")
        logger.info(f"RESULTS SUMMARY - {experiment_name}")
        logger.info(f"{'='*50}")
        for _, row in results_df.iterrows():
            logger.info(f"{row['Model']}: F1={row['F1-Score']:.4f}, Acc={row['Accuracy']:.4f}")
        
        return results
    
    else:
        logger.error(f"❌ No models completed successfully for {experiment_name}")
        return []

def run_experiment(experiment_config):
    """Main function to run any experiment based on configuration"""
    
    experiment_type = experiment_config.get('type', 'single')
    
    if experiment_type == 'single':
        return run_single_feature_experiment(experiment_config)
    elif experiment_type == 'combined':
        return run_combined_feature_experiment(experiment_config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function - DO NOT RUN DIRECTLY. Use train.py instead."""
    logger.warning("="*80)
    logger.warning("WARNING: Do not run this file directly!")
    logger.warning("Use 'python train.py' to run experiments instead.")
    logger.warning("This file contains utility functions for the training pipeline.")
    logger.warning("="*80)
    
    logger.info("This module provides the core training engine for stress classification experiments.")
    logger.info("To run experiments, use the train.py file which orchestrates the pipeline.")
    logger.info("Available functions:")
    logger.info("- run_experiment(): Run a single experiment")
    logger.info("- create_comprehensive_summary(): Create summary visualizations")
    logger.info("- Other utility functions for data processing and evaluation")

def create_comprehensive_summary(all_results_df, summary_dir):
    """Create comprehensive summary visualizations"""
    
    plt.figure(figsize=(20, 15))
    
    # Plot 1: F1-Score comparison across all experiments
    plt.subplot(3, 3, 1)
    experiment_f1 = all_results_df.groupby('Experiment')['F1-Score'].mean().sort_values(ascending=False)
    plt.bar(range(len(experiment_f1)), experiment_f1.values, color='skyblue', alpha=0.7)
    plt.xticks(range(len(experiment_f1)), experiment_f1.index, rotation=45, ha='right')
    plt.ylabel('F1-Score')
    plt.title('F1-Score by Experiment')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Accuracy comparison
    plt.subplot(3, 3, 2)
    experiment_acc = all_results_df.groupby('Experiment')['Accuracy'].mean().sort_values(ascending=False)
    plt.bar(range(len(experiment_acc)), experiment_acc.values, color='lightgreen', alpha=0.7)
    plt.xticks(range(len(experiment_acc)), experiment_acc.index, rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Experiment')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Model performance within experiments
    plt.subplot(3, 3, 3)
    pivot_f1 = all_results_df.pivot(index='Experiment', columns='Model', values='F1-Score')
    pivot_f1.plot(kind='bar', ax=plt.gca(), alpha=0.7)
    plt.ylabel('F1-Score')
    plt.title('Model Performance by Experiment')
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Feature type performance
    plt.subplot(3, 3, 4)
    feature_f1 = all_results_df.groupby('Feature_Type')['F1-Score'].mean().sort_values(ascending=False)
    plt.bar(range(len(feature_f1)), feature_f1.values, color='orange', alpha=0.7)
    plt.xticks(range(len(feature_f1)), feature_f1.index, rotation=45, ha='right')
    plt.ylabel('F1-Score')
    plt.title('Performance by Feature Type')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: PCA vs No PCA comparison
    plt.subplot(3, 3, 5)
    pca_f1 = all_results_df.groupby('PCA')['F1-Score'].mean()
    plt.bar(['No PCA', 'PCA'], pca_f1.values, color=['red', 'blue'], alpha=0.7)
    plt.ylabel('F1-Score')
    plt.title('PCA vs No PCA Performance')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Feature count comparison
    plt.subplot(3, 3, 6)
    experiment_features = all_results_df.groupby('Experiment')['Total Features'].first().sort_values(ascending=False)
    plt.bar(range(len(experiment_features)), experiment_features.values, color='purple', alpha=0.7)
    plt.xticks(range(len(experiment_features)), experiment_features.index, rotation=45, ha='right')
    plt.ylabel('Number of Features')
    plt.title('Feature Count by Experiment')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Temporal strategy performance
    plt.subplot(3, 3, 7)
    temporal_f1 = all_results_df.groupby('Temporal_Strategy')['F1-Score'].mean().sort_values(ascending=False)
    plt.bar(range(len(temporal_f1)), temporal_f1.values, color='brown', alpha=0.7)
    plt.xticks(range(len(temporal_f1)), temporal_f1.index, rotation=45, ha='right')
    plt.ylabel('F1-Score')
    plt.title('Performance by Temporal Strategy')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Best model per experiment
    plt.subplot(3, 3, 8)
    best_models = all_results_df.loc[all_results_df.groupby('Experiment')['F1-Score'].idxmax()]
    plt.bar(range(len(best_models)), best_models['F1-Score'].values, color='gold', alpha=0.7)
    plt.xticks(range(len(best_models)), best_models['Experiment'], rotation=45, ha='right')
    plt.ylabel('F1-Score')
    plt.title('Best Model per Experiment')
    plt.grid(True, alpha=0.3)
    
    # Plot 9: Overall model ranking
    plt.subplot(3, 3, 9)
    model_ranking = all_results_df.groupby('Model')['F1-Score'].mean().sort_values(ascending=False)
    plt.bar(range(len(model_ranking)), model_ranking.values, color='teal', alpha=0.7)
    plt.xticks(range(len(model_ranking)), model_ranking.index, rotation=45, ha='right')
    plt.ylabel('F1-Score')
    plt.title('Overall Model Ranking')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, "comprehensive_experiment_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Comprehensive summary visualization created")

if __name__ == "__main__":
    main()
