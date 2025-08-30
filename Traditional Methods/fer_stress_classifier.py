import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import logging
from tqdm import tqdm
import random
import tensorflow as tf
from tensorflow import keras
import h5py

# Configuration
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_labels_and_create_image_mapping():
    """Load labels and create mapping between images and labels using ID column."""
    logger.info("Loading labels and creating image mapping using ID matching...")
    
    # Load labels
    labels_df = pd.read_csv("Processed Data/Expert Majority/All_labels.csv")
    labels_df = labels_df.reset_index(drop=True)
    
    frames_dir = "Processed Data/Face Data"
    image_mapping = []
    missing_labels = 0
    total_images = 0
    
    # Walk through all participant/game/interval folders
    for participant_folder in os.listdir(frames_dir):
        part_path = os.path.join(frames_dir, participant_folder)
        if not os.path.isdir(part_path):
            continue
            
        # For each interval folder
        for interval_folder in os.listdir(part_path):
            interval_path = os.path.join(part_path, interval_folder)
            if not os.path.isdir(interval_path):
                continue
                
            # Convert folder name to label ID format by adding '_t' before the second
            folder_parts = interval_folder.split('_')
            if len(folder_parts) >= 3:
                # e.g., C1_CP_10 -> C1_CP_t10
                label_id = f"{folder_parts[0]}_{folder_parts[1]}_t{folder_parts[2]}"
            else:
                logger.warning(f"Could not parse folder name: {interval_folder}")
                continue
            
            # Find the corresponding label row
            label_row = labels_df[labels_df['ID'] == label_id]
            if label_row.empty:
                missing_labels += 1
                logger.warning(f"No label found for folder ID: {label_id}")
                continue
                
            label_row = label_row.iloc[0]
            
            # Find the highest numbered second file in the interval folder
            second_files = [f for f in os.listdir(interval_path) if f.startswith('second_') and f.endswith('.png')]
            if second_files:
                # Extract numbers and find the highest
                second_numbers = [int(f.replace('second_', '').replace('.png', '')) for f in second_files]
                highest_second = max(second_numbers)
                last_image_path = os.path.join(interval_path, f"second_{highest_second}.png")
                
                total_images += 1
                
                image_mapping.append({
                    'participant_id': label_row['Participant ID'],
                    'game': label_row['Game'],
                    'second': label_row['Second'],
                    'stress_label': label_row['Stress/No Stress'],
                    'image_path': last_image_path,
                    'condition': label_row['Condition'],
                    'folder_id': interval_folder
                })
    
    image_mapping_df = pd.DataFrame(image_mapping)
    logger.info(f"Found {len(image_mapping_df)} valid image-label pairs out of {total_images} images.")
    if len(image_mapping_df) > 0:
        logger.info(f"Stress distribution: {image_mapping_df['stress_label'].value_counts().to_dict()}")
        logger.info(f"Condition distribution: {image_mapping_df['condition'].value_counts().to_dict()}")
    else:
        logger.warning("No valid image-label pairs found! Check folder structure and label matching.")
    logger.info(f"Folders with missing labels: {missing_labels}")
    
    return image_mapping_df

def create_participant_based_split(image_mapping_df, test_size=0.1, random_state=42):
    """Create train/test split with participant-based selection"""
    np.random.seed(random_state)
    
    # Analyze participants
    unique_participants = image_mapping_df['participant_id'].unique()
    participant_info = []
    
    for participant in unique_participants:
        participant_mask = image_mapping_df['participant_id'] == participant
        participant_data = image_mapping_df[participant_mask]
        participant_condition = participant_data['condition'].iloc[0]
        participant_stress_labels = (participant_data['stress_label'] == 'Stress').astype(int)
        stress_ratio = participant_stress_labels.mean()
        
        participant_info.append({
            'participant': participant,
            'condition': participant_condition,
            'dominant_stress': 'Stress' if stress_ratio > 0.5 else 'No_Stress',
            'stress_ratio': stress_ratio,
            'total_samples': len(participant_stress_labels)
        })
    
    participant_df = pd.DataFrame(participant_info)
    participant_df['stratify_label'] = participant_df['condition'] + '_' + participant_df['dominant_stress']
    
    # Calculate proportions and select test participants
    combined_proportions = participant_df['stratify_label'].value_counts(normalize=True).to_dict()
    n_test_participants = max(1, int(len(unique_participants) * test_size))
    
    target_test_composition = {}
    for label, proportion in combined_proportions.items():
        target_count = max(1, round(n_test_participants * proportion))
        target_test_composition[label] = target_count
    
    # Select participants for test set
    test_participants = []
    remaining_participants = participant_df.copy()
    
    for stratify_label, target_count in target_test_composition.items():
        candidates = remaining_participants[remaining_participants['stratify_label'] == stratify_label]
        if len(candidates) > 0:
            n_to_select = min(target_count, len(candidates))
            selected_indices = np.random.choice(candidates.index, size=n_to_select, replace=False)
            selected_participants = candidates.loc[selected_indices, 'participant'].tolist()
            test_participants.extend(selected_participants)
            remaining_participants = remaining_participants.drop(selected_indices)
    
    train_participants = remaining_participants['participant'].tolist()
    
    # Create train/test masks
    train_mask = image_mapping_df['participant_id'].isin(train_participants)
    test_mask = image_mapping_df['participant_id'].isin(test_participants)
    
    train_data = image_mapping_df[train_mask]
    test_data = image_mapping_df[test_mask]
    
    logger.info(f"Final split: {len(train_participants)} train, {len(test_participants)} test participants")
    logger.info(f"Train: {len(train_data)} samples, Test: {len(test_data)} samples")
    
    return train_data, test_data

# ============================================================================
# DATASET AND DATA LOADER
# ============================================================================

class StressImageDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        image_path = row['image_path']
        label = 1 if row['stress_label'] == 'Stress' else 0
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ============================================================================
# FER MODELS
# ============================================================================

class FERModelWrapper(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(FERModelWrapper, self).__init__()
        
        # Load the Keras model
        self.keras_model = keras.models.load_model(model_path, compile=False)
        
        # Get the base model (remove the last classification layer)
        self.base_model = keras.Model(
            inputs=self.keras_model.input,
            outputs=self.keras_model.layers[-2].output  # Get features before last layer
        )
        
        # Freeze the base model
        self.base_model.trainable = False
        
        # Add new classification head for stress detection
        feature_dim = self.base_model.output_shape[-1]
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Convert PyTorch tensor to numpy for Keras model
        # PyTorch: (batch, channels, height, width) -> Keras: (batch, height, width, channels)
        x_np = x.permute(0, 2, 3, 1).cpu().numpy()
        
        # Get features from Keras model
        with torch.no_grad():
            features = self.base_model.predict(x_np, verbose=0)
        
        # Convert back to PyTorch tensor
        features = torch.FloatTensor(features).to(x.device)
        
        # Apply classifier
        return self.classifier(features)

def load_fer_models():
    """Load all available FER models"""
    models_dir = "models"
    fer_models = {}
    
    model_files = {
        'adults_8': 'adults_8.h5',
        'kids_8': 'kids_8.h5', 
        'mixed_1': 'mixed_1.h5'
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(models_dir, filename)
        if os.path.exists(model_path):
            try:
                logger.info(f"Loading FER model: {model_name}")
                model = FERModelWrapper(model_path, num_classes=2)
                fer_models[model_name] = model
                logger.info(f"Successfully loaded {model_name}")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
        else:
            logger.warning(f"Model file not found: {model_path}")
    
    return fer_models

# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_name):
    """Train the model"""
    logger.info(f"Training {model_name} for {num_epochs} epochs...")
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model with validation accuracy: {best_val_acc:.2f}%")
    
    return model

def evaluate_model(model, test_loader, device, model_name, results_dir):
    """Evaluate the model on test set"""
    logger.info(f"Evaluating {model_name}...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    logger.info(f"{model_name} Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    
    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Stress', 'Stress'],
                yticklabels=['No Stress', 'Stress'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()
    
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Features': 'FER Pre-trained Features',
        'Total Features': 'FER Model Features',
        'Data Augmentation': 'Standard transforms',
        'Experiment': 'FER_Stress_Classification'
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("Starting FER-based stress classification...")
    
    # Create results directory
    results_dir = "Results/FER_Results_Face_batch"
    if os.path.exists(results_dir):
        import shutil
        shutil.rmtree(results_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load data
    image_mapping_df = load_labels_and_create_image_mapping()
    train_data, test_data = create_participant_based_split(image_mapping_df)
    
    # Create validation split from training data
    train_participants = train_data['participant_id'].unique()
    val_participants = np.random.choice(train_participants, size=max(1, len(train_participants)//5), replace=False)
    val_mask = train_data['participant_id'].isin(val_participants)
    val_data = train_data[val_mask]
    train_data = train_data[~val_mask]
    
    logger.info(f"Final splits: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Define transforms (matching original FER usage - no normalization)
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        # No normalization - matching original FER usage
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        # No normalization - matching original FER usage
    ])
    
    # Create datasets and loaders
    train_dataset = StressImageDataset(train_data, transform=train_transform)
    val_dataset = StressImageDataset(val_data, transform=test_transform)
    test_dataset = StressImageDataset(test_data, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)  # Batch size 1 for FER models
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load FER models
    fer_models = load_fer_models()
    
    if not fer_models:
        logger.error("No FER models loaded! Exiting.")
        return
    
    # Define model configurations
    models_config = {}
    for model_name in fer_models.keys():
        models_config[model_name] = {
            'model': fer_models[model_name],
            'lr': 0.0001,
            'epochs': 5
        }
    
    all_results = []
    
    for model_name, config in models_config.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING {model_name}")
        logger.info(f"{'='*60}")
        
        # Initialize model
        model = config['model'].to(device)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-4)
        
        # Train model
        model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                          config['epochs'], device, model_name)
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device, model_name, results_dir)
        all_results.append(results)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(results_dir, f"{model_name.lower().replace(' ', '_')}_model.pth"))
    
    # Create summary
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(results_dir, "fer_results_summary.csv"), index=False)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("FER STRESS CLASSIFICATION RESULTS SUMMARY")
    logger.info("="*80)
    
    for _, row in results_df.iterrows():
        logger.info(f"{row['Model']}: F1={row['F1-Score']:.4f}, Acc={row['Accuracy']:.4f}")
    
    # Find best model
    best_model = results_df.loc[results_df['F1-Score'].idxmax()]
    logger.info(f"\n🏆 BEST MODEL: {best_model['Model']}")
    logger.info(f"   F1-Score: {best_model['F1-Score']:.4f}")
    logger.info(f"   Accuracy: {best_model['Accuracy']:.4f}")
    
    logger.info(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main() 