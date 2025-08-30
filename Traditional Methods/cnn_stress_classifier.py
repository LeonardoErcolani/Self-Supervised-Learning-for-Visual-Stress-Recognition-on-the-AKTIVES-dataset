import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import logging
from tqdm import tqdm
import random
import glob

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
                
            # The interval folder name (e.g., C1_CP_10) should match the ID in labels
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
            
            # For each image in the interval folder
            for img_file in os.listdir(interval_path):
                if not img_file.endswith('.png'):
                    continue
                total_images += 1
                
                image_mapping.append({
                    'participant_id': label_row['Participant ID'],
                    'game': label_row['Game'],
                    'second': label_row['Second'],
                    'stress_label': label_row['Stress/No Stress'],
                    'image_path': os.path.join(interval_path, img_file),
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
# CNN MODELS
# ============================================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet18Model(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18Model, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace final layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

class VGGFaceModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(VGGFaceModel, self).__init__()
        # Use VGG16 as base (VGG-Face is based on VGG16)
        self.model = models.vgg16(pretrained=pretrained)
        
        # Freeze early layers for transfer learning
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        
        # Replace the final classifier
        num_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

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
        'Features': 'Raw Images (1 per 10s interval)',
        'Total Features': '224x224x3 RGB',
        'Data Augmentation': 'Standard transforms',
        'Experiment': 'CNN_Stress_Classification'
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("Starting CNN-based stress classification...")
    
    # Create results directory
    results_dir = "Results/CNN_Results"
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
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and loaders
    train_dataset = StressImageDataset(train_data, transform=train_transform)
    val_dataset = StressImageDataset(val_data, transform=test_transform)
    test_dataset = StressImageDataset(test_data, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Define models
    models_config = {
        
        'ResNet18': {
            'model': ResNet18Model(num_classes=2, pretrained=True),
            'lr': 0.0001,
            'epochs': 5
        },
        'VGGFace': {
            'model': VGGFaceModel(num_classes=2, pretrained=True),
            'lr': 0.0001,
            'epochs': 5
        }
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
    results_df.to_csv(os.path.join(results_dir, "cnn_results_summary.csv"), index=False)
    
    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("CNN STRESS CLASSIFICATION RESULTS SUMMARY")
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