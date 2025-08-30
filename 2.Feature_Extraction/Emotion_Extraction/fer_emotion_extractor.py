import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import tensorflow as tf
import keras
from PIL import Image
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
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
            
            # Get all second files in the interval folder
            second_files = [f for f in os.listdir(interval_path) if f.startswith('second_') and f.endswith('.png')]
            if not second_files:
                continue
                
            # Sort by second number
            second_files.sort(key=lambda x: int(x.replace('second_', '').replace('.png', '')))
            
            # Add each frame with its metadata
            for frame_idx, second_file in enumerate(second_files):
                image_path = os.path.join(interval_path, second_file)
                total_images += 1
                
                image_mapping.append({
                    'ID': label_row['ID'],
                    'Participant ID': label_row['Participant ID'],
                    'Name': label_row['Name'],
                    'Condition': label_row['Condition'],
                    'Game': label_row['Game'],
                    'Minute': label_row['Minute'],
                    'Second': label_row['Second'],
                    'Stress/No Stress': label_row['Stress/No Stress'],
                    'Reaction/No Reaction': label_row['Reaction/No Reaction'],
                    'image_path': image_path,
                    'folder_id': interval_folder,
                    'Frame_Index': frame_idx
                })
    
    image_mapping_df = pd.DataFrame(image_mapping)
    logger.info(f"Found {len(image_mapping_df)} valid image-label pairs out of {total_images} images.")
    if len(image_mapping_df) > 0:
        logger.info(f"Stress distribution: {image_mapping_df['Stress/No Stress'].value_counts().to_dict()}")
        logger.info(f"Condition distribution: {image_mapping_df['Condition'].value_counts().to_dict()}")
    else:
        logger.warning("No valid image-label pairs found! Check folder structure and label matching.")
    logger.info(f"Folders with missing labels: {missing_labels}")
    
    return image_mapping_df

# ============================================================================
# DATASET AND DATA LOADER
# ============================================================================

class EmotionImageDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data_df = data_df
        self.transform = transform
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        image_path = row['image_path']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (48, 48), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, idx

# ============================================================================
# FER MODEL LOADING
# ============================================================================

def load_kids_8_model():
    """Load the kids_8 FER model"""
    model_path = "models/kids_8.h5"
    
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        logger.info(f"Loading FER model: kids_8")
        model = keras.models.load_model(model_path, compile=False)
        logger.info(f"Successfully loaded kids_8 model")
        logger.info(f"Model output shape: {model.output_shape}")
        return model
    except Exception as e:
        logger.error(f"Failed to load kids_8 model: {e}")
        raise

# ============================================================================
# EMOTION PREDICTION
# ============================================================================

def predict_emotions(model, data_loader, data_df, device):
    """Predict emotions using the FER model"""
    logger.info("Predicting emotions...")
    
    all_results = []
    
    # Define emotion labels (assuming 6 emotions based on typical FER models)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    
    for batch in tqdm(data_loader, desc="Predicting emotions"):
        images, idx = batch
        
        # Convert PyTorch tensor to numpy for Keras model
        # PyTorch: (batch, channels, height, width) -> Keras: (batch, height, width, channels)
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()
        
        # Get emotion predictions from Keras model
        emotion_probs = model.predict(images_np, verbose=0)
        
        # Convert back to PyTorch tensor for processing
        emotion_probs = torch.FloatTensor(emotion_probs)
        
        # Get predicted emotion class
        _, predicted_emotions = torch.max(emotion_probs, 1)
        
        # Get the original row data
        row = data_df.iloc[idx.item()]
        
        # Create result row
        result_row = {
            'ID': row['ID'],
            'Participant ID': row['Participant ID'],
            'Name': row['Name'],
            'Condition': row['Condition'],
            'Game': row['Game'],
            'Minute': row['Minute'],
            'Second': row['Second'],
            'Stress/No Stress': row['Stress/No Stress'],
            'Reaction/No Reaction': row['Reaction/No Reaction'],
            'Frame_Index': row['Frame_Index'],
            'predicted_emotion': emotion_labels[predicted_emotions[0].item()],
            'predicted_emotion_idx': predicted_emotions[0].item(),
        }
        
        # Add probability for each emotion
        for j, emotion in enumerate(emotion_labels):
            result_row[f'prob_{emotion}'] = emotion_probs[0, j].item()
        
        all_results.append(result_row)
    
    return all_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    logger.info("Starting emotion extraction...")
    
    # Create output directory
    output_dir = "Processed Data/FER_Emotion_Features"
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    image_mapping_df = load_labels_and_create_image_mapping()
    
    if len(image_mapping_df) == 0:
        logger.error("No valid data found! Exiting.")
        return
    
    # Define transforms (matching original FER usage - no normalization)
    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        # No normalization - matching original FER usage
    ])
    
    # Create dataset and loader
    dataset = EmotionImageDataset(image_mapping_df, transform=test_transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load FER model
    try:
        fer_model = load_kids_8_model()
    except Exception as e:
        logger.error(f"Failed to load FER model: {e}")
        return
    
    # Predict emotions
    results_data = predict_emotions(fer_model, data_loader, image_mapping_df, device)
    results_df = pd.DataFrame(results_data)
    
    # Save per-second emotions
    per_second_path = os.path.join(output_dir, "all_emotions_per_second.csv")
    results_df.to_csv(per_second_path, index=False)
    logger.info(f"Per-second emotions saved to: {per_second_path}")
    
    # Create per-interval emotions (average across frames for each interval)
    logger.info("Creating per-interval emotions...")
    interval_emotions = []
    
    # Group by ID to average emotions across frames
    for interval_id in results_df['ID'].unique():
        interval_frames = results_df[results_df['ID'] == interval_id]
        
        if len(interval_frames) == 0:
            continue
            
        # Get metadata from first frame
        first_frame = interval_frames.iloc[0]
        
        # Calculate average emotion probabilities
        emotion_cols = [col for col in interval_frames.columns if col.startswith('prob_')]
        avg_emotions = interval_frames[emotion_cols].mean()
        
        # Get most common predicted emotion
        most_common_emotion = interval_frames['predicted_emotion'].mode().iloc[0]
        most_common_idx = interval_frames['predicted_emotion_idx'].mode().iloc[0]
        
        # Create interval row
        interval_row = {
            'ID': interval_id,
            'Participant ID': first_frame['Participant ID'],
            'Name': first_frame['Name'],
            'Condition': first_frame['Condition'],
            'Game': first_frame['Game'],
            'Minute': first_frame['Minute'],
            'Second': first_frame['Second'],
            'Stress/No Stress': first_frame['Stress/No Stress'],
            'Reaction/No Reaction': first_frame['Reaction/No Reaction'],
            'predicted_emotion': most_common_emotion,
            'predicted_emotion_idx': most_common_idx,
        }
        
        # Add averaged emotion probabilities
        for emotion_col in emotion_cols:
            interval_row[emotion_col] = avg_emotions[emotion_col]
        
        interval_emotions.append(interval_row)
    
    # Save per-interval emotions
    interval_df = pd.DataFrame(interval_emotions)
    interval_path = os.path.join(output_dir, "all_emotions_per_interval.csv")
    interval_df.to_csv(interval_path, index=False)
    logger.info(f"Per-interval emotions saved to: {interval_path}")
    
    # Print summary statistics
    logger.info(f"\n=== Emotion Extraction Summary ===")
    logger.info(f"Total frames processed: {len(results_df)}")
    logger.info(f"Total intervals processed: {len(interval_df)}")
    logger.info(f"Emotion dimensions: {len(['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise'])}")
    
    # Print condition distribution
    logger.info(f"\nCondition distribution:")
    condition_counts = interval_df['Condition'].value_counts()
    for condition, count in condition_counts.items():
        logger.info(f"  {condition}: {count} intervals")
    
    # Print game distribution
    logger.info(f"\nGame distribution:")
    game_counts = interval_df['Game'].value_counts()
    for game, count in game_counts.items():
        logger.info(f"  {game}: {count} intervals")
    
    # Print emotion distribution
    logger.info(f"\nPredicted emotion distribution:")
    emotion_counts = interval_df['predicted_emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        logger.info(f"  {emotion}: {count} ({count/len(interval_df)*100:.1f}%)")
    
    # Print stress distribution
    logger.info(f"\nStress distribution:")
    stress_counts = interval_df['Stress/No Stress'].value_counts()
    for stress, count in stress_counts.items():
        logger.info(f"  {stress}: {count} ({count/len(interval_df)*100:.1f}%)")
    
    # Create emotion probability distribution plot
    plt.figure(figsize=(15, 10))
    
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    
    for i, emotion in enumerate(emotion_labels):
        plt.subplot(2, 3, i+1)
        prob_col = f'prob_{emotion}'
        plt.hist(interval_df[prob_col], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{emotion.capitalize()} Probability Distribution')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "emotion_probability_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create emotion vs stress analysis
    plt.figure(figsize=(12, 8))
    
    # Emotion distribution by stress
    plt.subplot(2, 2, 1)
    emotion_stress_crosstab = pd.crosstab(interval_df['predicted_emotion'], interval_df['Stress/No Stress'])
    emotion_stress_crosstab.plot(kind='bar', ax=plt.gca())
    plt.title('Predicted Emotion by Stress Level')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Stress Level')
    
    # Average emotion probabilities by stress
    plt.subplot(2, 2, 2)
    stress_emotion_probs = interval_df.groupby('Stress/No Stress')[['prob_angry', 'prob_disgust', 'prob_fear', 'prob_happy', 'prob_sad', 'prob_surprise']].mean()
    stress_emotion_probs.plot(kind='bar', ax=plt.gca())
    plt.title('Average Emotion Probabilities by Stress Level')
    plt.xlabel('Stress Level')
    plt.ylabel('Average Probability')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Emotion distribution by condition
    plt.subplot(2, 2, 3)
    emotion_condition_crosstab = pd.crosstab(interval_df['predicted_emotion'], interval_df['Condition'])
    emotion_condition_crosstab.plot(kind='bar', ax=plt.gca())
    plt.title('Predicted Emotion by Condition')
    plt.xlabel('Predicted Emotion')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Average emotion probabilities by condition
    plt.subplot(2, 2, 4)
    condition_emotion_probs = interval_df.groupby('Condition')[['prob_angry', 'prob_disgust', 'prob_fear', 'prob_happy', 'prob_sad', 'prob_surprise']].mean()
    condition_emotion_probs.plot(kind='bar', ax=plt.gca())
    plt.title('Average Emotion Probabilities by Condition')
    plt.xlabel('Condition')
    plt.ylabel('Average Probability')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "emotion_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Analysis plots saved to: {output_dir}")
    logger.info("Emotion extraction completed successfully!")

if __name__ == "__main__":
    main() 