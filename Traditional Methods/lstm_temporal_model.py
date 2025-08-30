import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import torch.nn.functional as F

class TemporalDataset(Dataset):
    def __init__(self, features_df, labels_df):
        self.features_df = features_df
        self.labels_df = labels_df
        
        # Group features by ID and Second to create sequences
        self.sequences = []
        self.labels = []
        self.sequence_lengths = []  # Store actual sequence lengths
        
        for (id_val, second), group in features_df.groupby(['ID', 'Second']):
            # Get the corresponding label
            label_row = labels_df[(labels_df['ID'] == id_val) & (labels_df['Second'] == second)]
            if len(label_row) == 0:
                continue
                
            # Get features for all frames in this interval
            frame_features = group.sort_values('Frame_Index')
            
            # Extract features
            features = frame_features[[col for col in frame_features.columns if col.startswith('feat_')]].values
            
            # Get stress label
            label = 1 if label_row['Stress/No Stress'].values[0] == 'Stress' else 0
            
            self.sequences.append(features)
            self.labels.append(label)
            self.sequence_lengths.append(len(features))
        
        # Convert to tensors
        self.labels = torch.LongTensor(self.labels)
        self.sequence_lengths = torch.LongTensor(self.sequence_lengths)
        
        # Pad sequences to max length in batch
        self.sequences = self.sequences  # Keep as list for dynamic padding in DataLoader
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx], self.sequence_lengths[idx]

class TemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.5, bidirectional=True, use_attention=True):
        super(TemporalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x, lengths):
        # Pack padded sequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed_x)
        
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        if self.use_attention:
            # Calculate attention weights
            attention_weights = self.attention(output)
            attention_weights = F.softmax(attention_weights, dim=1)
            
            # Apply attention
            attended = torch.sum(attention_weights * output, dim=1)
            return self.fc(attended)
        else:
            # Use last hidden state
            if self.bidirectional:
                hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            else:
                hidden = hidden[-1,:,:]
            return self.fc(hidden)

def collate_fn(batch):
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Separate sequences, labels, and lengths
    sequences, labels, lengths = zip(*batch)
    
    # Pad sequences
    padded_sequences = nn.utils.rnn.pad_sequence([torch.FloatTensor(seq) for seq in sequences], 
                                               batch_first=True)
    
    return padded_sequences, torch.LongTensor(labels), torch.LongTensor(lengths)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_model():
    """Train LSTM model for stress prediction"""
    # Create output directory
    output_dir = "Results/LSTM Results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    features_df = pd.read_csv("Processed Data/DinoV2 Features Extracted/all_features_per_second.csv")
    labels_df = pd.read_csv("Processed Data/Expert Majority/all_labels.csv")
    
    # Ensure we have the Condition column for stratification
    if 'Condition' not in features_df.columns:
        print("Warning: 'Condition' column not found in features. Adding default condition...")
        features_df['Condition'] = 'Unknown'
    
    print(f"Features shape: {features_df.shape}")
    print(f"Labels shape: {labels_df.shape}")
    
    # Create dataset
    dataset = TemporalDataset(features_df, labels_df)
    
    # Calculate class weights for imbalance
    label_counts = Counter(dataset.labels.numpy())
    total_samples = len(dataset.labels)
    class_weights = {label: total_samples / (len(label_counts) * count) 
                    for label, count in label_counts.items()}
    class_weights = torch.FloatTensor([class_weights[i] for i in range(len(label_counts))])
    
    # Save class distribution
    class_distribution = pd.DataFrame({
        'Class': list(label_counts.keys()),
        'Count': list(label_counts.values()),
        'Weight': [class_weights[i].item() for i in range(len(label_counts))]
    })
    class_distribution.to_csv(os.path.join(output_dir, 'class_distribution.csv'), index=False)
    
    # Create participant-based split (90/10) using all_labels.csv for efficiency
    print("Creating participant-based train/test split...")
    
    # Get unique participants and their stress ratios from labels_df (much faster)
    # Extract actual participant ID (C1, C2, etc.) from video session names
    labels_df['participant_id'] = labels_df['ID'].str.extract(r'(C\d+)')[0]
    
    participant_info = labels_df.groupby('participant_id').agg({
        'Stress/No Stress': lambda x: (x == 'Stress').mean(),  # Stress ratio
        'Condition': 'first'  # Condition (should be same for all rows of same participant)
    }).reset_index()
    
    participant_info.columns = ['participant', 'stress_ratio', 'condition']
    participant_info['dominant_stress'] = ['Stress' if ratio > 0.5 else 'No_Stress' for ratio in participant_info['stress_ratio']]
    participant_info['stratify_label'] = participant_info['condition'] + '_' + participant_info['dominant_stress']
    
    print(f"Total participants: {len(participant_info)}")
    print(f"Condition distribution: {participant_info['condition'].value_counts().to_dict()}")
    print(f"Stress distribution: {participant_info['dominant_stress'].value_counts().to_dict()}")
    
    # Calculate target test set composition (10% of participants)
    test_size = 0.1
    n_test_participants = max(1, int(len(participant_info) * test_size))
    print(f"Target test participants: {n_test_participants} ({test_size*100:.0f}% of {len(participant_info)})")
    
    # Calculate proportions for each stratify label
    combined_proportions = participant_info['stratify_label'].value_counts(normalize=True).to_dict()
    
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
    
    print(f"Target test composition: {target_test_composition}")
    
    # Select participants for test set based on target composition
    test_participants = []
    remaining_participants = participant_info.copy()
    
    for stratify_label, target_count in target_test_composition.items():
        candidates = remaining_participants[remaining_participants['stratify_label'] == stratify_label]
        
        if len(candidates) == 0:
            print(f"Warning: No candidates available for {stratify_label}")
            continue
            
        n_to_select = min(target_count, len(candidates))
        
        if n_to_select > 0:
            selected_indices = np.random.choice(candidates.index, size=n_to_select, replace=False)
            selected_participants = candidates.loc[selected_indices, 'participant'].tolist()
            test_participants.extend(selected_participants)
            
            remaining_participants = remaining_participants.drop(selected_indices)
            print(f"Selected {n_to_select} participants for {stratify_label}")
    
    # All remaining participants go to train set
    train_participants = remaining_participants['participant'].tolist()
    
    print(f"Final split: {len(train_participants)} train, {len(test_participants)} test participants")
    
    # Create train/test masks for the dataset
    train_mask = []
    test_mask = []
    
    # Extract participant IDs from features_df for matching
    features_df['participant_id'] = features_df['ID'].str.extract(r'(C\d+)')[0]
    
    # Get all unique (ID, Second) combinations from features_df
    feature_groups = list(features_df.groupby(['ID', 'Second']).groups.keys())
    
    for i, (id_val, second) in enumerate(feature_groups):
        # Extract participant ID from the full video session ID
        participant_id = re.search(r'(C\d+)', id_val).group(1)
        if participant_id in train_participants:
            train_mask.append(i)
        elif participant_id in test_participants:
            test_mask.append(i)
    
    # Verify no overlap
    train_participant_set = set(train_participants)
    test_participant_set = set(test_participants)
    overlap = train_participant_set.intersection(test_participant_set)
    
    if overlap:
        raise ValueError(f"Participant overlap detected: {overlap}")
    
    print(f"✅ NO OVERLAP: Train and test sets are completely separate")
    print(f"Train participants: {sorted(list(train_participant_set))}")
    print(f"Test participants: {sorted(list(test_participant_set))}")
    
    # Create train/test datasets
    train_dataset = torch.utils.data.Subset(dataset, train_mask)
    test_dataset = torch.utils.data.Subset(dataset, test_mask)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders with custom collate function
    print(f"Creating data loaders for {len(train_dataset)} train and {len(test_dataset)} test samples...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model with enhanced architecture
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TemporalLSTM(
        input_dim=768,  # DINOv2 feature dimension
        hidden_dim=256,
        num_layers=2,
        num_classes=2,  # Binary classification
        dropout=0.5,
        bidirectional=True,
        use_attention=True
    ).to(device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training loop
    print("Training model for stress prediction...")
    num_epochs = 50
    best_accuracy = 0
    training_history = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for sequences, labels, lengths in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss/len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        
        # Update learning rate
        scheduler.step(accuracy)
        
        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'accuracy': accuracy
        })
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': accuracy,
                'loss': avg_loss,
            }, os.path.join(output_dir, 'best_temporal_model.pth'))
    
    # Save training history
    pd.DataFrame(training_history).to_csv(os.path.join(output_dir, 'training_history.csv'), index=False)
    
    # Evaluate on test set
    print("\nEvaluating model...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_temporal_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences = sequences.to(device)
            outputs = model(sequences, lengths)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Save predictions and probabilities
    results_df = pd.DataFrame({
        'True_Label': all_labels,
        'Predicted_Label': all_preds,
        **{f'Prob_Class_{i}': [p[i] for p in all_probs] for i in range(2)}
    })
    results_df.to_csv(os.path.join(output_dir, 'test_predictions.csv'), index=False)
    
    # Print and save classification report
    report = classification_report(all_labels, all_preds, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_dir, 'classification_report.csv'))
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    
    # Plot and save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes=['No Stress', 'Stress'])
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f"\nAll outputs have been saved to: {output_dir}")

if __name__ == "__main__":
    train_model() 