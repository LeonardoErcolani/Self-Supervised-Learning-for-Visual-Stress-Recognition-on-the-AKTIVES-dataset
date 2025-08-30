import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FacialExpressionPoseAnalyzer:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.face_data_dir = os.path.join(root_dir, "Processed Data", "Frames")
        self.output_dir = os.path.join(root_dir, "Processed Data", "Expression_Pose_Analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize MediaPipe solutions
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Emotion analysis disabled for performance
        self.emotion_classifier = None
        
        # Define facial landmark indices for key features
        self.facial_features = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 296, 334, 293, 300, 276, 283, 282, 295, 285, 336],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318, 291, 303, 267, 269, 270, 267, 271, 272],
            'nose': [1, 2, 5, 4, 6, 19, 20, 94, 125, 141, 235, 236, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 218, 356, 389, 455, 358, 429, 437, 355, 371, 360, 344],
            'jaw': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        }
    
    def extract_facial_landmarks(self, image):
        """Extract facial landmarks using MediaPipe Face Mesh"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
        
        landmarks = results.multi_face_landmarks[0]
        h, w = image.shape[:2]
        
        # Convert normalized coordinates to pixels
        landmark_coords = []
        for landmark in landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmark_coords.append([x, y])
        
        return np.array(landmark_coords)
    
    def extract_pose_landmarks(self, image):
        """Extract pose landmarks using MediaPipe Pose"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return None
        
        h, w = image.shape[:2]
        pose_coords = []
        
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            z = landmark.z
            visibility = landmark.visibility
            pose_coords.append([x, y, z, visibility])
        
        return np.array(pose_coords)
    
    def calculate_facial_metrics(self, landmarks):
        """Calculate facial expression metrics from landmarks"""
        if landmarks is None:
            return {}
        
        metrics = {}
        
        try:
            # Eye aspect ratios
            left_eye_points = landmarks[self.facial_features['left_eye']]
            right_eye_points = landmarks[self.facial_features['right_eye']]
            
            # Calculate eye aspect ratio (EAR)
            def eye_aspect_ratio(eye_points):
                # Vertical distances
                A = np.linalg.norm(eye_points[1] - eye_points[5])
                B = np.linalg.norm(eye_points[2] - eye_points[4])
                # Horizontal distance
                C = np.linalg.norm(eye_points[0] - eye_points[3])
                return (A + B) / (2.0 * C)
            
            metrics['left_eye_ear'] = eye_aspect_ratio(left_eye_points[:6])
            metrics['right_eye_ear'] = eye_aspect_ratio(right_eye_points[:6])
            metrics['avg_eye_ear'] = (metrics['left_eye_ear'] + metrics['right_eye_ear']) / 2
            
            # Mouth aspect ratio
            mouth_points = landmarks[self.facial_features['mouth']]
            mouth_width = np.linalg.norm(mouth_points[0] - mouth_points[6])
            mouth_height = np.linalg.norm(mouth_points[3] - mouth_points[9])
            metrics['mouth_aspect_ratio'] = mouth_height / mouth_width if mouth_width > 0 else 0
            
            # Eyebrow position (relative to eye)
            eyebrow_points = landmarks[self.facial_features['eyebrows']]
            eye_center = np.mean(np.concatenate([left_eye_points, right_eye_points]), axis=0)
            eyebrow_center = np.mean(eyebrow_points, axis=0)
            metrics['eyebrow_distance'] = np.linalg.norm(eyebrow_center - eye_center)
            
            # Face symmetry
            face_center_x = np.mean(landmarks[:, 0])
            left_side = landmarks[landmarks[:, 0] < face_center_x]
            right_side = landmarks[landmarks[:, 0] >= face_center_x]
            
            if len(left_side) > 0 and len(right_side) > 0:
                left_mean = np.mean(left_side, axis=0)
                right_mean = np.mean(right_side, axis=0)
                metrics['face_symmetry'] = np.linalg.norm(left_mean - right_mean)
            else:
                metrics['face_symmetry'] = 0
            
        except Exception as e:
            print(f"Error calculating facial metrics: {e}")
            return {}
        
        return metrics
    
    def calculate_pose_metrics(self, pose_landmarks):
        """Calculate pose-based metrics"""
        if pose_landmarks is None:
            return {}
        
        metrics = {}
        
        try:
            # Head pose estimation
            nose = pose_landmarks[0]  # Nose tip
            left_ear = pose_landmarks[7]  # Left ear
            right_ear = pose_landmarks[8]  # Right ear
            
            # Head tilt calculation
            if left_ear[3] > 0.5 and right_ear[3] > 0.5:  # Check visibility
                ear_diff = left_ear[1] - right_ear[1]  # Y difference
                ear_distance = abs(left_ear[0] - right_ear[0])  # X distance
                metrics['head_tilt'] = np.arctan2(ear_diff, ear_distance) * 180 / np.pi
            else:
                metrics['head_tilt'] = 0
            
            # Shoulder position
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            
            if left_shoulder[3] > 0.5 and right_shoulder[3] > 0.5:
                shoulder_diff = left_shoulder[1] - right_shoulder[1]
                shoulder_distance = abs(left_shoulder[0] - right_shoulder[0])
                metrics['shoulder_tilt'] = np.arctan2(shoulder_diff, shoulder_distance) * 180 / np.pi
                
                # Body lean (forward/backward)
                metrics['body_lean'] = (left_shoulder[2] + right_shoulder[2]) / 2
            else:
                metrics['shoulder_tilt'] = 0
                metrics['body_lean'] = 0
            
            # Hand positions (if visible)
            left_wrist = pose_landmarks[15]
            right_wrist = pose_landmarks[16]
            
            metrics['left_hand_visible'] = left_wrist[3] > 0.5
            metrics['right_hand_visible'] = right_wrist[3] > 0.5
            
            if metrics['left_hand_visible']:
                metrics['left_hand_y'] = left_wrist[1]
            if metrics['right_hand_visible']:
                metrics['right_hand_y'] = right_wrist[1]
            
        except Exception as e:
            print(f"Error calculating pose metrics: {e}")
            return {}
        
        return metrics
    
    def analyze_emotion_features(self, image):
        """Emotion analysis disabled - returns empty dict"""
        return {}
    
    def analyze_interval(self, participant_id, game_type, interval_second):
        """Analyze a single 10-second interval"""
        interval_dir = os.path.join(
            self.face_data_dir, 
            f"{participant_id}_{game_type}_frames",
            f"{participant_id}_{game_type}_{interval_second}"
        )
        
        if not os.path.exists(interval_dir):
            return None
        
        # Get all frame files in the interval
        frame_files = sorted([f for f in os.listdir(interval_dir) if f.endswith('.png')])
        
        interval_data = []
        
        for frame_file in frame_files:
            frame_path = os.path.join(interval_dir, frame_file)
            image = cv2.imread(frame_path)
            
            if image is None:
                continue
            
            # Extract features
            facial_landmarks = self.extract_facial_landmarks(image)
            pose_landmarks = self.extract_pose_landmarks(image)
            
            facial_metrics = self.calculate_facial_metrics(facial_landmarks)
            pose_metrics = self.calculate_pose_metrics(pose_landmarks)
            emotion_metrics = self.analyze_emotion_features(image)
            
            # Combine all metrics
            frame_data = {
                'Participant ID': participant_id,
                'Game': game_type,
                'Second': interval_second,
                'frame_file': frame_file,
                **facial_metrics,
                **pose_metrics,
                **emotion_metrics
            }
            
            interval_data.append(frame_data)
        
        return interval_data
    
    def calculate_interval_statistics(self, interval_data):
        """Calculate only mean values for each feature (one metric per part)"""
        if not interval_data:
            return {}
        
        df = pd.DataFrame(interval_data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats_data = {}
        
        for col in numeric_cols:
            if col in ['Participant ID', 'Second']:
                continue
                
            values = df[col].values
            if len(values) > 0:
                # Only calculate mean - one metric per feature
                stats_data[col] = np.mean(values)
        
        return stats_data
    
    def process_all_intervals(self):
        """Process all intervals and save results - following DinoV2 pattern"""
        # Create output directory and clear existing files
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load metadata
        print("Loading metadata...")
        metadata_df = pd.read_csv(os.path.join(self.root_dir, "Processed Data", "Expert Majority", "All_labels.csv"))
        
        # Create a list to store per-second features
        per_second_features = []
        
        print("Processing images and extracting features...")
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            # Use proper metadata columns instead of parsing ID
            participant_id = row['Participant ID']  # e.g., C1
            game_type = row['Game']                 # e.g., LeapBall or CatchAPet
            
            # Convert game name to abbreviation for directory path
            game_abbr = "CP" if "catchapet" in game_type.lower() else "LP"
            second = int(row['Second'])
            
            # Analyze interval
            interval_data = self.analyze_interval(participant_id, game_abbr, second)
            
            if interval_data:
                # Store per-second features with metadata first
                for frame_idx, frame_data in enumerate(interval_data):
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
                        **{k: v for k, v in frame_data.items() if k not in ['Participant ID', 'Game', 'Second', 'frame_file']}
                    }
                    per_second_features.append(per_second_row)
        
        # Convert per_second_features to DataFrame for easier processing
        per_second_df = pd.DataFrame(per_second_features)
        
        # Now calculate interval averages from the per-second data
        print("Calculating interval averages...")
        for idx, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
            # Find all per-second rows for this interval
            interval_rows = per_second_df[per_second_df['ID'] == row['ID']]
            
            if len(interval_rows) > 0:
                # Get feature columns (exclude metadata)
                feature_cols = [col for col in interval_rows.columns if col not in ['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction', 'Frame_Index']]
                
                # Calculate averages for each feature
                for feature_col in feature_cols:
                    if pd.api.types.is_numeric_dtype(interval_rows[feature_col]):
                        avg_value = interval_rows[feature_col].mean()
                        metadata_df.at[idx, feature_col] = avg_value
        
        # Drop rows with missing values
        print("Dropping rows with missing values...")
        metadata_df = metadata_df.dropna()
        print(f"Remaining rows after dropping missing values: {len(metadata_df)}")
        
        # Save the averaged features
        output_path = os.path.join(self.output_dir, "all_features_per_interval.csv")
        metadata_df.to_csv(output_path, index=False)
        print(f"Averaged features saved to: {output_path}")
        
        # Save the per-second features (already created above)
        per_second_path = os.path.join(self.output_dir, "all_features_per_second.csv")
        per_second_df.to_csv(per_second_path, index=False)
        print(f"Per-second features saved to: {per_second_path}")
        
        # Print summary statistics
        print(f"\n=== Feature Extraction Summary ===")
        print(f"Total intervals processed: {len(metadata_df)}")
        print(f"Total frames processed: {len(per_second_df)}")
        
        # Count feature columns (exclude metadata)
        feature_cols = [col for col in metadata_df.columns if col not in ['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction']]
        print(f"Feature dimensions: {len(feature_cols)}")
        
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
        
        return metadata_df, per_second_df
    
    def create_overlapping_intervals(self, interval_df, per_second_df):
        """Create overlapping 5-second intervals - following DinoV2 pattern"""
        print("\nCreating overlapping intervals...")
        
        # Get the exact column structure from the original interval data
        original_columns = interval_df.columns.tolist()
        feature_cols = [col for col in per_second_df.columns if col not in ['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction', 'Frame_Index']]
        metadata_cols = [col for col in original_columns if col not in feature_cols]
        
        print(f"Original dataset has {len(metadata_cols)} metadata columns and {len(feature_cols)} feature columns")
        print(f"Metadata columns: {metadata_cols}")
        print(f"Feature columns: {feature_cols[:5]}...")  # Show first 5 feature columns
        
        # Create a mapping from ID to all metadata and labels
        label_mapping = {}
        for _, row in interval_df.iterrows():
            id_key = row['ID']
            # Store all metadata fields exactly as they appear in original
            label_mapping[id_key] = {col: row[col] for col in metadata_cols if col != 'ID'}
        
        grouped = per_second_df.groupby('ID')
        all_intervals = []
        
        for id_name, group in tqdm(grouped, desc="Processing intervals"):
            if id_name not in label_mapping:
                print(f"Warning: No label found for {id_name}")
                continue
                
            # Sort by frame index (which represents the second within the 10-second interval)
            group = group.sort_values('Frame_Index')
            
            # Get the metadata info for this ID
            metadata_info = label_mapping[id_name]
            
            # Create three overlapping 5-second intervals based on Frame_Index
            intervals = [
                (0, 4, "early"),    # Frame_Index 0-4 (seconds 0-5)
                (3, 7, "middle"),   # Frame_Index 3-7 (seconds 3-8)
                (5, 9, "late")      # Frame_Index 5-9 (seconds 5-10)
            ]
            
            for start_frame, end_frame, interval_name in intervals:
                # Filter frames within this interval using Frame_Index
                interval_frames = group[(group['Frame_Index'] >= start_frame) & (group['Frame_Index'] <= end_frame)]
                
                if len(interval_frames) == 0:
                    continue
                
                # Calculate mean of features for this interval
                interval_features = interval_frames[feature_cols].mean()
                
                # Create new row with interval suffix
                new_row = {}
                
                # Add ID with interval suffix
                new_row['ID'] = f"{id_name}_{interval_name}"
                new_row['Original_ID'] = id_name
                
                # Copy all original metadata fields exactly as they appear
                for col in metadata_cols:
                    if col != 'ID':
                        new_row[col] = metadata_info[col]
                
                # Add interval-specific metadata
                new_row['Interval_Type'] = interval_name
                new_row['Start_Frame'] = start_frame
                new_row['End_Frame'] = end_frame
                new_row['Num_Frames_Used'] = len(interval_frames)
                
                # Add the averaged features
                for feature_col in feature_cols:
                    new_row[feature_col] = interval_features[feature_col]
                
                all_intervals.append(new_row)
        
        # Create DataFrame
        result_df = pd.DataFrame(all_intervals)
        
        # Reorder columns to match original structure + new columns
        new_metadata_cols = ['ID', 'Original_ID', 'Interval_Type', 'Start_Frame', 'End_Frame', 'Num_Frames_Used']
        remaining_metadata = [col for col in metadata_cols if col not in ['ID']]
        final_column_order = new_metadata_cols + remaining_metadata + feature_cols
        
        # Ensure all columns exist and reorder
        result_df = result_df.reindex(columns=final_column_order)
        
        # Save overlapping intervals
        output_path = os.path.join(self.output_dir, "all_features_overlapping_intervals.csv")
        result_df.to_csv(output_path, index=False)
        
        print(f"✅ Created {len(result_df)} overlapping intervals")
        print(f"✅ Data augmentation factor: {len(result_df) / len(interval_df):.1f}x")
        print(f"✅ Saved overlapping intervals: {output_path}")
        
        # Print interval type distribution
        print("\nInterval type distribution:")
        print(result_df['Interval_Type'].value_counts())
        
        return result_df
    
    def generate_analysis_report(self, results_df):
        """Generate comprehensive analysis report"""
        print("\n" + "="*60)
        print("FACIAL EXPRESSION & POSE ANALYSIS REPORT")
        print("="*60)
        
        # Basic statistics
        print(f"\nDataset Overview:")
        print(f"- Total intervals analyzed: {len(results_df)}")
        print(f"- Participants: {results_df['Participant ID'].nunique()}")
        print(f"- Conditions: {results_df['Condition'].value_counts().to_dict()}")
        print(f"- Games: {results_df['Game'].value_counts().to_dict()}")
        
        # Stress vs No Stress analysis
        print(f"\nStress Analysis:")
        stress_counts = results_df['Stress/No Stress'].value_counts()
        print(f"- Stress: {stress_counts.get('Stress', 0)}")
        print(f"- No Stress: {stress_counts.get('No Stress', 0)}")
        
        # Feature importance analysis
        feature_cols = [col for col in results_df.columns if col not in ['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction']]
        
        if len(feature_cols) > 0:
            print(f"\nFacial Expression & Pose Features (Mean Values):")
            for condition in results_df['Condition'].unique():
                condition_data = results_df[results_df['Condition'] == condition]
                print(f"\n{condition}:")
                
                for feature in feature_cols[:8]:  # Top 8 features
                    if feature in condition_data.columns:
                        mean_val = condition_data[feature].mean()
                        print(f"  - {feature}: {mean_val:.4f}")
        
        return results_df

if __name__ == "__main__":
    root_dir = os.getcwd()
    analyzer = FacialExpressionPoseAnalyzer(root_dir)
    
    print("🚀 Starting Facial Expression & Pose Analysis...")
    print("="*60)
    
    # Process all intervals
    interval_df, per_second_df = analyzer.process_all_intervals()
    
    if len(interval_df) > 0:
        # Create overlapping intervals
        overlapping_df = analyzer.create_overlapping_intervals(interval_df, per_second_df)
        
        # Generate analysis report
        analyzer.generate_analysis_report(interval_df)
        
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"📁 Output files saved to: {analyzer.output_dir}")
        print(f"📊 Files created:")
        print(f"   • all_features_per_interval.csv ({len(interval_df)} rows)")
        print(f"   • all_features_per_second.csv ({len(per_second_df)} rows)")
        print(f"   • all_features_overlapping_intervals.csv ({len(overlapping_df)} rows)")
    else:
        print("❌ No data processed. Check if videos and label files exist.") 