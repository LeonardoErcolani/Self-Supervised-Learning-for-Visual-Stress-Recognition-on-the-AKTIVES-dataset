import os
import pandas as pd
import numpy as np
import shutil

class LabelExtractor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.output_root = os.path.join(root_dir, "Processed Data")
        self.expert_majority_dir = os.path.join(self.output_root, "Expert Majority")
        
        # Clear existing Expert Majority directory if it exists
        if os.path.exists(self.expert_majority_dir):
            print(f"Clearing existing Expert Majority directory: {self.expert_majority_dir}")
            shutil.rmtree(self.expert_majority_dir)
        
        # Create fresh directory
        os.makedirs(self.expert_majority_dir, exist_ok=True)
        
        # Define allowed conditions
        self.allowed_conditions = ["Brachial Pleksus", "Disleksi", "Mental Retardasyon", "Normal Gelişim"]
        
        # Condition name mapping (folder name -> display name)
        self.condition_mapping = {
            "Brachial Pleksus": "Brachial Plexus Injury",
            "Disleksi": "Dyslexia", 
            "Mental Retardasyon": "Intellectual Disability",
            "Normal Gelişim": "Typical Development"
        }
        
        # Game name mapping (folder name -> standardized name)
        self.game_mapping = {
            "CatchAPet": "CatchAPet",
            "catchapet": "CatchAPet",
            "CatchApet": "CatchAPet",
            "LeapBall": "LeapBall",
            "Leap": "LeapBall",
            "leap": "LeapBall",
            "LP": "LeapBall"
        }
        
        # List to store all DataFrames for combined output
        self.all_labels = []
        
    def find_excel_files_recursive(self, folder_path):
        """Recursively find all Excel files in a folder and its subfolders"""
        excel_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.xlsx') and not file.startswith('~'):
                    excel_files.append(os.path.join(root, file))
        return excel_files

    def preprocess_expert(self, df, expert_id):
        """Rename and normalize a single expert file"""
        # Drop rows where stress/reaction columns are empty
        df = df.dropna(subset=[df.columns[3], df.columns[4]])
        
        # If no valid rows left, return empty DataFrame
        if len(df) == 0:
            return pd.DataFrame(columns=['Minute', 'Second', f'Stress/No Stress-{expert_id}', f'Reaction/No Reaction-{expert_id}'])

        df.rename(columns={
            df.columns[0]: 'Minute',
            df.columns[1]: 'Second',
            df.columns[3]: f'Stress/No Stress-{expert_id}',
            df.columns[4]: f'Reaction/No Reaction-{expert_id}'
        }, inplace=True)

        # Normalize stress values
        df[f'Stress/No Stress-{expert_id}'] = (
            df[f'Stress/No Stress-{expert_id}']
            .astype(str).str.strip().str.lower()
            .replace({
                'stresli': 'Stress',
                'streslì': 'Stress',
                'stresli̇': 'Stress',
                'stressli': 'Stress',
                'stresiz': 'No Stress',
                'stressiz': 'No Stress',
                'nan': np.nan  
            })
        )

        # Normalize reaction values
        df[f'Reaction/No Reaction-{expert_id}'] = (
            df[f'Reaction/No Reaction-{expert_id}']
            .astype(str).str.strip().str.lower()
            .replace({
                'tepki var': 'Reaction',
                'tepki yok': 'No Reaction',
                'nan': np.nan 
            })
        )

        df = df.dropna(subset=[f'Stress/No Stress-{expert_id}', f'Reaction/No Reaction-{expert_id}'])

        return df[['Minute', 'Second', f'Stress/No Stress-{expert_id}', f'Reaction/No Reaction-{expert_id}']]

    def sync_labels(self, row, label_type):
        """Apply majority voting logic for label consensus"""
        labels = [row[f'{label_type}-1'], row[f'{label_type}-2'], row[f'{label_type}-3']]
        if labels.count(labels[0]) >= 2:
            return labels[0]
        elif labels.count(labels[1]) >= 2:
            return labels[1]
        else:
            return labels[1]

    def process_expert_files(self, folder_path, participant_name, participant_code):
        """Process three expert files in a folder and save the synchronized result"""
        excel_files = self.find_excel_files_recursive(folder_path)

        if len(excel_files) != 3:
            print(f"Skipping {folder_path}: Found {len(excel_files)} Excel files, expected 3")
            return None

        try:
            expert1 = self.preprocess_expert(pd.read_excel(excel_files[0]), expert_id=1)
            expert2 = self.preprocess_expert(pd.read_excel(excel_files[1]), expert_id=2)
            expert3 = self.preprocess_expert(pd.read_excel(excel_files[2]), expert_id=3)

            # Skip if any expert has no valid data
            if len(expert1) == 0 or len(expert2) == 0 or len(expert3) == 0:
                print(f"Skipping {folder_path}: One or more experts have no valid data")
                return None

            merged_df = pd.merge(expert1, expert2, on='Second', how='inner')  
            merged_df = pd.merge(merged_df, expert3, on='Second', how='inner')  
            merged_df.drop(columns=['Minute_x', 'Minute_y'], errors='ignore', inplace=True)
            merged_df.dropna(inplace=True)

            # Apply majority voting for consensus labels
            merged_df['Stress/No Stress'] = merged_df.apply(lambda x: self.sync_labels(x, 'Stress/No Stress'), axis=1)
            merged_df['Reaction/No Reaction'] = merged_df.apply(lambda x: self.sync_labels(x, 'Reaction/No Reaction'), axis=1)

            # Extract metadata from folder structure
            rel_path = os.path.normpath(os.path.relpath(folder_path, self.root_dir))
            parts = rel_path.split(os.sep)
            if len(parts) >= 3:
                folder_condition = parts[0]
                condition = self.condition_mapping.get(folder_condition, folder_condition)
                folder_game_name = parts[2]
                game_name = self.game_mapping.get(folder_game_name, folder_game_name)
            else:
                condition = game_name = "Unknown"

            # Determine game abbreviation
            game_abbr = "CP" if game_name == "CatchAPet" else "LP"

            # Create final DataFrame with metadata
            final_df = merged_df[['Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction']].reset_index(drop=True)
            final_df['ID'] = final_df.apply(lambda row: f"{participant_code}_{game_abbr}_t{int(row['Second'])}", axis=1)
            
            # Insert metadata columns
            final_df.insert(0, 'Game', game_name)
            final_df.insert(0, 'Condition', condition)
            final_df.insert(0, 'Name', participant_name)
            final_df.insert(0, 'Participant ID', participant_code)
            final_df.insert(0, 'ID', final_df.pop('ID'))  # Move ID to first position

            return final_df, game_abbr

        except Exception as e:
            print(f"❌ Error processing {folder_path}: {str(e)}")
            return None

    def process_all_labels(self):
        """Process all expert labels and save them"""
        participant_counter = 1
        
        for condition in self.allowed_conditions:
            condition_path = os.path.join(self.root_dir, condition)
            if not os.path.isdir(condition_path):
                continue

            for participant in sorted(os.listdir(condition_path), key=lambda s: s.lower()):
                participant_path = os.path.join(condition_path, participant)
                if not os.path.isdir(participant_path):
                    continue

                participant_code = f"C{participant_counter}"
                participant_counter += 1

                for game in os.listdir(participant_path):
                    game_path = os.path.join(participant_path, game)
                    if not os.path.isdir(game_path):
                        continue

                    result = self.process_expert_files(game_path, participant, participant_code)
                    if result is None:
                        continue
                    
                    label_df, game_abbr = result
                    
                    # Save individual label file
                    output_filename = f"{participant_code}_{game_abbr}_labels.csv"
                    output_file = os.path.join(self.expert_majority_dir, output_filename)
                    label_df.to_csv(output_file, index=False)
                    print(f"\n✅ Saved labels: {output_file}")
                    
                    # Add to combined labels list
                    self.all_labels.append(label_df)
        
        # Create and save combined labels file
        if self.all_labels:
            combined_df = pd.concat(self.all_labels, ignore_index=True)
            
            # Print info about NaN values before dropping
            print("\nChecking for NaN values in columns:")
            for col in combined_df.columns:
                nan_count = combined_df[col].isna().sum()
                if nan_count > 0:
                    print(f"Column '{col}': {nan_count} NaN values")
            
            # Drop NaN values
            original_len = len(combined_df)
            combined_df = combined_df.dropna(how='any', subset=['ID', 'Participant ID', 'Name', 'Condition', 'Game', 'Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction'])
            dropped_count = original_len - len(combined_df)
            
            combined_output = os.path.join(self.expert_majority_dir, "All_labels.csv")
            combined_df.to_csv(combined_output, index=False)
            print(f"\n✅ Saved combined labels: {combined_output}")
            print(f"Total number of labeled frames: {len(combined_df)}")
            if dropped_count > 0:
                print(f"Dropped {dropped_count} rows with missing values")

if __name__ == "__main__":
    root_dir = os.getcwd()
    extractor = LabelExtractor(root_dir)
    extractor.process_all_labels() 