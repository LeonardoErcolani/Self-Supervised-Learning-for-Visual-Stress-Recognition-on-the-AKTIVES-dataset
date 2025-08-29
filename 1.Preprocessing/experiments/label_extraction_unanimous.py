import os
import pandas as pd
import numpy as np
import shutil
from collections import defaultdict

class LabelExtractor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.expert_unanimous_dir = "Processed Data/Expert Unanimous"
        
        # Create output directory
        os.makedirs(self.expert_unanimous_dir, exist_ok=True)
        
        # Condition and game mappings
        self.condition_mapping = {
            "Brachial Pleksus": "Brachial Plexus Injury",
            "Disleksi": "Dyslexia", 
            "Mental Retardasyon": "Intellectual Disability",
            "Normal Gelişim": "Typical Development"
        }
        
        self.game_mapping = {
            "CatchAPet": "CatchAPet",
            "LeapBall": "LeapBall",
            "catchapet": "CatchAPet",
            "leap": "LeapBall",
            "leapball": "LeapBall"
        }
        
        # Allowed conditions
        self.allowed_conditions = ["Brachial Pleksus", "Disleksi", "Mental Retardasyon", "Normal Gelişim"]
        
        # Statistics tracking
        self.stats = {
            'total_participants': 0,
            'total_games': 0,
            'unanimous_cases': 0,
            'majority_cases': 0,
            'disagreement_cases': 0
        }
        
        # Expert agreement statistics
        self.agreement_stats = {
            'stress_agreement': defaultdict(int),
            'reaction_agreement': defaultdict(int),
            'both_agreement': defaultdict(int),
            'expert_pair_agreement': defaultdict(lambda: defaultdict(int)),
            'condition_agreement': defaultdict(lambda: defaultdict(int)),
            'game_agreement': defaultdict(lambda: defaultdict(int))
        }
        
        # Store all merged data for detailed analysis
        self.all_merged_data = []
        self.all_labels = []

    def find_excel_files_recursive(self, folder_path):
        """Find all Excel files in a folder recursively"""
        excel_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(('.xlsx', '.xls')):
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

        # Drop rows where stress/reaction values are NaN
        df = df.dropna(subset=[f'Stress/No Stress-{expert_id}', f'Reaction/No Reaction-{expert_id}'])

        return df[['Minute', 'Second', f'Stress/No Stress-{expert_id}', f'Reaction/No Reaction-{expert_id}']]

    def check_unanimous_agreement(self, row, label_type):
        """Check if all three experts agree on a label"""
        labels = [row[f'{label_type}-1'], row[f'{label_type}-2'], row[f'{label_type}-3']]
        return len(set(labels)) == 1  # All labels are the same

    def get_unanimous_label(self, row, label_type):
        """Get the unanimous label if all experts agree"""
        if self.check_unanimous_agreement(row, label_type):
            return row[f'{label_type}-1']  # All are the same, so return any
        return None

    def get_majority_label(self, row, label_type):
        """Get the majority label (2 out of 3 agree)"""
        labels = [row[f'{label_type}-1'], row[f'{label_type}-2'], row[f'{label_type}-3']]
        if labels.count(labels[0]) >= 2:
            return labels[0]
        elif labels.count(labels[1]) >= 2:
            return labels[1]
        else:
            return labels[1]

    def calculate_expert_pair_agreement(self, row, label_type):
        """Calculate agreement between each pair of experts"""
        expert1_label = row[f'{label_type}-1']
        expert2_label = row[f'{label_type}-2']
        expert3_label = row[f'{label_type}-3']
        
        # Expert pairs: (1,2), (1,3), (2,3)
        pairs = [
            (expert1_label, expert2_label),
            (expert1_label, expert3_label),
            (expert2_label, expert3_label)
        ]
        
        agreements = []
        for pair in pairs:
            agreements.append(pair[0] == pair[1])
        
        return agreements

    def process_expert_files(self, folder_path, participant_name, participant_code):
        """Process three expert files in a folder and save only unanimous cases"""
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

            merged_df = pd.merge(expert1, expert2, on='Second', how='inner')  # Changed to inner join
            merged_df = pd.merge(merged_df, expert3, on='Second', how='inner')  # Changed to inner join
            merged_df.drop(columns=['Minute_x', 'Minute_y'], errors='ignore', inplace=True)
            merged_df.dropna(inplace=True)

            # Extract metadata
            rel_path = os.path.normpath(os.path.relpath(folder_path, self.root_dir))
            parts = rel_path.split(os.sep)
            if len(parts) >= 3:
                folder_condition = parts[0]
                condition = self.condition_mapping.get(folder_condition, folder_condition)
                folder_game_name = parts[2]
                game_name = self.game_mapping.get(folder_game_name, folder_game_name)
            else:
                condition = game_name = "Unknown"

            # Store merged data for analysis
            merged_df_copy = merged_df.copy()
            merged_df_copy['Condition'] = condition
            merged_df_copy['Game'] = game_name
            merged_df_copy['Participant'] = participant_code
            self.all_merged_data.append(merged_df_copy)

            # Analyze agreement patterns directly on merged_df
            unanimous_cases, majority_cases, disagreement_cases = self.analyze_expert_agreement(merged_df, condition, game_name)
            
            if not unanimous_cases:
                print(f"No unanimous cases found for {folder_path}")
                return None

            # Create DataFrame with only unanimous cases
            unanimous_df = pd.DataFrame(unanimous_cases)
            
            # Stress: Unanimous agreement required
            unanimous_df['Stress/No Stress'] = unanimous_df.apply(lambda x: self.get_unanimous_label(x, 'Stress/No Stress'), axis=1)
            # Reaction: Majority voting (2 out of 3 agree)
            unanimous_df['Reaction/No Reaction'] = unanimous_df.apply(lambda x: self.get_majority_label(x, 'Reaction/No Reaction'), axis=1)

            # Determine game abbreviation
            game_abbr = "CP" if game_name == "CatchAPet" else "LP"

            # Final DataFrame with metadata
            final_df = unanimous_df[['Minute', 'Second', 'Stress/No Stress', 'Reaction/No Reaction']].reset_index(drop=True)
            final_df['ID'] = final_df.apply(lambda row: f"{participant_code}_{game_abbr}_t{int(row['Second'])}", axis=1)
            
            # Insert metadata columns
            final_df.insert(0, 'Game', game_name)
            final_df.insert(0, 'Condition', condition)
            final_df.insert(0, 'Name', participant_name)
            final_df.insert(0, 'Participant ID', participant_code)
            final_df.insert(0, 'ID', final_df.pop('ID'))  # Move ID to first position

            # Print agreement statistics for this participant/game
            total_cases = len(unanimous_cases) + len(majority_cases) + len(disagreement_cases)
            unanimous_percentage = (len(unanimous_cases) / total_cases * 100) if total_cases > 0 else 0
            print(f"  {participant_code}_{game_abbr}: {len(unanimous_cases)}/{total_cases} unanimous cases ({unanimous_percentage:.1f}%)")

            return final_df, game_abbr

        except Exception as e:
            print(f"❌ Error processing {folder_path}: {str(e)}")
            return None

    def analyze_expert_agreement(self, merged_df, condition, game_name):
        """Analyze expert agreement patterns and return unanimous cases only for stress"""
        unanimous_cases = []
        majority_cases = []
        disagreement_cases = []
        
        for _, row in merged_df.iterrows():
            stress_agreement = self.check_unanimous_agreement(row, 'Stress/No Stress')
            reaction_agreement = self.check_unanimous_agreement(row, 'Reaction/No Reaction')
            
            # Track agreement statistics
            if stress_agreement:
                self.agreement_stats['stress_agreement']['unanimous'] += 1
            else:
                self.agreement_stats['stress_agreement']['disagreement'] += 1
                
            if reaction_agreement:
                self.agreement_stats['reaction_agreement']['unanimous'] += 1
            else:
                self.agreement_stats['reaction_agreement']['disagreement'] += 1
                
            # Only consider stress agreement for unanimous cases
            if stress_agreement:
                self.agreement_stats['both_agreement']['unanimous'] += 1
                unanimous_cases.append(row.to_dict())
                self.stats['unanimous_cases'] += 1
            elif stress_agreement or reaction_agreement:
                self.agreement_stats['both_agreement']['partial'] += 1
                majority_cases.append(row.to_dict())
                self.stats['majority_cases'] += 1
            else:
                self.agreement_stats['both_agreement']['disagreement'] += 1
                disagreement_cases.append(row.to_dict())
                self.stats['disagreement_cases'] += 1
            
            # Track condition and game specific agreement
            self.agreement_stats['condition_agreement'][condition]['total'] += 1
            if stress_agreement:
                self.agreement_stats['condition_agreement'][condition]['unanimous'] += 1
                
            self.agreement_stats['game_agreement'][game_name]['total'] += 1
            if stress_agreement:
                self.agreement_stats['game_agreement'][game_name]['unanimous'] += 1
            
            # Track expert pair agreements
            stress_pairs = self.calculate_expert_pair_agreement(row, 'Stress/No Stress')
            reaction_pairs = self.calculate_expert_pair_agreement(row, 'Reaction/No Reaction')
            
            pair_names = ['E1-E2', 'E1-E3', 'E2-E3']
            for i, (stress_agree, reaction_agree) in enumerate(zip(stress_pairs, reaction_pairs)):
                pair_name = pair_names[i]
                if stress_agree:
                    self.agreement_stats['expert_pair_agreement'][pair_name]['stress_agreement'] += 1
                else:
                    self.agreement_stats['expert_pair_agreement'][pair_name]['stress_disagreement'] += 1
                    
                if reaction_agree:
                    self.agreement_stats['expert_pair_agreement'][pair_name]['reaction_agreement'] += 1
                else:
                    self.agreement_stats['expert_pair_agreement'][pair_name]['reaction_disagreement'] += 1
                
                self.agreement_stats['expert_pair_agreement'][pair_name]['total'] += 1
        
        return unanimous_cases, majority_cases, disagreement_cases

    def process_all_labels(self):
        """Process all expert labels and save only unanimous cases"""
        participant_counter = 1
        
        for condition in self.allowed_conditions:
            condition_path = os.path.join(self.root_dir, condition)
            if not os.path.isdir(condition_path):
                continue

            print(f"\n📁 Processing condition: {condition}")

            for participant in sorted(os.listdir(condition_path), key=lambda s: s.lower()):
                participant_path = os.path.join(condition_path, participant)
                if not os.path.isdir(participant_path):
                    continue

                participant_code = f"C{participant_counter}"
                participant_counter += 1
                self.stats['total_participants'] += 1

                for game in os.listdir(participant_path):
                    game_path = os.path.join(participant_path, game)
                    if not os.path.isdir(game_path):
                        continue

                    self.stats['total_games'] += 1

                    result = self.process_expert_files(game_path, participant, participant_code)
                    if result is None:
                        continue
                    
                    label_df, game_abbr = result
                    
                    # Save individual label file
                    output_filename = f"{participant_code}_{game_abbr}_unanimous_stress_labels.csv"
                    output_file = os.path.join(self.expert_unanimous_dir, output_filename)
                    label_df.to_csv(output_file, index=False)
                    print(f"  ✅ Saved unanimous stress labels: {output_filename}")
                    
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
            
            combined_output = os.path.join(self.expert_unanimous_dir, "All_unanimous_stress_labels.csv")
            combined_df.to_csv(combined_output, index=False)
            print(f"\n✅ Saved combined unanimous stress labels: {combined_output}")
            print(f"Total number of unanimous stress labeled frames: {len(combined_df)}")
            if dropped_count > 0:
                print(f"Dropped {dropped_count} rows with missing values")
            
            # Print final statistics
            self.print_final_statistics()
            self.print_detailed_agreement_statistics()
        else:
            print("\n❌ No unanimous cases found across all participants!")

    def print_final_statistics(self):
        """Print comprehensive statistics about the extraction process"""
        print("\n" + "="*60)
        print("📈 FINAL STATISTICS")
        print("="*60)
        print(f"👥 Total Participants Processed: {self.stats['total_participants']}")
        print(f"🎮 Total Games Processed: {self.stats['total_games']}")
        print(f"✅ Unanimous Stress Cases: {self.stats['unanimous_cases']}")
        print(f"🔄 Majority Cases (not unanimous): {self.stats['majority_cases']}")
        print(f"❌ Disagreement Cases: {self.stats['disagreement_cases']}")
        
        total_cases = self.stats['unanimous_cases'] + self.stats['majority_cases'] + self.stats['disagreement_cases']
        if total_cases > 0:
            unanimous_percentage = (self.stats['unanimous_cases'] / total_cases) * 100
            print(f"📊 Unanimous Stress Agreement Rate: {unanimous_percentage:.1f}%")
            print(f"📊 Data Reduction: {((total_cases - self.stats['unanimous_cases']) / total_cases * 100):.1f}% of cases removed")
        print("="*60)

    def print_detailed_agreement_statistics(self):
        """Print detailed inter-rater reliability statistics"""
        print("\n" + "="*80)
        print("🔬 DETAILED INTER-RATER RELIABILITY STATISTICS")
        print("="*80)
        
        # Overall agreement rates
        print("\n📊 OVERALL AGREEMENT RATES:")
        print("-" * 40)
        
        total_stress = self.agreement_stats['stress_agreement']['unanimous'] + self.agreement_stats['stress_agreement']['disagreement']
        total_reaction = self.agreement_stats['reaction_agreement']['unanimous'] + self.agreement_stats['reaction_agreement']['disagreement']
        total_both = self.agreement_stats['both_agreement']['unanimous'] + self.agreement_stats['both_agreement']['partial'] + self.agreement_stats['both_agreement']['disagreement']
        
        if total_stress > 0:
            stress_agreement_rate = (self.agreement_stats['stress_agreement']['unanimous'] / total_stress) * 100
            print(f"Stress Labels: {stress_agreement_rate:.1f}% unanimous agreement")
            
        if total_reaction > 0:
            reaction_agreement_rate = (self.agreement_stats['reaction_agreement']['unanimous'] / total_reaction) * 100
            print(f"Reaction Labels: {reaction_agreement_rate:.1f}% unanimous agreement")
            
        if total_both > 0:
            both_agreement_rate = (self.agreement_stats['both_agreement']['unanimous'] / total_both) * 100
            print(f"Stress Labels (Unanimous): {both_agreement_rate:.1f}% unanimous agreement")
        
        # Expert pair agreement rates
        print("\n👥 EXPERT PAIR AGREEMENT RATES:")
        print("-" * 40)
        for pair_name, pair_stats in self.agreement_stats['expert_pair_agreement'].items():
            total = pair_stats['total']
            if total > 0:
                stress_rate = (pair_stats['stress_agreement'] / total) * 100
                reaction_rate = (pair_stats['reaction_agreement'] / total) * 100
                print(f"{pair_name}:")
                print(f"  Stress: {stress_rate:.1f}% agreement ({pair_stats['stress_agreement']}/{total})")
                print(f"  Reaction: {reaction_rate:.1f}% agreement ({pair_stats['reaction_agreement']}/{total})")
        
        # Condition-specific agreement rates
        print("\n🏥 CONDITION-SPECIFIC AGREEMENT RATES:")
        print("-" * 40)
        for condition, condition_stats in self.agreement_stats['condition_agreement'].items():
            total = condition_stats['total']
            if total > 0:
                agreement_rate = (condition_stats['unanimous'] / total) * 100
                print(f"{condition}: {agreement_rate:.1f}% unanimous agreement ({condition_stats['unanimous']}/{total})")
        
        # Game-specific agreement rates
        print("\n🎮 GAME-SPECIFIC AGREEMENT RATES:")
        print("-" * 40)
        for game, game_stats in self.agreement_stats['game_agreement'].items():
            total = game_stats['total']
            if total > 0:
                agreement_rate = (game_stats['unanimous'] / total) * 100
                print(f"{game}: {agreement_rate:.1f}% unanimous agreement ({game_stats['unanimous']}/{total})")
        
        # Cohen's Kappa approximation (simple agreement vs. expected agreement)
        print("\n📈 RELIABILITY METRICS:")
        print("-" * 40)
        if total_both > 0:
            # Calculate expected agreement by chance (assuming equal distribution)
            expected_agreement = total_both / 2  # 2 possible combinations: (Stress, NoStress)
            observed_agreement = self.agreement_stats['both_agreement']['unanimous']
            
            if expected_agreement > 0:
                kappa_approx = (observed_agreement - expected_agreement) / (total_both - expected_agreement)
                print(f"Cohen's Kappa Approximation: {kappa_approx:.3f}")
                print(f"  Observed Agreement: {observed_agreement}/{total_both} ({observed_agreement/total_both*100:.1f}%)")
                print(f"  Expected Agreement by Chance: {expected_agreement:.1f}/{total_both} ({expected_agreement/total_both*100:.1f}%)")
        
        print("="*80)

if __name__ == "__main__":
    root_dir = os.getcwd()
    extractor = LabelExtractor(root_dir)
    extractor.process_all_labels()
