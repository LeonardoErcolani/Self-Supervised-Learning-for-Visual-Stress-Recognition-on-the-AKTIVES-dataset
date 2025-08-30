#!/usr/bin/env python3
"""
🚀 TRAINING SCRIPT - TEMPLATE FOR RUNNING EXPERIMENTS

This script is your template to run stress classification experiments on the AKTIVES dataset.
It contains pre-configured experiments from thesis results and shows you how to create your own.

📋 HOW TO USE:
1. To run all experiments: python train.py
2. To run specific experiments: Edit the EXPERIMENTS list below
3. To add new experiments: Copy and modify existing experiment configurations

🔧 EXPERIMENT TYPES:
- "single": Use one feature type (DINOv2, Emotions, or Landmarks)
- "combined": Combine two feature types (e.g., DINOv2 + Emotions)

⏱️ TEMPORAL STRATEGIES:
- "Baseline 10 seconds": Single frame at 10th second
- "Average per interval": Average features across interval
- "Triple intervals": Three samples per interval (early, middle, late)
- "Overlapping intervals": 5-second sliding windows

📊 FEATURE OPTIONS:
- "DINOv2": Self-supervised visual features (768 dimensions)
- "Emotions": FER emotion probabilities (6 dimensions)
- "Expression_Pose": MediaPipe landmarks (468+33 points)

⚙️ PCA OPTIONS:
- True: Apply dimensionality reduction (10 components)
- False: Use original feature dimensions

🎯 QUICK START:
1. Ensure feature extraction is complete
2. Edit experiments below if needed
3. Run: python train.py
4. Check Results/ folder for outputs

⚠️ IMPORTANT: All experiments use participant-based splitting to prevent data leakage.
"""

from unified_experiment_runner import run_experiment

# ============================================================================
# 🎯 EXPERIMENT CONFIGURATIONS - EDIT THESE TO CUSTOMIZE YOUR EXPERIMENTS
# ============================================================================
"""
📝 TO ADD YOUR OWN EXPERIMENT:
1. Copy any experiment configuration below
2. Change the "name" to something descriptive
3. Modify feature types, temporal strategies, or PCA settings
4. Ensure the CSV file paths exist in your Processed Data folder

📁 REQUIRED CSV FILES:
- DINOv2: "Processed Data/DinoV2 Features Extracted/all_features_*.csv"
- Emotions: "Processed Data/FER_Emotion_Features/all_emotions_*.csv"
- Landmarks: "Processed Data/Expression_Pose_Analysis/all_features_*.csv"

🔍 AVAILABLE CSV SUFFIXES:
- "_at_sec_10.csv" (Baseline 10 seconds)
- "_per_interval.csv" (Average per interval)
- "_triple_intervals.csv" (Triple intervals)
- "_overlapping_intervals.csv" (Overlapping intervals)
"""

EXPERIMENTS = [
    {
        "name": "DINOv2_PCA_Overlapping_Best",
        "type": "single",
        "input_csv": "Processed Data/DinoV2 Features Extracted/all_features_overlapping_intervals.csv",
        "feature_type": "DINOv2",
        "temporal_strategy": "Overlapping intervals",
        "use_pca": True
    },
    
    {
        "name": "DINOv2_Emotion_PCA_Averaged",
        "type": "combined",
        "dinov2_csv": "Processed Data/DinoV2 Features Extracted/all_features_overlapping_intervals.csv",
        "second_csv": "Processed Data/FER_Emotion_Features/all_emotions_overlapping_intervals.csv",
        "second_feature_type": "Emotions",
        "temporal_strategy": "Average per interval",
        "use_pca": True
    },
    
    {
        "name": "DINOv2_FullFrame_Overlapping",
        "type": "single",
        "input_csv": "Processed Data/DinoV2 Features Extracted/all_features_overlapping_intervals.csv",
        "feature_type": "DINOv2",
        "temporal_strategy": "Overlapping intervals",
        "use_pca": False
    },
    
    {
        "name": "DINOv2_Emotion_Triple_Intervals",
        "type": "combined",
        "dinov2_csv": "Processed Data/DinoV2 Features Extracted/all_features_triple_intervals.csv",
        "second_csv": "Processed Data/FER_Emotion_Features/all_emotions_triple_intervals.csv",
        "second_feature_type": "Emotions",
        "temporal_strategy": "Triple intervals",
        "use_pca": False
    },
    
    {
        "name": "DINOv2_Landmarks_Overlapping",
        "type": "combined",
        "dinov2_csv": "Processed Data/DinoV2 Features Extracted/all_features_overlapping_intervals.csv",
        "second_csv": "Processed Data/Expression_Pose_Analysis/all_features_overlapping_intervals.csv",
        "second_feature_type": "Expression_Pose",
        "temporal_strategy": "Overlapping intervals",
        "use_pca": False
    },
    
    {
        "name": "DINOv2_PCA_Averaged",
        "type": "single",
        "input_csv": "Processed Data/DinoV2 Features Extracted/all_features_per_interval.csv",
        "feature_type": "DINOv2",
        "temporal_strategy": "Average per interval",
        "use_pca": True
    },
    
    {
        "name": "DINOv2_Emotion_PCA_Overlapping",
        "type": "combined",
        "dinov2_csv": "Processed Data/DinoV2 Features Extracted/all_features_overlapping_intervals.csv",
        "second_csv": "Processed Data/FER_Emotion_Features/all_emotions_overlapping_intervals.csv",
        "second_feature_type": "Emotions",
        "temporal_strategy": "Overlapping intervals",
        "use_pca": True
    },
    
    {
        "name": "DINOv2_Landmarks_PCA_Averaged",
        "type": "combined",
        "dinov2_csv": "Processed Data/DinoV2 Features Extracted/all_features_per_interval.csv",
        "second_csv": "Processed Data/Expression_Pose_Analysis/all_features_per_interval.csv",
        "second_feature_type": "Expression_Pose",
        "temporal_strategy": "Average per interval",
        "use_pca": True
    }
    # ADD MORE EXPERIMENTS HERE - just copy and paste the examples above and modify them
]

# ============================================================================
# AVAILABLE OPTIONS
# ============================================================================
"""
FEATURE TYPES:
- "DINOv2"           - Full frame DINOv2 features
- "Face DINOv2"      - Face-only DINOv2 features  
- "Upper Body DINOv2" - Upper body DINOv2 features
- "Emotions"         - FER emotion probabilities
- "Expression_Pose"  - Expression and pose landmarks

EXPERIMENT TYPES:
- "single"           - Use one feature type
- "combined"         - Combine DINOv2 + another feature type

TEMPORAL STRATEGIES:
- "Baseline 10 seconds"     - Single frame at 10th second
- "Average per interval"    - Average features per game interval
- "Triple intervals"        - Features from three consecutive intervals
- "Overlapping intervals"   - 5-second overlapping windows

PCA:
- True               - Apply PCA dimensionality reduction (10 components for DINOv2)
- False              - Use original features
"""

# ============================================================================
# 🚀 EXECUTION SECTION - RUN YOUR EXPERIMENTS HERE
# ============================================================================
"""
🎯 TO RUN YOUR EXPERIMENTS:
1. Make sure your experiment configurations above are correct
2. Ensure all CSV files exist in the specified paths
3. Run this script: python train.py
4. Watch the progress and check Results/ folder for outputs

📋 WHAT HAPPENS WHEN YOU RUN:
- Each experiment will be executed sequentially
- Progress and results will be displayed in real-time
- All results will be saved to the Results/ folder
- A final summary will show the best performing configurations

⚠️ DON'T MODIFY BELOW THIS LINE - This section handles the execution logic
"""

if __name__ == "__main__":
    print("🚀 Starting Best Performing Experiments from Thesis...")
    print(f"📋 Total experiments to run: {len(EXPERIMENTS)}")
    print("\n" + "="*60)

    all_results = []
    successful_experiments = 0
    failed_experiments = 0
    
    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n🔬 EXPERIMENT {i}/{len(EXPERIMENTS)}")
        print(f"📋 Name: {exp_config['name']}")
        print(f"🔧 Type: {exp_config['type']}")
        
        if exp_config['type'] == 'single':
            print(f"📊 Feature Type: {exp_config['feature_type']}")
            print(f"⏱️  Temporal Strategy: {exp_config['temporal_strategy']}")
            print(f"📉 PCA: {exp_config['use_pca']}")
        else:
            print(f"🔗 Combining: DINOv2 + {exp_config['second_feature_type']}")
            print(f"⏱️  Temporal Strategy: {exp_config['temporal_strategy']}")
            print(f"📉 PCA: {exp_config['use_pca']}")
        
        print("-" * 40)
        
        try:
            results = run_experiment(exp_config)
            
            if results:
                print(f"✅ Experiment {i} completed successfully!")
                print(f"📊 Trained {len(results)} models")
                print(f"🏆 Best F1-Score: {max([r['F1-Score'] for r in results]):.4f}")
                
                for result in results:
                    result['Experiment_Number'] = i
                    result['Total_Experiments'] = len(EXPERIMENTS)
                    all_results.append(result)
                
                successful_experiments += 1
            else:
                print(f"❌ Experiment {i} failed - no models completed")
                failed_experiments += 1
                
        except Exception as e:
            print(f"💥 Error in experiment {i}: {e}")
            failed_experiments += 1
            continue
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 FINAL SUMMARY")
    print("="*60)
    print(f"✅ Successful experiments: {successful_experiments}")
    print(f"❌ Failed experiments: {failed_experiments}")
    print(f"📊 Total experiments: {len(EXPERIMENTS)}")
    
    if all_results:
        print(f"🤖 Total models trained: {len(all_results)}")
        print(f"🏆 Overall best F1-Score: {max([r['F1-Score'] for r in all_results]):.4f}")
        print(f"📁 Results saved to: Results/")
        
        best_result = max(all_results, key=lambda x: x['F1-Score'])
        print(f"🥇 Best experiment: {best_result['Experiment']} (F1: {best_result['F1-Score']:.4f})")
        
        best_model = max(all_results, key=lambda x: x['F1-Score'])
        print(f"🤖 Best model: {best_model['Model']} (F1: {best_model['F1-Score']:.4f})")
    else:
        print("💥 No experiments completed successfully")
        print("\n🔍 Troubleshooting tips:")
        print("1. Check that the CSV file paths exist")
        print("2. Verify the feature_type matches your data")
        print("3. Ensure all required fields are filled in")
        print("4. Check the console output for detailed error messages")
