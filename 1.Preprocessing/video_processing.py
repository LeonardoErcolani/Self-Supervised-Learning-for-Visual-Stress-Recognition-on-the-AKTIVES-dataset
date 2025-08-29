import os
import cv2
import pandas as pd
import numpy as np

class VideoProcessor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.output_root = os.path.join(root_dir, "Processed Data")
        self.frames_dir = os.path.join(self.output_root, "Frames")
        if os.path.exists(self.frames_dir):
            import shutil
            shutil.rmtree(self.frames_dir)
        os.makedirs(self.frames_dir, exist_ok=True)
        
        # Define allowed conditions
        self.allowed_conditions = ["Brachial Pleksus", "Disleksi", "Mental Retardasyon", "Normal Gelişim"]

    def extract_frames(self, video_path, label_df, participant_code, game_abbr):
        """Extract frames from video every second and organize them in 10-second interval groups"""
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = total_frames / fps

        # Create output directory for frames
        frame_output_dir = os.path.join(self.frames_dir, f"{participant_code}_{game_abbr}_frames")
        os.makedirs(frame_output_dir, exist_ok=True)

        # Get unique 10-second intervals from the label dataframe
        intervals = label_df['Second'].unique()
        
        # For each 10-second interval
        for interval in intervals:
            # Create a directory for this 10-second interval
            interval_dir = os.path.join(frame_output_dir, f"{participant_code}_{game_abbr}_{int(interval)}")
            os.makedirs(interval_dir, exist_ok=True)
            
            # Extract frames for each second in the 10 seconds before this interval
            start_second = max(0, int(interval) - 9)  # Start 9 seconds before the interval
            end_second = int(interval)  # End at the interval point
            
            for second in range(start_second, end_second + 1):
                if second >= duration:
                    continue
                    
                cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
                success, frame = cap.read()
                if not success:
                    print(f"⚠️ Failed to read frame at {second}s in {video_path}")
                    continue

                # Save frame with the second as part of the filename
                frame_filename = f"second_{second}.png"
                frame_path = os.path.join(interval_dir, frame_filename)
                cv2.imwrite(frame_path, frame)

        # Count total frames saved
        num_saved_frames = sum(len([f for f in os.listdir(d) if f.endswith(".png")]) 
                             for d in [os.path.join(frame_output_dir, d) for d in os.listdir(frame_output_dir)])
        print(f"--Total frames saved: {num_saved_frames} in {frame_output_dir}")

        cap.release()
        print(f"✅ Completed processing frames for {participant_code} / {game_abbr}")
        return num_saved_frames

    def process_all_videos(self):
        """Process all videos and extract frames for all participants and conditions"""
        participant_counter = 1
        total_frames_all_videos = 0
        
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

                    # Find the corresponding label file
                    game_abbr = "CP" if "catchapet" in game.lower() else "LP"
                    label_file = os.path.join(self.output_root, "Expert Majority", f"{participant_code}_{game_abbr}_labels.csv")
                    
                    if not os.path.exists(label_file):
                        print(f"❌ Label file not found: {label_file}") # This can be normal as some candidates have no labels
                        continue

                    # Read the label file
                    label_df = pd.read_csv(label_file)

                    # Find and process video
                    for dirpath, _, filenames in os.walk(game_path):
                        for file in filenames:
                            if file.endswith(".mp4"):
                                video_path = os.path.join(dirpath, file)
                                num_frames = self.extract_frames(video_path, label_df, participant_code, game_abbr)
                                total_frames_all_videos += num_frames
                                break  # Process only the first video found
        print(f"Total frames extracted from all videos: {total_frames_all_videos}")

if __name__ == "__main__":
    root_dir = os.getcwd()
    processor = VideoProcessor(root_dir)
    processor.process_all_videos()