import os
from label_extraction import LabelExtractor
from video_processing import VideoProcessor
from face_extraction import FaceExtractor

def main():
    root_dir = os.getcwd()
    
    # Step 1: Extract and process labels
    print("\n=== Step 1: Processing Expert Labels ===")
    label_extractor = LabelExtractor(root_dir)
    label_extractor.process_all_labels()
    
    # Step 2: Extract frames from videos
    print("\n=== Step 2: Extracting Video Frames ===")
    video_processor = VideoProcessor(root_dir)
    video_processor.process_all_videos()
    
    # Step 3: Extract faces from frames
    print("\n=== Step 3: Extracting Faces ===")
    face_extractor = FaceExtractor(root_dir)
    face_extractor.process_all_faces()
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    main() 