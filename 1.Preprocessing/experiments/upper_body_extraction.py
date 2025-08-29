import os
import cv2
import numpy as np
import mediapipe as mp

class UpperBodyExtractor:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.output_root = os.path.join(root_dir, "Processed Data")
        self.frames_dir = os.path.join(self.output_root, "Frames")
        self.upper_body_data_dir = os.path.join(self.output_root, "UpperBody Data")
        if os.path.exists(self.upper_body_data_dir):
            import shutil
            shutil.rmtree(self.upper_body_data_dir)
        os.makedirs(self.upper_body_data_dir, exist_ok=True)

        # Initialize MediaPipe face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )

    def detect_faces(self, image):
        """Detect faces in an image using MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        if not results.detections:
            return []
        height, width, _ = image.shape
        faces = []
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x = int(box.xmin * width)
            y = int(box.ymin * height)
            w = int(box.width * width)
            h = int(box.height * height)
            faces.append((x, y, w, h))
        return faces

    def select_likely_child_face(self, faces, image_shape):
        """Select the most likely child face from multiple detections using position and size scoring"""
        if len(faces) == 1:
            return faces[0]
        img_h, img_w = image_shape[:2]
        scores = []
        for x, y, w, h in faces:
            # Calculate position score (closer to center = higher score)
            center_x = x + w / 2
            center_y = y + h / 2
            position_score = 1 - ((abs(center_x - img_w / 2) / (img_w / 2)) +
                                (abs(center_y - img_h / 2) / (img_h / 2))) / 2
            # Calculate size score (larger face = higher score)
            size_score = (w * h) / (img_w * img_h)
            # Combine scores with weights
            total_score = 0.7 * size_score + 0.3 * position_score
            scores.append(total_score)
        return faces[np.argmax(scores)]

    def extract_upper_body(self, image, face_coords, expand_x=1.0, expand_y_top=0.5, expand_y_bottom=2.5):
        """Extract upper body region from image by expanding around the detected face"""
        x, y, w, h = face_coords
        # Expand the box: more to the sides and much more below the face
        x1 = max(0, int(x - w * expand_x))
        y1 = max(0, int(y - h * expand_y_top))
        x2 = min(image.shape[1], int(x + w + w * expand_x))
        y2 = min(image.shape[0], int(y + h + h * expand_y_bottom))
        return image[y1:y2, x1:x2]

    def process_frames(self, participant_code, game_abbr):
        """Process frames to extract upper body regions for a specific participant and game"""
        frame_input_dir = os.path.join(self.frames_dir, f"{participant_code}_{game_abbr}_frames")
        upper_body_output_dir = os.path.join(self.upper_body_data_dir, f"{participant_code}_{game_abbr}_frames")
        os.makedirs(upper_body_output_dir, exist_ok=True)
        if not os.path.exists(frame_input_dir):
            print(f"❌ Frame directory not found: {frame_input_dir}")
            return
        interval_dirs = [d for d in os.listdir(frame_input_dir) if os.path.isdir(os.path.join(frame_input_dir, d))]
        processed_count = 0
        for interval_dir in interval_dirs:
            interval_upper_body_dir = os.path.join(upper_body_output_dir, interval_dir)
            os.makedirs(interval_upper_body_dir, exist_ok=True)
            interval_path = os.path.join(frame_input_dir, interval_dir)
            image_files = [f for f in os.listdir(interval_path) if f.endswith('.png')]
            for image_file in image_files:
                image_full_path = os.path.join(interval_path, image_file)
                img = cv2.imread(image_full_path)
                if img is None:
                    continue
                faces = self.detect_faces(img)
                if not faces:
                    continue
                face_coords = self.select_likely_child_face(faces, img.shape)
                upper_body_img = self.extract_upper_body(img, face_coords)
                save_path = os.path.join(interval_upper_body_dir, image_file)
                cv2.imwrite(save_path, upper_body_img)
                processed_count += 1
            print(f"✅ Processed {processed_count} upper body crops for {participant_code}_{game_abbr}")

    def process_all_upper_bodies(self):
        """Process all frames to extract upper body regions for all participants and games"""
        frame_dirs = [d for d in os.listdir(self.frames_dir) if os.path.isdir(os.path.join(self.frames_dir, d))]
        for frame_dir in frame_dirs:
            parts = frame_dir.split('_')
            if len(parts) >= 2:
                participant_code = parts[0]
                game_abbr = parts[1]
                self.process_frames(participant_code, game_abbr)

if __name__ == "__main__":
    root_dir = os.getcwd()
    extractor = UpperBodyExtractor(root_dir)
    extractor.process_all_upper_bodies() 