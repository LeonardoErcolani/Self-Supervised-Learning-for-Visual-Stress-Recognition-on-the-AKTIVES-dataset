# Preprocessing Pipeline

This folder contains the preprocessing pipeline for the AKTIVES dataset. The pipeline processes raw video data and expert annotations to prepare them for feature extraction and machine learning analysis.

## 🏗️ Pipeline Overview

The preprocessing pipeline consists of three main stages that process data sequentially:

```
Raw Data → Labels → Frames → Feature Extraction
```

## 📁 File Structure

```
1.Preprocessing/
├── README.md                 # This file
├── main.py                   # Main pipeline orchestrator
├── label_extraction.py       # Expert label processing
├── video_processing.py       # Video frame extraction
├── face_extraction.py        # Face detection and extraction
└── experiments/              # Experimental scripts and variations
```

## 🔧 Components

### 1. Label Extraction (`label_extraction.py`)

**Purpose**: Processes expert annotations from Excel files to create standardized CSV labels.

**Key Features**:
- **Multi-expert consensus**: Combines annotations from multiple experts
- **Label normalization**: Standardizes stress and reaction labels
- **Condition mapping**: Maps Turkish condition names to English equivalents
- **Game standardization**: Normalizes game name variations

**Output**: 
- `Processed Data/Expert Majority/` - Consensus labels for each participant-game combination

**Supported Conditions**:
- Brachial Plexus Injury (`Brachial Pleksus`)
- Dyslexia (`Disleksi`) 
- Intellectual Disability (`Mental Retardasyon`)
- Typical Development (`Normal Gelişim`)

**Supported Games**:
- CatchAPet 
- LeapBall 

### 2. Video Processing (`video_processing.py`)

**Purpose**: Extracts frames from video files at specific time intervals based on expert annotations.

**Key Features**:
- **Temporal alignment**: Extracts frames aligned with expert annotation timestamps
- **Interval-based extraction**: Creates 10-second interval groups
- **Frame organization**: Organizes frames by participant, game, and time interval
- **Automatic cleanup**: Removes existing frame data before processing

**Output**:
- `Processed Data/Frames/` - Organized frame collections for each participant-game-interval

**Frame Structure**:
```
Frames/
├── C1_CP_frames/                    # Participant C1, CatchAPet
│   ├── C1_CP_10/                    # 10-second interval at 10s
│   │   ├── second_1.png             # Frame at 1 second
│   │   ├── second_2.png             # Frame at 2 seconds
│   │   └── ...                      # ... up to 10 seconds
│   └── C1_CP_20/                    # 10-second interval at 20s
└── C1_LP_frames/                    # Participant C1, LeapBall
```

### 3. Face Extraction (`face_extraction.py`)

**Purpose**: Detects and extracts face regions from extracted frames using MediaPipe.

**Key Features**:
- **MediaPipe integration**: Uses Google's MediaPipe for robust face detection
- **Child face selection**: Intelligent selection of the most likely child face
- **Multi-face handling**: Processes cases with multiple detected faces
- **Padding control**: Adds configurable padding around detected faces
- **Quality scoring**: Uses position and size metrics to select optimal faces

**Output**:
- `Processed Data/Face Data/` - Face crops organized by participant, game, and interval

**Face Selection Algorithm**:
- **Position score**: Faces closer to image center get higher scores
- **Size score**: Larger faces (likely closer to camera) get higher scores
- **Combined scoring**: 70% size weight + 30% position weight

## 🚀 Usage

### Quick Start

Run the complete preprocessing pipeline:

```bash
cd "1.Preprocessing"
python main.py
```

### Step-by-Step Execution

If you want to run stages individually:

```python
from label_extraction import LabelExtractor
from video_processing import VideoProcessor
from face_extraction import FaceExtractor

# Step 1: Process labels
label_extractor = LabelExtractor(root_dir)
label_extractor.process_all_labels()

# Step 2: Extract video frames
video_processor = VideoProcessor(root_dir)
video_processor.process_all_videos()

# Step 3: Extract faces
face_extractor = FaceExtractor(root_dir)
face_extractor.process_all_faces()
```

### Custom Configuration

Modify the pipeline behavior by editing the respective classes:

```python
# Adjust face detection confidence
face_detection = mp.solutions.face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.7  # Increase for stricter detection
)

# Adjust face padding
face_extractor.extract_face(image, face_coords, padding=30)  # More padding
```

## 📋 Requirements

### Python Dependencies

```bash
pip install opencv-python
pip install mediapipe
pip install pandas
pip install numpy
pip install openpyxl 
```

### System Requirements

- **OpenCV**: Video processing and image manipulation
- **MediaPipe**: Face detection and landmark extraction
- **Pandas**: Data manipulation and CSV processing
- **NumPy**: Numerical operations

### Input Data Structure

The pipeline expects the following directory structure:

```
Root Directory/
├── Brachial Pleksus/
│   ├── Participant1/
│   │   ├── CatchAPet/
│   │   │   └── video.mp4
│   │   └── LeapBall/
│   │       └── video.mp4
│   └── Participant2/
├── Disleksi/
├── Mental Retardasyon/
├── Normal Gelişim/
└── Processed Data/  # Created automatically
```

## 📊 Output Structure

After running the pipeline, you'll have:

```
Processed Data/
├── Expert Majority/           # Consensus expert labels
├── Frames/                   # Extracted video frames
└── Face Data/                # Extracted face crops
```

## ⚠️ Important Notes

### Data Cleanup
- The pipeline automatically removes existing processed data before starting
- This ensures clean, consistent outputs but means you'll lose previous results

### Memory Usage
- Video processing can be memory-intensive for large datasets
- Consider processing in batches for very large video collections

### File Naming
- Participant codes are automatically assigned (C1, C2, C3, etc.)
- Game abbreviations are standardized (CP for CatchAPet, LP for LeapBall)

### Error Handling
- The pipeline continues processing even if individual files fail
- Check console output for warnings and errors
- Missing label files are logged but don't stop the pipeline

## 🔍 Troubleshooting

### Common Issues

1. **Missing dependencies**: Install required packages with pip
2. **Video format issues**: Ensure videos are in common formats (MP4, AVI, etc.)
3. **Memory errors**: Process smaller batches or reduce video resolution
4. **Face detection failures**: Adjust `min_detection_confidence` parameter


## 🔗 Next Steps

After preprocessing, the data is ready for:

1. **Feature Extraction**: DINOv2 embeddings, emotion probabilities, landmarks
2. **Temporal Aggregation**: Creating interval-based feature representations
3. **Machine Learning**: Training stress classification models

## 📚 References

- **MediaPipe**: [Google MediaPipe Documentation](https://mediapipe.dev/)
- **OpenCV**: [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- **Pandas**: [Pandas User Guide](https://pandas.pydata.org/docs/user_guide/index.html)

---

**Note**: This preprocessing pipeline is designed specifically for the AKTIVES dataset structure.
