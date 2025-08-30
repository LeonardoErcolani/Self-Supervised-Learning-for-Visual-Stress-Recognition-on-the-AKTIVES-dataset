# Traditional Methods

Implementation of conventional deep learning approaches for stress detection in children, used as baseline comparisons for the DINOv2 self-supervised learning approach.

## 🎯 Overview

This folder contains three traditional supervised learning approaches that were tested on the AKTIVES dataset:

1. **CNN Stress Classifier** - ResNet18-based image classification
2. **FER Stress Classifier** - Pre-trained VGG-Face emotion recognition
3. **LSTM Temporal Model** - Bidirectional LSTM with attention for temporal sequences

## 📁 Structure

```
Traditional Methods/
├── cnn_stress_classifier.py      # ResNet18 CNN classifier
├── fer_stress_classifier.py      # VGG-Face emotion recognition
├── lstm_temporal_model.py        # Bidirectional LSTM with attention
└── README.md                     # This file
```

## 🔬 Implemented Methods

### 1. CNN Stress Classifier (`cnn_stress_classifier.py`)

**Architecture**: Pre-trained ResNet18 with custom classification head

**Features**:
- **Model**: ResNet18 from torchvision
- **Input**: Face images (224x224 RGB)
- **Augmentation**: Random horizontal flip, rotation, color jitter
- **Training**: Transfer learning with fine-tuning
- **Loss**: Cross-entropy with class weights

**Performance**: 
- **Accuracy**: 0.686
- **F1-Score**: 0.0126
- **Recall**: 0.0064
- **Precision**: 0.45

### 2. FER Stress Classifier (`fer_stress_classifier.py`)

**Architecture**: Pre-trained VGG-Face model for facial emotion recognition

**Features**:
- **Model**: VGG-Face architecture (TensorFlow/Keras)
- **Input**: Face images (48x48 grayscale)
- **Emotions**: 6 classes (Happy, Sad, Angry, Fear, Surprise, Disgust)
- **Training**: Fine-tuning on stress/no-stress labels
- **Loss**: Categorical crossentropy

**Performance**:
- **Accuracy**: 0.707
- **F1-Score**: 0.0
- **Recall**: 0.0
- **Precision**: 0.0

### 3. LSTM Temporal Model (`lstm_temporal_model.py`)

**Architecture**: Bidirectional LSTM with attention mechanism

**Features**:
- **Model**: Custom LSTM with attention
- **Input**: Temporal sequences of DINOv2 features
- **Architecture**: Bidirectional LSTM + attention + classification head
- **Training**: End-to-end training on temporal sequences
- **Loss**: Cross-entropy with focal loss option

**Performance**: 
- **Accuracy**: Lower than DINOv2 approaches
- **Issue**: Poor generalization on limited temporal data

## 🚀 Usage

### Run CNN Classifier
```bash
cd "Traditional Methods"
python cnn_stress_classifier.py
```

### Run FER Classifier
```bash
cd "Traditional Methods"
python fer_stress_classifier.py
```

### Run LSTM Model
```bash
cd "Traditional Methods"
python lstm_temporal_model.py
```

## 🔗 References

- **ResNet18**: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **VGG-Face**: [Deep Face Recognition](https://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
- **AKTIVES Dataset**: [Coşkun et al. (2023)](https://www.nature.com/articles/s41597-023-02272-2#Sec10)

## 📚 Next Steps

After running traditional methods, proceed to:

1. **Feature Extraction**: DINOv2, emotions, landmarks
2. **Temporal Aggregation**: Multiple temporal strategies
3. **Advanced Training**: Unified experiment pipeline
4. **Performance Analysis**: Compare with DINOv2 results

---

**Note**: These traditional methods serve as important baselines that demonstrate why self-supervised learning approaches like DINOv2 are necessary for pediatric stress detection with limited labeled data.
