Emotion Recognition App

A professional emotion recognition system using ResNet50 with quantum-inspired optimization techniques.

Quick Start

1. Install dependencies:
```powershell
pip install -r requirements.txt
```

2. Run the application:
```powershell
streamlit run emotion_recognition_app.py
```

3. Open your browser and navigate to http://localhost:8501

Features

- Real-time emotion detection from images
- Support for multiple faces per image
- Two model options with different training splits
- 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- Professional, clean user interface
- Detailed accuracy metrics and confusion matrix analysis

Project Structure

```
minor project demo/
├── emotion_recognition_app.py          Main Streamlit application
├── train_resnet50_quantum.py           Model training script
├── final_model70.keras                 Model 1 (70-15-15 split) - Recommended
├── final_model80.keras                 Model 2 (80-10-10 split)
├── training_metadata_70.json           Model 1 metadata
├── training_metadata_80.json           Model 2 metadata
├── fer2013.csv                         FER2013 dataset
├── requirements.txt                    Python dependencies
└── Documentation/                      All documentation files
    ├── README.md                       Documentation index
    ├── HOW_TO_RUN.md                   Setup and running guide
    ├── QUICK_COMMANDS.md               Quick command reference
    ├── MODEL_PERFORMANCE.md            Performance metrics
    └── CHANGELOG.md                    Version history
```

Models

**Model 1 (70-15-15 split) - Recommended**
- Training: 70%, Validation: 15%, Test: 15%
- Better generalization and less overfitting
- File: final_model70.keras

**Model 2 (80-10-10 split)**
- Training: 80%, Validation: 10%, Test: 10%
- More training data exposure
- File: final_model80.keras

Both models use ResNet50 with quantum-inspired optimization techniques for advanced emotion recognition.

Requirements

- Python 3.8 or higher
- TensorFlow 2.x
- Streamlit
- OpenCV
- NumPy, Pandas, Pillow
- Scikit-learn

System Requirements

Minimum:
- CPU: Dual-core processor
- RAM: 4 GB
- Storage: 500 MB free space

Recommended:
- CPU: Quad-core processor or better
- RAM: 8 GB or more
- GPU: NVIDIA GPU with CUDA support (for training)
- Storage: 1 GB free space

Usage

1. Launch the app using the Quick Start instructions above
2. Select your preferred model from the dropdown menu
3. Upload an image containing faces
4. Click "Analyze Emotion" to detect emotions
5. View results with confidence scores and visualizations

Documentation

For detailed documentation, please refer to the Documentation folder:
- Full setup guide: Documentation/HOW_TO_RUN.md
- Quick commands: Documentation/QUICK_COMMANDS.md
- Model performance: Documentation/MODEL_PERFORMANCE.md

Technology Stack

- Deep Learning Framework: TensorFlow/Keras
- Web Framework: Streamlit
- Computer Vision: OpenCV
- Model Architecture: ResNet50 with quantum-inspired layers
- Dataset: FER2013 (35,887 facial expression images)

License

This project is for educational and research purposes.
