Emotion Recognition App - Documentation

Welcome to the Emotion Recognition App documentation. This folder contains all the documentation files you need to understand, run, and use the application.

Documentation Files

1. HOW_TO_RUN.md
   Complete step-by-step guide to set up and run the application
   Includes installation instructions, troubleshooting tips, and system requirements

2. QUICK_COMMANDS.md
   Quick reference for essential commands
   Useful for developers who want fast access to common operations

3. MODEL_PERFORMANCE.md
   Detailed model performance metrics and analysis
   Includes accuracy scores, dataset distribution, and usage recommendations

4. CHANGELOG.md
   Version history and changes made to the application
   Documents the evolution of the app and new features

5. EMOTION_ACCURACY_REPORT.md
   Emotion-wise accuracy breakdown
   Detailed metrics for each emotion class

Getting Started

If you're new to this application, start with HOW_TO_RUN.md for complete setup instructions.

For quick reference, check QUICK_COMMANDS.md.

Project Overview

This is a ResNet50-based emotion recognition system with two available models:

**Model 1 (70-15-15 split) - Recommended**
- File: final_model70.keras
- Better generalization and less overfitting
- Training: 70%, Validation: 15%, Test: 15%

**Model 2 (80-10-10 split)**
- File: final_model80.keras
- More training data exposure
- Training: 80%, Validation: 10%, Test: 10%

Key Features

- Real-time emotion detection from images
- Support for multiple faces in a single image
- 7 emotion classes: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- ResNet50 backbone with quantum-inspired layers
- Clean, professional user interface
- Model selection option for comparison

Technology Stack

- Python 3.8+
- TensorFlow/Keras
- Streamlit
- OpenCV
- NumPy, Pandas
- Scikit-learn

Support

For issues or questions, refer to the troubleshooting section in HOW_TO_RUN.md.
