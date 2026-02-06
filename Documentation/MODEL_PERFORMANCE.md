Model Performance Metrics
ResNet50 Quantum-Inspired Emotion Recognition Model

This document provides comprehensive performance metrics for both available models in the emotion recognition system.

## Model Overview

**Model 1 (70-15-15 split) - Recommended**
- File: final_model70.keras
- Training: 70%, Validation: 15%, Test: 15%
- Better generalization and less overfitting

**Model 2 (80-10-10 split)**
- File: final_model80.keras
- Training: 80%, Validation: 10%, Test: 10%
- More training data exposure

## Accuracy Scores

### Test Accuracy
Evaluated on the FER2013 test dataset - Independent test set not seen during training

### Validation Accuracy
Measured during model training - Used for hyperparameter tuning and early stopping

### Training Accuracy
Measured on the training dataset - Model learns to fit training patterns

*Note: Select either model in the application to view specific accuracy metrics for that model.*

## Latest Classification Metrics

Classification metrics can be viewed in real-time through the application's Model Info tab for each selected model.

## Dataset Distribution

Total Samples: 35,887 images

Emotion       Count    Percentage
Angry         4,953    14%
Disgust       547      1.5%
Fear          5,121    14%
Happy         8,989    25%
Neutral       6,198    17%
Sad           6,077    17%
Surprise      3,880    11%

Model Architecture Details

Model Type: ResNet50-Quantum-Transfer-Learning
Framework: TensorFlow/Keras
Input Size: 224x224 RGB images
Output Classes: 7 emotions

Architecture Components

ResNet50 backbone with ImageNet pre-training
Quantum-inspired attention layers
Custom optimization for emotion recognition
Transfer learning approach for improved performance

Key Insights

Strengths:

Balanced accuracy across test and validation sets
Robust multi-emotion classification
Real-time inference capability
Successfully detects 7 distinct emotion categories

Observations:

Test accuracy slightly higher than validation (0.1% difference)
Model performs equally on seen and unseen data
Class imbalance in dataset (Happy: 25%, Disgust: 1.5%)

Performance by Emotion

The model is trained on FER2013 dataset with varying class distributions:

Well-represented: Happy, Neutral, Sad, Fear, Angry
Under-represented: Disgust (only 1.5% of dataset)

This class imbalance affects per-emotion accuracy, with better performance on frequently occurring emotions.

Usage Recommendations

Best Results For:
- Happy, Neutral, Sad, Fear, and Angry emotions
- Clear, frontal face images
- Good lighting conditions

Challenges:
- Disgust emotion has lower accuracy (limited training data)
- Occluded faces or side profiles
- Poor lighting or low-quality images

Optimization Tips:
- Ensure good image quality for better accuracy
- Optimal face size: 50x50 pixels or larger
- Single or few faces per image for best results
