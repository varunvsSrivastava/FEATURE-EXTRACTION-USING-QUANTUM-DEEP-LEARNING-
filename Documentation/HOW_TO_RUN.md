How to Run the Emotion Recognition App

Prerequisites

Python 3.8 or higher
Webcam (for real-time emotion detection)

Step-by-Step Instructions

1. Install Required Packages

Open PowerShell or Command Prompt in the project directory and run:

```powershell
pip install -r requirements.txt
```

This will install all necessary dependencies including:
- TensorFlow
- Streamlit
- OpenCV
- NumPy
- Pillow

2. Run the Application

Execute the following command:

```powershell
streamlit run emotion_recognition_app.py
```

3. Using the App

Once the app starts:
- Your default browser will open automatically
- If not, navigate to: http://localhost:8501

Available Tabs:

Upload and Analyze
1. Upload an image (JPG, PNG, BMP)
2. Click Analyze Emotion button
3. View detected faces with emotion labels
4. See confidence scores and detailed predictions

Model Info
1. View model architecture details
2. Check test and validation accuracy
3. See all emotion classes

Confusion Matrix
1. Click Generate Confusion Matrix button
2. The model will be evaluated on the FER2013 test dataset
3. View the confusion matrix heatmap
4. See overall test accuracy
5. Review classification report with precision, recall, and F1-scores
6. Analyze per-class accuracy bar chart

About
1. Learn about the model architecture
2. View key features
3. Read dataset information

4. Stop the Application

Press Ctrl + C in the terminal where Streamlit is running

Optional: Train Your Own Model

If you want to retrain the model with your own data:

1. Prepare the Dataset

Ensure fer2013.csv is in the project directory

2. Run Training Script

```powershell
python train_resnet50_quantum.py --epochs 50 --batch_size 32
```

Training Parameters (optional):
- epochs: Number of training epochs (default: 50)
- batch_size: Batch size for training (default: 32)
- learning_rate: Initial learning rate (default: 0.0001)

3. Monitor Training

Training logs will be saved in the logs/ directory. The best model will be saved based on your training configuration.

Troubleshooting

Issue: "Module not found" error
Solution: Run pip install -r requirements.txt again

Issue: Webcam not detected
Solution:
- Check if your webcam is connected
- Ensure no other application is using the webcam
- Try refreshing the browser page

Issue: App runs slowly
Solution:
- Close other applications
- Use a GPU-enabled system for better performance
- Reduce image size if uploading large images

Issue: "Port already in use"
Solution: Stop other Streamlit apps or specify a different port:
```powershell
streamlit run emotion_recognition_app.py --server.port 8502
```

Project Files

emotion_recognition_app.py - Main application
fer_model_resnet50_quantum_best.h5 - Trained model (275 MB)
fer2013.csv - Training dataset
train_resnet50_quantum.py - Model training script
requirements.txt - Python dependencies

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
