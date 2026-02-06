Quick Commands - Emotion Recognition App

Essential Commands

1. Install Dependencies
```powershell
pip install -r requirements.txt
```

2. Run the Streamlit App
```powershell
streamlit run emotion_recognition_app.py
```

3. Train the Model (Optional)
```powershell
python train_resnet50_quantum.py --epochs 50 --batch_size 32
```

4. Compute Metrics (Precision/Recall/F1 + Inference Time)
```powershell
python scripts/evaluate_metrics.py
```

5. Stop the App
```powershell
Ctrl + C
```

6. Run on Different Port (if 8501 is busy)
```powershell
streamlit run emotion_recognition_app.py --server.port 8502
```

Default Access

Once running, open your browser and go to:
```
http://localhost:8501
```

Project Structure

emotion_recognition_app.py - Main Streamlit application
train_resnet50_quantum.py - Model training script
final_model70.keras - Pre-trained Model 1 (70-15-15 split)
final_model80.keras - Pre-trained Model 2 (80-10-10 split)
training_metadata_70.json - Model 1 metadata
training_metadata_80.json - Model 2 metadata
fer2013.csv - Training dataset
requirements.txt - Python dependencies
