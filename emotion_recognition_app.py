"""
Emotion Recognition App - ResNet50 Quantum Model
Advanced Transfer Learning with Quantum-Inspired Architecture
"""

import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import pandas as pd

# Define custom quantum layers for model deserialization
@tf.keras.utils.register_keras_serializable()
class QuantumQubitCircuit(tf.keras.layers.Layer):
    """Quantum circuit simulator with qubit operations (training-compatible)."""
    def __init__(self, num_qubits=6, num_layers=3, **kwargs):
        super(QuantumQubitCircuit, self).__init__(**kwargs)
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.rotation_params = None
        self.entangle_params = None
        self.measurement_angles = None

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        # Variables order/name match training script for weight loading
        self.rotation_params = self.add_weight(
            name="rotation_params",
            shape=(self.num_layers, self.num_qubits, 2),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.entangle_params = self.add_weight(
            name="entangle_params",
            shape=(self.num_layers, self.num_qubits),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.measurement_angles = self.add_weight(
            name="measurement_angles",
            shape=(self.num_qubits,),
            initializer="zeros",
            trainable=True,
        )
        super(QuantumQubitCircuit, self).build(input_shape)

    def _ensure_tensor(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def call(self, x):
        x = self._ensure_tensor(x)
        # Flatten to (batch, features) if needed
        if x.shape.rank is None or x.shape.rank > 2:
            x = tf.reshape(x, [tf.shape(x)[0], -1])

        batch_size = tf.shape(x)[0]
        feature_dim = tf.shape(x)[-1]
        dtype = x.dtype

        # Project/pad/crop to num_qubits deterministically (training-compatible)
        feat_static = x.shape[-1]
        if feat_static is not None and feat_static > self.num_qubits:
            x_projected = x[:, : self.num_qubits]
        elif feat_static is not None and feat_static < self.num_qubits:
            padding = tf.zeros([batch_size, self.num_qubits - feat_static], dtype=dtype)
            x_projected = tf.concat([x, padding], axis=1)
        else:
            # Fallback for dynamic shapes
            def crop():
                return x[:, : self.num_qubits]
            def pad():
                pad_len = self.num_qubits - tf.cast(feature_dim, tf.int32)
                return tf.concat([x, tf.zeros([batch_size, pad_len], dtype=dtype)], axis=1)
            x_projected = tf.cond(feature_dim > self.num_qubits, crop, pad)

        angles = x_projected
        ry_cos = tf.cos(angles / 2)
        ry_sin = tf.sin(angles / 2)
        qubit_real = ry_cos ** 2
        qubit_imag = ry_sin ** 2

        for layer_idx in range(self.num_layers):
            ry_angles = tf.cast(self.rotation_params[layer_idx, :, 0], dtype)
            rz_angles = tf.cast(self.rotation_params[layer_idx, :, 1], dtype)

            # RY
            ry_c = tf.cos(ry_angles / 2)
            ry_s = tf.sin(ry_angles / 2)
            real_old = qubit_real
            imag_old = qubit_imag
            qubit_real = real_old * ry_c - imag_old * ry_s
            qubit_imag = real_old * ry_s + imag_old * ry_c

            # RZ
            rz_c = tf.cos(rz_angles / 2)
            rz_s = tf.sin(rz_angles / 2)
            new_real = qubit_real * rz_c - qubit_imag * rz_s
            new_imag = qubit_real * rz_s + qubit_imag * rz_c
            qubit_real = new_real
            qubit_imag = new_imag

            # Entanglement (pairwise mixing)
            entangle_phases = tf.cast(self.entangle_params[layer_idx, :], dtype)
            ent_c = tf.cos(entangle_phases)
            ent_s = tf.sin(entangle_phases)
            for i in range(self.num_qubits - 1):
                alpha = ent_c[i]
                beta = ent_s[i]

                new_real_i = alpha * qubit_real[:, i] + beta * qubit_real[:, i + 1]
                new_real_ip1 = beta * qubit_real[:, i] + alpha * qubit_real[:, i + 1]
                new_imag_i = alpha * qubit_imag[:, i] + beta * qubit_imag[:, i + 1]
                new_imag_ip1 = beta * qubit_imag[:, i] + alpha * qubit_imag[:, i + 1]

                tail_real = qubit_real[:, i + 2 :] if i < self.num_qubits - 2 else tf.zeros([batch_size, 0], dtype=dtype)
                tail_imag = qubit_imag[:, i + 2 :] if i < self.num_qubits - 2 else tf.zeros([batch_size, 0], dtype=dtype)

                qubit_real = tf.concat([
                    qubit_real[:, :i], tf.expand_dims(new_real_i, 1), tf.expand_dims(new_real_ip1, 1), tail_real
                ], axis=1)
                qubit_imag = tf.concat([
                    qubit_imag[:, :i], tf.expand_dims(new_imag_i, 1), tf.expand_dims(new_imag_ip1, 1), tail_imag
                ], axis=1)

        amplitude = tf.sqrt(qubit_real ** 2 + qubit_imag ** 2)
        measurement_cos = tf.cos(tf.cast(self.measurement_angles, dtype))
        result = amplitude * measurement_cos
        return result

    def compute_output_shape(self, input_shape):
        batch = None
        try:
            ts = tf.TensorShape(input_shape)
            if ts.rank:
                batch = ts[0]
        except Exception:
            pass
        return (batch, self.num_qubits)

    def get_config(self):
        config = super().get_config()
        config.update({"num_qubits": self.num_qubits, "num_layers": self.num_layers})
        return config


@tf.keras.utils.register_keras_serializable()
class QuantumEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, latent_dim=256, **kwargs):
        super(QuantumEncodingLayer, self).__init__(**kwargs)
        self.latent_dim = latent_dim
        self.encoding_matrix = None
        self.phase_bias = None

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        if input_shape is None or input_shape[-1] is None:
            # Fallback to training-time expected dim to match saved weights
            input_dim = 512
        else:
            input_dim = int(input_shape[-1])
        self.encoding_matrix = self.add_weight(
            name="encoding_matrix",
            shape=(input_dim, self.latent_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.phase_bias = self.add_weight(
            name="phase_bias",
            shape=(self.latent_dim,),
            initializer="zeros",
            trainable=True,
        )
        super(QuantumEncodingLayer, self).build(input_shape)

    def _ensure_tensor(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def call(self, x):
        x = self._ensure_tensor(x)
        if x.shape.rank is None or x.shape.rank > 2:
            x = tf.reshape(x, [tf.shape(x)[0], -1])
        encoded = tf.matmul(x, self.encoding_matrix)
        phase_cos = tf.cos(self.phase_bias)
        phase_sin = tf.sin(self.phase_bias)
        output = encoded * phase_cos + encoded * phase_sin
        return output

    def compute_output_shape(self, input_shape):
        batch = None
        try:
            ts = tf.TensorShape(input_shape)
            if ts.rank:
                batch = ts[0]
        except Exception:
            pass
        return (batch, self.latent_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"latent_dim": self.latent_dim})
        return config


@tf.keras.utils.register_keras_serializable()
class QuantumAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(QuantumAttentionLayer, self).__init__(**kwargs)
        self.attention_params = None

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        feat_dim = int(input_shape[-1]) if input_shape is not None and input_shape[-1] is not None else 256
        self.attention_params = self.add_weight(
            name="attention_params",
            shape=(feat_dim,),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(QuantumAttentionLayer, self).build(input_shape)

    def _ensure_tensor(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def call(self, x):
        x = self._ensure_tensor(x)
        if x.shape.rank is None or x.shape.rank > 2:
            x = tf.reshape(x, [tf.shape(x)[0], -1])
        amplitudes = tf.nn.softmax(self.attention_params, axis=-1)
        return x * amplitudes

    def compute_output_shape(self, input_shape):
        try:
            ts = tf.TensorShape(input_shape)
            return tuple(ts.as_list())
        except Exception:
            return (None, 256)

    def get_config(self):
        return super().get_config()


st.set_page_config(
    page_title="ResNet50 Emotion Recognition",
    page_icon="ER",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        text-align: center;
        padding: 30px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    .header h1 {
        margin: 0;
        font-size: 2.5em;
    }
    .emotion-card {
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #667eea;
        background: #f8f9fa;
        margin-bottom: 10px;
    }
    .confidence {
        font-size: 28px;
        font-weight: bold;
        color: #667eea;
    }
    .emotion-name {
        font-size: 20px;
        font-weight: bold;
        margin: 10px 0;
    }
    .accuracy-box {
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 8px;
        text-align: center;
        font-size: 16px;
        font-weight: bold;
    }
    .model-info {
        padding: 15px;
        background: #e8f4f8;
        border-left: 4px solid #667eea;
        border-radius: 4px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Emotion mappings
EMOTION_EMOJI = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "sad": "Sad",
    "surprise": "Surprise",
    "neutral": "Neutral"
}

EMOTION_COLORS = {
    "angry": "#ff4757",
    "disgust": "#2ed573",
    "fear": "#5f27cd",
    "happy": "#ffa502",
    "sad": "#3742fa",
    "surprise": "#ee5a6f",
    "neutral": "#95afc0"
}

EMOTION_DESCRIPTIONS = {
    "angry": "Expression of anger or strong displeasure",
    "disgust": "Expression of disapproval or revulsion",
    "fear": "Expression of anxiety or concern",
    "happy": "Expression of joy or contentment",
    "sad": "Expression of sadness or sorrow",
    "surprise": "Expression of surprise or shock",
    "neutral": "No strong emotional expression"
}


from pathlib import Path


@st.cache_resource
def load_model(model_choice):
    """Load selected ResNet50 Quantum Model"""
    base_dir = Path(__file__).resolve().parent
    if model_choice == "Model 1 (70% Training Split)":
        model_path = base_dir / 'final_model70.keras'
        model_type = "ResNet50-Quantum (70% Training)"
    else:
        model_path = base_dir / 'final_model80.keras'
        model_type = "ResNet50-Quantum (80% Training)"

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        return None, None

    model_path_str = str(model_path)

    try:
        # Both models are in .keras format
        custom_objects = {
            'QuantumQubitCircuit': QuantumQubitCircuit,
            'QuantumEncodingLayer': QuantumEncodingLayer,
            'QuantumAttentionLayer': QuantumAttentionLayer
        }
        try:
            import keras
            model = keras.models.load_model(
                model_path_str,
                custom_objects=custom_objects,
                compile=False,
                safe_mode=False
            )
        except Exception as e_keras:
            # Fallback to tf.keras loader
            try:
                model = tf.keras.models.load_model(
                    model_path_str,
                    custom_objects=custom_objects,
                    compile=False
                )
            except Exception as e_tf:
                raise RuntimeError(f"Keras load failed ({e_keras}); tf.keras fallback failed ({e_tf})")
        return model, model_type
    except Exception as e:
        st.error(f"Could not load model: {e}")
        return None, None


def load_metadata(model_choice):
    """Load model metadata based on selected model"""
    base_dir = Path(__file__).resolve().parent
    
    # Load different metadata files for each model
    if model_choice == "Model 1 (70% Training Split)":
        metadata_path = base_dir / 'training_metadata_70.json'
        split = '70-15-15 (Train-Val-Test)'
        fallback_train_acc = 72.62
        fallback_test_acc = 59.64
        fallback_val_acc = 61.08
    else:
        metadata_path = base_dir / 'training_metadata_80.json'
        split = '80-10-10 (Train-Val-Test)'
        fallback_train_acc = 99.60
        fallback_test_acc = 53.66
        fallback_val_acc = 51.77
    
    if metadata_path.exists():
        try:
            import json
            with open(metadata_path, 'r') as f:
                data = json.load(f)
                
                return {
                    'emotions': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
                    'training_accuracy': data.get('train_accuracy', fallback_train_acc) / 100.0,
                    'test_accuracy': data.get('test_accuracy', fallback_test_acc) / 100.0,
                    'val_accuracy': data.get('val_accuracy', fallback_val_acc) / 100.0,
                    'model_type': 'ResNet50-Quantum-Transfer-Learning',
                    'framework': 'TensorFlow/Keras',
                    'architecture': 'ResNet50 (ImageNet pretrained) + Quantum-inspired layers',
                    'split': split
                }
        except Exception as e:
            st.warning(f"Could not load metadata: {e}")
    
    # Default fallback metadata
    return {
        'emotions': ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'],
        'training_accuracy': fallback_train_acc / 100.0,
        'test_accuracy': fallback_test_acc / 100.0,
        'val_accuracy': fallback_val_acc / 100.0,
        'model_type': 'ResNet50-Quantum-Transfer-Learning',
        'framework': 'TensorFlow/Keras',
        'architecture': 'ResNet50 (ImageNet pretrained) + Quantum-inspired layers',
        'split': split
    }


def preprocess_image_for_resnet(image_array, target_size=(224, 224)):
    """Preprocess image for ResNet-like models (RGB + ImageNet normalization)"""
    pil_img = Image.fromarray(image_array)
    pil_img = pil_img.resize(tuple(target_size), Image.BILINEAR)
    
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')
    
    img_array = np.array(pil_img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    return img_array


def detect_faces_and_emotions(image_array, model, emotion_list, target_size=(224, 224)):
    """Detect faces and predict emotions"""
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Apply histogram equalization to improve face detection in varied lighting
    gray = cv2.equalizeHist(gray)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Balanced face detection parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,  # Smaller scale factor for better detection
        minNeighbors=4,    # Balanced to detect faces while reducing false positives
        minSize=(40, 40),  # Reasonable minimum size
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    results = []
    
    if len(faces) == 0:
        return None, "No faces detected in the image"
    
    # Filter overlapping faces (keep largest)
    faces_filtered = []
    for i, (x1, y1, w1, h1) in enumerate(faces):
        is_duplicate = False
        for j, (x2, y2, w2, h2) in enumerate(faces):
            if i != j:
                # Check if faces overlap significantly
                overlap_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
                overlap_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                area1 = w1 * h1
                area2 = w2 * h2
                
                # If overlap is more than 50% and current face is smaller, skip it
                if overlap_area > 0.5 * min(area1, area2) and area1 < area2:
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            faces_filtered.append((x1, y1, w1, h1))
    
    for (x, y, w, h) in faces_filtered:
        face_region = image_array[y:y+h, x:x+w]
        
        try:
            processed_face = preprocess_image_for_resnet(face_region, target_size=target_size)
            predictions = model.predict(processed_face, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion_name = emotion_list[emotion_idx]
            confidence = float(predictions[emotion_idx])
            
            results.append({
                'face_region': (x, y, w, h),
                'emotion': emotion_name,
                'confidence': confidence,
                'all_predictions': predictions,
                'emotion_list': emotion_list
            })
        except Exception as e:
            st.warning(f"Error processing face: {e}")
            continue
    
    if not results:
        return None, f"Detected {len(faces)} face(s), but couldn't process them"
    
    return results, f"Successfully detected and processed {len(results)} face(s)"


def draw_boxes_with_emotions(image, results):
    """Draw bounding boxes with emotion labels"""
    image_copy = image.copy()
    
    for result in results:
        x, y, w, h = result['face_region']
        emotion = result['emotion']
        confidence = result['confidence']
        
        color_hex = EMOTION_COLORS.get(emotion, "#0000FF")
        color_tuple = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = tuple(reversed(color_tuple))
        
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), color_bgr, 3)
        label = f"{emotion.upper()} ({confidence*100:.1f}%)"
        cv2.putText(image_copy, label, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)
    
    return image_copy


@st.cache_data
def load_test_data(target_size=(224, 224)):
    """Load test data from FER2013 dataset and resize to target_size"""
    csv_path = 'fer2013.csv'
    
    if not os.path.exists(csv_path):
        return None, None, "Dataset file not found"
    
    try:
        df = pd.read_csv(csv_path)
        test_df = df[df['Usage'] == 'PrivateTest']
        
        X_test = []
        y_test = []
        
        for idx, row in test_df.iterrows():
            pixels = np.array(row['pixels'].split(), dtype='float32')
            img = pixels.reshape(48, 48)
            # Resize to model's expected size
            img_resized = cv2.resize(img, tuple(target_size))
            # Convert to RGB
            img_rgb = cv2.cvtColor(img_resized.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            X_test.append(img_rgb)
            y_test.append(row['emotion'])
        
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test)
        
        return X_test, y_test, f"Loaded {len(X_test)} test samples"
    except Exception as e:
        return None, None, f"Error loading dataset: {e}"


def evaluate_model_and_generate_confusion_matrix(model, X_test, y_test, emotion_list, X_train=None, y_train=None, X_val=None, y_val=None):
    """Evaluate model and generate confusion matrix with batch processing to save memory"""
    batch_size = 32
    
    def get_predictions(X_data, y_data, data_name):
        """Helper function to get predictions for a dataset"""
        y_pred = []
        with st.spinner(f'Evaluating model on {data_name} dataset...'):
            num_batches = (len(X_data) + batch_size - 1) // batch_size
            progress_bar = st.progress(0)
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(X_data))
                
                # Get batch
                X_batch = X_data[start_idx:end_idx].copy()
                
                # Preprocess batch
                X_batch_preprocessed = tf.keras.applications.resnet50.preprocess_input(X_batch)
                
                # Get predictions for batch
                batch_predictions = model.predict(X_batch_preprocessed, verbose=0)
                y_pred.extend(np.argmax(batch_predictions, axis=1))
                
                # Update progress
                progress_bar.progress((i + 1) / num_batches)
        
        return np.array(y_pred)
    
    # Get predictions for all datasets
    y_pred_test = get_predictions(X_test, y_test, 'test')
    
    results = {
        'test': {
            'cm': confusion_matrix(y_test, y_pred_test),
            'report': classification_report(y_test, y_pred_test, target_names=emotion_list, output_dict=True),
            'y_pred': y_pred_test
        }
    }
    
    # Evaluate train data if provided
    if X_train is not None and y_train is not None:
        y_pred_train = get_predictions(X_train, y_train, 'train')
        results['train'] = {
            'cm': confusion_matrix(y_train, y_pred_train),
            'report': classification_report(y_train, y_pred_train, target_names=emotion_list, output_dict=True),
            'y_pred': y_pred_train
        }
    
    # Evaluate validation data if provided
    if X_val is not None and y_val is not None:
        y_pred_val = get_predictions(X_val, y_val, 'validation')
        results['val'] = {
            'cm': confusion_matrix(y_val, y_pred_val),
            'report': classification_report(y_val, y_pred_val, target_names=emotion_list, output_dict=True),
            'y_pred': y_pred_val
        }
    
    # Return for backward compatibility
    return results['test']['cm'], results['test']['report'], results['test']['y_pred'], results


def plot_confusion_matrix(cm, emotion_list):
    """Plot confusion matrix with seaborn"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_list, yticklabels=emotion_list,
                ax=ax, cbar_kws={'label': 'Number of Predictions'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Emotion Recognition Model', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_roc_curve(model, X_test, y_test, emotion_list):
    """Plot ROC curves for multi-class emotion recognition using One-vs-Rest approach"""
    # Preprocess test data
    X_test_preprocessed = tf.keras.applications.resnet50.preprocess_input(X_test.copy())
    
    # Get probability predictions
    y_pred_proba = model.predict(X_test_preprocessed, verbose=0)
    
    # Binarize y_test for multi-class ROC
    n_classes = len(emotion_list)
    y_test_binarized = label_binarize(y_test, classes=list(range(n_classes)))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Calculate ROC curve and AUC for each class
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    for i, emotion in enumerate(emotion_list[:4]):  # Show first 4 classes
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
        # Plot ROC curve
        axes[i].plot(fpr_dict[i], tpr_dict[i], color='darkorange', lw=2.5,
                    label=f'ROC curve (AUC = {roc_auc_dict[i]:.3f})')
        axes[i].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        axes[i].set_xlim([0.0, 1.0])
        axes[i].set_ylim([0.0, 1.05])
        axes[i].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
        axes[i].set_title(f'ROC Curve - {emotion.upper()}', fontsize=12, fontweight='bold')
        axes[i].legend(loc="lower right", fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    # If there are more than 4 classes, show remaining classes in remaining subplots
    for i in range(4, min(7, n_classes)):
        emotion = emotion_list[i]
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
        
        axes[i % 4].plot(fpr_dict[i], tpr_dict[i], color='green', lw=2.5,
                        label=f'ROC curve (AUC = {roc_auc_dict[i]:.3f})')
    
    plt.tight_layout()
    return fig, roc_auc_dict


# Main App
def main():
    # Header
    st.markdown("""
    <div class="header">
        <h1>ResNet50 Quantum Emotion Recognition</h1>
        <p>Emotion Recognition with Advanced Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Selection
    st.markdown("### Select Model")
    model_choice = st.selectbox(
        "Choose which model to use for emotion recognition:",
        [
            "Model 1 (70% Training Split)",
            "Model 2 (80% Training Split)"
        ],
        help="Model 1: Trained with 70% training, 15% validation, 15% test split (Better generalization)\nModel 2: Trained with 80% training, 10% validation, 10% test split (More training data)"
    )
    
    # Display model info box
    if model_choice == "Model 1 (70% Training Split)":
        st.info("**Model 1**: Trained with 70% training data, 15% validation, 15% test split. Uses final_model70.keras - Recommended for better generalization")
    else:
        st.info("**Model 2**: Trained with 80% training data, 10% validation, 10% test split. Uses final_model80.keras - More training data exposure")
    
    # Load model and metadata
    model, model_name = load_model(model_choice)
    metadata = load_metadata(model_choice)
    emotion_list = metadata.get('emotions', [])
    
    # Model Info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="accuracy-box">
            Training Accuracy<br><span style="font-size: 24px;">{metadata.get('training_accuracy', 0)*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="accuracy-box">
            Test Accuracy<br><span style="font-size: 24px;">{metadata.get('test_accuracy', 0)*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="accuracy-box">
            Validation Accuracy<br><span style="font-size: 24px;">{metadata.get('val_accuracy', 0)*100:.2f}%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="accuracy-box">
            Data Split<br><span style="font-size: 16px;">{metadata.get('split', 'N/A')}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if model is None or not emotion_list:
        st.error("Error: Could not load model or emotion list")
        return

    # Determine model input size (height, width) for preprocessing
    model_input_size = (224, 224)
    try:
        shp = getattr(model, 'input_shape', None)
        if shp is not None and len(shp) == 4 and shp[1] is not None and shp[2] is not None:
            model_input_size = (int(shp[1]), int(shp[2]))
    except Exception:
        pass
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Analyze", "Model Info", "Confusion Matrix", "About"])
    
    with tab1:
        st.markdown("### Upload an Image to Detect Emotions")
        
        uploaded_file = st.file_uploader(
            "Choose an image (JPG, PNG, BMP)",
            type=["jpg", "jpeg", "png", "bmp"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Emotion", key="analyze"):
                with st.spinner("Detecting faces and analyzing emotions..."):
                    results, status_msg = detect_faces_and_emotions(
                        image_array, model, emotion_list, target_size=model_input_size
                    )
                    
                    st.info(status_msg)
                    
                    if results:
                        annotated_image = draw_boxes_with_emotions(image_array, results)
                        col_img, col_pred = st.columns([1.15, 0.85])

                        with col_img:
                            st.image(annotated_image, caption="Detection Results", use_container_width=True)

                        with col_pred:
                            st.markdown("### Results")
                            for idx, result in enumerate(results):
                                with st.container():
                                    st.markdown(f"**Face {idx + 1}**")
                                    emotion = result['emotion']
                                    confidence = result['confidence']
                                    emotion_label = EMOTION_EMOJI.get(emotion, emotion)

                                    st.markdown(f"""
                                    <div class="emotion-card">
                                        <div class="emotion-name">{emotion_label}</div>
                                        <div class="confidence">{confidence*100:.2f}%</div>
                                        <p>{EMOTION_DESCRIPTIONS.get(emotion, '')}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                                    st.markdown(f"**Top Prediction:** {emotion.upper()} - {confidence*100:.2f}%")

                                    pred_dict = {emotion_list[i]: result['all_predictions'][i]*100 
                                               for i in range(len(emotion_list))}
                                    st.bar_chart(pred_dict)
    
    with tab2:
        st.markdown("### Model Architecture")
        st.markdown(f"""
        <div class="model-info">
            <strong>Model Type:</strong> {metadata.get('model_type', 'N/A')}<br>
            <strong>Framework:</strong> {metadata.get('framework', 'N/A')}<br>
            <strong>Input Size:</strong> 224x224 RGB<br>
            <strong>Output Classes:</strong> 7 emotions<br>
            <strong>Architecture:</strong> {metadata.get('architecture', 'N/A')}<br>
            <strong>Data Split:</strong> {metadata.get('split', 'N/A')}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Performance Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Accuracy", f"{metadata.get('training_accuracy', 0)*100:.2f}%")
        with col2:
            st.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0)*100:.2f}%")
        with col3:
            st.metric("Validation Accuracy", f"{metadata.get('val_accuracy', 0)*100:.2f}%")
        
        st.markdown("### Emotion Classes")
        emotion_cols = st.columns(7)
        for i, emotion in enumerate(emotion_list):
            with emotion_cols[i]:
                st.markdown(f"""
                <div style="text-align:center; padding:10px;">
                    <div style="font-size:0.9em; font-weight:bold;">{EMOTION_EMOJI.get(emotion, emotion).upper()}</div>
                </div>
                """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("### Model Evaluation on Test Dataset")
        st.info("This section evaluates the model on the FER2013 test dataset and displays a confusion matrix.")
        
        if st.button("Generate Confusion Matrix", key="generate_cm"):
            # Load test data sized to model input
            X_test, y_test, status = load_test_data(target_size=model_input_size)
            
            if X_test is None:
                st.error(status)
            else:
                st.success(status)
                
                # Evaluate model
                cm, report, y_pred, all_results = evaluate_model_and_generate_confusion_matrix(
                    model, X_test, y_test, emotion_list
                )
                
                # Display confusion matrix
                st.markdown("#### Confusion Matrix")
                fig = plot_confusion_matrix(cm, emotion_list)
                st.pyplot(fig)
                
                # Display overall accuracy comparison (test, train, val if available)
                st.markdown("### Overall Accuracy Comparison")
                col1, col2, col3 = st.columns(3)
                
                test_accuracy = np.trace(cm) / np.sum(cm)
                with col1:
                    st.metric("Test Accuracy", f"{test_accuracy*100:.2f}%")
                
                # Display metrics for other splits if available (placeholder for future implementation)
                with col2:
                    st.metric("Train Accuracy", "N/A")
                
                with col3:
                    st.metric("Validation Accuracy", "N/A")
                
                # Display overall accuracy description
                st.markdown("#### 📊 What is Overall Accuracy?")
                st.info(
                    f"**Overall Accuracy ({test_accuracy*100:.2f}%)** represents the percentage of emotions "
                    f"that the model correctly predicted across all test samples in the FER2013 dataset. "
                    f"It is calculated as: (Total Correct Predictions) / (Total Predictions) × 100%\n\n"
                    f"**Interpretation:** Out of all {np.sum(cm)} test images, the model successfully identified "
                    f"the correct emotion for {int(np.trace(cm))} images. An accuracy of {test_accuracy*100:.2f}% indicates "
                    f"{'excellent' if test_accuracy >= 0.90 else 'very good' if test_accuracy >= 0.80 else 'good' if test_accuracy >= 0.70 else 'moderate'} "
                    f"model performance in emotion recognition across diverse facial expressions."
                )
                
                # Display classification report
                st.markdown("#### Classification Report (Test Set)")
                
                report_df = pd.DataFrame(report).transpose()
                report_df = report_df[report_df.index.isin(emotion_list)]
                
                # Format percentages
                for col in ['precision', 'recall', 'f1-score']:
                    if col in report_df.columns:
                        report_df[col] = report_df[col].apply(lambda x: f"{x*100:.2f}%")
                
                st.dataframe(report_df, use_container_width=True)
                
                # Per-class accuracy with precision, recall, f1-score
                st.markdown("#### Per-Class Metrics (Test Set)")
                class_accuracy = {}
                class_precision = {}
                class_recall = {}
                class_f1 = {}
                
                for i, emotion in enumerate(emotion_list):
                    if cm[i].sum() > 0:
                        class_acc = cm[i, i] / cm[i].sum()
                        class_accuracy[emotion] = class_acc * 100
                        # Get precision, recall, f1 from report
                        if emotion in report:
                            class_precision[emotion] = report[emotion].get('precision', 0) * 100
                            class_recall[emotion] = report[emotion].get('recall', 0) * 100
                            class_f1[emotion] = report[emotion].get('f1-score', 0) * 100
                
                # Display metrics in a detailed table
                metrics_data = []
                for emotion in emotion_list:
                    metrics_data.append({
                        'Emotion': emotion.upper(),
                        'Accuracy': f"{class_accuracy.get(emotion, 0):.2f}%",
                        'Precision': f"{class_precision.get(emotion, 0):.2f}%",
                        'Recall': f"{class_recall.get(emotion, 0):.2f}%",
                        'F1-Score': f"{class_f1.get(emotion, 0):.2f}%"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # ROC Curve Analysis
                st.markdown("#### 📈 ROC Curves (One-vs-Rest Analysis)")
                st.info(
                    "ROC (Receiver Operating Characteristic) curves show the trade-off between True Positive Rate and "
                    "False Positive Rate for each emotion class. Higher AUC (Area Under the Curve) values indicate better "
                    "model performance. AUC ranges from 0 to 1, where 1.0 is perfect and 0.5 is random."
                )
                
                with st.spinner("Generating ROC curves..."):
                    roc_fig, roc_auc_dict = plot_roc_curve(model, X_test, y_test, emotion_list)
                    st.pyplot(roc_fig)
                
                # Display AUC scores for each emotion
                st.markdown("#### AUC Scores by Emotion Class")
                auc_data = []
                for i, emotion in enumerate(emotion_list):
                    if i in roc_auc_dict:
                        auc_data.append({
                            'Emotion': emotion.upper(),
                            'AUC Score': f"{roc_auc_dict[i]:.4f}",
                            'Performance': 'Excellent' if roc_auc_dict[i] >= 0.9 else 'Very Good' if roc_auc_dict[i] >= 0.8 else 'Good' if roc_auc_dict[i] >= 0.7 else 'Fair'
                        })
                
                if auc_data:
                    auc_df = pd.DataFrame(auc_data)
                    st.dataframe(auc_df, use_container_width=True)
                
                # Visualization
                st.bar_chart(class_accuracy)
    
    with tab4:
        st.markdown("### About This Model")
        st.info(
            "This emotion recognition system uses ResNet50 transfer learning with quantum-inspired "
            "optimization techniques. The model is trained on the FER2013 dataset containing 35,887 "
            "images of facial expressions across 7 emotion categories."
        )
        
        st.markdown("### Key Features")
        st.markdown("""
        - ResNet50 architecture with ImageNet pre-training
        - Quantum-inspired attention mechanisms
        - Real-time face detection and emotion recognition
        - Supports multiple faces in a single image
        - Confidence scores for each emotion
        - Beautiful visualization with emotion labels
        - Multiple model options with different data splits
        """)
        
        st.markdown("### Dataset Information")
        st.markdown("""
        - **Name:** FER2013 (Facial Expression Recognition 2013)
        - **Total Samples:** 35,887 images
        - **Image Size:** 48x48 pixels (converted to 224x224 for ResNet50)
        - **Emotion Classes:** 7 (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
        - **Model 1 Split:** 70% Train / 15% Validation / 15% Test (Better generalization)
        - **Model 2 Split:** 80% Train / 10% Validation / 10% Test (More training data)
        """)


if __name__ == "__main__":
    main()
