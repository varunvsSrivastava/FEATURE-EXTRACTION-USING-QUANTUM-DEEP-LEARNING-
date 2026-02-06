#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate both saved models (final_model70.keras, final_model80.keras) on FER2013
- Computes accuracy on all three splits: Training, PublicTest (Validation), PrivateTest (Test)
- Reports per-class (7 emotions) accuracy for each split
- Saves JSON summaries to disk and prints a concise table to console

Usage (Windows PowerShell):
  python evaluate_models.py --csv fer2013.csv --batch-size 64

Optional:
  python evaluate_models.py --csv fer2013.csv --model 70
  python evaluate_models.py --csv fer2013.csv --model 80
  python evaluate_models.py --max-samples 2000  # quick sanity run
"""
import os
import sys
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

# -----------------------------------------------------------------------------
# Custom layers used in the .keras models (from app) so loading works
# -----------------------------------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class QuantumQubitCircuit(tf.keras.layers.Layer):
    def __init__(self, num_qubits=6, num_layers=3, **kwargs):
        super(QuantumQubitCircuit, self).__init__(**kwargs)
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        self.rotation_params = self.add_weight(
            name="rotation_params", shape=(self.num_layers, self.num_qubits, 2),
            initializer="glorot_uniform", trainable=True,
        )
        self.entangle_params = self.add_weight(
            name="entangle_params", shape=(self.num_layers, self.num_qubits),
            initializer="glorot_uniform", trainable=True,
        )
        self.measurement_angles = self.add_weight(
            name="measurement_angles", shape=(self.num_qubits,),
            initializer="zeros", trainable=True,
        )
        super(QuantumQubitCircuit, self).build(input_shape)

    def _ensure_tensor(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def call(self, x):
        x = self._ensure_tensor(x)
        if x.shape.rank is None or x.shape.rank > 2:
            x = tf.reshape(x, [tf.shape(x)[0], -1])
        batch_size = tf.shape(x)[0]
        feature_dim = tf.shape(x)[-1]
        dtype = x.dtype
        feat_static = x.shape[-1]
        if feat_static is not None and feat_static > self.num_qubits:
            x_projected = x[:, : self.num_qubits]
        elif feat_static is not None and feat_static < self.num_qubits:
            padding = tf.zeros([batch_size, self.num_qubits - feat_static], dtype=dtype)
            x_projected = tf.concat([x, padding], axis=1)
        else:
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
            ry_c = tf.cos(ry_angles / 2)
            ry_s = tf.sin(ry_angles / 2)
            real_old = qubit_real
            imag_old = qubit_imag
            qubit_real = real_old * ry_c - imag_old * ry_s
            qubit_imag = real_old * ry_s + imag_old * ry_c
            rz_c = tf.cos(rz_angles / 2)
            rz_s = tf.sin(rz_angles / 2)
            new_real = qubit_real * rz_c - qubit_imag * rz_s
            new_imag = qubit_real * rz_s + qubit_imag * rz_c
            qubit_real = new_real
            qubit_imag = new_imag
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

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)):
            input_shape = input_shape[0]
        input_dim = 512 if (input_shape is None or input_shape[-1] is None) else int(input_shape[-1])
        self.encoding_matrix = self.add_weight(
            name="encoding_matrix", shape=(input_dim, self.latent_dim),
            initializer="glorot_uniform", trainable=True,
        )
        self.phase_bias = self.add_weight(
            name="phase_bias", shape=(self.latent_dim,),
            initializer="zeros", trainable=True,
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
            name="attention_params", shape=(feat_dim,),
            initializer="glorot_uniform", trainable=True,
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

# -----------------------------------------------------------------------------
# Evaluation utilities
# -----------------------------------------------------------------------------
FER_EMOTION_ORDER = [
    "angry",    # 0
    "disgust",  # 1
    "fear",     # 2
    "happy",    # 3
    "sad",      # 4
    "surprise", # 5
    "neutral"   # 6
]

# App display order used elsewhere in this repo (if you prefer):
APP_EMOTION_ORDER = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def preprocess_batch_gray48_to_resnet_rgb(images_48x48: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Convert a batch of 48x48 grayscale (uint8) to target_size RGB and apply ResNet50 preprocessing."""
    batch = []
    h, w = int(target_size[0]), int(target_size[1])
    for img in images_48x48:
        img_resized = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB).astype(np.float32)
        batch.append(img_rgb)
    x = np.stack(batch, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)
    return x


def load_split_from_csv(csv_path: str, usage: str, max_samples: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load a split by Usage from FER2013 CSV: 'Training' | 'PublicTest' | 'PrivateTest'"""
    df = pd.read_csv(csv_path)
    df = df[df['Usage'] == usage]
    if max_samples:
        df = df.iloc[:max_samples]
    # Parse pixels -> (n, 48, 48) uint8
    pixels = df['pixels'].tolist()
    images = []
    for p in pixels:
        arr = np.fromstring(p, sep=' ', dtype=np.uint8)
        images.append(arr.reshape(48, 48))
    X = np.stack(images, axis=0)
    y = df['emotion'].values.astype(np.int32)
    return X, y


def evaluate_split(model: tf.keras.Model, X48: np.ndarray, y_true: np.ndarray, batch_size: int = 64, target_size: Tuple[int, int] = (224, 224)) -> Dict:
    """Run batched inference and compute overall + per-class metrics."""
    n = len(X48)
    num_classes = 7
    correct = 0
    y_pred_all: List[int] = []

    # Per-class counters
    class_totals = np.zeros(num_classes, dtype=np.int64)
    class_correct = np.zeros(num_classes, dtype=np.int64)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = preprocess_batch_gray48_to_resnet_rgb(X48[start:end], target_size)
        preds = model.predict(X_batch, verbose=0)
        y_pred = np.argmax(preds, axis=1)

        y_pred_all.extend(y_pred.tolist())
        batch_y = y_true[start:end]
        correct += int(np.sum(y_pred == batch_y))

        # Update per-class tallies
        for c in range(num_classes):
            mask = (batch_y == c)
            class_totals[c] += int(mask.sum())
            if class_totals[c] > 0:
                class_correct[c] += int(np.sum(y_pred[mask] == c))

    overall_acc = correct / float(n) if n > 0 else 0.0
    per_class_acc = {}
    for c in range(num_classes):
        denom = class_totals[c]
        per_class_acc[FER_EMOTION_ORDER[c]] = (class_correct[c] / float(denom)) if denom > 0 else 0.0

    return {
        'samples': n,
        'overall_accuracy': overall_acc,
        'per_class_accuracy': per_class_acc,
    }


def load_model_file(model_path: Path) -> tf.keras.Model:
    """Load .keras model with necessary custom_objects, compile=False."""
    custom_objects = {
        'QuantumQubitCircuit': QuantumQubitCircuit,
        'QuantumEncodingLayer': QuantumEncodingLayer,
        'QuantumAttentionLayer': QuantumAttentionLayer,
    }
    # Prefer keras loader, then tf.keras fallback
    try:
        import keras
        return keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False, safe_mode=False)
    except Exception as e_keras:
        try:
            return tf.keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)
        except Exception as e_tf:
            raise RuntimeError(f"Failed to load model {model_path}: Keras({e_keras}) / tf.keras({e_tf})")


def evaluate_models(csv_path: str, batch_size: int = 64, which_model: str = 'both', max_samples: int = None) -> Dict[str, Dict]:
    base_dir = Path(__file__).resolve().parent
    results: Dict[str, Dict] = {}

    targets = []
    if which_model in ('70', 'both'):
        targets.append(('model70', base_dir / 'final_model70.keras'))
    if which_model in ('80', 'both'):
        targets.append(('model80', base_dir / 'final_model80.keras'))

    # Load splits once to avoid re-reading CSV repeatedly
    splits = {
        'Training': load_split_from_csv(csv_path, 'Training', max_samples=max_samples),
        'PublicTest': load_split_from_csv(csv_path, 'PublicTest', max_samples=max_samples),
        'PrivateTest': load_split_from_csv(csv_path, 'PrivateTest', max_samples=max_samples),
    }

    for tag, path in targets:
        if not path.exists():
            print(f"[WARN] Model file not found: {path}")
            continue
        print(f"\n[LOAD] Loading {tag} from {path.name}...")
        model = load_model_file(path)
        # Determine input size from model
        input_shape = getattr(model, 'input_shape', None)
        target_size = (224, 224)
        if input_shape and len(input_shape) == 4 and input_shape[1] and input_shape[2]:
            target_size = (int(input_shape[1]), int(input_shape[2]))
        print(f"[OK] Model loaded. Input shape: {input_shape} -> using target_size={target_size}")

        tag_result = {}
        for usage, (X, y) in splits.items():
            print(f"[EVAL] {tag} on {usage} split (n={len(X)})...")
            start_t = time.time()
            split_res = evaluate_split(model, X, y, batch_size=batch_size, target_size=target_size)
            elapsed = time.time() - start_t
            tag_result[usage] = split_res
            print(f"  - Overall Acc: {split_res['overall_accuracy']*100:.2f}%  | time {elapsed:.1f}s")

        results[tag] = tag_result
    return results


def save_json(path: Path, data: Dict):
    path.write_text(json.dumps(data, indent=2))
    print(f"[SAVE] Wrote {path}")


def format_console_summary(res: Dict[str, Dict]):
    def fmt_pct(x):
        return f"{x*100:.2f}%"
    print("\n================ Evaluation Summary ================")
    for tag, splits in res.items():
        print(f"\nModel: {tag}")
        
        # Display train, validation, and test accuracy
        train_acc = splits.get('Training', {}).get('overall_accuracy', 0.0)
        val_acc = splits.get('PublicTest', {}).get('overall_accuracy', 0.0)
        test_acc = splits.get('PrivateTest', {}).get('overall_accuracy', 0.0)
        
        print(f"  Train Accuracy:      {fmt_pct(train_acc)}")
        print(f"  Validation Accuracy: {fmt_pct(val_acc)}")
        print(f"  Test Accuracy:       {fmt_pct(test_acc)}")
        print()
        
        # Display per-class accuracy across train/validation/test
        train_pcs = splits.get('Training', {}).get('per_class_accuracy', {})
        val_pcs = splits.get('PublicTest', {}).get('per_class_accuracy', {})
        test_pcs = splits.get('PrivateTest', {}).get('per_class_accuracy', {})
        
        if train_pcs:
            print("  Per-Class Accuracy (Train | Validation | Test):")
            for emotion in train_pcs.keys():
                train_e = train_pcs.get(emotion, 0.0)
                val_e = val_pcs.get(emotion, 0.0)
                test_e = test_pcs.get(emotion, 0.0)
                print(f"    {emotion:12s}: {fmt_pct(train_e):>8s} | {fmt_pct(val_e):>8s} | {fmt_pct(test_e):>8s}")
        print()
        
        for usage, metrics in splits.items():
            print(f"  {usage}: overall {fmt_pct(metrics['overall_accuracy'])}")
            pcs = metrics['per_class_accuracy']
            pcs_str = ", ".join([f"{k}: {fmt_pct(v)}" for k, v in pcs.items()])
            print(f"    per-class: {pcs_str}")
    print("===================================================\n")


def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate saved models on FER2013 across splits with per-class metrics")
    p.add_argument('--csv', type=str, default='fer2013.csv', help='Path to FER2013 CSV')
    p.add_argument('--batch-size', type=int, default=64, help='Batch size for inference')
    p.add_argument('--model', type=str, choices=['70','80','both'], default='both', help='Which model to evaluate')
    p.add_argument('--max-samples', type=int, default=None, help='Limit samples per split for quick runs')
    return p.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}")
        sys.exit(1)

    res = evaluate_models(args.csv, batch_size=args.batch_size, which_model=args.model, max_samples=args.max_samples)

    # Save JSON outputs per model
    base_dir = Path(__file__).resolve().parent
    for tag, splits in res.items():
        out_path = base_dir / f"evaluation_results_{'70' if tag=='model70' else '80'}.json"
        save_json(out_path, splits)

    # Print summary to console
    format_console_summary(res)


if __name__ == '__main__':
    main()
