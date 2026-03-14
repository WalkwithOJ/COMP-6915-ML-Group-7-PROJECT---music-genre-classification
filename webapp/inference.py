"""
Inference pipeline for the Flask web app.
Takes an uploaded audio file, processes it, and returns genre predictions.
"""
import os

import numpy as np
import librosa
import joblib

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.mel_spectrogram import generate_mel_spectrogram
from src.noise_robustness import extract_features_from_audio


class GenrePredictor:
    """Loads trained models and provides genre prediction for uploaded audio."""

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.label_encoder = None
        self._load_models()

    def _load_models(self):
        """Load all available trained models."""
        # Label encoder
        le_path = os.path.join(config.MODELS_DIR, "label_encoder.joblib")
        if os.path.isfile(le_path):
            self.label_encoder = joblib.load(le_path)

        # Scaler
        scaler_path = os.path.join(config.MODELS_DIR, "scaler.joblib")
        if os.path.isfile(scaler_path):
            self.scaler = joblib.load(scaler_path)

        # SVM
        svm_path = os.path.join(config.MODELS_DIR, "svm_best.joblib")
        if os.path.isfile(svm_path):
            self.models["SVM"] = joblib.load(svm_path)

        # Random Forest
        rf_path = os.path.join(config.MODELS_DIR, "rf_best.joblib")
        if os.path.isfile(rf_path):
            self.models["Random Forest"] = joblib.load(rf_path)

        # CNN
        cnn_path = os.path.join(config.MODELS_DIR, "cnn_best.keras")
        if os.path.isfile(cnn_path):
            import tensorflow as tf
            self.models["CNN"] = tf.keras.models.load_model(cnn_path)

    def predict(self, audio_path: str, model_name: str = "CNN") -> dict:
        """
        Predict genre for an uploaded audio file.

        Returns dict with:
            - genre: predicted genre name
            - confidence: prediction confidence (0-1)
            - probabilities: dict of genre -> probability
            - model_used: which model was used
        """
        y, sr = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)

        # Trim to 30 seconds if longer
        max_samples = config.TRACK_DURATION * sr
        if len(y) > max_samples:
            y = y[:max_samples]

        if model_name == "CNN" and "CNN" in self.models:
            return self._predict_cnn(y, sr)
        elif model_name in self.models:
            return self._predict_traditional(y, sr, model_name)
        else:
            # Fall back to any available model
            if self.models:
                fallback = list(self.models.keys())[0]
                if fallback == "CNN":
                    return self._predict_cnn(y, sr)
                return self._predict_traditional(y, sr, fallback)
            return {"error": "No trained models found"}

    def _predict_traditional(self, y: np.ndarray, sr: int, model_name: str) -> dict:
        """Predict using SVM or Random Forest on extracted features."""
        features = extract_features_from_audio(y, sr)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        model = self.models[model_name]
        pred_idx = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]

        genre = self.label_encoder.inverse_transform([pred_idx])[0]
        probabilities = {
            self.label_encoder.inverse_transform([i])[0]: float(p)
            for i, p in enumerate(proba)
        }

        return {
            "genre": genre,
            "confidence": float(max(proba)),
            "probabilities": probabilities,
            "model_used": model_name,
        }

    def _predict_cnn(self, y: np.ndarray, sr: int) -> dict:
        """Predict using CNN with majority voting over 3-sec segments."""
        samples_per_seg = config.SEGMENT_DURATION * sr
        num_segments = len(y) // samples_per_seg

        segment_probs = []
        for i in range(num_segments):
            start = i * samples_per_seg
            end = start + samples_per_seg
            segment = y[start:end]

            mel = generate_mel_spectrogram(segment, sr)
            if mel.shape[1] < config.MEL_SPEC_WIDTH:
                mel = np.pad(mel, ((0, 0), (0, config.MEL_SPEC_WIDTH - mel.shape[1])))
            elif mel.shape[1] > config.MEL_SPEC_WIDTH:
                mel = mel[:, :config.MEL_SPEC_WIDTH]

            mel_input = mel[np.newaxis, ..., np.newaxis]
            proba = self.models["CNN"].predict(mel_input, verbose=0)[0]
            segment_probs.append(proba)

        # Average probabilities across segments
        avg_proba = np.mean(segment_probs, axis=0)
        pred_idx = np.argmax(avg_proba)
        genre = config.GENRES[pred_idx]

        probabilities = {
            config.GENRES[i]: float(p) for i, p in enumerate(avg_proba)
        }

        return {
            "genre": genre,
            "confidence": float(avg_proba[pred_idx]),
            "probabilities": probabilities,
            "model_used": "CNN",
        }
