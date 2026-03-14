"""
Phase 4C: Noise robustness experiment (Research Question 3).
Tests how synthetic noise at different SNR levels degrades model accuracy.
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
import librosa
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import get_logger, ensure_dirs, set_seed
from src.mel_spectrogram import generate_mel_spectrogram

logger = get_logger(__name__)


def add_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """Add white Gaussian noise to a signal at a specified SNR level."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def extract_features_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract the same set of features as in features_30_sec.csv from audio.
    Returns a 1D feature vector matching the CSV columns.
    """
    features = []

    # Chroma STFT
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend([chroma.mean(), chroma.var()])

    # RMS energy
    rms = librosa.feature.rms(y=y)
    features.extend([rms.mean(), rms.var()])

    # Spectral centroid
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.extend([cent.mean(), cent.var()])

    # Spectral bandwidth
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.extend([bw.mean(), bw.var()])

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.extend([rolloff.mean(), rolloff.var()])

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features.extend([zcr.mean(), zcr.var()])

    # Harmony and perceptual features
    harmony = librosa.effects.harmonic(y)
    percept = librosa.effects.percussive(y)
    features.extend([harmony.mean(), harmony.var()])
    features.extend([percept.mean(), percept.var()])

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if hasattr(tempo, '__len__'):
        tempo = tempo[0] if len(tempo) > 0 else 0.0
    features.append(float(tempo))

    # MFCCs (20 coefficients, mean and variance each)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=config.N_MFCC)
    for i in range(config.N_MFCC):
        features.extend([mfccs[i].mean(), mfccs[i].var()])

    return np.array(features, dtype=np.float32)


def evaluate_at_snr(snr_db: float, test_tracks: list, models: dict, scaler):
    """
    Add noise at given SNR to all test tracks, extract features / spectrograms,
    and evaluate each model. Returns accuracy dict.
    """
    set_seed(config.RANDOM_SEED)
    results = {}

    # Collect noisy features and labels for traditional ML
    X_noisy, y_true = [], []
    mel_segments, mel_labels = [], []

    for track_id in test_tracks:
        genre = track_id.rsplit(".", 1)[0]
        genre_idx = config.GENRES.index(genre)
        wav_path = os.path.join(config.AUDIO_DIR, genre, f"{track_id}.wav")

        if not os.path.isfile(wav_path):
            continue

        y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
        y_noisy = add_noise(y, snr_db)

        # Extract tabular features from noisy 30-sec audio
        feats = extract_features_from_audio(y_noisy, sr)
        X_noisy.append(feats)
        y_true.append(genre_idx)

        # Generate mel-spectrograms from noisy 3-sec segments
        samples_per_seg = config.SEGMENT_DURATION * sr
        for seg_idx in range(config.SEGMENTS_PER_TRACK):
            start = seg_idx * samples_per_seg
            end = start + samples_per_seg
            if end > len(y_noisy):
                break
            segment = y_noisy[start:end]
            mel = generate_mel_spectrogram(segment, sr)
            if mel.shape[1] < config.MEL_SPEC_WIDTH:
                mel = np.pad(mel, ((0, 0), (0, config.MEL_SPEC_WIDTH - mel.shape[1])))
            elif mel.shape[1] > config.MEL_SPEC_WIDTH:
                mel = mel[:, :config.MEL_SPEC_WIDTH]
            mel_segments.append(mel)
            mel_labels.append(genre_idx)

    X_noisy = np.array(X_noisy, dtype=np.float32)
    y_true = np.array(y_true)

    # Scale features using the trained scaler
    X_scaled = scaler.transform(X_noisy)

    # Evaluate SVM
    if "svm" in models:
        pred = models["svm"].predict(X_scaled)
        results["SVM"] = float(np.mean(pred == y_true))

    # Evaluate Random Forest
    if "rf" in models:
        pred = models["rf"].predict(X_scaled)
        results["Random Forest"] = float(np.mean(pred == y_true))

    # Evaluate CNN (segment-level, then majority vote per track)
    if "cnn" in models and mel_segments:
        mel_array = np.array(mel_segments)[..., np.newaxis]
        cnn_preds = models["cnn"].predict(mel_array, verbose=0)
        cnn_preds = np.argmax(cnn_preds, axis=1)

        # Majority vote per track
        seg_idx = 0
        track_preds = []
        for track_id in test_tracks:
            genre = track_id.rsplit(".", 1)[0]
            wav_path = os.path.join(config.AUDIO_DIR, genre, f"{track_id}.wav")
            if not os.path.isfile(wav_path):
                continue

            seg_preds = cnn_preds[seg_idx:seg_idx + config.SEGMENTS_PER_TRACK]
            seg_idx += config.SEGMENTS_PER_TRACK
            if len(seg_preds) > 0:
                vote = max(set(seg_preds.tolist()), key=seg_preds.tolist().count)
                track_preds.append(vote)

        track_preds = np.array(track_preds)
        # Align lengths
        min_len = min(len(track_preds), len(y_true))
        results["CNN"] = float(np.mean(track_preds[:min_len] == y_true[:min_len]))

    return results


def run_noise_robustness():
    """Execute the full noise robustness experiment."""
    logger.info("=" * 60)
    logger.info("PHASE 4C: Noise Robustness Experiment")
    logger.info("=" * 60)

    with open(config.SPLITS_FILE, "r") as f:
        splits = json.load(f)

    test_tracks = splits["test"]

    # Load models
    models = {}
    svm_path = os.path.join(config.MODELS_DIR, "svm_best.joblib")
    if os.path.isfile(svm_path):
        models["svm"] = joblib.load(svm_path)

    rf_path = os.path.join(config.MODELS_DIR, "rf_best.joblib")
    if os.path.isfile(rf_path):
        models["rf"] = joblib.load(rf_path)

    cnn_path = os.path.join(config.MODELS_DIR, "cnn_best.keras")
    if os.path.isfile(cnn_path):
        import tensorflow as tf
        models["cnn"] = tf.keras.models.load_model(cnn_path)

    scaler = joblib.load(os.path.join(config.MODELS_DIR, "scaler.joblib"))

    # Run experiment at each SNR level
    all_results = {}
    for snr_db in config.NOISE_SNR_LEVELS:
        logger.info("Evaluating at SNR = %d dB ...", snr_db)
        results = evaluate_at_snr(snr_db, test_tracks, models, scaler)
        all_results[snr_db] = results
        for model_name, acc in results.items():
            logger.info("  %s: %.4f", model_name, acc)

    # Plot results
    ensure_dirs(config.RESULTS_DIR)
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = set()
    for r in all_results.values():
        model_names.update(r.keys())

    colors = {"SVM": "blue", "Random Forest": "green", "CNN": "red"}
    for model_name in sorted(model_names):
        snrs = sorted(all_results.keys())
        accs = [all_results[snr].get(model_name, 0) for snr in snrs]
        ax.plot(snrs, accs, "o-", label=model_name,
                color=colors.get(model_name, "gray"), linewidth=2)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy vs. Noise Level")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Lower SNR = more noise, shown right-to-left
    plt.tight_layout()

    path = os.path.join(config.RESULTS_DIR, "noise_robustness.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved noise robustness plot: %s", path)

    # Save raw data
    results_path = os.path.join(config.RESULTS_DIR, "noise_robustness.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_noise_robustness()
