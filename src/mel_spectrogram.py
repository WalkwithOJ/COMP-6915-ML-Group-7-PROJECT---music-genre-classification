"""
Phase 2B: Mel-spectrogram generation for the CNN model.
Loads WAV files, slices into 3-second segments, computes log-mel
spectrograms, and saves as .npy arrays.
"""
import os
import json

import numpy as np
import librosa

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import get_logger, ensure_dirs

logger = get_logger(__name__)


def generate_mel_spectrogram(y: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """
    Compute a log-mel spectrogram from an audio signal.
    Returns a normalized array of shape (n_mels, time_frames).
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_mels=config.N_MELS,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalize to [0, 1]
    S_min, S_max = S_db.min(), S_db.max()
    if S_max - S_min > 0:
        S_norm = (S_db - S_min) / (S_max - S_min)
    else:
        S_norm = np.zeros_like(S_db)

    return S_norm.astype(np.float32)


def process_track(track_id: str) -> list:
    """
    Load a 30-second track, slice it into 3-second segments,
    generate mel-spectrograms, and save as .npy files.
    Returns list of (segment_id, spectrogram_path) tuples.
    """
    genre = track_id.rsplit(".", 1)[0]
    wav_path = os.path.join(config.AUDIO_DIR, genre, f"{track_id}.wav")

    if not os.path.isfile(wav_path):
        logger.warning("WAV file not found: %s", wav_path)
        return []

    y, sr = librosa.load(wav_path, sr=config.SAMPLE_RATE, mono=True)
    samples_per_segment = config.SEGMENT_DURATION * sr  # 3 * 22050 = 66150

    output_dir = os.path.join(config.MEL_SPEC_DIR, genre)
    ensure_dirs(output_dir)

    results = []
    for seg_idx in range(config.SEGMENTS_PER_TRACK):
        start = seg_idx * samples_per_segment
        end = start + samples_per_segment

        if end > len(y):
            break  # skip incomplete final segment

        segment = y[start:end]
        mel_spec = generate_mel_spectrogram(segment, sr)

        # Pad or trim to ensure consistent width
        if mel_spec.shape[1] < config.MEL_SPEC_WIDTH:
            pad_width = config.MEL_SPEC_WIDTH - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)))
        elif mel_spec.shape[1] > config.MEL_SPEC_WIDTH:
            mel_spec = mel_spec[:, :config.MEL_SPEC_WIDTH]

        segment_id = f"{track_id}.{seg_idx}"
        npy_path = os.path.join(output_dir, f"{segment_id}.npy")
        np.save(npy_path, mel_spec)
        results.append((segment_id, npy_path))

    return results


def generate_all_spectrograms():
    """
    Generate mel-spectrograms for all tracks listed in splits.json.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2B: Mel-Spectrogram Generation")
    logger.info("=" * 60)

    with open(config.SPLITS_FILE, "r") as f:
        splits = json.load(f)

    all_tracks = splits["train"] + splits["val"] + splits["test"]
    logger.info("Processing %d tracks ...", len(all_tracks))

    total_segments = 0
    for i, track_id in enumerate(all_tracks):
        results = process_track(track_id)
        total_segments += len(results)
        if (i + 1) % 100 == 0:
            logger.info("  Processed %d / %d tracks", i + 1, len(all_tracks))

    logger.info("Generated %d mel-spectrogram segments total.", total_segments)


def load_mel_data(splits: dict):
    """
    Load all mel-spectrograms into memory as numpy arrays,
    split by partition. Returns dict with X/y arrays for each set.

    Each X array has shape (num_samples, height, width, 1).
    """
    data = {}

    for partition in ["train", "val", "test"]:
        X_list, y_list = [], []
        for track_id in splits[partition]:
            genre = track_id.rsplit(".", 1)[0]
            genre_idx = config.GENRES.index(genre)

            for seg_idx in range(config.SEGMENTS_PER_TRACK):
                segment_id = f"{track_id}.{seg_idx}"
                npy_path = os.path.join(
                    config.MEL_SPEC_DIR, genre, f"{segment_id}.npy"
                )
                if os.path.isfile(npy_path):
                    mel = np.load(npy_path)
                    X_list.append(mel[..., np.newaxis])  # add channel dim
                    y_list.append(genre_idx)

        data[f"X_{partition}"] = np.array(X_list, dtype=np.float32)
        data[f"y_{partition}"] = np.array(y_list, dtype=np.int32)
        logger.info(
            "  %s: %d segments, shape %s",
            partition, len(y_list), data[f"X_{partition}"].shape,
        )

    return data


if __name__ == "__main__":
    generate_all_spectrograms()
