"""
Phase 1: Data pipeline — extract archive, validate audio files,
create track-level stratified train/val/test split.
"""
import os
import json
import zipfile

import librosa
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import get_logger, ensure_dirs, set_seed

logger = get_logger(__name__)


def extract_archive():
    """Extract archive.zip into data/raw/ if not already extracted."""
    if os.path.isdir(config.AUDIO_DIR) and len(os.listdir(config.AUDIO_DIR)) > 0:
        logger.info("Audio directory already exists, skipping extraction.")
        return

    logger.info("Extracting archive.zip ...")
    ensure_dirs(config.DATA_RAW)

    with zipfile.ZipFile(config.ARCHIVE_PATH, "r") as zf:
        for member in zf.namelist():
            # Files inside the zip are under "Data/..." — strip the prefix
            if member.startswith("Data/"):
                target = os.path.join(config.DATA_RAW, member[len("Data/"):])
                if member.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        dst.write(src.read())

    logger.info("Extraction complete.")


def validate_audio():
    """
    Validate all WAV files. Returns a list of valid track IDs and
    a list of corrupted track IDs.

    A track ID looks like 'blues.00000' (genre.NNNNN).
    """
    valid_tracks = []
    corrupted_tracks = []

    for genre in config.GENRES:
        genre_dir = os.path.join(config.AUDIO_DIR, genre)
        if not os.path.isdir(genre_dir):
            logger.warning("Missing genre directory: %s", genre)
            continue

        wav_files = sorted(f for f in os.listdir(genre_dir) if f.endswith(".wav"))
        logger.info("Genre '%s': found %d WAV files", genre, len(wav_files))

        for wav_file in wav_files:
            track_id = wav_file.replace(".wav", "")  # e.g. blues.00000
            filepath = os.path.join(genre_dir, wav_file)

            # Check against known corrupted files
            if wav_file in config.CORRUPTED_FILES:
                logger.warning("Skipping known corrupted file: %s", wav_file)
                corrupted_tracks.append(track_id)
                continue

            try:
                y, sr = librosa.load(filepath, sr=config.SAMPLE_RATE, mono=True)
                duration = len(y) / sr
                if duration < (config.TRACK_DURATION - 2):
                    logger.warning(
                        "Short track (%0.1fs): %s", duration, wav_file
                    )
                    corrupted_tracks.append(track_id)
                else:
                    valid_tracks.append(track_id)
            except Exception as e:
                logger.error("Failed to load %s: %s", wav_file, e)
                corrupted_tracks.append(track_id)

    logger.info(
        "Validation complete: %d valid, %d corrupted",
        len(valid_tracks), len(corrupted_tracks),
    )
    return valid_tracks, corrupted_tracks


def create_splits(valid_tracks: list):
    """
    Create a track-level stratified train/val/test split.
    Saves the result to data/processed/splits.json.

    This is the critical step that prevents data leakage: all 10 segments
    from the same 30-second track will end up in the same partition.
    """
    set_seed(config.RANDOM_SEED)
    ensure_dirs(config.DATA_PROCESSED)

    # Extract genre label from track ID (e.g. 'blues' from 'blues.00000')
    labels = [tid.rsplit(".", 1)[0] for tid in valid_tracks]

    # First split: separate test set
    train_val, test, labels_tv, _ = train_test_split(
        valid_tracks, labels,
        test_size=config.TEST_RATIO,
        stratify=labels,
        random_state=config.RANDOM_SEED,
    )

    # Second split: separate validation from training
    val_ratio_adjusted = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train, val, _, _ = train_test_split(
        train_val, labels_tv,
        test_size=val_ratio_adjusted,
        stratify=labels_tv,
        random_state=config.RANDOM_SEED,
    )

    splits = {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }

    # Sanity checks
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    assert len(all_ids) == len(valid_tracks), "Track count mismatch after splitting"
    assert not (set(splits["train"]) & set(splits["val"])), "Leakage: train/val overlap"
    assert not (set(splits["train"]) & set(splits["test"])), "Leakage: train/test overlap"
    assert not (set(splits["val"]) & set(splits["test"])), "Leakage: val/test overlap"

    with open(config.SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)

    logger.info(
        "Splits created — train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )

    # Log per-genre distribution
    for partition_name in ["train", "val", "test"]:
        genre_counts = {}
        for tid in splits[partition_name]:
            g = tid.rsplit(".", 1)[0]
            genre_counts[g] = genre_counts.get(g, 0) + 1
        logger.info("  %s genre distribution: %s", partition_name, genre_counts)

    return splits


def load_splits() -> dict:
    """Load the saved splits from disk."""
    with open(config.SPLITS_FILE, "r") as f:
        return json.load(f)


def run_pipeline():
    """Execute the full Phase 1 pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 1: Data Pipeline")
    logger.info("=" * 60)

    extract_archive()
    valid_tracks, corrupted_tracks = validate_audio()

    if corrupted_tracks:
        logger.info("Corrupted/excluded tracks: %s", corrupted_tracks)

    splits = create_splits(valid_tracks)
    return splits


if __name__ == "__main__":
    run_pipeline()
