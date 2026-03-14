"""
Phase 3b: CNN training loop with callbacks, data augmentation,
and track-level majority voting for inference.
"""
import os
import json

import numpy as np
import tensorflow as tf

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import get_logger, ensure_dirs, set_seed
from src.cnn_model import build_cnn
from src.mel_spectrogram import load_mel_data

logger = get_logger(__name__)


# -- Data Augmentation (SpecAugment-style) ---------------------------------

def spec_augment(spectrogram, freq_mask_num=2, time_mask_num=2,
                 freq_mask_width=10, time_mask_width=10):
    """
    Apply SpecAugment-style masking to a mel-spectrogram.
    Input shape: (height, width, 1)
    """
    spec = tf.identity(spectrogram)
    h, w = config.MEL_SPEC_HEIGHT, config.MEL_SPEC_WIDTH

    # Frequency masking
    for _ in range(freq_mask_num):
        f = tf.random.uniform([], 0, freq_mask_width, dtype=tf.int32)
        f0 = tf.random.uniform([], 0, h - f, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([f0, w, 1]),
            tf.zeros([f, w, 1]),
            tf.ones([h - f0 - f, w, 1]),
        ], axis=0)
        spec = spec * mask

    # Time masking
    for _ in range(time_mask_num):
        t = tf.random.uniform([], 0, time_mask_width, dtype=tf.int32)
        t0 = tf.random.uniform([], 0, w - t, dtype=tf.int32)
        mask = tf.concat([
            tf.ones([h, t0, 1]),
            tf.zeros([h, t, 1]),
            tf.ones([h, w - t0 - t, 1]),
        ], axis=1)
        spec = spec * mask

    return spec


def create_datasets(mel_data: dict):
    """Create tf.data.Datasets with augmentation for training."""
    # Training set with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices(
        (mel_data["X_train"], mel_data["y_train"])
    )
    train_ds = train_ds.shuffle(buffer_size=len(mel_data["y_train"]))
    train_ds = train_ds.map(
        lambda x, y: (spec_augment(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    train_ds = train_ds.batch(config.CNN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Validation set (no augmentation)
    val_ds = tf.data.Dataset.from_tensor_slices(
        (mel_data["X_val"], mel_data["y_val"])
    )
    val_ds = val_ds.batch(config.CNN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Test set (no augmentation)
    test_ds = tf.data.Dataset.from_tensor_slices(
        (mel_data["X_test"], mel_data["y_test"])
    )
    test_ds = test_ds.batch(config.CNN_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def train_cnn(mel_data: dict):
    """Train the CNN and return the trained model + training history."""
    logger.info("Building CNN model ...")
    model = build_cnn()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.CNN_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(print_fn=logger.info)

    train_ds, val_ds, test_ds = create_datasets(mel_data)

    ensure_dirs(config.MODELS_DIR)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.CNN_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, "cnn_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    logger.info("Starting CNN training ...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.CNN_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate on test set (segment-level)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    logger.info("Test segment-level accuracy: %.4f, loss: %.4f", test_acc, test_loss)

    return model, history


def majority_vote_predict(model, splits: dict):
    """
    Track-level prediction using majority voting across 10 segments.
    This gives a fair comparison against SVM/RF which operate on 30-sec features.
    """
    logger.info("Performing track-level majority vote prediction ...")
    track_preds = []
    track_labels = []

    for track_id in splits["test"]:
        genre = track_id.rsplit(".", 1)[0]
        genre_idx = config.GENRES.index(genre)
        track_labels.append(genre_idx)

        segment_preds = []
        for seg_idx in range(config.SEGMENTS_PER_TRACK):
            npy_path = os.path.join(
                config.MEL_SPEC_DIR, genre, f"{track_id}.{seg_idx}.npy"
            )
            if os.path.isfile(npy_path):
                mel = np.load(npy_path)
                mel = mel[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
                pred = model.predict(mel, verbose=0)
                segment_preds.append(np.argmax(pred, axis=1)[0])

        if segment_preds:
            # Majority vote
            vote = max(set(segment_preds), key=segment_preds.count)
            track_preds.append(vote)
        else:
            track_preds.append(-1)

    return np.array(track_preds), np.array(track_labels)


def run_cnn_training():
    """Execute the full Phase 3b pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 3b: CNN Training")
    logger.info("=" * 60)

    set_seed(config.RANDOM_SEED)

    with open(config.SPLITS_FILE, "r") as f:
        splits = json.load(f)

    mel_data = load_mel_data(splits)
    model, history = train_cnn(mel_data)

    # Track-level majority voting
    track_preds, track_labels = majority_vote_predict(model, splits)
    track_acc = np.mean(track_preds == track_labels)
    logger.info("Test track-level accuracy (majority vote): %.4f", track_acc)

    # Save training history
    ensure_dirs(config.RESULTS_DIR)
    hist_path = os.path.join(config.RESULTS_DIR, "cnn_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)

    return model, history


if __name__ == "__main__":
    run_cnn_training()
