"""
Phase 3b: CNN architecture for mel-spectrogram classification.
4-block convolutional network with BatchNorm, Dropout, and L2 regularization.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def build_cnn(
    input_shape=config.MEL_SPEC_SHAPE,
    num_classes=config.NUM_GENRES,
    l2_lambda=config.L2_LAMBDA,
    dropout_rate=config.DROPOUT_RATE,
):
    """
    Build a 4-block CNN for mel-spectrogram classification.

    Architecture:
        4x [Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout]
        -> GlobalAveragePooling -> Dense(256) -> BatchNorm -> ReLU
        -> Dropout(0.5) -> Dense(num_classes, softmax)
    """
    from tensorflow.keras import models, layers, regularizers

    model = models.Sequential([
        # Block 1: 32 filters
        layers.Conv2D(
            32, (3, 3), padding="same",
            kernel_regularizer=regularizers.l2(l2_lambda),
            input_shape=input_shape,
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Block 2: 64 filters
        layers.Conv2D(
            64, (3, 3), padding="same",
            kernel_regularizer=regularizers.l2(l2_lambda),
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Block 3: 128 filters
        layers.Conv2D(
            128, (3, 3), padding="same",
            kernel_regularizer=regularizers.l2(l2_lambda),
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Block 4: 256 filters
        layers.Conv2D(
            256, (3, 3), padding="same",
            kernel_regularizer=regularizers.l2(l2_lambda),
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(dropout_rate),

        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(
            256,
            kernel_regularizer=regularizers.l2(l2_lambda),
        ),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
