"""
Central configuration for the Music Genre Classification project.
All paths, hyperparameters, and constants are defined here.
"""
import os

# -- Paths -----------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_PATH = os.path.join(PROJECT_ROOT, "archive.zip")

DATA_RAW       = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
AUDIO_DIR      = os.path.join(DATA_RAW, "genres_original")
CSV_30S        = os.path.join(DATA_RAW, "features_30_sec.csv")
CSV_3S         = os.path.join(DATA_RAW, "features_3_sec.csv")
SPLITS_FILE    = os.path.join(DATA_PROCESSED, "splits.json")
MEL_SPEC_DIR   = os.path.join(DATA_PROCESSED, "mel_spectrograms")
MODELS_DIR     = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "results")

# -- Dataset ---------------------------------------------------------------
GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]
NUM_GENRES     = 10
SAMPLE_RATE    = 22050
TRACK_DURATION = 30   # seconds
SEGMENT_DURATION = 3  # seconds
SEGMENTS_PER_TRACK = TRACK_DURATION // SEGMENT_DURATION  # 10

# Known corrupted/truncated files in GTZAN
CORRUPTED_FILES = {"jazz.00054.wav"}

# -- Train / Val / Test Split ----------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42

# -- Feature Engineering ---------------------------------------------------
N_MFCC     = 20
N_FFT      = 2048
HOP_LENGTH = 512
N_MELS     = 128

# Mel-spectrogram shape for CNN input (height, width, channels)
# width = ceil(SEGMENT_DURATION * SAMPLE_RATE / HOP_LENGTH) ≈ 130
MEL_SPEC_HEIGHT = 128
MEL_SPEC_WIDTH  = 130
MEL_SPEC_SHAPE  = (MEL_SPEC_HEIGHT, MEL_SPEC_WIDTH, 1)

# -- Traditional ML Hyperparameter Grids -----------------------------------
SVM_PARAM_GRID = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale", "auto"],
}

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5],
}

# -- CNN Hyperparameters ---------------------------------------------------
CNN_EPOCHS     = 100
CNN_BATCH_SIZE = 32
CNN_LR         = 0.001
CNN_PATIENCE   = 10   # EarlyStopping patience
L2_LAMBDA      = 0.001
DROPOUT_RATE   = 0.3

# -- Noise Robustness Experiment -------------------------------------------
NOISE_SNR_LEVELS = [40, 30, 20, 10, 5]  # dB
