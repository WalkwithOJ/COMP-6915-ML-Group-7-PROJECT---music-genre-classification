"""
Shared utility functions: seeding, logging, directory helpers.
"""
import os
import random
import logging
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across numpy, random, and TF."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Create a console logger with a standard format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_dirs(*paths):
    """Create directories if they don't exist."""
    for p in paths:
        os.makedirs(p, exist_ok=True)
