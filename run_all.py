"""
Master orchestrator: runs the full ML pipeline from data extraction to evaluation.

Usage:
    python run_all.py                  # Run all phases
    python run_all.py --phase 1        # Run only Phase 1 (data pipeline)
    python run_all.py --phase 2        # Run only Phase 2 (feature engineering)
    python run_all.py --phase 3        # Run only Phase 3 (model training)
    python run_all.py --phase 4        # Run only Phase 4 (evaluation)
"""
import argparse
import time

from src.utils import set_seed, get_logger
import config

logger = get_logger("run_all")


def phase1():
    """Phase 1: Data Pipeline — extract, validate, split."""
    from src.data_pipeline import run_pipeline
    run_pipeline()


def phase2():
    """Phase 2: Feature Engineering — tabular features + mel-spectrograms."""
    from src.feature_engineering import prepare_features
    from src.mel_spectrogram import generate_all_spectrograms

    prepare_features(use_3sec=False)
    generate_all_spectrograms()


def phase3():
    """Phase 3: Model Training — SVM, RF, CNN."""
    from src.traditional_ml import run_traditional_ml
    from src.train_cnn import run_cnn_training

    run_traditional_ml()
    run_cnn_training()


def phase4():
    """Phase 4: Evaluation — metrics, feature importance, noise robustness."""
    from src.evaluate import run_evaluation
    from src.feature_importance import run_feature_importance
    from src.noise_robustness import run_noise_robustness

    run_evaluation()
    run_feature_importance()
    run_noise_robustness()


def main():
    parser = argparse.ArgumentParser(description="Music Genre Classification Pipeline")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4],
        help="Run a specific phase only (1-4). Omit to run all.",
    )
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    start = time.time()

    phases = {1: phase1, 2: phase2, 3: phase3, 4: phase4}

    if args.phase:
        logger.info("Running Phase %d only", args.phase)
        phases[args.phase]()
    else:
        logger.info("Running full pipeline (Phases 1-4)")
        for i in [1, 2, 3, 4]:
            phases[i]()

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
