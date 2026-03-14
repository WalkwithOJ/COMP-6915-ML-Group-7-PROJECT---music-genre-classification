"""
Phase 4B: Feature importance analysis (Research Question 2).
Identifies which audio features contribute most to genre distinction.
"""
import os

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import get_logger, ensure_dirs
from src.feature_engineering import prepare_features

logger = get_logger(__name__)


def rf_feature_importance(model, feature_cols: list, top_n: int = 20):
    """
    Extract and plot Gini importance from the Random Forest model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_cols[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_scores[::-1], align="center", color="steelblue")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Gini Importance")
    ax.set_title(f"Top {top_n} Features — Random Forest")
    plt.tight_layout()

    ensure_dirs(config.RESULTS_DIR)
    path = os.path.join(config.RESULTS_DIR, "feature_importance_rf.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved RF feature importance plot: %s", path)

    return list(zip(top_features, top_scores.tolist()))


def svm_permutation_importance(model, X_val, y_val, feature_cols: list, top_n: int = 20):
    """
    Compute permutation importance for SVM on the validation set.
    This works for any model (including RBF kernel SVM).
    """
    logger.info("Computing permutation importance for SVM (this may take a few minutes) ...")
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=10,
        random_state=config.RANDOM_SEED,
        scoring="f1_macro",
        n_jobs=-1,
    )

    importances = result.importances_mean
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_cols[i] for i in indices]
    top_scores = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), top_scores[::-1], align="center", color="coral")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Permutation Importance (F1 decrease)")
    ax.set_title(f"Top {top_n} Features — SVM (Permutation)")
    plt.tight_layout()

    path = os.path.join(config.RESULTS_DIR, "feature_importance_svm.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved SVM permutation importance plot: %s", path)

    return list(zip(top_features, top_scores.tolist()))


def run_feature_importance():
    """Execute the full feature importance analysis."""
    logger.info("=" * 60)
    logger.info("PHASE 4B: Feature Importance Analysis")
    logger.info("=" * 60)

    data, feature_cols, le = prepare_features(use_3sec=False)
    ensure_dirs(config.RESULTS_DIR)

    results = {}

    # Random Forest
    rf_path = os.path.join(config.MODELS_DIR, "rf_best.joblib")
    if os.path.isfile(rf_path):
        rf = joblib.load(rf_path)
        rf_top = rf_feature_importance(rf, feature_cols)
        results["rf"] = rf_top
        logger.info("RF top 5: %s", rf_top[:5])

    # SVM
    svm_path = os.path.join(config.MODELS_DIR, "svm_best.joblib")
    if os.path.isfile(svm_path):
        svm = joblib.load(svm_path)
        svm_top = svm_permutation_importance(
            svm, data["X_val"], data["y_val"], feature_cols,
        )
        results["svm"] = svm_top
        logger.info("SVM top 5: %s", svm_top[:5])

    # Cross-model consensus
    if "rf" in results and "svm" in results:
        rf_names = {name for name, _ in results["rf"]}
        svm_names = {name for name, _ in results["svm"]}
        consensus = rf_names & svm_names
        logger.info("Consensus top features (both models): %s", consensus)

    return results


if __name__ == "__main__":
    run_feature_importance()
