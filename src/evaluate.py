"""
Phase 4A: Unified evaluation across all models.
Generates confusion matrices, classification reports, and comparison tables.
"""
import os
import json

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import get_logger, ensure_dirs

logger = get_logger(__name__)


def plot_confusion_matrix(y_true, y_pred, model_name: str, labels=None):
    """Plot and save a confusion matrix heatmap."""
    if labels is None:
        labels = config.GENRES

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("Predicted Genre")
    ax.set_ylabel("True Genre")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()

    ensure_dirs(config.RESULTS_DIR)
    path = os.path.join(config.RESULTS_DIR, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix: %s", path)
    return cm


def evaluate_model(y_true, y_pred, model_name: str, labels=None):
    """
    Compute all evaluation metrics for a single model.
    Returns a dict of metrics.
    """
    if labels is None:
        labels = config.GENRES

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    logger.info("-" * 40)
    logger.info("Model: %s", model_name)
    logger.info("  Accuracy:       %.4f", acc)
    logger.info("  F1 (macro):     %.4f", f1_macro)
    logger.info("  F1 (weighted):  %.4f", f1_weighted)

    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        output_dict=True,
    )
    report_str = classification_report(y_true, y_pred, target_names=labels)
    logger.info("\n%s", report_str)

    cm = plot_confusion_matrix(y_true, y_pred, model_name, labels)

    return {
        "model": model_name,
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }


def plot_training_curves(history_path: str):
    """Plot CNN training/validation loss and accuracy curves."""
    with open(history_path, "r") as f:
        history = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(history["loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(history["accuracy"], label="Train Accuracy")
    ax2.plot(history["val_accuracy"], label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.RESULTS_DIR, "training_curves_cnn.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved training curves: %s", path)


def generate_comparison_table(all_results: list):
    """Create and save the model comparison table."""
    rows = []
    for r in all_results:
        rows.append({
            "Model": r["model"],
            "Accuracy": f"{r['accuracy']:.4f}",
            "F1 (Macro)": f"{r['f1_macro']:.4f}",
            "F1 (Weighted)": f"{r['f1_weighted']:.4f}",
        })

    df = pd.DataFrame(rows)
    path = os.path.join(config.RESULTS_DIR, "comparison_table.csv")
    df.to_csv(path, index=False)
    logger.info("Saved comparison table:\n%s", df.to_string(index=False))
    return df


def run_evaluation():
    """
    Run full evaluation on all trained models.
    Assumes models are already trained and saved.
    """
    logger.info("=" * 60)
    logger.info("PHASE 4A: Model Evaluation")
    logger.info("=" * 60)

    ensure_dirs(config.RESULTS_DIR)

    # Load splits and features
    from src.feature_engineering import prepare_features
    data, feature_cols, le = prepare_features(use_3sec=False)

    all_results = []

    # Evaluate SVM
    svm_path = os.path.join(config.MODELS_DIR, "svm_best.joblib")
    if os.path.isfile(svm_path):
        svm = joblib.load(svm_path)
        svm_pred = svm.predict(data["X_test"])
        result = evaluate_model(data["y_test"], svm_pred, "SVM")
        all_results.append(result)

    # Evaluate Random Forest
    rf_path = os.path.join(config.MODELS_DIR, "rf_best.joblib")
    if os.path.isfile(rf_path):
        rf = joblib.load(rf_path)
        rf_pred = rf.predict(data["X_test"])
        result = evaluate_model(data["y_test"], rf_pred, "Random Forest")
        all_results.append(result)

    # Evaluate CNN (track-level majority vote)
    cnn_path = os.path.join(config.MODELS_DIR, "cnn_best.keras")
    if os.path.isfile(cnn_path):
        import tensorflow as tf
        cnn = tf.keras.models.load_model(cnn_path)

        with open(config.SPLITS_FILE, "r") as f:
            splits = json.load(f)

        from src.train_cnn import majority_vote_predict
        cnn_pred, cnn_labels = majority_vote_predict(cnn, splits)
        result = evaluate_model(cnn_labels, cnn_pred, "CNN (Majority Vote)")
        all_results.append(result)

    # Plot CNN training curves if available
    hist_path = os.path.join(config.RESULTS_DIR, "cnn_history.json")
    if os.path.isfile(hist_path):
        plot_training_curves(hist_path)

    # Comparison table
    if all_results:
        comparison_df = generate_comparison_table(all_results)

    return all_results


if __name__ == "__main__":
    run_evaluation()
