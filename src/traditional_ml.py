"""
Phase 3a: Traditional ML models — SVM and Random Forest.
Uses GridSearchCV with 5-fold cross-validation on the training set.
"""
import os
import json

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from src.utils import get_logger, ensure_dirs, set_seed
from src.feature_engineering import prepare_features

logger = get_logger(__name__)


def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM with GridSearchCV and return the best estimator."""
    logger.info("Training SVM with GridSearchCV ...")
    logger.info("  Parameter grid: %s", config.SVM_PARAM_GRID)

    svm = SVC(probability=True, random_state=config.RANDOM_SEED)
    grid = GridSearchCV(
        svm,
        param_grid=config.SVM_PARAM_GRID,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    logger.info("  Best params: %s", grid.best_params_)
    logger.info("  Best CV F1 (macro): %.4f", grid.best_score_)

    # Validation performance
    val_pred = best.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    logger.info("  Val accuracy: %.4f, Val F1 (macro): %.4f", val_acc, val_f1)

    # Save model
    ensure_dirs(config.MODELS_DIR)
    model_path = os.path.join(config.MODELS_DIR, "svm_best.joblib")
    joblib.dump(best, model_path)
    logger.info("  SVM saved to %s", model_path)

    return best, grid.best_params_, grid.best_score_


def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest with GridSearchCV and return the best estimator."""
    logger.info("Training Random Forest with GridSearchCV ...")
    logger.info("  Parameter grid: %s", config.RF_PARAM_GRID)

    rf = RandomForestClassifier(random_state=config.RANDOM_SEED)
    grid = GridSearchCV(
        rf,
        param_grid=config.RF_PARAM_GRID,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    logger.info("  Best params: %s", grid.best_params_)
    logger.info("  Best CV F1 (macro): %.4f", grid.best_score_)

    # Validation performance
    val_pred = best.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_f1 = f1_score(y_val, val_pred, average="macro")
    logger.info("  Val accuracy: %.4f, Val F1 (macro): %.4f", val_acc, val_f1)

    # Save model
    model_path = os.path.join(config.MODELS_DIR, "rf_best.joblib")
    joblib.dump(best, model_path)
    logger.info("  RF saved to %s", model_path)

    return best, grid.best_params_, grid.best_score_


def run_traditional_ml():
    """Execute the full Phase 3a pipeline."""
    logger.info("=" * 60)
    logger.info("PHASE 3a: Traditional ML (SVM + Random Forest)")
    logger.info("=" * 60)

    set_seed(config.RANDOM_SEED)
    data, feature_cols, le = prepare_features(use_3sec=False)

    svm_model, svm_params, svm_cv_score = train_svm(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
    )

    rf_model, rf_params, rf_cv_score = train_random_forest(
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
    )

    results = {
        "svm": {"best_params": svm_params, "cv_f1_macro": svm_cv_score},
        "rf": {"best_params": rf_params, "cv_f1_macro": rf_cv_score},
    }

    # Save training summary
    ensure_dirs(config.RESULTS_DIR)
    summary_path = os.path.join(config.RESULTS_DIR, "traditional_ml_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return svm_model, rf_model, data, le


if __name__ == "__main__":
    run_traditional_ml()
