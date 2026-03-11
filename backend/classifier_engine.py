"""
Classifier engine for multiverse analysis.
Runs multiple classifiers (RF, XGBoost, Logistic, SVM) across predictor/covariate
combinations and reports accuracy, AUC, and feature importances.
"""

import uuid
import itertools
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False


def _prepare_data(df: pd.DataFrame, outcome: str, features: list[str]):
    """Prepare X, y arrays from dataframe, handling missing values and encoding."""
    cols = [outcome] + features
    data = df[cols].dropna()
    if len(data) < 20:
        return None, None, None

    y = data[outcome].values
    # Encode outcome if not already numeric
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        # For continuous numeric outcomes, derive a binary target via median split
        # so classifiers can run consistently across all analyses.
        unique_count = len(np.unique(y))
        if unique_count > 10:
            median_val = float(np.median(y))
            y = (y > median_val).astype(int)

    X = data[features].copy()
    # Encode any categorical features
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X = X.values.astype(float)
    y = y.astype(float)

    # Need at least 2 classes
    if len(np.unique(y)) < 2:
        return None, None, None

    return X, y, features


def _get_classifiers() -> list[tuple[str, object]]:
    """Return list of (name, classifier) tuples."""
    classifiers = [
        ("Random Forest", RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
        )),
        ("Gradient Boosting", GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
        )),
        ("Logistic Regression", LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs"
        )),
        ("SVM (RBF)", SVC(
            kernel="rbf", probability=True, random_state=42
        )),
    ]
    if HAS_XGBOOST:
        classifiers.insert(2, ("XGBoost", XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric="logloss",
            verbosity=0
        )))
    return classifiers


def _compute_feature_importance(clf, clf_name: str, X: np.ndarray, y: np.ndarray,
                                 feature_names: list[str]) -> dict[str, float]:
    """Extract or compute feature importances for a fitted classifier."""
    # Tree-based models have native feature_importances_
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        return dict(zip(feature_names, importances.tolist()))

    # For logistic regression, use absolute coefficient values
    if hasattr(clf, "coef_"):
        coefs = np.abs(clf.coef_).mean(axis=0) if clf.coef_.ndim > 1 else np.abs(clf.coef_[0])
        # Normalize to sum to 1
        total = coefs.sum()
        if total > 0:
            coefs = coefs / total
        return dict(zip(feature_names, coefs.tolist()))

    # Fallback: permutation importance
    try:
        result = permutation_importance(clf, X, y, n_repeats=5, random_state=42, n_jobs=-1)
        importances = np.maximum(result.importances_mean, 0)
        total = importances.sum()
        if total > 0:
            importances = importances / total
        return dict(zip(feature_names, importances.tolist()))
    except Exception:
        return {name: 1.0 / len(feature_names) for name in feature_names}


def generate_feature_subsets(predictors: list[str], confounders: list[str]) -> list[list[str]]:
    """Generate feature subsets similar to regression covariate subsets."""
    subsets = []

    # Each predictor + all confounders
    if confounders:
        for p in predictors:
            subsets.append([p] + confounders)

    # All predictors together
    if len(predictors) > 1:
        subsets.append(list(predictors))

    # All predictors + all confounders
    if confounders and len(predictors) > 1:
        subsets.append(list(predictors) + confounders)

    # All features
    all_features = list(predictors) + confounders
    if all_features not in subsets:
        subsets.append(all_features)

    # Deduplicate
    seen = set()
    unique = []
    for s in subsets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique.append(s)

    return unique


def run_classifiers(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    confounders: list[str],
) -> dict:
    """Run multiple classifiers across feature subsets.

    Returns dict with:
      - classifier_results: list of individual classifier results
      - summary: overall summary statistics
    """
    feature_subsets = generate_feature_subsets(predictors, confounders)
    classifiers = _get_classifiers()
    results = []
    cv_folds = int(os.environ.get("CLASSIFIER_CV_FOLDS", "3") or "3")
    cv = StratifiedKFold(n_splits=max(2, cv_folds), shuffle=True, random_state=42)

    for features in feature_subsets:
        X, y, feat_names = _prepare_data(df, outcome, features)
        if X is None:
            continue

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        for clf_name, clf_template in classifiers:

            try:
                # Clone classifier for fresh fit
                from sklearn.base import clone
                clf = clone(clf_template)

                # Single cross-validation pass for all metrics to avoid repeated fitting.
                cv_scores = cross_validate(
                    clf,
                    X_scaled,
                    y,
                    cv=cv,
                    scoring={
                        "accuracy": "accuracy",
                        "roc_auc": "roc_auc",
                        "precision": "precision_weighted",
                        "recall": "recall_weighted",
                    },
                    n_jobs=-1,
                    error_score=np.nan,
                )
                acc_scores = cv_scores.get("test_accuracy", np.array([np.nan]))
                auc_scores = cv_scores.get("test_roc_auc", np.array([np.nan]))
                prec_scores = cv_scores.get("test_precision", np.array([np.nan]))
                recall_scores = cv_scores.get("test_recall", np.array([np.nan]))

                accuracy = float(np.nanmean(acc_scores))
                accuracy_std = float(np.nanstd(acc_scores))
                auc_mean = float(np.nanmean(auc_scores))
                auc = auc_mean if np.isfinite(auc_mean) else None
                auc_std = float(np.nanstd(auc_scores)) if auc is not None else None

                # Fit on full data for feature importance
                clf.fit(X_scaled, y)
                feature_importance = _compute_feature_importance(
                    clf, clf_name, X_scaled, y, feat_names
                )
                precision_mean = float(np.nanmean(prec_scores))
                recall_mean = float(np.nanmean(recall_scores))
                precision = precision_mean if np.isfinite(precision_mean) else None
                recall = recall_mean if np.isfinite(recall_mean) else None

                results.append({
                    "spec_id": str(uuid.uuid4()),
                    "classifier": clf_name,
                    "features": feat_names,
                    "predictors_used": [f for f in feat_names if f in predictors],
                    "covariates_used": [f for f in feat_names if f in confounders],
                    "accuracy": round(accuracy, 4),
                    "accuracy_std": round(accuracy_std, 4),
                    "auc": round(auc, 4) if auc is not None else None,
                    "auc_std": round(auc_std, 4) if auc_std is not None else None,
                    "precision": round(precision, 4) if precision is not None else None,
                    "recall": round(recall, 4) if recall is not None else None,
                    "feature_importance": feature_importance,
                    "n_obs": int(len(y)),
                    "n_features": len(feat_names),
                })
            except Exception as e:
                print(f"Classifier {clf_name} failed on features {feat_names}: {e}")
                continue

    # Summary
    if results:
        best = max(results, key=lambda r: r["accuracy"])
        mean_acc = float(np.mean([r["accuracy"] for r in results]))
        aucs = [r["auc"] for r in results if r["auc"] is not None]
        mean_auc = float(np.mean(aucs)) if aucs else None
    else:
        best = None
        mean_acc = 0.0
        mean_auc = None

    return {
        "classifier_results": results,
        "total_specs": len(results),
        "best_classifier": best["classifier"] if best else None,
        "best_accuracy": best["accuracy"] if best else None,
        "mean_accuracy": round(mean_acc, 4),
        "mean_auc": round(mean_auc, 4) if mean_auc is not None else None,
    }
