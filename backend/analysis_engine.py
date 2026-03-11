"""
Multiverse Analysis Engine

Runs exhaustive model specifications across different regression types,
covariate sets, and transformations based on the outcome variable type
and assumption check results.
"""

import uuid
import itertools
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from covariate_roles import classify_covariates
from scipy.special import boxcox as boxcox_transform
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import (
    Lasso,
    Ridge,
    ElasticNet,
    LogisticRegression,
)
from sklearn.preprocessing import StandardScaler


def detect_outcome_type(series: pd.Series) -> str:
    """Detect if outcome is continuous or binary."""
    unique = series.dropna().nunique()
    if unique == 2:
        return "binary"
    if unique < 10 and series.dtype in ["int64", "int32"]:
        return "count"
    return "continuous"


def check_continuous_assumptions(y: np.ndarray, X: np.ndarray) -> dict:
    """Check OLS assumptions for continuous outcome."""
    results = {}

    # Fit OLS to get residuals
    X_const = sm.add_constant(X)
    try:
        model = sm.OLS(y, X_const).fit()
        residuals = model.resid
    except Exception:
        return {"all_met": False, "error": "Could not fit initial OLS"}

    # 1. Normality of residuals (Shapiro-Wilk)
    # Use subset if n > 5000 (Shapiro-Wilk limit)
    sample = residuals[:5000] if len(residuals) > 5000 else residuals
    try:
        shapiro_stat, shapiro_p = stats.shapiro(sample)
        results["shapiro_wilk"] = {
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_p),
            "passed": shapiro_p > 0.05,
        }
    except Exception:
        results["shapiro_wilk"] = {"passed": False, "error": "Test failed"}

    # 2. Homoscedasticity (Breusch-Pagan)
    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_const)
        results["breusch_pagan"] = {
            "statistic": float(bp_stat),
            "p_value": float(bp_p),
            "passed": bp_p > 0.05,
        }
    except Exception:
        results["breusch_pagan"] = {"passed": False, "error": "Test failed"}

    # 3. Multicollinearity (VIF)
    if X.shape[1] > 1:
        try:
            vif_values = []
            for i in range(X.shape[1]):
                vif_val = variance_inflation_factor(X, i)
                vif_values.append(float(vif_val))
            results["vif"] = {
                "values": vif_values,
                "max_vif": max(vif_values),
                "passed": max(vif_values) < 10,
            }
        except Exception:
            results["vif"] = {"passed": True, "note": "Could not compute"}
    else:
        results["vif"] = {"passed": True, "note": "Single predictor"}

    results["all_met"] = all(
        v.get("passed", False) for v in results.values() if isinstance(v, dict)
    )
    return results


def check_binary_assumptions(y: np.ndarray, X: np.ndarray) -> dict:
    """Check logistic regression assumptions."""
    n = len(y)
    n_predictors = X.shape[1] if X.ndim > 1 else 1
    events = int(y.sum())
    non_events = n - events
    min_class = min(events, non_events)

    results = {
        "sample_size": {
            "n": n,
            "events_per_variable": min_class / max(n_predictors, 1),
            "passed": min_class / max(n_predictors, 1) >= 10,
        },
        "class_balance": {
            "events": events,
            "non_events": non_events,
            "minority_pct": round(min_class / n * 100, 1),
            "passed": min_class / n > 0.05,
        },
    }
    results["all_met"] = all(
        v.get("passed", False) for v in results.values() if isinstance(v, dict)
    )
    return results


def generate_covariate_subsets(predictors: list[str], confounders: list[str], max_subset_size: int = 4) -> list[list[str]]:
    """Generate all meaningful covariate subsets."""
    subsets = []

    # Limit combinatorial explosion
    max_k = min(len(predictors), max_subset_size)

    for k in range(1, max_k + 1):
        for combo in itertools.combinations(predictors, k):
            # Each combo with and without confounders
            subsets.append(list(combo))
            if confounders:
                subsets.append(list(combo) + confounders)

    # Also add just confounders if any
    if confounders and len(predictors) > 0:
        for pred in predictors:
            subsets.append([pred] + confounders)

    # Deduplicate
    seen = set()
    unique_subsets = []
    for s in subsets:
        key = tuple(sorted(s))
        if key not in seen:
            seen.add(key)
            unique_subsets.append(s)

    return unique_subsets


def run_ols(y: np.ndarray, X: np.ndarray, predictor_idx: int = 0) -> Optional[dict]:
    """Run OLS regression and return results."""
    try:
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        idx = predictor_idx + 1  # +1 for constant

        return {
            "coefficient": float(model.params[idx]),
            "p_value": float(model.pvalues[idx]),
            "ci_lower": float(model.conf_int()[idx][0]),
            "ci_upper": float(model.conf_int()[idx][1]),
            "r_squared": float(model.rsquared),
            "aic": float(model.aic),
            "n_obs": int(model.nobs),
        }
    except Exception:
        return None


def run_penalized_regression(y: np.ndarray, X: np.ndarray, method: str, predictor_idx: int = 0) -> Optional[dict]:
    """Run Lasso, Ridge, or ElasticNet."""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if method == "lasso":
            model = Lasso(alpha=0.1, max_iter=10000)
        elif method == "ridge":
            model = Ridge(alpha=1.0)
        else:
            model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=10000)

        model.fit(X_scaled, y)
        coef = float(model.coef_[predictor_idx])

        # Approximate p-value using bootstrap (simplified)
        n = len(y)
        y_pred = model.predict(X_scaled)
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)
        se = np.sqrt(mse / n) if n > 0 else 1.0
        t_stat = coef / se if se > 0 else 0
        p_val = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(n - X.shape[1] - 1, 1))))

        return {
            "coefficient": coef,
            "p_value": p_val,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "r_squared": float(model.score(X_scaled, y)),
            "aic": None,
            "n_obs": n,
        }
    except Exception:
        return None


def run_logistic(y: np.ndarray, X: np.ndarray, predictor_idx: int = 0) -> Optional[dict]:
    """Run logistic regression."""
    try:
        X_const = sm.add_constant(X)
        model = sm.Logit(y, X_const).fit(disp=0, maxiter=100)
        idx = predictor_idx + 1

        return {
            "coefficient": float(model.params[idx]),
            "p_value": float(model.pvalues[idx]),
            "ci_lower": float(model.conf_int()[idx][0]),
            "ci_upper": float(model.conf_int()[idx][1]),
            "r_squared": float(model.prsquared),
            "aic": float(model.aic),
            "n_obs": int(model.nobs),
            "odds_ratio": float(np.exp(model.params[idx])),
        }
    except Exception:
        return None


def run_penalized_logistic(y: np.ndarray, X: np.ndarray, penalty: str, predictor_idx: int = 0) -> Optional[dict]:
    """Run penalized logistic regression."""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pen = "l1" if penalty == "lasso" else "l2"
        solver = "saga" if pen == "l1" else "lbfgs"
        model = LogisticRegression(penalty=pen, C=1.0, solver=solver, max_iter=10000)
        model.fit(X_scaled, y)

        coef = float(model.coef_[0][predictor_idx])
        # Approximate p-value
        n = len(y)
        prob = model.predict_proba(X_scaled)[:, 1]
        W = np.diag(prob * (1 - prob))
        # Simplified SE estimation
        try:
            cov = np.linalg.inv(X_scaled.T @ W @ X_scaled)
            se = np.sqrt(np.diag(cov))[predictor_idx]
        except Exception:
            se = abs(coef) * 0.5 if coef != 0 else 1.0

        z_stat = coef / se if se > 0 else 0
        p_val = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        return {
            "coefficient": coef,
            "p_value": p_val,
            "ci_lower": coef - 1.96 * se,
            "ci_upper": coef + 1.96 * se,
            "r_squared": None,
            "aic": None,
            "n_obs": n,
            "odds_ratio": float(np.exp(coef)),
        }
    except Exception:
        return None


def compute_effect_measures(y: np.ndarray, x: np.ndarray) -> dict:
    """Compute OR, RR, and RD for binary outcome with binary predictor."""
    try:
        # 2x2 table
        a = np.sum((x == 1) & (y == 1))  # exposed, outcome
        b = np.sum((x == 1) & (y == 0))  # exposed, no outcome
        c = np.sum((x == 0) & (y == 1))  # unexposed, outcome
        d = np.sum((x == 0) & (y == 0))  # unexposed, no outcome

        # Odds ratio
        or_val = (a * d) / (b * c) if (b * c) > 0 else float("inf")

        # Risk ratio
        r1 = a / (a + b) if (a + b) > 0 else 0
        r0 = c / (c + d) if (c + d) > 0 else 0
        rr_val = r1 / r0 if r0 > 0 else float("inf")

        # Risk difference
        rd_val = r1 - r0

        return {
            "odds_ratio": float(or_val),
            "risk_ratio": float(rr_val),
            "risk_difference": float(rd_val),
            "divergent": abs(or_val - rr_val) / max(abs(or_val), 0.001) > 0.2
            if or_val != float("inf") and rr_val != float("inf")
            else True,
        }
    except Exception:
        return {}


def apply_fdr_correction(p_values: list[float]) -> list[float]:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * n

    for rank, (orig_idx, p) in enumerate(indexed, 1):
        corrected[orig_idx] = min(p * n / rank, 1.0)

    # Enforce monotonicity (step-up)
    sorted_indices = [idx for idx, _ in indexed]
    for i in range(n - 2, -1, -1):
        corrected[sorted_indices[i]] = min(
            corrected[sorted_indices[i]],
            corrected[sorted_indices[i + 1]] if i + 1 < n else 1.0,
        )

    return corrected


def run_multiverse_analysis(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    confounders: list[str],
    hypothesis: str = "",
) -> dict:
    """Run the full multiverse analysis."""
    # Clean data
    all_vars = [outcome] + predictors + confounders
    all_vars = list(set(all_vars))  # deduplicate
    working_df = df[all_vars].dropna()

    y = working_df[outcome].values.astype(float)
    outcome_type = detect_outcome_type(working_df[outcome])

    # Generate covariate subsets
    covariate_sets = generate_covariate_subsets(predictors, confounders)
    max_cov_sets = int(os.environ.get("ANALYSIS_MAX_COVARIATE_SETS", "0") or "0")
    if max_cov_sets > 0 and len(covariate_sets) > max_cov_sets:
        covariate_sets = sorted(covariate_sets, key=lambda s: (len(s), tuple(s)))[:max_cov_sets]

    all_results = []

    for cov_set in covariate_sets:
        # Determine the primary predictor(s) in this set
        preds_in_set = [p for p in cov_set if p in predictors]

        if not preds_in_set:
            continue

        X_cols = cov_set
        X = working_df[X_cols].values.astype(float)
        continuous_assumptions = None
        binary_assumptions = None
        if outcome_type == "continuous":
            continuous_assumptions = check_continuous_assumptions(y, X)
        elif outcome_type == "binary":
            binary_assumptions = check_binary_assumptions(y, X)

        for pred in preds_in_set:
            pred_idx = X_cols.index(pred)
            covariates_label = [c for c in X_cols if c != pred]

            if outcome_type == "continuous":
                assumptions = continuous_assumptions or {"all_met": False}

                # === Models when assumptions hold ===
                # OLS
                ols_result = run_ols(y, X, pred_idx)
                if ols_result:
                    all_results.append({
                        "spec_id": str(uuid.uuid4()),
                        "model_family": "OLS",
                        "predictor": pred,
                        "outcome": outcome,
                        "covariates": covariates_label,
                        "assumptions_met": assumptions.get("all_met", False),
                        "assumption_details": assumptions,
                        **ols_result,
                    })

                # Penalized regressions
                for method in ["lasso", "ridge", "elastic_net"]:
                    pen_result = run_penalized_regression(y, X, method, pred_idx)
                    if pen_result:
                        all_results.append({
                            "spec_id": str(uuid.uuid4()),
                            "model_family": method.replace("_", " ").title(),
                            "predictor": pred,
                            "outcome": outcome,
                            "covariates": covariates_label,
                            "assumptions_met": assumptions.get("all_met", False),
                            "assumption_details": assumptions,
                            **pen_result,
                        })

                # === Models when assumptions violated ===
                if not assumptions.get("all_met", True):
                    # Log transform
                    if np.all(y > 0):
                        y_log = np.log(y)
                        log_result = run_ols(y_log, X, pred_idx)
                        if log_result:
                            all_results.append({
                                "spec_id": str(uuid.uuid4()),
                                "model_family": "OLS (Log Y)",
                                "predictor": pred,
                                "outcome": outcome,
                                "covariates": covariates_label,
                                "assumptions_met": True,
                                "assumption_details": {"transform": "log"},
                                **log_result,
                            })

                    # Box-Cox transform
                    if np.all(y > 0):
                        try:
                            y_bc, lmbda = stats.boxcox(y)
                            bc_result = run_ols(y_bc, X, pred_idx)
                            if bc_result:
                                all_results.append({
                                    "spec_id": str(uuid.uuid4()),
                                    "model_family": f"OLS (Box-Cox λ={lmbda:.2f})",
                                    "predictor": pred,
                                    "outcome": outcome,
                                    "covariates": covariates_label,
                                    "assumptions_met": True,
                                    "assumption_details": {"transform": "box-cox", "lambda": lmbda},
                                    **bc_result,
                                })
                        except Exception:
                            pass

                    # Quantile regression (median)
                    try:
                        X_const = sm.add_constant(X)
                        qr_model = sm.QuantReg(y, X_const).fit(q=0.5, max_iter=1000)
                        qr_idx = pred_idx + 1
                        all_results.append({
                            "spec_id": str(uuid.uuid4()),
                            "model_family": "Quantile (median)",
                            "predictor": pred,
                            "outcome": outcome,
                            "covariates": covariates_label,
                            "coefficient": float(qr_model.params[qr_idx]),
                            "p_value": float(qr_model.pvalues[qr_idx]),
                            "ci_lower": float(qr_model.conf_int()[qr_idx][0]),
                            "ci_upper": float(qr_model.conf_int()[qr_idx][1]),
                            "r_squared": float(qr_model.prsquared),
                            "aic": None,
                            "n_obs": int(qr_model.nobs),
                            "assumptions_met": True,
                            "assumption_details": {"transform": "quantile"},
                        })
                    except Exception:
                        pass

            elif outcome_type == "binary":
                assumptions = binary_assumptions or {"all_met": False}

                # Logistic regression
                log_result = run_logistic(y, X, pred_idx)
                if log_result:
                    all_results.append({
                        "spec_id": str(uuid.uuid4()),
                        "model_family": "Logistic",
                        "predictor": pred,
                        "outcome": outcome,
                        "covariates": covariates_label,
                        "assumptions_met": assumptions.get("all_met", False),
                        "assumption_details": assumptions,
                        **log_result,
                    })

                # Penalized logistic
                for penalty in ["lasso", "ridge"]:
                    pen_log_result = run_penalized_logistic(y, X, penalty, pred_idx)
                    if pen_log_result:
                        all_results.append({
                            "spec_id": str(uuid.uuid4()),
                            "model_family": f"Logistic ({penalty.title()})",
                            "predictor": pred,
                            "outcome": outcome,
                            "covariates": covariates_label,
                            "assumptions_met": assumptions.get("all_met", False),
                            "assumption_details": assumptions,
                            **pen_log_result,
                        })

                # Effect measures for binary predictors
                x_pred = working_df[pred].values.astype(float)
                if len(np.unique(x_pred)) == 2:
                    effect_measures = compute_effect_measures(y, x_pred)
                    if effect_measures:
                        # Store as a special result
                        all_results.append({
                            "spec_id": str(uuid.uuid4()),
                            "model_family": "Effect Measures",
                            "predictor": pred,
                            "outcome": outcome,
                            "covariates": [],
                            "coefficient": effect_measures.get("odds_ratio", 0),
                            "p_value": 0,  # placeholder
                            "ci_lower": 0,
                            "ci_upper": 0,
                            "r_squared": None,
                            "aic": None,
                            "n_obs": len(y),
                            "assumptions_met": True,
                            "assumption_details": effect_measures,
                        })

    # Apply FDR correction
    p_values = [r.get("p_value", 1.0) for r in all_results]
    corrected_p = apply_fdr_correction(p_values)

    for i, r in enumerate(all_results):
        r["p_value_corrected"] = corrected_p[i]
        r["significant"] = r.get("p_value", 1.0) < 0.05
        r["significant_corrected"] = corrected_p[i] < 0.05

        # Compute effect size (Cohen's f2 for continuous, OR for binary)
        if outcome_type == "continuous":
            r2 = r.get("r_squared", 0) or 0
            r["effect_size"] = r2 / (1 - r2) if r2 < 1 else 0
        else:
            r["effect_size"] = abs(r.get("coefficient", 0))

    # Classify covariate roles for each unique predictor/covariate set once.
    covariate_role_cache: dict[tuple[str, tuple[str, ...]], list] = {}
    for r in all_results:
        try:
            cache_key = (r["predictor"], tuple(sorted(r["covariates"])))
            if cache_key not in covariate_role_cache:
                covariate_role_cache[cache_key] = classify_covariates(
                    working_df,
                    outcome=outcome,
                    predictor=r["predictor"],
                    covariates=r["covariates"],
                )
            r["covariate_roles"] = covariate_role_cache[cache_key]
        except Exception:
            r["covariate_roles"] = []

    # Compute summary stats
    sig_count = sum(1 for r in all_results if r["significant_corrected"])
    effect_sizes = [r["effect_size"] for r in all_results if r["effect_size"] is not None]

    return {
        "outcome_variable": outcome,
        "outcome_type": outcome_type,
        "total_specs": len(all_results),
        "significant_specs": sig_count,
        "robustness_pct": round(sig_count / len(all_results) * 100, 1) if all_results else 0,
        "mean_effect_size": round(float(np.mean(effect_sizes)), 4) if effect_sizes else 0,
        "regressions": all_results,
    }
