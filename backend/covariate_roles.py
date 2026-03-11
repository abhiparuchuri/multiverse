"""
Covariate role classification via change-in-estimate.

For each covariate in a regression specification, compares the predictor's
coefficient WITH vs WITHOUT the covariate. The percentage change determines
the covariate's likely causal role:

  - Confounder:  coefficient changes ≥10% (covariate biases the estimate)
  - Mediator:    coefficient changes ≥10% AND covariate correlates with
                 predictor more strongly than with outcome (lies on causal path)
  - Precision:   coefficient changes <10% but SE shrinks (reduces noise)
  - Neutral:     no meaningful change in coefficient or SE

The confounder/mediator distinction uses a heuristic: if the covariate
correlates more strongly with the predictor than the outcome, it is more
likely a mediator (predictor → covariate → outcome). This is a statistical
signal, not definitive causal proof — domain knowledge is required to confirm.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Optional


CHANGE_THRESHOLD = 0.10  # 10% change-in-estimate threshold


def _safe_ols_coeff(y: np.ndarray, X: np.ndarray, pred_idx: int) -> Optional[tuple]:
    """Run OLS and return (coefficient, std_error) for the predictor, or None."""
    try:
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()
        idx = pred_idx + 1  # +1 for constant
        return float(model.params[idx]), float(model.bse[idx])
    except Exception:
        return None


def classify_covariates(
    df: pd.DataFrame,
    outcome: str,
    predictor: str,
    covariates: list,
) -> list:
    """Classify each covariate's role relative to a predictor→outcome relationship.

    Returns a list of dicts, one per covariate:
      {
        "variable": str,
        "role": "confounder" | "mediator" | "precision" | "neutral",
        "coeff_change_pct": float,  # % change in predictor coefficient
      }
    """
    if not covariates:
        return []

    all_vars = list(set([outcome, predictor] + covariates))
    working = df[all_vars].dropna()
    if len(working) < 20:
        return [{"variable": c, "role": "neutral", "coeff_change_pct": 0.0} for c in covariates]

    y = working[outcome].values.astype(float)

    # Baseline: predictor alone
    X_base = working[[predictor]].values.astype(float)
    base = _safe_ols_coeff(y, X_base, 0)
    if base is None:
        return [{"variable": c, "role": "neutral", "coeff_change_pct": 0.0} for c in covariates]

    base_coeff, base_se = base

    # Avoid division by zero
    if abs(base_coeff) < 1e-12:
        return [{"variable": c, "role": "neutral", "coeff_change_pct": 0.0} for c in covariates]

    results = []
    for cov in covariates:
        if cov == predictor or cov == outcome:
            continue

        # Adjusted: predictor + this covariate
        X_adj = working[[predictor, cov]].values.astype(float)
        adj = _safe_ols_coeff(y, X_adj, 0)

        if adj is None:
            results.append({"variable": cov, "role": "neutral", "coeff_change_pct": 0.0})
            continue

        adj_coeff, adj_se = adj
        pct_change = (adj_coeff - base_coeff) / abs(base_coeff)

        if abs(pct_change) >= CHANGE_THRESHOLD:
            # Coefficient changed substantially — confounder or mediator?
            # Heuristic: if covariate correlates more with predictor than outcome,
            # it likely sits on the causal path (mediator).
            try:
                r_pred = abs(float(working[cov].corr(working[predictor])))
                r_out = abs(float(working[cov].corr(working[outcome])))
                role = "mediator" if r_pred > r_out else "confounder"
            except Exception:
                role = "confounder"
        elif adj_se < base_se * 0.95:
            role = "precision"
        else:
            role = "neutral"

        results.append({
            "variable": cov,
            "role": role,
            "coeff_change_pct": round(pct_change * 100, 1),
        })

    return results
