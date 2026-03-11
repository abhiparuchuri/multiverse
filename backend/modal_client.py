"""
Modal client helpers for the FastAPI backend.

When USE_MODAL=1 is set, the /analyze endpoint calls Modal functions
instead of running compute locally. Results (including plot PNGs) are
transferred back and written to the local plots/ directory.

When USE_MODAL is not set, falls back to direct local imports.
"""

import os
import io
import json
import base64
from pathlib import Path

USE_MODAL = os.environ.get("USE_MODAL", "").strip() in ("1", "true", "yes")

# Local plot directory (same as plot_generator.py)
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def _write_plot_files(plot_files: dict):
    """Write base64-encoded PNGs from Modal to local plots/ directory."""
    for filename, b64_data in plot_files.items():
        if b64_data:
            filepath = PLOT_DIR / filename
            filepath.write_bytes(base64.b64decode(b64_data))


def run_analysis(df, outcome, predictors, confounders, hypothesis=""):
    """Run multiverse analysis — on Modal if enabled, else locally."""
    if USE_MODAL:
        import modal
        f = modal.Function.from_name("multiverse-analysis", "run_analysis_remote")
        df_json = df.to_json(orient="split")
        results = f.remote(
            df_json=df_json,
            outcome=outcome,
            predictors=predictors,
            confounders=confounders,
            hypothesis=hypothesis,
        )
        return results
    else:
        from analysis_engine import run_multiverse_analysis
        return run_multiverse_analysis(
            df=df,
            outcome=outcome,
            predictors=predictors,
            confounders=confounders,
            hypothesis=hypothesis,
        )


def run_classifiers_call(df, outcome, predictors, confounders):
    """Run classifier analysis — on Modal if enabled, else locally."""
    if USE_MODAL:
        import modal
        f = modal.Function.from_name("multiverse-analysis", "run_classifiers_remote")
        df_json = df.to_json(orient="split")
        return f.remote(
            df_json=df_json,
            outcome=outcome,
            predictors=predictors,
            confounders=confounders,
        )
    else:
        from classifier_engine import run_classifiers
        return run_classifiers(
            df=df,
            outcome=outcome,
            predictors=predictors,
            confounders=confounders,
        )


def generate_all_plots_call(results, df=None):
    """Generate all regression plots + DAGs — on Modal if enabled, else locally.

    Returns (plot_map, dag_map) just like the local function.
    """
    if USE_MODAL:
        import modal
        f = modal.Function.from_name("multiverse-analysis", "generate_plots_remote")
        results_json = json.dumps(results, default=str)
        df_json = df.to_json(orient="split") if df is not None else None
        remote_result = f.remote(results_json=results_json, df_json=df_json)
        # Write plot PNGs to local disk for StaticFiles serving
        _write_plot_files(remote_result.get("plot_files", {}))
        return remote_result["plot_map"], remote_result["dag_map"]
    else:
        from plot_generator import generate_all_plots
        return generate_all_plots(results, df=df)


def generate_feature_importance_plot_call(classifier_result):
    """Generate a single feature importance plot."""
    if USE_MODAL:
        import modal
        f = modal.Function.from_name("multiverse-analysis", "generate_single_plot_remote")
        result = f.remote(
            plot_type="feature_importance",
            data_json=json.dumps(classifier_result, default=str),
        )
        _write_plot_files({result["filename"]: result["png_b64"]})
        return result["filename"]
    else:
        from plot_generator import generate_feature_importance_plot
        return generate_feature_importance_plot(classifier_result)


def generate_distribution_plot_call(df, col_name, role):
    """Generate a single distribution plot."""
    if USE_MODAL:
        import modal
        f = modal.Function.from_name("multiverse-analysis", "generate_single_plot_remote")
        result = f.remote(
            plot_type="distribution",
            data_json=json.dumps({"col_name": col_name, "role": role}),
            df_json=df.to_json(orient="split"),
        )
        _write_plot_files({result["filename"]: result["png_b64"]})
        return result["filename"]
    else:
        from plot_generator import generate_distribution_plot
        return generate_distribution_plot(df, col_name, role)


def generate_dag_plot_call(outcome, predictors, covariates, spec_id, covariate_roles=None):
    """Generate a single DAG plot."""
    if USE_MODAL:
        import modal
        f = modal.Function.from_name("multiverse-analysis", "generate_single_plot_remote")
        result = f.remote(
            plot_type="dag",
            data_json=json.dumps({
                "outcome": outcome,
                "predictors": predictors,
                "covariates": covariates,
                "spec_id": spec_id,
                "covariate_roles": covariate_roles or [],
            }),
        )
        _write_plot_files({result["filename"]: result["png_b64"]})
        return result["filename"]
    else:
        from plot_generator import generate_dag_plot
        return generate_dag_plot(
            outcome=outcome,
            predictors=predictors,
            covariates=covariates,
            spec_id=spec_id,
            covariate_roles=covariate_roles or [],
        )


def generate_classifier_artifacts_call(classifier_results, outcome):
    """
    Batch-generate classifier feature-importance plots and DAGs.

    Returns: (classifier_plot_map, classifier_dag_map)
    """
    if USE_MODAL:
        import modal
        f = modal.Function.from_name("multiverse-analysis", "generate_classifier_artifacts_remote")
        remote_result = f.remote(
            classifier_results_json=json.dumps(classifier_results, default=str),
            outcome=outcome,
        )
        _write_plot_files(remote_result.get("plot_files", {}))
        return (
            remote_result.get("classifier_plot_map", {}),
            remote_result.get("classifier_dag_map", {}),
        )
    else:
        from plot_generator import generate_feature_importance_plot, generate_dag_plot

        classifier_plot_map = {}
        classifier_dag_map = {}

        for cr in classifier_results:
            try:
                fname = generate_feature_importance_plot(cr)
                classifier_plot_map[cr["spec_id"]] = fname
            except Exception:
                pass
            try:
                dag_fname = generate_dag_plot(
                    outcome=outcome,
                    predictors=cr.get("predictors_used", []),
                    covariates=cr.get("covariates_used", []),
                    spec_id=f"clf_{cr['spec_id']}",
                )
                classifier_dag_map[cr["spec_id"]] = dag_fname
            except Exception:
                pass

        return classifier_plot_map, classifier_dag_map


def generate_distribution_plots_call(df, variable_roles):
    """
    Batch-generate distribution plots.

    variable_roles: list of {"col_name": str, "role": str}
    Returns: distribution_plot_map dict (col_name -> filename)
    """
    if USE_MODAL:
        import modal
        f = modal.Function.from_name("multiverse-analysis", "generate_distribution_plots_remote")
        remote_result = f.remote(
            df_json=df.to_json(orient="split"),
            variable_roles_json=json.dumps(variable_roles, default=str),
        )
        _write_plot_files(remote_result.get("plot_files", {}))
        return remote_result.get("distribution_plot_map", {})
    else:
        from plot_generator import generate_distribution_plot

        distribution_plot_map = {}
        for item in variable_roles:
            col_name = item.get("col_name")
            role = item.get("role", "covariate")
            if not col_name or col_name not in df.columns:
                continue
            try:
                distribution_plot_map[col_name] = generate_distribution_plot(df, col_name, role)
            except Exception:
                pass
        return distribution_plot_map
