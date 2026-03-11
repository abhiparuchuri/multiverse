"""
Modal serverless compute functions for multiverse analysis.

Offloads CPU-intensive work (regressions, classifiers, plot generation)
to Modal's cloud infrastructure. FastAPI remains the orchestrator.

Usage:
  - Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET env vars (from modal.com dashboard)
  - Set USE_MODAL=1 to enable Modal; otherwise falls back to local execution
"""

import io
import os
import json
import base64
from pathlib import Path
from typing import Optional

import modal

# ── Modal App & Image ──────────────────────────────────────────────────────

app = modal.App("multiverse-analysis")

# Build a container image with all scientific Python deps + local source
analysis_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "pandas",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "matplotlib",
        "xgboost",
    )
    .workdir("/root")
    .add_local_file(
        str(Path(__file__).parent / "analysis_engine.py"),
        "/root/analysis_engine.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "classifier_engine.py"),
        "/root/classifier_engine.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "plot_generator.py"),
        "/root/plot_generator.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "covariate_roles.py"),
        "/root/covariate_roles.py",
    )
)


# ── Remote Analysis Function ──────────────────────────────────────────────

@app.function(image=analysis_image, timeout=300)
def run_analysis_remote(
    df_json: str,
    outcome: str,
    predictors: list,
    confounders: list,
    hypothesis: str = "",
) -> dict:
    """Run multiverse analysis on Modal."""
    import pandas as pd
    from analysis_engine import run_multiverse_analysis

    df = pd.read_json(io.StringIO(df_json), orient="split")
    results = run_multiverse_analysis(
        df=df,
        outcome=outcome,
        predictors=predictors,
        confounders=confounders,
        hypothesis=hypothesis,
    )
    return results


# ── Remote Classifier Function ─────────────────────────────────────────────

@app.function(image=analysis_image, timeout=300)
def run_classifiers_remote(
    df_json: str,
    outcome: str,
    predictors: list,
    confounders: list,
) -> dict:
    """Run classifier analysis on Modal."""
    import pandas as pd
    from classifier_engine import run_classifiers

    df = pd.read_json(io.StringIO(df_json), orient="split")
    return run_classifiers(
        df=df,
        outcome=outcome,
        predictors=predictors,
        confounders=confounders,
    )


# ── Remote Plot Generation ─────────────────────────────────────────────────

@app.function(image=analysis_image, timeout=300)
def generate_plots_remote(
    results_json: str,
    df_json: Optional[str] = None,
) -> dict:
    """Generate all regression plots + DAGs on Modal.

    Returns a dict of {filename: base64_png_bytes} so the caller can
    write them to disk for StaticFiles serving.
    """
    import json as json_mod
    import pandas as pd

    from plot_generator import (
        generate_all_plots,
        PLOT_DIR,
    )

    PLOT_DIR.mkdir(exist_ok=True)

    results = json_mod.loads(results_json)
    df = pd.read_json(io.StringIO(df_json), orient="split") if df_json else None

    plot_map, dag_map = generate_all_plots(results, df=df)

    # Read generated PNGs and encode as base64 for transfer.
    # Collect filenames from both maps explicitly (same spec_id keys exist in both).
    plot_files = {}
    filenames = set(plot_map.values()) | set(dag_map.values())
    for filename in filenames:
        filepath = PLOT_DIR / filename
        if filepath.exists():
            plot_files[filename] = base64.b64encode(filepath.read_bytes()).decode()

    return {
        "plot_map": plot_map,
        "dag_map": dag_map,
        "plot_files": plot_files,
    }


@app.function(image=analysis_image, timeout=120)
def generate_single_plot_remote(
    plot_type: str,
    data_json: str,
    df_json: Optional[str] = None,
) -> dict:
    """Generate a single plot on Modal.

    plot_type: "feature_importance" | "distribution" | "dag"
    data_json: JSON-serialized input data for the plot function
    df_json: DataFrame as JSON (needed for distribution plots)

    Returns {filename: str, png_b64: str}
    """
    import json as json_mod
    import pandas as pd

    from plot_generator import (
        generate_feature_importance_plot,
        generate_distribution_plot,
        generate_dag_plot,
        PLOT_DIR,
    )

    PLOT_DIR.mkdir(exist_ok=True)
    data = json_mod.loads(data_json)

    if plot_type == "feature_importance":
        filename = generate_feature_importance_plot(data)
    elif plot_type == "distribution":
        df = pd.read_json(io.StringIO(df_json), orient="split") if df_json else None
        if df is None:
            raise ValueError("df_json required for distribution plots")
        filename = generate_distribution_plot(df, data["col_name"], data["role"])
    elif plot_type == "dag":
        filename = generate_dag_plot(
            outcome=data["outcome"],
            predictors=data["predictors"],
            covariates=data["covariates"],
            spec_id=data["spec_id"],
            covariate_roles=data.get("covariate_roles", []),
        )
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    filepath = PLOT_DIR / filename
    png_b64 = base64.b64encode(filepath.read_bytes()).decode() if filepath.exists() else ""

    return {"filename": filename, "png_b64": png_b64}


@app.function(image=analysis_image, timeout=300)
def generate_classifier_artifacts_remote(
    classifier_results_json: str,
    outcome: str,
) -> dict:
    """
    Batch-generate classifier feature-importance plots and DAGs in one remote call.

    Returns:
      {
        "classifier_plot_map": {spec_id: filename},
        "classifier_dag_map": {spec_id: filename},
        "plot_files": {filename: base64_png}
      }
    """
    import json as json_mod

    from plot_generator import (
        generate_feature_importance_plot,
        generate_dag_plot,
        PLOT_DIR,
    )

    PLOT_DIR.mkdir(exist_ok=True)
    classifier_results = json_mod.loads(classifier_results_json)

    classifier_plot_map = {}
    classifier_dag_map = {}

    for cr in classifier_results:
        spec_id = cr.get("spec_id")
        if not spec_id:
            continue
        try:
            plot_file = generate_feature_importance_plot(cr)
            classifier_plot_map[spec_id] = plot_file
        except Exception:
            pass
        try:
            dag_file = generate_dag_plot(
                outcome=outcome,
                predictors=cr.get("predictors_used", []),
                covariates=cr.get("covariates_used", []),
                spec_id=f"clf_{spec_id}",
            )
            classifier_dag_map[spec_id] = dag_file
        except Exception:
            pass

    plot_files = {}
    filenames = set(classifier_plot_map.values()) | set(classifier_dag_map.values())
    for filename in filenames:
        filepath = PLOT_DIR / filename
        if filepath.exists():
            plot_files[filename] = base64.b64encode(filepath.read_bytes()).decode()

    return {
        "classifier_plot_map": classifier_plot_map,
        "classifier_dag_map": classifier_dag_map,
        "plot_files": plot_files,
    }


@app.function(image=analysis_image, timeout=300)
def generate_distribution_plots_remote(
    df_json: str,
    variable_roles_json: str,
) -> dict:
    """
    Batch-generate distribution plots in one remote call.

    variable_roles_json should serialize a list of:
      [{"col_name": str, "role": "outcome"|"predictor"|"covariate"}]
    """
    import json as json_mod
    import pandas as pd

    from plot_generator import (
        generate_distribution_plot,
        PLOT_DIR,
    )

    PLOT_DIR.mkdir(exist_ok=True)
    df = pd.read_json(io.StringIO(df_json), orient="split")
    variable_roles = json_mod.loads(variable_roles_json)

    distribution_plot_map = {}
    for item in variable_roles:
        col_name = item.get("col_name")
        role = item.get("role", "covariate")
        if not col_name or col_name not in df.columns:
            continue
        try:
            filename = generate_distribution_plot(df, col_name, role)
            distribution_plot_map[col_name] = filename
        except Exception:
            continue

    plot_files = {}
    for _key, filename in distribution_plot_map.items():
        filepath = PLOT_DIR / filename
        if filepath.exists():
            plot_files[filename] = base64.b64encode(filepath.read_bytes()).decode()

    return {
        "distribution_plot_map": distribution_plot_map,
        "plot_files": plot_files,
    }
