"""
Publication-ready plot generation for multiverse analysis results.
Uses matplotlib with a clean scientific aesthetic.
"""

import os
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker
from scipy.special import expit, logit

# Output directory for generated plots
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# ── Style constants ─────────────────────────────────────────────────────────

FONT_FAMILY = "Inter, Helvetica Neue, Arial, sans-serif"
COLOR_SIG = "#16a34a"
COLOR_NS = "#94a3b8"
COLOR_AXIS = "#64748b"
COLOR_GRID = "#e2e8f0"
COLOR_BG = "#ffffff"
COLOR_TEXT = "#0f172a"
COLOR_SUBTITLE = "#64748b"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_GRID,
    "axes.grid": False,
    "text.color": COLOR_TEXT,
    "axes.labelcolor": COLOR_AXIS,
    "xtick.color": COLOR_AXIS,
    "ytick.color": COLOR_AXIS,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "figure.dpi": 200,
})


def generate_forest_plot(regression: dict) -> str:
    """Generate a publication-ready forest plot for a single regression.
    Returns the filename (relative to PLOT_DIR)."""
    coeff = regression["coefficient"]
    ci_lower = regression["ci_lower"]
    ci_upper = regression["ci_upper"]
    sig = regression.get("significant_corrected", False)
    predictor = regression["predictor"]
    outcome = regression["outcome"]
    model_family = regression["model_family"]
    p_val = regression.get("p_value_corrected", regression.get("p_value", 1.0))
    covariates = regression.get("covariates", [])

    color = COLOR_SIG if sig else COLOR_NS

    fig, ax = plt.subplots(figsize=(5.5, 2.0))

    # CI whisker
    ax.plot([ci_lower, ci_upper], [0, 0], color=color, linewidth=2, solid_capstyle="round", zorder=2)

    # CI caps
    cap_h = 0.15
    ax.plot([ci_lower, ci_lower], [-cap_h, cap_h], color=color, linewidth=1.5, solid_capstyle="round", zorder=2)
    ax.plot([ci_upper, ci_upper], [-cap_h, cap_h], color=color, linewidth=1.5, solid_capstyle="round", zorder=2)

    # Point estimate
    ax.scatter([coeff], [0], color=color, s=70, zorder=3, edgecolors="white", linewidths=1.2)

    # Null reference line
    ax.axvline(x=0, color="#cbd5e1", linestyle="--", linewidth=1, zorder=1)

    # Axis formatting
    abs_max = max(abs(ci_lower), abs(ci_upper), abs(coeff)) * 1.4
    ax.set_xlim(-abs_max, abs_max)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRID)

    ax.set_xlabel("Coefficient estimate", fontsize=9, labelpad=6)

    # Annotation: coefficient value and CI
    p_str = f"p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    annotation = f"β = {coeff:.4f}  [{ci_lower:.3f}, {ci_upper:.3f}]  {p_str}"
    ax.text(
        0.5, -0.42, annotation,
        transform=ax.transAxes, ha="center", fontsize=8,
        color=COLOR_SUBTITLE, fontstyle="italic"
    )

    # Title
    title = f"{predictor} → {outcome}"
    subtitle = model_family
    if covariates:
        subtitle += f"  |  adjusted for {', '.join(covariates[:3])}"
        if len(covariates) > 3:
            subtitle += f" +{len(covariates) - 3} more"

    ax.set_title(title, fontsize=11, fontweight="600", pad=10, loc="left", color=COLOR_TEXT)
    ax.text(
        0.0, 1.02, subtitle,
        transform=ax.transAxes, fontsize=8, color=COLOR_SUBTITLE,
        va="bottom"
    )

    filename = f"{regression['spec_id']}.png"
    filepath = PLOT_DIR / filename
    fig.savefig(filepath, facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    return filename


def generate_scatter_model_plot(df: pd.DataFrame, regression: dict) -> str:
    """Generate scatter plot with model overlay using raw data for predictor/outcome."""
    predictor = regression["predictor"]
    outcome = regression["outcome"]
    coeff = regression.get("coefficient", 0.0)
    model_family = regression.get("model_family", "Model")
    sig = regression.get("significant_corrected", False)
    p_val = regression.get("p_value_corrected", regression.get("p_value", 1.0))
    covariates = regression.get("covariates", [])

    if predictor not in df.columns or outcome not in df.columns:
        raise ValueError("Predictor or outcome column missing in dataframe")

    data = df[[predictor, outcome]].dropna()
    if len(data) < 5:
        raise ValueError("Not enough non-missing points for scatter plot")

    x = pd.to_numeric(data[predictor], errors="coerce")
    y = pd.to_numeric(data[outcome], errors="coerce")
    mask = x.notna() & y.notna()
    x = x[mask].to_numpy()
    y = y[mask].to_numpy()
    if len(x) < 5:
        raise ValueError("Not enough numeric points for scatter plot")

    color = COLOR_SIG if sig else COLOR_NS
    fig, ax = plt.subplots(figsize=(5.5, 3.3))

    is_binary_y = np.unique(y).size <= 2 and set(np.unique(y)).issubset({0, 1})
    if is_binary_y:
        jitter = np.random.default_rng(42).uniform(-0.03, 0.03, size=len(y))
        y_plot = np.clip(y + jitter, -0.05, 1.05)
        ax.scatter(x, y_plot, s=14, alpha=0.35, color=COLOR_NS, edgecolors="none", zorder=1)

        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 120)
        base_p = float(np.clip(np.mean(y), 1e-4, 1 - 1e-4))
        y_hat = expit(logit(base_p) + coeff * (x_grid - np.mean(x)))
        ax.plot(x_grid, y_hat, color=color, linewidth=2.2, zorder=3)
        ax.set_ylim(-0.08, 1.08)
        ax.set_ylabel(f"{outcome} (probability)", fontsize=9, labelpad=8)
    else:
        ax.scatter(x, y, s=14, alpha=0.35, color=COLOR_NS, edgecolors="none", zorder=1)
        x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 120)
        y_hat = np.mean(y) + coeff * (x_grid - np.mean(x))
        ax.plot(x_grid, y_hat, color=color, linewidth=2.2, zorder=3)
        ax.set_ylabel(outcome, fontsize=9, labelpad=8)

    ax.set_xlabel(predictor, fontsize=9, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_color(COLOR_GRID)

    p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
    stats_line = f"β = {coeff:.4f}  |  {p_str}  |  n = {len(x)}"

    subtitle = model_family
    if covariates:
        subtitle += f"  |  adjusted for {', '.join(covariates[:2])}"
        if len(covariates) > 2:
            subtitle += f" +{len(covariates) - 2} more"

    ax.set_title(f"{predictor} → {outcome}", fontsize=11, fontweight="600", pad=36, loc="left", color=COLOR_TEXT)
    ax.text(0.0, 1.15, stats_line, transform=ax.transAxes, fontsize=8, color=COLOR_SUBTITLE, va="bottom")
    ax.text(0.0, 1.03, subtitle, transform=ax.transAxes, fontsize=8, color=COLOR_SUBTITLE, va="bottom")

    filename = f"{regression['spec_id']}.png"
    filepath = PLOT_DIR / filename
    fig.savefig(filepath, facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    return filename


def generate_spec_curve_plot(regressions: list[dict], outcome: str, outcome_type: str) -> str:
    """Generate a specification curve plot showing all regressions sorted by effect size.
    Returns the filename."""
    sorted_regs = sorted(regressions, key=lambda r: r["coefficient"])
    n = len(sorted_regs)

    coefficients = [r["coefficient"] for r in sorted_regs]
    ci_lowers = [r["ci_lower"] for r in sorted_regs]
    ci_uppers = [r["ci_upper"] for r in sorted_regs]
    significant = [r.get("significant_corrected", False) for r in sorted_regs]
    colors = [COLOR_SIG if s else COLOR_NS for s in significant]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Error bars (CI whiskers)
    for i in range(n):
        ax.plot(
            [i, i], [ci_lowers[i], ci_uppers[i]],
            color=colors[i], linewidth=0.8, alpha=0.4, zorder=1
        )

    # Points
    ax.scatter(
        range(n), coefficients,
        c=colors, s=12, zorder=2, edgecolors="none", alpha=0.85
    )

    # Null reference
    ax.axhline(y=0, color="#cbd5e1", linestyle="--", linewidth=1, zorder=0)

    # Styling
    ax.set_xlabel("Specification (sorted by effect size)", fontsize=9, labelpad=8)
    ax.set_ylabel("Coefficient", fontsize=9, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_color(COLOR_GRID)

    sig_count = sum(significant)
    sig_pct = round(sig_count / n * 100, 1) if n > 0 else 0

    ax.set_title(
        f"Specification Curve: {outcome} ({outcome_type})",
        fontsize=12, fontweight="600", pad=14, loc="left", color=COLOR_TEXT
    )
    ax.text(
        0.0, 1.02,
        f"{n} specifications  |  {sig_count} significant ({sig_pct}%)  |  green = sig. after FDR",
        transform=ax.transAxes, fontsize=8, color=COLOR_SUBTITLE, va="bottom"
    )

    # Legend dots
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_SIG, markersize=6, label="Significant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_NS, markersize=6, label="Not significant"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right", fontsize=8,
        frameon=True, fancybox=True, framealpha=0.9,
        edgecolor=COLOR_GRID
    )

    filename = "spec_curve.png"
    filepath = PLOT_DIR / filename
    fig.savefig(filepath, facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    return filename


def generate_feature_importance_plot(classifier_result: dict) -> str:
    """Generate a horizontal bar chart of feature importances for a classifier.
    Returns the filename."""
    fi = classifier_result["feature_importance"]
    clf_name = classifier_result["classifier"]
    features = classifier_result.get("features", list(fi.keys()))

    # Sort by importance
    sorted_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    n_features = len(names)
    fig_height = max(1.8, 0.35 * n_features + 0.8)
    fig, ax = plt.subplots(figsize=(5.5, fig_height))

    y_pos = np.arange(n_features)
    colors = [COLOR_SIG if v == max(values) else "#64748b" for v in values]

    ax.barh(y_pos, values, height=0.6, color=colors, edgecolor="none", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=9, labelpad=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_GRID)
    ax.tick_params(axis="y", length=0)

    acc = classifier_result.get("accuracy", 0)
    auc = classifier_result.get("auc")
    stats_parts = [f"Accuracy: {acc:.1%}"]
    if auc is not None:
        stats_parts.append(f"AUC: {auc:.3f}")
    stats_parts.append(f"n = {classifier_result.get('n_obs', '?')}")

    ax.set_title(clf_name, fontsize=11, fontweight="600", pad=20, loc="left", color=COLOR_TEXT)
    ax.text(0.0, 1.02, "  |  ".join(stats_parts),
            transform=ax.transAxes, fontsize=8, color=COLOR_SUBTITLE, va="bottom")

    filename = f"{classifier_result['spec_id']}.png"
    filepath = PLOT_DIR / filename
    fig.savefig(filepath, facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    return filename


def generate_distribution_plot(df: pd.DataFrame, col_name: str, role: str) -> str:
    """Generate a distribution plot for a single variable.
    role is 'outcome', 'predictor', or 'covariate'.
    Returns the filename."""
    data = df[col_name].dropna()
    if len(data) == 0:
        raise ValueError(f"No data for {col_name}")

    is_numeric = np.issubdtype(data.dtype, np.number)
    n_unique = data.nunique()
    is_binary = is_numeric and n_unique <= 2

    role_colors = {
        "outcome": "#8b5cf6",
        "predictor": COLOR_SIG,
        "covariate": "#64748b",
    }
    color = role_colors.get(role, "#64748b")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    if is_binary:
        counts = data.value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values, color=color, alpha=0.8,
               width=0.5, edgecolor="none")
        ax.set_ylabel("Count", fontsize=9, labelpad=8)
    elif is_numeric:
        n_bins = min(30, max(10, int(np.sqrt(len(data)))))
        ax.hist(data.values, bins=n_bins, color=color, alpha=0.7, edgecolor="white",
                linewidth=0.5)
        ax.set_ylabel("Frequency", fontsize=9, labelpad=8)
        # Add KDE overlay
        try:
            from scipy.stats import gaussian_kde
            kde_x = np.linspace(float(data.min()), float(data.max()), 200)
            kde = gaussian_kde(data.values)
            kde_y = kde(kde_x)
            ax2 = ax.twinx()
            ax2.plot(kde_x, kde_y, color=color, linewidth=1.5, alpha=0.9)
            ax2.set_yticks([])
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
        except Exception:
            pass
    else:
        # Categorical
        counts = data.value_counts().head(15)
        ax.barh(range(len(counts)), counts.values, color=color, alpha=0.8,
                height=0.6, edgecolor="none")
        ax.set_yticks(range(len(counts)))
        ax.set_yticklabels(counts.index.astype(str), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Count", fontsize=9, labelpad=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COLOR_GRID)
    ax.spines["bottom"].set_color(COLOR_GRID)

    # Stats line
    stats_parts = [f"n = {len(data)}"]
    if is_numeric and not is_binary:
        stats_parts.extend([
            f"mean = {data.mean():.2f}",
            f"std = {data.std():.2f}",
            f"median = {data.median():.2f}",
        ])
        skew = data.skew()
        if abs(skew) > 1:
            stats_parts.append(f"skew = {skew:.2f}")
    elif is_binary:
        pct = data.mean() * 100
        stats_parts.append(f"{pct:.1f}% positive")
    else:
        stats_parts.append(f"{n_unique} categories")
    missing_pct = df[col_name].isna().mean() * 100
    if missing_pct > 0:
        stats_parts.append(f"{missing_pct:.1f}% missing")

    role_label = role.capitalize()
    ax.set_title(col_name, fontsize=11, fontweight="600", pad=20, loc="left", color=COLOR_TEXT)
    ax.text(0.0, 1.08, "  |  ".join(stats_parts),
            transform=ax.transAxes, fontsize=8, color=COLOR_SUBTITLE, va="bottom")
    ax.text(0.0, 1.02, role_label,
            transform=ax.transAxes, fontsize=8, color=color, va="bottom", fontweight="600")

    filename = f"dist_{col_name}.png"
    filepath = PLOT_DIR / filename
    fig.savefig(filepath, facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    return filename


def generate_dag_plot(
    outcome: str,
    predictors: list[str],
    covariates: list[str],
    spec_id: str,
    covariate_roles: list = None,
) -> str:
    """Generate a publication-ready DAG showing predictor→outcome and covariate→outcome
    relationships with causal role classification. Returns the filename.

    covariate_roles: list of {"variable": str, "role": str, "coeff_change_pct": float}
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # Build role lookup from classification results
    role_map = {}
    if covariate_roles:
        for cr in covariate_roles:
            role_map[cr["variable"]] = cr

    # Role → visual style
    ROLE_STYLES = {
        "confounder": {"face": "#ea580c", "edge": "#c2410c", "arrow": "#ea580c", "label": "Confounder"},
        "mediator":   {"face": "#2563eb", "edge": "#1d4ed8", "arrow": "#2563eb", "label": "Mediator"},
        "precision":  {"face": "#64748b", "edge": "#475569", "arrow": "#94a3b8", "label": "Precision"},
        "neutral":    {"face": "#64748b", "edge": "#475569", "arrow": "#94a3b8", "label": "Neutral"},
    }

    all_left = predictors + covariates
    n_left = len(all_left)

    # Layout parameters
    node_w, node_h = 2.0, 0.5
    gap_y = 0.85
    left_x = 0.0
    right_x = 5.5
    total_h = max(2.0, (n_left - 1) * gap_y + node_h + 1.0)
    fig_w = 8.0
    fig_h = max(2.0, total_h + 0.8)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(-0.8, right_x + node_w + 0.8)
    ax.set_ylim(-0.4, total_h + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")

    # Compute Y positions for left-side nodes (centered vertically)
    left_ys = []
    if n_left > 0:
        block_h = (n_left - 1) * gap_y
        start_y = (total_h - block_h) / 2
        for i in range(n_left):
            left_ys.append(start_y + i * gap_y)
        left_ys.reverse()

    # Outcome node (right side, centered)
    outcome_cy = total_h / 2
    outcome_rect = FancyBboxPatch(
        (right_x, outcome_cy - node_h / 2), node_w, node_h,
        boxstyle="round,pad=0.1", facecolor="#8b5cf6", edgecolor="#7c3aed",
        linewidth=1.5, alpha=0.9, zorder=3,
    )
    ax.add_patch(outcome_rect)
    ax.text(
        right_x + node_w / 2, outcome_cy, outcome,
        ha="center", va="center", fontsize=9, fontweight="600",
        color="white", zorder=4,
    )

    # Draw left-side nodes and arrows
    for i, (var, y_pos) in enumerate(zip(all_left, left_ys)):
        is_predictor = var in predictors

        if is_predictor:
            face, edge = "#16a34a", "#15803d"
            arrow_color = "#16a34a"
            lw = 1.8
            linestyle = "-"
        else:
            role_info = role_map.get(var, {})
            role = role_info.get("role", "neutral")
            style = ROLE_STYLES.get(role, ROLE_STYLES["neutral"])
            face, edge = style["face"], style["edge"]
            arrow_color = style["arrow"]
            lw = 1.5 if role in ("confounder", "mediator") else 1.2
            linestyle = "-" if role in ("confounder", "mediator") else "--"

        rect = FancyBboxPatch(
            (left_x, y_pos - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.1", facecolor=face, edgecolor=edge,
            linewidth=1.5, alpha=0.9, zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            left_x + node_w / 2, y_pos, var,
            ha="center", va="center", fontsize=9, fontweight="600",
            color="white", zorder=4,
        )

        # Arrow from left node to outcome
        arrow = FancyArrowPatch(
            (left_x + node_w + 0.05, y_pos),
            (right_x - 0.05, outcome_cy),
            arrowstyle="-|>",
            mutation_scale=14,
            color=arrow_color,
            linewidth=lw,
            linestyle=linestyle,
            zorder=2,
            connectionstyle="arc3,rad=0",
        )
        ax.add_patch(arrow)

        # Annotate covariate arrows with role + coefficient change %
        if not is_predictor:
            role_info = role_map.get(var, {})
            role = role_info.get("role", "neutral")
            pct = role_info.get("coeff_change_pct", 0.0)
            style = ROLE_STYLES.get(role, ROLE_STYLES["neutral"])

            if role in ("confounder", "mediator") and abs(pct) > 0:
                label_text = f"{style['label']} ({pct:+.0f}%)"
            elif role == "precision":
                label_text = style["label"]
            else:
                label_text = ""

            if label_text:
                mid_x = (left_x + node_w + right_x) / 2
                mid_y = (y_pos + outcome_cy) / 2
                ax.text(
                    mid_x, mid_y + 0.18, label_text,
                    ha="center", va="bottom", fontsize=6.5,
                    color=arrow_color, fontstyle="italic", zorder=5,
                )

    # Legend — include only roles that appear
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#16a34a",
               markersize=8, label="Predictor"),
    ]
    seen_roles = set()
    for var in covariates:
        role_info = role_map.get(var, {})
        role = role_info.get("role", "neutral")
        seen_roles.add(role)

    for role_key in ["confounder", "mediator", "precision", "neutral"]:
        if role_key in seen_roles:
            style = ROLE_STYLES[role_key]
            legend_items.append(
                Line2D([0], [0], marker="s", color="w", markerfacecolor=style["face"],
                       markersize=8, label=style["label"]),
            )

    legend_items.append(
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#8b5cf6",
               markersize=8, label="Outcome"),
    )
    ax.legend(
        handles=legend_items, loc="lower center", fontsize=7,
        frameon=True, fancybox=True, framealpha=0.9,
        edgecolor=COLOR_GRID, ncol=min(len(legend_items), 5),
        bbox_to_anchor=(0.5, -0.02),
    )

    filename = f"dag_{spec_id}.png"
    filepath = PLOT_DIR / filename
    fig.savefig(filepath, facecolor=COLOR_BG, edgecolor="none")
    plt.close(fig)
    return filename


def generate_all_plots(results: dict, df: Optional[pd.DataFrame] = None) -> dict:
    """Generate all plots for the analysis results.
    Returns a dict mapping spec_id -> plot filename, plus 'spec_curve' key."""
    plot_map = {}
    regressions = list(results.get("regressions", []))
    max_regression_plots = int(os.environ.get("ANALYSIS_MAX_REGRESSION_PLOTS", "0") or "0")
    max_dag_plots = int(os.environ.get("ANALYSIS_MAX_DAG_PLOTS", "0") or "0")

    # Prioritize significant specs and low corrected p-values for eager plotting.
    ranked = sorted(
        regressions,
        key=lambda r: (
            0 if r.get("significant_corrected") else 1,
            float(r.get("p_value_corrected", 1.0)),
            -abs(float(r.get("coefficient", 0.0))),
        ),
    )
    plot_regs = ranked[:max_regression_plots] if max_regression_plots > 0 else ranked
    dag_regs = ranked[:max_dag_plots] if max_dag_plots > 0 else ranked

    # Individual regression plots (scatter + model overlay when data is available)
    for reg in plot_regs:
        try:
            if df is not None:
                filename = generate_scatter_model_plot(df, reg)
            else:
                filename = generate_forest_plot(reg)
            plot_map[reg["spec_id"]] = filename
        except Exception as e:
            # Fallback to forest plot so UI always has a figure.
            try:
                filename = generate_forest_plot(reg)
                plot_map[reg["spec_id"]] = filename
            except Exception as e2:
                print(f"Plot generation failed for {reg['spec_id']}: {e} / fallback: {e2}")

    # DAG plots for each regression
    dag_map = {}
    for reg in dag_regs:
        try:
            dag_file = generate_dag_plot(
                outcome=reg["outcome"],
                predictors=[reg["predictor"]],
                covariates=reg.get("covariates", []),
                spec_id=reg["spec_id"],
                covariate_roles=reg.get("covariate_roles", []),
            )
            dag_map[reg["spec_id"]] = dag_file
        except Exception as e:
            print(f"DAG generation failed for {reg['spec_id']}: {e}")

    # Specification curve
    try:
        spec_curve = generate_spec_curve_plot(
            results["regressions"],
            results["outcome_variable"],
            results["outcome_type"],
        )
        plot_map["spec_curve"] = spec_curve
    except Exception as e:
        print(f"Spec curve generation failed: {e}")

    return plot_map, dag_map
