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
    "axes.grid": True,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.8,
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
    subtitle = model_family
    if covariates:
        subtitle += f"  |  adjusted for {', '.join(covariates[:3])}"
        if len(covariates) > 3:
            subtitle += f" +{len(covariates) - 3} more"

    ax.set_title(f"{predictor} → {outcome}", fontsize=11, fontweight="600", pad=10, loc="left", color=COLOR_TEXT)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=8, color=COLOR_SUBTITLE, va="bottom")
    ax.text(
        1.0, 1.02,
        f"β = {coeff:.4f}  |  {p_str}  |  n = {len(x)}",
        transform=ax.transAxes, ha="right", fontsize=8, color=COLOR_SUBTITLE, va="bottom",
    )

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


def generate_all_plots(results: dict, df: Optional[pd.DataFrame] = None) -> dict:
    """Generate all plots for the analysis results.
    Returns a dict mapping spec_id -> plot filename, plus 'spec_curve' key."""
    # Clear previous plots
    for f in PLOT_DIR.glob("*.png"):
        f.unlink()

    plot_map = {}

    # Individual regression plots (scatter + model overlay when data is available)
    for reg in results.get("regressions", []):
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

    return plot_map
