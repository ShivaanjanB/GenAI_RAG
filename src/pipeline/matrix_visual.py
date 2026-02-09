"""
Matrix visualisation for the strategic fit framework (Task 2 Chunk 3).

This module creates a two-dimensional scatter plot representing the
strategic fit matrix. Each company placement (with valid scores on
both axes) is plotted on an X–Y plane scaled from 0 to 5. A vertical
and horizontal line mark the default thresholds defined in the
configuration. Companies are labelled with a short name for clarity.

If the number of companies to display exceeds a configurable
threshold, the function switches to a numbered points mode. In this
mode each point is annotated with a simple integer identifier and a
separate legend CSV is written to disk mapping each identifier to
company metadata. This avoids excessive label overlap on the plot.

Usage
-----

The primary entry point is the ``generate_matrix_plot`` function. It
takes a matrix configuration object (as defined in
``strategic_framework.py``), a list of ``CompanyPlacement`` instances,
an output directory and an optional threshold for numbered mode.

Example:

    from src.pipeline.matrix_visual import generate_matrix_plot
    from src.pipeline.strategic_framework import load_matrix_config
    from src.pipeline.strategic_models import CompanyPlacement

    config = load_matrix_config(Path('config/strategic_matrix.yaml'))
    placements = [...]  # obtained from JSONL or scoring step
    generate_matrix_plot(config, placements, Path('data/output'))

"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import matplotlib

# Use non‑interactive backend for environments without a display
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from urllib.parse import urlparse  # noqa: E402

from src.pipeline.strategic_framework import MatrixConfig
from src.pipeline.strategic_models import CompanyPlacement


def _shorten_name(name: str) -> str:
    """Return a shortened version of the company name for labelling.

    This helper strips common corporate suffixes (Inc, LLC, Ltd, PLC,
    Corporation) and truncates long names to improve legibility on the
    plot. If the resulting name is still lengthy (> 15 characters),
    only the first word is returned.

    Parameters
    ----------
    name : str
        Full company name.

    Returns
    -------
    str
        Shortened company name.
    """
    if not name:
        return ""
    n = name.strip()
    # Remove corporate suffixes
    suffixes = ["inc", "inc.", "llc", "ltd", "ltd.", "plc", "corp", "corp.", "corporation", "co."]
    parts = n.split()
    if parts:
        last = parts[-1].lower().strip(',.')
        if last in suffixes:
            parts = parts[:-1]
    short = " ".join(parts)
    # Truncate if too long
    if len(short) > 15:
        # Use first word or first 15 chars
        words = short.split()
        if words:
            short = words[0]
        # If still long, slice
        if len(short) > 15:
            short = short[:15]
    return short


def generate_matrix_plot(
    matrix_config: MatrixConfig,
    placements: List[CompanyPlacement],
    output_dir: Path,
    numbered_mode_threshold: int = 15,
) -> Tuple[bool, Optional[Path]]:
    """Create and save the strategic fit matrix scatter plot.

    Parameters
    ----------
    matrix_config : MatrixConfig
        Loaded matrix configuration defining axes and thresholds.
    placements : List[CompanyPlacement]
        Placements to be plotted; only those with both axis scores
        defined are plotted. Other placements are ignored for the
        graphic.
    output_dir : Path
        Directory in which to save the PNG image (and legend if
        applicable). The directory is created if it does not exist.
    numbered_mode_threshold : int, optional
        Maximum number of points to label directly. If the number of
        valid placements exceeds this threshold, numbered mode is used
        and a legend CSV is written. Default is 15.

    Returns
    -------
    Tuple[bool, Optional[Path]]
        A tuple (numbered_mode_used, legend_csv_path). If numbered
        mode is not used, the second element is None.
    """
    # Extract axis labels and thresholds
    x_label = matrix_config.axis_x.name
    y_label = matrix_config.axis_y.name
    thresholds = matrix_config.placement_policy.default_thresholds
    x_thresh = float(thresholds.get("x", 3.0))
    y_thresh = float(thresholds.get("y", 3.0))
    # Collect points: (x, y, company_name, entity_type, quadrant, status)
    points: List[Tuple[float, float, str, str, Optional[str], str]] = []
    for p in placements:
        if p.axis_x_score is None or p.axis_y_score is None:
            continue
        points.append(
            (
                float(p.axis_x_score),
                float(p.axis_y_score),
                p.company_name,
                p.entity_type,
                p.quadrant,
                p.status,
            )
        )
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    # Determine mode
    N = len(points)
    numbered_mode = N > numbered_mode_threshold
    # Prepare figure
    fig, ax = plt.subplots(figsize=(8, 8))
    # Plot vertical/horizontal threshold lines
    ax.axvline(x=x_thresh, color="grey", linestyle="--", linewidth=1)
    ax.axhline(y=y_thresh, color="grey", linestyle="--", linewidth=1)
    # Prepare scatter points by entity_type for different markers
    # We'll use 'o' for private and 's' for public; colours are default cycle
    xs_private: List[float] = []
    ys_private: List[float] = []
    labels_private: List[str] = []
    xs_public: List[float] = []
    ys_public: List[float] = []
    labels_public: List[str] = []
    # For numbered mode, store legend rows
    legend_rows: List[Dict[str, str]] = []
    # Determine labels
    for idx, (x, y, name, etype, quadrant, status) in enumerate(points, 1):
        if numbered_mode:
            label = str(idx)
        else:
            label = _shorten_name(name)
        if etype.lower() == "private":
            xs_private.append(x)
            ys_private.append(y)
            labels_private.append(label)
        else:
            xs_public.append(x)
            ys_public.append(y)
            labels_public.append(label)
        # Build legend row if in numbered mode
        if numbered_mode:
            legend_rows.append(
                {
                    "point_id": str(idx),
                    "company_name": name,
                    "entity_type": etype,
                    "quadrant": quadrant or "",
                    "status": status,
                }
            )
    # Scatter points
    if xs_private:
        ax.scatter(xs_private, ys_private, marker="o", label="Private")
    if xs_public:
        ax.scatter(xs_public, ys_public, marker="s", label="Public")
    # Annotate points
    def annotate_points(x_vals: List[float], y_vals: List[float], labels: List[str]) -> None:
        # Alternate offsets to reduce overlap
        for i, (x, y, lbl) in enumerate(zip(x_vals, y_vals, labels)):
            # Determine offset direction based on position
            # Slight jitter based on index
            delta_x = 0.03 if (i % 2 == 0) else -0.03
            delta_y = 0.03 if (i % 3 == 0) else -0.03
            ax.text(
                x + delta_x,
                y + delta_y,
                lbl,
                fontsize=8,
                va='center',
                ha='center',
            )
    if not numbered_mode:
        annotate_points(xs_private, ys_private, labels_private)
        annotate_points(xs_public, ys_public, labels_public)
    else:
        # numbered mode: annotate with numbers
        annotate_points(xs_private, ys_private, labels_private)
        annotate_points(xs_public, ys_public, labels_public)
    # Axis settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_title("Strategic Fit Matrix")
    ax.grid(True, linestyle=':', linewidth=0.5, color='lightgray')
    # Add legend if both entity types present
    if xs_private and xs_public:
        ax.legend(loc='upper right')
    # Save figure
    png_path = output_dir / "strategic_fit_matrix.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    legend_csv_path: Optional[Path] = None
    if numbered_mode:
        # Write legend CSV
        legend_csv_path = output_dir / "strategic_matrix_legend.csv"
        import csv
        with legend_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["point_id", "company_name", "entity_type", "quadrant", "status"])
            writer.writeheader()
            for row in legend_rows:
                writer.writerow(row)
    return numbered_mode, legend_csv_path


__all__ = ["generate_matrix_plot"]
