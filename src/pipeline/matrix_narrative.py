"""
Narrative generation for the strategic fit matrix (Task 2 Chunk 3).

This module produces a slide‑ready narrative in Markdown format based
on the quadrant definitions and company placements. For each quadrant
it summarises the strategic meaning, M&A perspective and the key
companies placed in that quadrant along with their review status.
A final method note section reiterates the evidence policy,
override handling and temporal context.

Usage
-----
Call the ``generate_narrative`` function with a matrix configuration,
a list of ``CompanyPlacement`` objects, the destination path for the
Markdown file and the AS_OF_DATE. The function writes the file and
returns None.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from src.pipeline.strategic_framework import MatrixConfig
from src.pipeline.strategic_models import CompanyPlacement


def generate_narrative(
    matrix_config: MatrixConfig,
    placements: List[CompanyPlacement],
    output_path: Path,
    as_of_date: str,
) -> None:
    """Generate a narrative markdown file describing the strategic matrix.

    Parameters
    ----------
    matrix_config : MatrixConfig
        The configuration defining axes and quadrant meanings.
    placements : List[CompanyPlacement]
        Placements computed for all companies.
    output_path : Path
        Path where the Markdown narrative will be written.
    as_of_date : str
        The AS_OF_DATE used for context, e.g., '2025-04-30'.
    """
    # Group placements by quadrant label
    quad_to_placements: Dict[str, List[CompanyPlacement]] = {}
    for p in placements:
        if p.quadrant:
            quad_to_placements.setdefault(p.quadrant, []).append(p)
    # Build content lines
    lines: List[str] = []
    lines.append(f"# Strategic Fit Matrix (as of {as_of_date})\n")
    lines.append(
        "This narrative summarises the interpretation of each quadrant, the M&A perspective and the key companies placed in each quadrant.\n"
    )
    # For each quadrant definition in order
    for qdef in matrix_config.quadrant_definitions:
        label = qdef.label
        interpretation = qdef.interpretation
        ma_angle = qdef.suggested_MA_angle
        lines.append(f"## {label}\n")
        # Description bullets
        lines.append(f"- {interpretation}\n")
        # Use M&A angle as second bullet; if it's long, split into two sentences
        if ma_angle:
            # Split by sentences to form up to 2 bullets
            sentences = [s.strip() for s in ma_angle.split('.') if s.strip()]
            if sentences:
                # First sentence becomes one bullet
                lines.append(f"- {sentences[0]}.\n")
                # Combine remaining sentences if present
                if len(sentences) > 1:
                    other = ". ".join(sentences[1:])
                    lines.append(f"- {other}.\n")
        # Top companies in this quadrant
        quad_pls = quad_to_placements.get(label, [])
        if quad_pls:
            # Sort by sum of scores descending
            quad_pls_sorted = sorted(
                quad_pls,
                key=lambda p: ((p.axis_x_score or 0) + (p.axis_y_score or 0)),
                reverse=True,
            )
            top = quad_pls_sorted[:5]
            comps_desc: List[str] = []
            for p in top:
                status_note = "" if p.status == "ok" else f" ({p.status})"
                comps_desc.append(f"{p.company_name}{status_note}")
            lines.append(f"- Top companies: {', '.join(comps_desc)}\n")
        else:
            lines.append("- No companies assigned to this quadrant.\n")
        lines.append("\n")
    # Method note
    lines.append("## Method Note\n")
    lines.append("- **Evidence policy:** Scores are based only on evidence with valid citations. Fields without supporting evidence are marked 'needs_review'.\n")
    # Override handling: mention if manual overrides path exists in config
    if matrix_config.manual_overrides_path:
        lines.append("- **Manual overrides:** Overrides from the manual inputs file were applied where provided; scores may be provisional if credible sources were missing.\n")
    else:
        lines.append("- **Manual overrides:** No manual overrides were configured.\n")
    lines.append(f"- **Temporal context:** All data and placements reflect information available as of {as_of_date}.\n")
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


__all__ = ["generate_narrative"]
