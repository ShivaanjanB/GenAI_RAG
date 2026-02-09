"""
Audit log generation for the strategic fit matrix (Task 2 Chunk 3).

This module produces a detailed Markdown report of the placements for
audit and review purposes. It summarises quadrant counts, lists
companies requiring attention (non-OK statuses), documents manual
overrides applied and enumerates the top evidence domains used in the
placement evidence.

Usage
-----
Call ``generate_audit`` with a list of ``CompanyPlacement`` objects
and a destination path. The report is written to the given path.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict
from collections import Counter, defaultdict
from urllib.parse import urlparse

from src.pipeline.strategic_models import CompanyPlacement, EvidenceRecord


def generate_audit(
    placements: List[CompanyPlacement],
    output_path: Path,
) -> None:
    """Generate an audit report for strategic matrix placements.

    Parameters
    ----------
    placements : List[CompanyPlacement]
        Placements generated for each company.
    output_path : Path
        Path where the audit markdown will be written.
    """
    lines: List[str] = []
    lines.append("# Strategic Matrix Audit Log\n")
    # Quadrant counts
    quad_counts: Dict[str, int] = defaultdict(int)
    for p in placements:
        quad_label = p.quadrant or "Unplaced"
        quad_counts[quad_label] += 1
    lines.append("## Quadrant Counts\n")
    for quad, count in sorted(quad_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {quad}: {count}\n")
    lines.append("\n")
    # Companies needing attention
    lines.append("## Companies Needing Attention\n")
    any_attention = False
    for p in placements:
        if p.status != "ok":
            any_attention = True
            missing_fields = ", ".join(p.missing_evidence_fields) if p.missing_evidence_fields else "None"
            lines.append(
                f"- {p.company_name} ({p.entity_type}) – Status: {p.status}, Missing evidence fields: {missing_fields}\n"
            )
    if not any_attention:
        lines.append("All companies have status 'ok'.\n")
    lines.append("\n")
    # Manual overrides
    lines.append("## Manual Overrides\n")
    overrides_present = False
    for p in placements:
        # Identify overrides by checking evidence records with field_name 'override'
        for ev in p.evidence:
            if ev.field_name == "override":
                overrides_present = True
                reason = ev.value
                url = ev.citation_url or ""
                lines.append(f"- {p.company_name} ({p.entity_type}): {reason} – Source: {url}\n")
    if not overrides_present:
        lines.append("No manual overrides applied.\n")
    lines.append("\n")
    # Top evidence domains
    lines.append("## Top Evidence Domains\n")
    domain_counts: Counter = Counter()
    for p in placements:
        for ev in p.evidence:
            url = ev.citation_url or ""
            if url:
                domain = urlparse(url).netloc.lower()
                if domain:
                    domain_counts[domain] += 1
    if domain_counts:
        for domain, count in domain_counts.most_common(10):
            lines.append(f"- {domain}: {count}\n")
    else:
        lines.append("No evidence citations recorded.\n")
    lines.append("\n")
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("".join(lines))


__all__ = ["generate_audit"]
