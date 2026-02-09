"""
Data quality report generation for the audit‑trail pipeline (Chunk 5).

This module analyses the outputs from the extraction stage (private and
public company records) and the audit trail to produce a Markdown
report summarising data quality, coverage, integrity and potential
issues. The report includes:

* Coverage and missingness statistics per field.
* Slide readiness status and missing fields for incomplete companies.
* Evidence integrity metrics derived from the audit trail (validation
  failures and failure reasons).
* Conflict detection across audit trail records.
* Outlier detection for financial multiples (EV/Revenue and EV/EBITDA).
* A source register summarising domains used as sources.

The report is written to the specified output path.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

try:
    import pandas as pd  # type: ignore
    import numpy as np  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    np = None  # type: ignore
    _HAS_PANDAS = False

from urllib.parse import urlparse


def _to_date(val: Union[str, date]) -> date:
    """Convert a string or date to a date object."""
    if isinstance(val, date):
        return val
    return datetime.fromisoformat(str(val)).date()


def _coverage(records: List[Dict[str, Any]], fields: List[str]) -> Dict[str, float]:
    """Compute coverage (1 - missingness) for the given fields.

    Returns a dict mapping field -> percentage missing (0–1).
    """
    coverage: Dict[str, float] = {}
    total = len(records)
    if total == 0:
        return {f: 0.0 for f in fields}
    for f in fields:
        missing = 0
        for rec in records:
            val = rec.get(f)
            # Treat empty string or empty list as missing
            if val is None or val == "" or (isinstance(val, list) and len(val) == 0):
                missing += 1
        coverage[f] = missing / total
    return coverage


def _list_missing_fields(rec: Dict[str, Any], fields: List[str]) -> List[str]:
    missing = []
    for f in fields:
        val = rec.get(f)
        if val is None or val == "" or (isinstance(val, list) and len(val) == 0):
            missing.append(f)
    return missing


def _format_table(rows: List[List[str]], headers: List[str]) -> str:
    """Format a simple Markdown table from rows and headers."""
    table_lines = []
    table_lines.append("| " + " | ".join(headers) + " |")
    table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


def generate_report(
    private_records: List[Dict[str, Any]],
    public_records: List[Dict[str, Any]],
    audit_records: List[Dict[str, Any]],
    output_path: Path,
    as_of_date: Union[str, date],
) -> None:
    """Generate the data quality report and write it to the specified path.

    Parameters
    ----------
    private_records : list of dict
        Private company records after processing.
    public_records : list of dict
        Public company records after processing and multiples computation.
    audit_records : list of dict
        Audit trail records from extraction.
    output_path : Path
        Destination path for the Markdown report.
    as_of_date : str or date
        The AS_OF_DATE from configuration.
    """
    as_of = _to_date(as_of_date)
    # Prepare base markdown lines
    md: List[str] = []
    md.append("# Data Quality Report")
    md.append("")
    md.append(f"**AS OF DATE:** {as_of.isoformat()}")
    md.append("")
    # Section A: Coverage / Missingness
    md.append("## A) Coverage / Missingness")
    private_fields = [
        "business_description",
        "latest_valuation_usd",
        "funding_to_date_usd",
        "investors",
        "key_customers",
        "headquarters",
    ]
    public_fields = [
        "ticker",
        "currency",
        "market_cap",
        "enterprise_value",
        "revenue",
        "ebitda",
        "net_income",
        "period_label",
        "ev_to_revenue",
        "ev_to_ebitda",
        "pe",
    ]
    priv_cov = _coverage(private_records, private_fields)
    pub_cov = _coverage(public_records, public_fields)
    md.append("### Private Companies")
    md.append("Coverage (missingness) per field:")
    rows_priv = [[f, f"{priv_cov[f]*100:.1f}%"] for f in private_fields]
    md.append(_format_table(rows_priv, ["Field", "% Missing"]))
    md.append("")
    md.append("### Public Companies")
    rows_pub = [[f, f"{pub_cov[f]*100:.1f}%"] for f in public_fields]
    md.append(_format_table(rows_pub, ["Field", "% Missing"]))
    md.append("")
    # Slide readiness missing fields
    md.append("### Non Slide‑Ready Companies")
    # Private
    not_ready_priv = [rec for rec in private_records if not rec.get("slide_ready_private")]
    if not_ready_priv:
        md.append("#### Private")
        for rec in not_ready_priv:
            missing = _list_missing_fields(rec, private_fields)
            md.append(f"- **{rec.get('company_name')}** missing: {', '.join(missing) if missing else 'N/A'}")
    else:
        md.append("#### Private: All companies are slide‑ready.")
    md.append("")
    # Public
    not_ready_pub = [rec for rec in public_records if not rec.get("slide_ready_public")]
    if not_ready_pub:
        md.append("#### Public")
        for rec in not_ready_pub:
            missing = _list_missing_fields(rec, public_fields)
            md.append(f"- **{rec.get('company_name')}** missing: {', '.join(missing) if missing else 'N/A'}")
    else:
        md.append("#### Public: All companies are slide‑ready.")
    md.append("")
    # Section B: Evidence integrity
    md.append("## B) Evidence Integrity")
    validation_failures = [rec for rec in audit_records if rec.get("validation") == "failed"]
    md.append(f"Total validation failures: **{len(validation_failures)}**")
    # Top reasons for failure
    reason_counts = Counter()
    for rec in validation_failures:
        reason = rec.get("failure_reason") or "unknown"
        reason_counts[reason] += 1
    if reason_counts:
        md.append("### Top failure reasons:")
        rows = []
        for reason, count in reason_counts.most_common(10):
            rows.append([reason, str(count)])
        md.append(_format_table(rows, ["Failure Reason", "Count"]))
    # Top 10 fields with most failures
    field_fail_counts = Counter()
    for rec in validation_failures:
        key = (rec.get("company_name"), rec.get("field_name"))
        field_fail_counts[key] += 1
    if field_fail_counts:
        md.append("")
        md.append("### Fields with most validation failures:")
        rows = []
        for (comp, field), count in field_fail_counts.most_common(10):
            rows.append([str(comp), str(field), str(count)])
        md.append(_format_table(rows, ["Company", "Field", "Fail Count"]))
    md.append("")
    # Section C: Conflicts
    md.append("## C) Conflicts Detected")
    conflict_groups: Dict[Tuple[str, str, str], Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for rec in audit_records:
        key = (rec.get("entity_type"), rec.get("company_name"), rec.get("field_name"))
        val = rec.get("field_value")
        conflict_groups[key][val].append(rec)  # type: ignore
    conflict_rows: List[List[str]] = []
    for key, val_dict in conflict_groups.items():
        # conflict if more than one distinct value for same entity + field
        if len(val_dict) > 1:
            entity_type, comp, field = key
            # Convert values to strings to avoid type errors when joining
            values = ["" if v is None else str(v) for v in val_dict.keys()]
            # For brevity, only show first citation for each value
            cites = []
            for v, recs in val_dict.items():
                cite = recs[0].get("citation", {}) or {}
                url = cite.get("final_url") or ""
                date_str = cite.get("published_date") or ""
                # Represent value as string for citation listing
                value_str = "" if v is None else str(v)
                cites.append(f"{value_str} (source: {url}, date: {date_str})")
            conflict_rows.append([
                str(comp),
                str(field),
                "; ".join(values),
                " | ".join(cites),
            ])
    if conflict_rows:
        md.append("Conflicting values were detected for the following fields:")
        md.append(_format_table(conflict_rows, ["Company", "Field", "Values", "Example Citations"]))
    else:
        md.append("No conflicts detected in the audit trail.")
    md.append("")
    # Section D: Outliers
    md.append("## D) Outliers")
    outlier_lines: List[str] = []
    if _HAS_PANDAS and pd is not None:
        df_pub = pd.DataFrame(public_records)
        for metric in ["ev_to_revenue", "ev_to_ebitda"]:
            if metric in df_pub.columns:
                # Convert to numeric
                metric_series = pd.to_numeric(df_pub[metric], errors="coerce")
                series = metric_series.dropna()
                if not series.empty:
                    q1 = series.quantile(0.25)
                    q3 = series.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outlier_mask = (metric_series < lower) | (metric_series > upper)
                    outliers = df_pub[outlier_mask]
                    if not outliers.empty:
                        comps = "; ".join(outliers["company_name"].astype(str).tolist())
                        outlier_lines.append(f"- **{metric}** outliers: {comps}")
    if outlier_lines:
        md.extend(outlier_lines)
    else:
        md.append("No significant outliers detected for multiples.")
    md.append("")
    # Section E: Source register
    md.append("## E) Source Register")
    domain_counts: Dict[str, int] = Counter()
    for rec in audit_records:
        cite = rec.get("citation") or {}
        url = cite.get("final_url") or ""
        if url:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain:
                domain_counts[domain] += 1
    if domain_counts:
        rows = []
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            rows.append([domain, str(count)])
        md.append("### Domains Used in Citations")
        md.append(_format_table(rows, ["Domain", "Fact Count"]))
    else:
        md.append("No sources registered in the audit trail.")
    md.append("")
    md.append(f"_Point‑in‑time data extraction used an AS_OF_DATE of {as_of.isoformat()}. All data is filtered accordingly._")
    # Write report to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(md))


__all__ = ["generate_report"]