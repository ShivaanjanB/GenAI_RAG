"""
Output table generation utilities for the audit‑trail pipeline (Chunk 5).

This module reads JSON Lines (JSONL) outputs produced by the extraction
stage (Chunk 4) and converts them into flat CSV tables ready for
consumption in client deliverables. It also handles citation
flattening, numeric type coercion and snippet truncation.

The functions in this module rely on the following input files in
`data/output/`:

* `private_targets.jsonl` – One JSON object per private company
  conforming to the PrivateTarget schema. Each object must include a
  `citations` dictionary mapping field names to citation objects.
* `public_comps.jsonl` – One JSON object per public company conforming
  to the PublicComp schema. Each object must include a `citations`
  dictionary mapping field names to citation objects.

If pandas is available, the module will use it for convenience; if
not, it will fall back to Python's built‑in csv module. The
flattened tables include additional columns for citations: for every
field present in the `citations` map, the following columns are
generated:

  `<field>_cite_url`, `<field>_cite_pub_date`,
  `<field>_cite_retrieved_at`, `<field>_cite_snippet`

The snippet column is truncated to 200 characters to keep the CSV size
manageable. Numeric columns are coerced to floats where possible; empty
cells are left blank for missing values.
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd  # type: ignore
    _HAS_PANDAS = True
except ImportError:
    pd = None  # type: ignore
    _HAS_PANDAS = False


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries.

    Parameters
    ----------
    path : Path
        Path to the JSON Lines file.

    Returns
    -------
    List[Dict[str, Any]]
        A list of parsed JSON objects. Malformed lines are skipped.
    """
    records: List[Dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
            except Exception:
                # Skip malformed lines
                continue
    return records


def _flatten_citation(prefix: str, citation: Optional[Dict[str, Any]], out: Dict[str, Any]) -> None:
    """Flatten a citation object into a dict with keys prefixed by the field name.

    Parameters
    ----------
    prefix : str
        The field name prefix (e.g. 'latest_valuation_usd').
    citation : dict or None
        Citation object with keys: final_url, published_date, retrieved_at, snippet.
    out : dict
        Dictionary to which flattened fields will be written.
    """
    cite_url = ""
    cite_pub_date = ""
    cite_retrieved = ""
    cite_snippet = ""
    if citation:
        cite_url = citation.get("final_url") or ""
        cite_pub_date = citation.get("published_date") or ""
        cite_retrieved = citation.get("retrieved_at") or ""
        snippet = citation.get("snippet") or ""
        # Truncate snippet to 200 characters
        cite_snippet = snippet[:200] if snippet else ""
    out[f"{prefix}_cite_url"] = cite_url
    out[f"{prefix}_cite_pub_date"] = cite_pub_date
    out[f"{prefix}_cite_retrieved_at"] = cite_retrieved
    out[f"{prefix}_cite_snippet"] = cite_snippet


def _flatten_private_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a single private company record into a flat dict suitable for CSV.

    Parameters
    ----------
    record : dict
        A PrivateTarget record loaded from JSONL.

    Returns
    -------
    Dict[str, Any]
        A flattened representation with citation columns.
    """
    out: Dict[str, Any] = {}
    # Core fields
    out["company_name"] = record.get("company_name")
    out["business_description"] = record.get("business_description")
    out["latest_valuation_usd"] = record.get("latest_valuation_usd")
    out["latest_valuation_date"] = record.get("latest_valuation_date")
    out["funding_to_date_usd"] = record.get("funding_to_date_usd")
    # List fields become semicolon‑separated strings
    investors = record.get("investors")
    if isinstance(investors, list):
        out["investors"] = "; ".join([str(i) for i in investors if i])
    else:
        out["investors"] = investors if investors is not None else None
    key_customers = record.get("key_customers")
    if isinstance(key_customers, list):
        out["key_customers"] = "; ".join([str(i) for i in key_customers if i])
    else:
        out["key_customers"] = key_customers if key_customers is not None else None
    out["headquarters"] = record.get("headquarters")
    out["notes"] = record.get("notes")
    # Slide readiness flag
    if "slide_ready_private" in record:
        out["slide_ready_private"] = bool(record.get("slide_ready_private"))
    # Computed fields list (if present)
    if "computed_fields" in record:
        cf = record.get("computed_fields")
        if isinstance(cf, list):
            out["computed_fields"] = "; ".join(cf)
        else:
            out["computed_fields"] = cf
    # Flatten citations
    citations = record.get("citations") or {}
    for field, citation in citations.items():
        _flatten_citation(field, citation, out)
    return out


def _flatten_public_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a single public company record into a flat dict suitable for CSV.

    Parameters
    ----------
    record : dict
        A PublicComp record loaded from JSONL.

    Returns
    -------
    Dict[str, Any]
        A flattened representation with citation columns.
    """
    out: Dict[str, Any] = {}
    # Core fields
    out["company_name"] = record.get("company_name")
    out["ticker"] = record.get("ticker")
    out["currency"] = record.get("currency")
    out["market_cap"] = record.get("market_cap")
    out["enterprise_value"] = record.get("enterprise_value")
    out["revenue"] = record.get("revenue")
    out["ebitda"] = record.get("ebitda")
    out["net_income"] = record.get("net_income")
    out["period_label"] = record.get("period_label")
    out["ev_to_revenue"] = record.get("ev_to_revenue")
    out["ev_to_ebitda"] = record.get("ev_to_ebitda")
    out["pe"] = record.get("pe")
    out["notes"] = record.get("notes")
    # Slide readiness flag
    if "slide_ready_public" in record:
        out["slide_ready_public"] = bool(record.get("slide_ready_public"))
    # Computed fields list
    if "computed_fields" in record:
        cf = record.get("computed_fields")
        if isinstance(cf, list):
            out["computed_fields"] = "; ".join(cf)
        else:
            out["computed_fields"] = cf
    # Flatten citations
    citations = record.get("citations") or {}
    for field, citation in citations.items():
        _flatten_citation(field, citation, out)
    return out


def private_records_to_dataframe(records: List[Dict[str, Any]]) -> "pd.DataFrame | List[Dict[str, Any]]":
    """Convert private company records to a pandas DataFrame (if available).

    Parameters
    ----------
    records : list of dict
        Private company records after slide readiness and multiples processing.

    Returns
    -------
    DataFrame or list
        If pandas is available, returns a DataFrame; otherwise, returns a list of
        flattened dictionaries.
    """
    flat = [_flatten_private_record(rec) for rec in records]
    if _HAS_PANDAS:
        df = pd.DataFrame(flat)
        # Numeric columns
        numeric_cols = ["latest_valuation_usd", "funding_to_date_usd"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        return flat


def public_records_to_dataframe(records: List[Dict[str, Any]]) -> "pd.DataFrame | List[Dict[str, Any]]":
    """Convert public company records to a pandas DataFrame (if available).

    Parameters
    ----------
    records : list of dict
        Public company records after multiples and slide readiness processing.

    Returns
    -------
    DataFrame or list
        If pandas is available, returns a DataFrame; otherwise, returns a list of
        flattened dictionaries.
    """
    flat = [_flatten_public_record(rec) for rec in records]
    if _HAS_PANDAS:
        df = pd.DataFrame(flat)
        # Numeric columns
        numeric_cols = [
            "market_cap",
            "enterprise_value",
            "revenue",
            "ebitda",
            "net_income",
            "ev_to_revenue",
            "ev_to_ebitda",
            "pe",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        return flat


def write_dataframe_to_csv(data: "pd.DataFrame | List[Dict[str, Any]]", path: Path) -> None:
    """Write data (DataFrame or list of dicts) to a CSV file.

    Parameters
    ----------
    data : DataFrame or list of dicts
        Flattened data to write.
    path : Path
        Output CSV path. Parent directory must exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use pandas if available
    if _HAS_PANDAS and isinstance(data, pd.DataFrame):
        # Replace NaN with empty string
        data = data.copy()
        data = data.where(~data.isna(), None)
        data.to_csv(path, index=False)
    else:
        if isinstance(data, list):
            if not data:
                # Write empty file if no data
                path.open("w", encoding="utf-8").close()
                return
            headers = list(data[0].keys())
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in data:
                    writer.writerow({k: row.get(k, "") for k in headers})
        else:
            # Unexpected data format
            pass


__all__ = [
    "_load_jsonl",
    "private_records_to_dataframe",
    "public_records_to_dataframe",
    "write_dataframe_to_csv",
]