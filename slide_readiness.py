"""
Slide readiness computation for the audit‑trail pipeline (Chunk 5).

This module defines helper functions to determine whether a private or
public company record is complete enough to be "slide‑ready" for a
presentation. The rules are defined as follows:

* A private company is slide‑ready if:
  - `business_description` is not null/empty.
  - `latest_valuation_usd` is not null.
  - `funding_to_date_usd` is not null.
  - `investors` list is non‑empty.
  - `key_customers` list is non‑empty.

* A public company is slide‑ready if:
  - `revenue` and `enterprise_value` exist (not null).
  - At least one of `ebitda` or `net_income` exists (not null).

The functions here update the records in place by adding a boolean
`slide_ready_private` or `slide_ready_public` field and return the
count of records that are slide‑ready.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def compute_slide_ready_private(private_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Determine slide readiness for private companies.

    Parameters
    ----------
    private_records : list of dict
        Private company records.

    Returns
    -------
    Tuple[list of dict, int]
        The updated records and the count of slide‑ready private companies.
    """
    ready_count = 0
    for rec in private_records:
        desc = rec.get("business_description")
        val = rec.get("latest_valuation_usd")
        fund = rec.get("funding_to_date_usd")
        investors = rec.get("investors")
        customers = rec.get("key_customers")

        # Normalise list fields for checking
        def _is_non_empty_list(x: Any) -> bool:
            if isinstance(x, list):
                return len([i for i in x if i]) > 0
            return bool(x)

        slide_ready = bool(desc) and (val is not None) and (fund is not None) and _is_non_empty_list(investors) and _is_non_empty_list(customers)
        rec["slide_ready_private"] = slide_ready
        if slide_ready:
            ready_count += 1
    return private_records, ready_count


def compute_slide_ready_public(public_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Determine slide readiness for public companies.

    Parameters
    ----------
    public_records : list of dict
        Public company records.

    Returns
    -------
    Tuple[list of dict, int]
        The updated records and the count of slide‑ready public companies.
    """
    ready_count = 0
    for rec in public_records:
        revenue = rec.get("revenue")
        ev = rec.get("enterprise_value")
        ebitda = rec.get("ebitda")
        net_income = rec.get("net_income")
        slide_ready = (revenue is not None) and (ev is not None) and ((ebitda is not None) or (net_income is not None))
        rec["slide_ready_public"] = slide_ready
        if slide_ready:
            ready_count += 1
    return public_records, ready_count


__all__ = [
    "compute_slide_ready_private",
    "compute_slide_ready_public",
]