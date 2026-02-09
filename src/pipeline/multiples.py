"""
Multiples computation utilities for the audit‑trail pipeline (Chunk 5).

This module provides functions to compute financial multiples for public
companies when they are missing from the extracted data. The multiples
computed include:

* `ev_to_revenue` – Enterprise value divided by revenue.
* `ev_to_ebitda` – Enterprise value divided by EBITDA.
* `pe` (P/E ratio) – Market capitalisation divided by net income.

When computing multiples, the module adds the name of each computed
multiple to the record's `computed_fields` list. It does not assign a
citation to the computed value itself; citations for the input values
remain unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def compute_multiples(public_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Compute missing multiples for public company records.

    Parameters
    ----------
    public_records : list of dict
        Records for public companies. Each record may contain numeric fields
        `enterprise_value`, `revenue`, `ebitda`, `net_income`, `ev_to_revenue`,
        `ev_to_ebitda` and `pe`. This function will update the records in place
        when a missing multiple can be computed.

    Returns
    -------
    Tuple[list of dict, dict]
        A tuple containing the updated list of records and a dictionary of
        counts: number of `ev_to_revenue`, `ev_to_ebitda` and `pe` ratios
        computed.
    """
    counts = {"ev_to_revenue": 0, "ev_to_ebitda": 0, "pe": 0}
    for rec in public_records:
        # Ensure computed_fields is a list
        cf: List[str] = rec.get("computed_fields") or []
        if not isinstance(cf, list):
            cf = [str(cf)] if cf else []

        # Helper to parse numeric values safely
        def parse_num(value: Any) -> float | None:
            if value is None:
                return None
            try:
                return float(value)
            except Exception:
                return None

        ev = parse_num(rec.get("enterprise_value"))
        rev = parse_num(rec.get("revenue"))
        ebitda = parse_num(rec.get("ebitda"))
        net_inc = parse_num(rec.get("net_income"))
        ev_rev = parse_num(rec.get("ev_to_revenue"))
        ev_ebitda = parse_num(rec.get("ev_to_ebitda"))
        pe_ratio = parse_num(rec.get("pe"))

        # Compute EV/Revenue if missing
        if ev_rev is None and ev is not None and rev is not None and rev > 0:
            rec["ev_to_revenue"] = ev / rev
            counts["ev_to_revenue"] += 1
            cf.append("ev_to_revenue")

        # Compute EV/EBITDA if missing
        if ev_ebitda is None and ev is not None and ebitda is not None and ebitda > 0:
            rec["ev_to_ebitda"] = ev / ebitda
            counts["ev_to_ebitda"] += 1
            cf.append("ev_to_ebitda")

        # Compute P/E ratio if missing and net income positive
        if pe_ratio is None and net_inc is not None and net_inc > 0:
            mc = parse_num(rec.get("market_cap"))
            if mc is not None and mc > 0:
                rec["pe"] = mc / net_inc
                counts["pe"] += 1
                cf.append("pe")

        # Update computed_fields list
        if cf:
            seen = set()
            cf_unique: List[str] = []
            for x in cf:
                if x not in seen:
                    cf_unique.append(x)
                    seen.add(x)
            rec["computed_fields"] = cf_unique
    return public_records, counts


__all__ = ["compute_multiples"]
