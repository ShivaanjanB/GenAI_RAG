"""Real estate listings CSV provider.

Loads property listing data fresh from a CSV file on every call —
no in-process caching — so that updates written to the file by
Zillow, Redfin, or Realtor.com feed-sync jobs are visible
immediately to each incoming request.

The CSV is expected to live at ``data/listings.csv`` relative to the
repository root, though callers may pass an explicit path.  Supported
source identifiers (case-insensitive) are:

* ``zillow``
* ``redfin``
* ``realtor``  (also matches ``realtor.com``)

Example
-------
>>> rows = load_listings(source="zillow")
>>> len(rows) >= 0
True
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

# Canonical source identifiers accepted by this module.
SUPPORTED_SOURCES = {"zillow", "redfin", "realtor"}

# Default CSV path relative to the repository root.
_DEFAULT_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "listings.csv"


def _normalise_source(source: str) -> str:
    """Return a canonical source key or raise ValueError."""
    key = source.strip().lower().replace(".com", "")
    if key not in SUPPORTED_SOURCES:
        raise ValueError(
            f"Unsupported source '{source}'. "
            f"Must be one of: {sorted(SUPPORTED_SOURCES)}"
        )
    return key


def load_listings(
    source: str,
    csv_path: Optional[Path] = None,
    *,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Load listings for *source* directly from the CSV — no cache.

    The CSV file is opened and parsed on every invocation so that
    feed-sync jobs can update it between requests without any restart
    or cache-invalidation step.

    Parameters
    ----------
    source:
        Platform originating the request — ``"zillow"``, ``"redfin"``,
        or ``"realtor"`` / ``"realtor.com"`` (case-insensitive).
    csv_path:
        Absolute path to the listings CSV.  Defaults to
        ``<repo_root>/data/listings.csv``.
    filters:
        Optional dict of additional column→value filters to apply
        after source filtering (e.g. ``{"status": "Active"}``).

    Returns
    -------
    list of dict
        Each dict represents one CSV row whose ``source`` column
        matches the requested platform.  All values are strings as
        read from the file.

    Raises
    ------
    ValueError
        If *source* is not one of the supported platform identifiers.
    FileNotFoundError
        If the CSV file does not exist at *csv_path*.
    """
    canonical = _normalise_source(source)
    path = csv_path or _DEFAULT_CSV

    if not path.exists():
        raise FileNotFoundError(f"Listings CSV not found: {path}")

    results: List[Dict[str, Any]] = []
    # Open fresh on every call — intentionally no module-level cache.
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_source = row.get("source", "").strip().lower().replace(".com", "")
            if row_source != canonical:
                continue
            if filters:
                if any(
                    str(row.get(col, "")).strip().lower() != str(val).strip().lower()
                    for col, val in filters.items()
                ):
                    continue
            results.append(dict(row))

    return results
