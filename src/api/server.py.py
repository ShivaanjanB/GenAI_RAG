"""
FastAPI server to expose the RAG retrieval and index pipeline as a REST API.

This module provides a minimal HTTP wrapper around the existing audit‑trail
retrieval logic implemented in ``src/pipeline/index.py``.  It allows
external clients (such as a Lovable application) to submit search queries
and receive structured results with citations and scoring.  Results are
filtered according to the configured ``as_of_date`` and can be further
filtered by source name or tags.  In addition, an optional reliability
ranking is applied when multiple sources are configured with differing
trust levels.

The API defines a single endpoint ``POST /query`` which accepts a JSON
payload with the query text and optional parameters:

* ``top_k`` – number of results to return (default 5).
* ``source_names`` – list of source identifiers to restrict retrieval.
* ``tags`` – list of tags to restrict retrieval.

The response is a JSON object containing the top‑k results with fields
matching those returned by ``query_index``.  If a source has a
``reliability`` value specified in the configuration, the raw score is
multiplied by this weight before ranking.

To run the server locally:

```
uvicorn src.api.server:app --reload
```

Note: Running the server requires ``fastapi`` and ``uvicorn`` in your
Python environment.  These dependencies are intentionally optional and
should be added to ``requirements.txt`` if you intend to deploy the API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.utils.config import load_settings, Settings
from src.pipeline.index import query_index
from src.data_providers.real_estate_csv import load_listings, SUPPORTED_SOURCES



def _build_reliability_map(settings: Settings) -> Dict[str, float]:
    """Build a mapping from source_name to reliability weight.

    The configuration file may specify a ``sources`` section where each
    entry can include a ``reliability`` field (a float >= 0).  A value of
    1.0 means neutral weight; values >1.0 increase the ranking score,
    values between 0 and 1 decrease it.  Missing or invalid values
    default to 1.0.

    Parameters
    ----------
    settings : Settings
        Parsed settings containing an optional ``sources`` structure.

    Returns
    -------
    dict
        Mapping from source_name to reliability weight.
    """
    reliability: Dict[str, float] = {}
    sources_config = settings.sources or {}
    entries: List[Dict[str, Any]]
    if isinstance(sources_config, dict) and "sources" in sources_config:
        entries = sources_config["sources"]
    elif isinstance(sources_config, list):
        entries = sources_config  # type: ignore[assignment]
    else:
        entries = []
    for entry in entries:
        name = entry.get("name")
        weight = entry.get("reliability")
        if name and isinstance(weight, (int, float)) and weight > 0:
            reliability[name] = float(weight)
    return reliability


class QueryRequest(BaseModel):
    """Schema for the /query request payload."""

    query: str = Field(..., description="User search query")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    source_names: Optional[List[str]] = Field(
        None,
        description="Optional list of source names to restrict retrieval",
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Optional list of tags to restrict retrieval",
    )


class QueryResponse(BaseModel):
    """Schema for the /query response payload."""

    results: List[Dict[str, Any]]


app = FastAPI(title="Audit‑Trail RAG API")


# ---------------------------------------------------------------------------
# Real-estate listings endpoint
# ---------------------------------------------------------------------------

class ListingsRequest(BaseModel):
    """Schema for the /listings request payload."""

    source: str = Field(
        ...,
        description=(
            "Platform originating the request: 'zillow', 'redfin', or 'realtor' / 'realtor.com'"
        ),
    )
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional column→value filters applied after source filtering (e.g. {\"status\": \"Active\"})",
    )


class ListingsResponse(BaseModel):
    """Schema for the /listings response payload."""

    source: str
    count: int
    listings: List[Dict[str, Any]]


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Handle search queries against the built index.

    This endpoint loads the pipeline settings, computes a reliability
    weighting map, queries the index and applies reliability weights to
    the raw scores.  Results are sorted by weighted score (descending)
    and truncated to ``top_k`` items.

    Parameters
    ----------
    req : QueryRequest
        Request body containing the query and optional filters.

    Returns
    -------
    QueryResponse
        Response containing the ranked retrieval results.
    """
    base_dir = Path(__file__).resolve().parent.parent
    settings = load_settings(base_dir / "config")
    # Build filters for query_index
    filters: Dict[str, Any] = {}
    if req.source_names:
        filters["source_name"] = req.source_names
    if req.tags:
        filters["tags"] = req.tags
    # Perform retrieval (returns list of dicts with score and metadata)
    raw_results = query_index(
        base_dir=base_dir,
        settings=settings,
        query=req.query,
        top_k=req.top_k,
        filters=filters if filters else None,
    )
    # Build reliability weighting
    reliability_map = _build_reliability_map(settings)
    # Apply reliability weighting to scores
    for res in raw_results:
        src_name = res.get("source_name")
        weight = reliability_map.get(src_name, 1.0)
        try:
            score = float(res.get("score", 0))
        except Exception:
            score = 0.0
        res["weighted_score"] = score * weight
    # Sort by weighted_score descending
    ranked = sorted(raw_results, key=lambda r: r.get("weighted_score", 0), reverse=True)
    # Truncate to requested top_k
    final = ranked[: req.top_k]
    return QueryResponse(results=final)


@app.post("/listings", response_model=ListingsResponse)
def listings_endpoint(req: ListingsRequest) -> ListingsResponse:
    """Return property listings for a given real-estate platform.

    The CSV is read fresh on **every** request — there is no in-process
    cache — so feed-sync jobs updating the file are reflected immediately
    without any server restart.

    Supported sources: ``zillow``, ``redfin``, ``realtor`` / ``realtor.com``.

    Parameters
    ----------
    req : ListingsRequest
        Request body with the ``source`` field and optional ``filters``.

    Returns
    -------
    ListingsResponse
        Canonical source name, total count, and the matching listing rows.

    Raises
    ------
    422
        If ``source`` is not one of the supported platform identifiers.
    404
        If the listings CSV file does not exist.
    """
    from fastapi import HTTPException

    try:
        rows = load_listings(source=req.source, filters=req.filters)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    canonical = req.source.strip().lower().replace(".com", "")
    return ListingsResponse(source=canonical, count=len(rows), listings=rows)
