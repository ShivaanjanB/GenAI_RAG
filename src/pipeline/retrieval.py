"""
Retrieval utilities for the audit‑trail pipeline.

This module exposes simple wrapper functions for building and querying
the vector index over document chunks. It delegates the heavy lifting
to the underlying `index` module but simplifies the interface by
handling configuration and logging for you. It is intended for use
both from the command line and from other Python code.

Functions
---------
build_index(force_rebuild: bool = False) -> Dict[str, Any]
    Build the retrieval index from eligible documents, optionally
    forcing a full rebuild of embeddings.

query_index(query: str, top_k: int = 10, filters: dict | None = None) -> List[Dict[str, Any]]
    Query the index for the given search string and return the top
    matching chunks along with their metadata.

Note
----
These functions rely on the presence of configuration files under
``config/`` and previously ingested and processed documents under
``data/processed/``. If no eligible documents exist or the index has not
yet been built, appropriate exceptions will be raised.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import load_settings
from src.pipeline import index as index_module


def _get_base_dir() -> Path:
    """
    Determine the repository root based on this file's location.

    Returns
    -------
    Path
        The absolute path to the repository root (two levels above this file).
    """
    return Path(__file__).resolve().parents[2]


def build_index(force_rebuild: bool = False) -> Dict[str, Any]:
    """
    Build or update the vector index over document chunks.

    This function loads the project settings, initialises a temporary
    logger (named "retrieval"), and delegates to
    ``src.pipeline.index.build_index``. The index artefacts are persisted
    under ``data/processed/index/``.

    Parameters
    ----------
    force_rebuild : bool, optional
        If True, force re‑embedding of all chunks and rebuild the index
        even if cached artefacts exist.

    Returns
    -------
    Dict[str, Any]
        A summary of the indexing process including counts of eligible
        documents, total chunks processed, and which backend was used.
    """
    base_dir = _get_base_dir()
    settings = load_settings(base_dir / "config")
    # Use a simple logger that writes only to the console for retrieval.
    logger = logging.getLogger("retrieval")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(settings.log_level)
    return index_module.build_index(
        base_dir=base_dir,
        settings=settings,
        logger=logger,
        force_rebuild=force_rebuild,
    )


def query_index(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query the vector index for a given search string.

    This function loads the project settings and delegates to
    ``src.pipeline.index.query_index``. If the index has not been built yet,
    a FileNotFoundError will be raised.

    Parameters
    ----------
    query : str
        The search query string.
    top_k : int, optional
        The maximum number of results to return.
    filters : dict, optional
        Additional filter options. See ``index.query_index`` for details.

    Returns
    -------
    List[Dict[str, Any]]
        A list of result dictionaries. Each dictionary contains the
        chunk_id, score, doc_id, source_name, final_url, published_date,
        retrieved_at, start_char, end_char and a text snippet.
    """
    base_dir = _get_base_dir()
    settings = load_settings(base_dir / "config")
    return index_module.query_index(
        base_dir=base_dir,
        settings=settings,
        query=query,
        top_k=top_k,
        filters=filters,
    )
