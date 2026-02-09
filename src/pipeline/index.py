"""
Vector indexing and retrieval for the audit‑trail pipeline.

This module builds a searchable index over document chunks produced by
``chunking.py``. Depending on the availability of external libraries, it
prefers using a SentenceTransformer model with a FAISS index for
high‑quality semantic search. If those packages are not available, it
falls back to a TF‑IDF representation with cosine similarity using
scikit‑learn. Both approaches persist their state to disk to avoid
recomputing embeddings on subsequent runs.

Key directories::

    * ``data/processed/chunks/`` – per‑document JSONL files containing chunk
      definitions.
    * ``data/processed/metadata/`` – per‑document metadata files.
    * ``data/processed/index/`` – artefacts produced by this module, including
      FAISS indices or TF‑IDF models, chunk metadata and embedding caches.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.pipeline.chunking import chunk_document, write_chunks_to_file
from src.utils.config import Settings

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    _HAS_ST = False

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore
    _HAS_FAISS = False


def _load_eligible_docs(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load eligible document metadata from ``eligible_docs.jsonl``.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping of doc_id to its metadata.
    """
    eligible_path = base_dir / "data" / "processed" / "eligible_docs.jsonl"
    docs: Dict[str, Dict[str, Any]] = {}
    if eligible_path.exists():
        with eligible_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    continue
                doc_id = record.get("doc_id")
                if doc_id:
                    docs[doc_id] = record
    return docs


def _read_chunk_file(doc_id: str, base_dir: Path) -> List[Dict[str, Any]]:
    """
    Read chunks for a single document from disk.

    Parameters
    ----------
    doc_id : str
        Document identifier.
    base_dir : Path
        Repository root.

    Returns
    -------
    List[Dict[str, Any]]
        List of chunk records.
    """
    chunks_path = base_dir / "data" / "processed" / "chunks" / f"{doc_id}.jsonl"
    chunks: List[Dict[str, Any]] = []
    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunks.append(json.loads(line))
                except Exception:
                    continue
    return chunks


def _load_doc_metadata(doc_id: str, base_dir: Path) -> Dict[str, Any]:
    """
    Load metadata JSON for a document.

    Returns an empty dict if not found.
    """
    meta_path = base_dir / "data" / "processed" / "metadata" / f"{doc_id}.json"
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _get_source_tags(settings: Settings, source_name: str) -> List[str]:
    """
    Retrieve tags from the configuration for a given source name.
    """
    sources_config = settings.sources or {}
    if isinstance(sources_config, dict) and "sources" in sources_config:
        entries = sources_config["sources"]
    elif isinstance(sources_config, list):
        entries = sources_config
    else:
        entries = []
    for entry in entries:
        if entry.get("name") == source_name:
            return entry.get("tags", []) or []
    return []


def build_index(
    base_dir: Path,
    settings: Settings,
    logger: Any,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Build or update the vector index from eligible documents.

    This function performs chunking, metadata joining and embedding. Depending
    on the available libraries it will either build a FAISS index over
    SentenceTransformer embeddings or a TF‑IDF matrix. The index and
    associated artefacts are persisted on disk.

    Parameters
    ----------
    base_dir : Path
        Repository root.
    settings : Settings
        Loaded configuration providing the AS_OF_DATE and sources information.
    logger : logging.Logger
        Logger for structured output.
    force_rebuild : bool, optional
        If True, re‑embed all chunks and rebuild the index even if
        artefacts already exist.

    Returns
    -------
    Dict[str, Any]
        Summary statistics including counts and backend used.
    """
    eligible_docs = _load_eligible_docs(base_dir)
    summary: Dict[str, Any] = {
        "eligible_docs": len(eligible_docs),
        "total_chunks": 0,
        "embedded_chunks": 0,
        "cached_chunks": 0,
        "backend": "",
    }
    if not eligible_docs:
        logger.info("No eligible documents to index.")
        return summary
    # Prepare directories
    chunks_dir = base_dir / "data" / "processed" / "chunks"
    index_dir = base_dir / "data" / "processed" / "index"
    embeddings_cache_dir = index_dir / "embeddings_cache"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    embeddings_cache_dir.mkdir(parents=True, exist_ok=True)
    chunk_metadata_path = index_dir / "chunk_metadata.jsonl"
    # Determine whether to use ST+FAISS or TF‑IDF
    use_faiss = _HAS_ST and _HAS_FAISS
    summary["backend"] = "faiss" if use_faiss else "tfidf"
    # Collect all chunks and metadata
    all_chunks: List[Dict[str, Any]] = []
    chunk_texts: List[str] = []
    # For caching counts
    embedding_new = 0
    embedding_cached = 0
    # Process each eligible doc
    for doc_id, meta in eligible_docs.items():
        # Read existing chunk file if exists and not forcing rebuild
        if not force_rebuild and (chunks_dir / f"{doc_id}.jsonl").exists():
            doc_chunks = _read_chunk_file(doc_id, base_dir)
        else:
            # Load text
            text_path = base_dir / "data" / "processed" / "text" / f"{doc_id}.txt"
            if not text_path.exists():
                logger.warning(f"Text file missing for {doc_id}, skipping")
                continue
            try:
                text = text_path.read_text(encoding="utf-8")
            except Exception as exc:
                logger.warning(f"Failed to read text for {doc_id}: {exc}")
                continue
            doc_chunks = chunk_document(doc_id, text)
            write_chunks_to_file(doc_id, doc_chunks, base_dir)
        # Join metadata for each chunk
        doc_meta = _load_doc_metadata(doc_id, base_dir)
        tags = _get_source_tags(settings, meta.get("source_name", doc_meta.get("source_name", "")))
        for chunk in doc_chunks:
            chunk_copy = dict(chunk)  # avoid mutating original
            # Merge metadata fields
            chunk_copy.update({
                "source_name": doc_meta.get("source_name"),
                "source_url": doc_meta.get("source_url"),
                "final_url": doc_meta.get("final_url"),
                "published_date": doc_meta.get("published_date"),
                "retrieved_at": doc_meta.get("retrieved_at"),
                "as_of_matched": (doc_meta.get("as_of_evidence") or {}).get("matched"),
                "tags": tags,
                "content_type": doc_meta.get("content_type"),
            })
            all_chunks.append(chunk_copy)
            chunk_texts.append(chunk_copy.get("text", ""))
    summary["total_chunks"] = len(all_chunks)
    if not all_chunks:
        logger.info("No chunks to index.")
        return summary
    # Save chunk metadata file
    # Each line: JSON record without text to reduce size, but include snippet
    try:
        with chunk_metadata_path.open("w", encoding="utf-8") as f:
            for chunk in all_chunks:
                # store snippet rather than full text for quick lookup; but keep start/end positions
                snippet = chunk.get("text", "")[:350]
                record = {k: v for k, v in chunk.items() if k != "text"}
                record["snippet"] = snippet
                f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write chunk metadata: {exc}")
    # Build embeddings / vectors
    if use_faiss:
        # Use sentence-transformers model
        model_name = "all-MiniLM-L6-v2"
        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning(f"Failed to load SentenceTransformer: {exc}, falling back to TF-IDF")
            use_faiss = False
    if use_faiss:
        # Determine embedding dimension by encoding a small sample if cache is empty
        embeddings: List[np.ndarray] = []
        dim: Optional[int] = None
        for chunk, text in zip(all_chunks, chunk_texts):
            chunk_id = chunk["chunk_id"]
            cache_file = embeddings_cache_dir / f"{chunk_id}.npy"
            if not force_rebuild and cache_file.exists():
                emb = np.load(cache_file)
                embedding_cached += 1
            else:
                emb = model.encode(text, show_progress_bar=False)
                embedding_new += 1
                try:
                    np.save(cache_file, emb)
                except Exception:
                    pass
            embeddings.append(emb)
            if dim is None:
                dim = emb.shape[0]
        # Build FAISS index
        if dim is None:
            logger.warning("No embeddings produced; skipping index build.")
            return summary
        # Convert to numpy array of shape (n_chunks, dim)
        emb_matrix = np.vstack(embeddings).astype("float32")
        # Normalise embeddings to unit length for cosine similarity
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        emb_matrix = emb_matrix / norms
        # Build FAISS index (inner product for cosine similarity)
        index_file = index_dir / "faiss.index"
        try:
            faiss_index = faiss.IndexFlatIP(dim)
            faiss_index.add(emb_matrix)
            faiss.write_index(faiss_index, str(index_file))
        except Exception as exc:
            logger.warning(f"Failed to build FAISS index: {exc}, falling back to TF-IDF")
            use_faiss = False
    if not use_faiss:
        # Build TF-IDF vectoriser and matrix
        vectoriser_path = index_dir / "tfidf.pkl"
        matrix_path = index_dir / "matrix.npz"
        # For simplicity, rebuild vectoriser each time
        vectoriser = TfidfVectorizer(stop_words='english', max_features=50000)
        tf_matrix = vectoriser.fit_transform(chunk_texts)
        try:
            with open(vectoriser_path, 'wb') as f:
                pickle.dump(vectoriser, f)
            scipy.sparse.save_npz(matrix_path, tf_matrix)
        except Exception as exc:
            logger.warning(f"Failed to persist TF-IDF artefacts: {exc}")
    # Write embedding model info
    model_info: Dict[str, Any] = {
        "backend": "faiss" if use_faiss else "tfidf",
        "created_at": datetime.utcnow().isoformat(),
    }
    if use_faiss:
        model_info.update({"model_name": "all-MiniLM-L6-v2", "embedding_dim": emb_matrix.shape[1]})
    else:
        model_info.update({"model_name": "tfidf", "embedding_dim": None})
    try:
        with (index_dir / "embedding_model.json").open("w", encoding="utf-8") as f:
            json.dump(model_info, f, indent=2)
    except Exception as exc:
        logger.warning(f"Failed to write embedding model info: {exc}")
    # Update summary counts
    summary["embedded_chunks"] = embedding_new
    summary["cached_chunks"] = embedding_cached
    return summary


def _load_index(base_dir: Path) -> Tuple[str, Any, Any]:
    """
    Load the persisted index artefacts.

    Returns a tuple (backend, index_object, vectoriser_or_model).
    """
    index_dir = base_dir / "data" / "processed" / "index"
    model_info_path = index_dir / "embedding_model.json"
    if not model_info_path.exists():
        raise FileNotFoundError("Index not built. Run build_index first.")
    with model_info_path.open("r", encoding="utf-8") as f:
        model_info = json.load(f)
    backend = model_info.get("backend")
    if backend == "faiss":
        # Load FAISS index
        index_file = index_dir / "faiss.index"
        if not index_file.exists():
            raise FileNotFoundError("FAISS index file not found.")
        faiss_index = faiss.read_index(str(index_file))
        # Load model
        model_name = model_info.get("model_name", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        return backend, faiss_index, model
    else:
        # Load TF-IDF artefacts
        vectoriser_path = index_dir / "tfidf.pkl"
        matrix_path = index_dir / "matrix.npz"
        with open(vectoriser_path, 'rb') as f:
            vectoriser = pickle.load(f)
        tf_matrix = scipy.sparse.load_npz(matrix_path)
        return backend, tf_matrix, vectoriser


def query_index(
    base_dir: Path,
    settings: Settings,
    query: str,
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Query the built index and return top matching chunks with metadata.

    Parameters
    ----------
    base_dir : Path
        Repository root.
    settings : Settings
        Loaded configuration to enforce AS_OF_DATE filtering.
    query : str
        The query string.
    top_k : int, optional
        Number of top results to return, by default 10.
    filters : dict, optional
        Additional filters to apply. Supported keys:
        - published_date_max : ISO date string
        - source_name : str or list of str
        - tags : str or list of str
        - doc_ids : list of doc_id

    Returns
    -------
    List[Dict[str, Any]]
        List of result dictionaries with fields:
        {chunk_id, score, doc_id, source_name, final_url, published_date,
         retrieved_at, start_char, end_char, snippet}
    """
    # Load index and metadata
    backend, index_obj, model_or_vectoriser = _load_index(base_dir)
    index_dir = base_dir / "data" / "processed" / "index"
    chunk_metadata_path = index_dir / "chunk_metadata.jsonl"
    # Load chunk metadata into list
    chunk_meta_list: List[Dict[str, Any]] = []
    with chunk_metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                chunk_meta_list.append(json.loads(line))
            except Exception:
                continue
    if not chunk_meta_list:
        return []
    # Build search
    scores = None
    indices: np.ndarray
    if backend == "faiss":
        # Compute query embedding
        model = model_or_vectoriser
        q_emb = model.encode(query, show_progress_bar=False)
        q_emb = np.asarray(q_emb).astype("float32")[None, :]
        # normalise query for cosine similarity
        q_norm = np.linalg.norm(q_emb, axis=1, keepdims=True)
        if q_norm.any():
            q_emb = q_emb / q_norm
        # Search
        scores_arr, idx_arr = index_obj.search(q_emb, min(top_k * 5, len(chunk_meta_list)))
        scores = scores_arr[0]
        indices = idx_arr[0]
    else:
        # TF-IDF
        tf_matrix = index_obj  # sparse matrix
        vectoriser = model_or_vectoriser
        q_vec = vectoriser.transform([query])
        # Compute cosine similarity
        sims = cosine_similarity(q_vec, tf_matrix).flatten()
        # Get indices of top scores
        idx_sorted = np.argsort(-sims)[: min(top_k * 5, sims.shape[0])]
        scores = sims[idx_sorted]
        indices = idx_sorted
    results: List[Dict[str, Any]] = []
    as_of_cutoff = settings.as_of_date
    published_date_max = None
    allowed_source_names: Optional[List[str]] = None
    allowed_tags: Optional[List[str]] = None
    allowed_doc_ids: Optional[List[str]] = None
    if filters:
        if isinstance(filters.get("published_date_max"), str):
            try:
                published_date_max = datetime.fromisoformat(filters["published_date_max"]).date()
            except Exception:
                published_date_max = None
        src = filters.get("source_name")
        if src:
            allowed_source_names = src if isinstance(src, list) else [src]
        tg = filters.get("tags")
        if tg:
            allowed_tags = tg if isinstance(tg, list) else [tg]
        ids = filters.get("doc_ids")
        if ids:
            allowed_doc_ids = ids if isinstance(ids, list) else [ids]
    # Iterate through candidate indices until enough results found
    for rank, idx in enumerate(indices):
        if len(results) >= top_k:
            break
        try:
            meta = chunk_meta_list[int(idx)]
        except Exception:
            continue
        score = float(scores[rank]) if scores is not None else 0.0
        # Filters
        doc_published = meta.get("published_date")
        # Always enforce AS_OF_DATE unless as_of_matched is true
        as_of_matched = meta.get("as_of_matched")
        if doc_published:
            try:
                date_obj = datetime.fromisoformat(doc_published).date()
            except Exception:
                date_obj = None
        else:
            date_obj = None
        if not as_of_matched:
            if date_obj and date_obj > as_of_cutoff:
                continue
        # Additional filter: published_date_max
        if published_date_max and date_obj and date_obj > published_date_max:
            continue
        # Filter by source_name
        if allowed_source_names and meta.get("source_name") not in allowed_source_names:
            continue
        # Filter by tags
        if allowed_tags:
            chunk_tags = meta.get("tags") or []
            if not any(t in chunk_tags for t in allowed_tags):
                continue
        # Filter by doc_id
        if allowed_doc_ids and meta.get("doc_id") not in allowed_doc_ids:
            continue
        # Snippet is already stored in metadata
        results.append(
            {
                "chunk_id": meta.get("chunk_id"),
                "score": score,
                "doc_id": meta.get("doc_id"),
                "source_name": meta.get("source_name"),
                "final_url": meta.get("final_url"),
                "published_date": meta.get("published_date"),
                "retrieved_at": meta.get("retrieved_at"),
                "start_char": meta.get("start_char"),
                "end_char": meta.get("end_char"),
                "snippet": meta.get("snippet"),
            }
        )
    return results
