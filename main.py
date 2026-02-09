#!/usr/bin/env python3
"""
CLI entry point for the audit‑trail pipeline.

Running this module from the command line triggers individual chunks of the
pipeline in sequence. It now supports:

  • Chunk 1: Repository scaffold.
  • Chunk 3: Processing + indexing (building the vector index).
  • Chunk 5: Final outputs and QA report generation.

Usage examples:

```bash
# Create scaffold
python main.py --chunk 1

# Build index (Chunk 3) and optionally query it
python main.py --chunk 3 --force
python main.py --chunk 3 --query "last‑mile delivery valuations" --top_k 5

# Generate final CSVs and QA report (Chunk 5)
python main.py --chunk 5
```
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.config import ConfigError, load_settings
from src.utils.logging import get_logger
from src.utils.run_context import (
    append_notes_log,
    ensure_directories,
    init_run,
)


def run_chunk1(base_dir: Path) -> int:
    """
    Execute the scaffold creation (chunk 1) of the pipeline.

    Parameters
    ----------
    base_dir : Path
        Root directory of the repository.

    Returns
    -------
    int
        Exit status code (0 for success).
    """
    # Step 1: create folder hierarchy if necessary.
    ensure_directories(base_dir)
    # Step 2: load configuration.
    config_dir = base_dir / "config"
    try:
        settings = load_settings(config_dir)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        return 1
    # Step 3: initialise run context.
    logs_root = base_dir / "logs"
    run_id, run_dir, manifest = init_run(settings.as_of_date, logs_root)
    # Step 4: configure logging.
    logger = get_logger(run_id, run_dir, log_level=settings.log_level)
    logger.info("Initialising chunk 1: scaffold")
    # Step 5: append to notes log.
    append_notes_log(
        logs_root=logs_root,
        run_id=run_id,
        started_at=manifest["started_at"],
        as_of_date=settings.as_of_date,
        chunk_desc="Chunk 1: scaffold",
    )
    # Step 6: summarise created paths.
    created_paths = [
        base_dir / "config",
        base_dir / "data" / "raw",
        base_dir / "data" / "manual_inputs",
        base_dir / "data" / "processed",
        base_dir / "data" / "output",
        base_dir / "logs",
        base_dir / "logs" / "runs" / run_id,
        base_dir / "src" / "pipeline",
        base_dir / "src" / "utils",
    ]
    logger.info("Directory scaffold created:")
    for path in created_paths:
        logger.info(f" - {path}")
    print("Chunk 1 complete.")
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    return 0


def run_chunk3(
    base_dir: Path,
    query: str | None = None,
    top_k: int = 10,
    force_rebuild: bool = False,
) -> int:
    """
    Execute the processing and indexing stage (chunk 3) of the pipeline.

    This function builds or updates the vector index from previously
    ingested and eligible documents. Optionally it can also perform a
    search query over the built index.

    Parameters
    ----------
    base_dir : Path
        Root directory of the repository.
    query : str, optional
        Query string to search against the index. If None, only index
        construction is performed.
    top_k : int, optional
        Number of search results to return if a query is provided.
    force_rebuild : bool, optional
        If True, force re‑embedding of all chunks and rebuild the index
        even if cached artefacts exist.

    Returns
    -------
    int
        Exit status code (0 for success).
    """
    # Ensure directories exist
    ensure_directories(base_dir)
    # Load configuration
    config_dir = base_dir / "config"
    try:
        settings = load_settings(config_dir)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        return 1
    # Initialise run context
    logs_root = base_dir / "logs"
    run_id, run_dir, manifest = init_run(settings.as_of_date, logs_root)
    # Configure logging
    logger = get_logger(run_id, run_dir, log_level=settings.log_level)
    logger.info("Initialising chunk 3: indexing")
    # Append entry to notes log
    append_notes_log(
        logs_root=logs_root,
        run_id=run_id,
        started_at=manifest["started_at"],
        as_of_date=settings.as_of_date,
        chunk_desc="Chunk 3: indexing",
    )
    # Build index
    from src.pipeline import index as index_module
    summary = index_module.build_index(
        base_dir=base_dir,
        settings=settings,
        logger=logger,
        force_rebuild=force_rebuild,
    )
    logger.info(
        "Indexing complete: eligible_docs=%d, total_chunks=%d, new_embeddings=%d, cached_embeddings=%d, backend=%s",
        summary.get("eligible_docs", 0),
        summary.get("total_chunks", 0),
        summary.get("embedded_chunks", 0),
        summary.get("cached_chunks", 0),
        summary.get("backend", ""),
    )
    print("Chunk 3 indexing complete.")
    print(f"Run ID: {run_id}")
    print(f"Run directory: {run_dir}")
    print(
        f"Indexed {summary.get('total_chunks', 0)} chunks from {summary.get('eligible_docs', 0)} documents using backend: {summary.get('backend', '')}."
    )
    # If a query is provided, perform search
    if query:
        logger.info("Querying index: %s", query)
        try:
            results = index_module.query_index(
                base_dir=base_dir,
                settings=settings,
                query=query,
                top_k=top_k,
                filters=None,
            )
        except Exception as exc:
            logger.error("Failed to query index: %s", exc)
            print(f"Error querying index: {exc}", file=sys.stderr)
            return 1
        if not results:
            print("No results found for query.")
        else:
            print(f"Top {min(top_k, len(results))} results:")
            for idx, hit in enumerate(results[:top_k], 1):
                pub_date = hit.get("published_date") or "unknown"
                src_name = hit.get("source_name") or "unknown"
                url = hit.get("final_url") or ""
                snippet = hit.get("snippet") or ""
                score = hit.get("score")
                print(
                    f"{idx}. doc_id={hit['doc_id']}, score={score:.4f}, source={src_name}, date={pub_date}, url={url}\n    {snippet}\n"
                )
    return 0


def run_chunk5(base_dir: Path) -> int:
    """
    Execute the final outputs and QA report generation stage (chunk 5).

    This stage reads extraction outputs (JSONL files) from `data/output/`,
    computes missing multiples for public companies, determines slide readiness
    for private and public records, writes flat CSV tables, and generates
    a data quality report in Markdown format.

    Parameters
    ----------
    base_dir : Path
        Root directory of the repository.

    Returns
    -------
    int
        Exit status code (0 for success).
    """
    # Import heavy modules lazily to reduce startup cost
    from src.pipeline import output_tables, multiples, slide_readiness, qa_report

    ensure_directories(base_dir)
    # Load configuration
    config_dir = base_dir / "config"
    try:
        settings = load_settings(config_dir)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        return 1
    # Initialise run context
    logs_root = base_dir / "logs"
    run_id, run_dir, manifest = init_run(settings.as_of_date, logs_root)
    # Configure logging
    logger = get_logger(run_id, run_dir, log_level=settings.log_level)
    logger.info("Initialising chunk 5: final outputs and QA report")
    # Append entry to notes log
    append_notes_log(
        logs_root=logs_root,
        run_id=run_id,
        started_at=manifest["started_at"],
        as_of_date=settings.as_of_date,
        chunk_desc="Chunk 5: outputs + QA",
    )
    # Load extraction outputs
    priv_jsonl = base_dir / "data" / "output" / "private_targets.jsonl"
    pub_jsonl = base_dir / "data" / "output" / "public_comps.jsonl"
    audit_jsonl = base_dir / "data" / "output" / "audit_trail.jsonl"
    private_records = output_tables._load_jsonl(priv_jsonl)
    public_records = output_tables._load_jsonl(pub_jsonl)
    audit_records = output_tables._load_jsonl(audit_jsonl)
    if not private_records and not public_records:
        logger.error("No extraction outputs found. Ensure chunk 4 has been executed before running chunk 5.")
        print("Error: No private or public records to process. Run extraction first.", file=sys.stderr)
        return 1
    # Compute missing multiples for public companies
    public_records, computed_counts = multiples.compute_multiples(public_records)
    # Determine slide readiness
    private_records, slide_ready_private_count = slide_readiness.compute_slide_ready_private(private_records)
    public_records, slide_ready_public_count = slide_readiness.compute_slide_ready_public(public_records)
    # Generate data quality report
    qa_report_path = base_dir / "data" / "output" / "data_quality_report.md"
    qa_report.generate_report(
        private_records=private_records,
        public_records=public_records,
        audit_records=audit_records,
        output_path=qa_report_path,
        as_of_date=settings.as_of_date,
    )
    # Convert records to CSV-ready forms and write CSVs
    priv_df = output_tables.private_records_to_dataframe(private_records)
    pub_df = output_tables.public_records_to_dataframe(public_records)
    priv_csv_path = base_dir / "data" / "output" / "private_targets.csv"
    pub_csv_path = base_dir / "data" / "output" / "public_comps.csv"
    output_tables.write_dataframe_to_csv(priv_df, priv_csv_path)
    output_tables.write_dataframe_to_csv(pub_df, pub_csv_path)
    # Count validation failures in audit records
    validation_failed_count = len([rec for rec in audit_records if rec.get("validation") == "failed"])
    # Print summary to stdout
    print("Chunk 5 completed.")
    print(f"Run ID: {run_id}")
    print(f"Private companies processed: {len(private_records)}")
    print(f"Public companies processed: {len(public_records)}")
    print(f"Slide‑ready private companies: {slide_ready_private_count}")
    print(f"Slide‑ready public companies: {slide_ready_public_count}")
    print(f"Computed multiples: {computed_counts}")
    print(f"Validation failures: {validation_failed_count}")
    print(f"Outputs saved to {priv_csv_path}, {pub_csv_path}, and {qa_report_path}")
    # Log summary
    logger.info(
        "Chunk 5 summary: private=%d, public=%d, slide_ready_private=%d, slide_ready_public=%d, computed=%s, validation_failures=%d",
        len(private_records),
        len(public_records),
        slide_ready_private_count,
        slide_ready_public_count,
        computed_counts,
        validation_failed_count,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point invoked by the command line.

    This function parses command line arguments to select which pipeline
    chunk to execute. Supported chunks are:

    * 1: Scaffold (create directory structure and initialise run context).
    * 3: Processing + indexing (chunk documents and build a retrieval index).
      Optionally performs a query if the `--query` flag is provided.
    * 5: Final outputs (generate CSVs and QA report).

    Parameters
    ----------
    argv : list of str, optional
        Command‑line arguments; defaults to sys.argv[1:].

    Returns
    -------
    int
        Exit status code.
    """
    parser = argparse.ArgumentParser(
        description="Audit‑trail pipeline CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chunk",
        type=int,
        required=True,
        help="Pipeline chunk to execute (1=scaffold, 3=indexing, 5=final outputs).",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="If provided with chunk 3, perform a query against the built index.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top results to return for a query (chunk 3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of embeddings and index (chunk 3).",
    )
    args = parser.parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    if args.chunk == 1:
        return run_chunk1(base_dir)
    elif args.chunk == 3:
        return run_chunk3(
            base_dir=base_dir,
            query=args.query,
            top_k=args.top_k,
            force_rebuild=args.force,
        )
    elif args.chunk == 5:
        return run_chunk5(base_dir)
    else:
        print(f"Unsupported chunk: {args.chunk}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())