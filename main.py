#!/usr/bin/env python3
"""
CLI entry point for the audit‑trail pipeline.

This script drives the various stages ("chunks") of the audit‑trail
pipeline and the strategic matrix workflow. Each stage can be invoked
independently from the command line by specifying a task and chunk
number.  The pipeline supports two high‑level tasks:

* **Task 1 – Audit‑trail pipeline:** Build the repository scaffold,
  index ingested documents for retrieval, and produce final outputs
  including CSV tables and a data quality report.
* **Task 2 – Strategic fit matrix:** Generate a strategic fit matrix
  visualisation along with a narrative summary and audit log based on
  previously computed placements.

Within each task there are numbered chunks.  For Task 1 the supported
chunks are:

  * **Chunk 1:** Scaffold (create the expected directory structure and
    initialise a run context).
  * **Chunk 3:** Processing and indexing (chunk documents, build a
    retrieval index, and optionally perform a search query).
  * **Chunk 5:** Final outputs (compute financial multiples, determine
    slide readiness, generate CSV tables and a data quality report).

For Task 2 the only defined chunk at present is:

  * **Chunk 3:** Generate the strategic fit matrix visual, narrative
    bullets for slides, and an audit log.  This chunk expects that
    strategic matrix placements have already been produced by a prior
    step (Task 2, Chunk 2) and written to
    ``data/output/strategic_matrix_placements.jsonl``.

Usage examples:

```bash
# Task 1: create scaffold
python main.py --task 1 --chunk 1

# Task 1: build index (Chunk 3) and optionally query it
python main.py --task 1 --chunk 3 --force
python main.py --task 1 --chunk 3 --query "last‑mile delivery valuations" --top_k 5

# Task 1: generate final CSVs and QA report (Chunk 5)
python main.py --task 1 --chunk 5

# Task 2: strategic fit matrix visual and narrative (Chunk 3)
python main.py --task 2 --chunk 3
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


def run_task2_chunk3(base_dir: Path) -> int:
    """
    Execute Task 2 Chunk 3: generate the strategic fit matrix visual,
    narrative bullets and audit log.

    This function reads strategic matrix placements (JSONL records) from
    ``data/output/strategic_matrix_placements.jsonl``, validates the matrix
    configuration, generates a high‑resolution scatter plot, a narrative
    description and an audit report.  Outputs are written under
    ``data/output/``.  It also logs the run context and summarises
    outcomes to the console.

    Parameters
    ----------
    base_dir : Path
        Root directory of the repository.

    Returns
    -------
    int
        Exit status code (0 for success).
    """
    from src.pipeline import strategic_framework, matrix_visual, matrix_narrative, matrix_audit
    from src.pipeline.strategic_models import CompanyPlacement  # type: ignore
    import json

    # Ensure directories exist
    ensure_directories(base_dir)

    # Load base settings
    config_dir = base_dir / "config"
    try:
        settings = load_settings(config_dir)
    except (ConfigError, FileNotFoundError) as exc:
        print(f"Error loading base configuration: {exc}", file=sys.stderr)
        return 1

    # Load matrix configuration
    try:
        matrix_config = strategic_framework.load_matrix_config(config_dir / "strategic_matrix.yaml")
        strategic_framework.validate_config(matrix_config)
    except Exception as exc:
        print(f"Error loading matrix configuration: {exc}", file=sys.stderr)
        return 1

    # Initialise run context
    logs_root = base_dir / "logs"
    run_id, run_dir, manifest = init_run(settings.as_of_date, logs_root)
    logger = get_logger(run_id, run_dir, log_level=settings.log_level)
    logger.info("Initialising Task 2 Chunk 3: matrix visual and narrative generation")

    # Append entry to notes log
    append_notes_log(
        logs_root=logs_root,
        run_id=run_id,
        started_at=manifest["started_at"],
        as_of_date=settings.as_of_date,
        chunk_desc="Task 2 Chunk 3: matrix visual and narrative",
    )

    # Load placements
    placements_path = base_dir / "data" / "output" / "strategic_matrix_placements.jsonl"
    if not placements_path.exists():
        logger.error("Placements file not found: %s", placements_path)
        print("Error: strategic_matrix_placements.jsonl not found. Run Task 2 Chunk 2 first.", file=sys.stderr)
        return 1

    placements: list[CompanyPlacement] = []
    with placements_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                try:
                    # Try Pydantic v2 API
                    placement = CompanyPlacement.model_validate(data)  # type: ignore[attr-defined]
                except Exception:
                    placement = CompanyPlacement.parse_obj(data)  # type: ignore[attr-defined]
                placements.append(placement)
            except Exception as exc:
                logger.warning("Skipping malformed placement line: %s", exc)
                continue

    if not placements:
        logger.error("No placements loaded from %s", placements_path)
        print("Error: no placements to visualise.", file=sys.stderr)
        return 1

    # Generate matrix plot
    output_dir = base_dir / "data" / "output"
    numbered_mode, legend_path = matrix_visual.generate_matrix_plot(
        matrix_config=matrix_config,
        placements=placements,
        output_dir=output_dir,
        numbered_mode_threshold=15,
    )

    # Generate narrative
    narrative_path = output_dir / "strategic_matrix_slide_bullets.md"
    matrix_narrative.generate_narrative(
        matrix_config=matrix_config,
        placements=placements,
        output_path=narrative_path,
        as_of_date=settings.as_of_date,
    )

    # Generate audit log
    audit_path = output_dir / "strategic_matrix_audit.md"
    matrix_audit.generate_audit(
        placements=placements,
        output_path=audit_path,
    )

    # Summarise counts
    total_placements = len(placements)
    plotted_count = len([p for p in placements if p.axis_x_score is not None and p.axis_y_score is not None])
    unplotted = total_placements - plotted_count
    quadrant_counts: dict[str, int] = {}
    for p in placements:
        quad = p.quadrant or "Unplaced"
        quadrant_counts[quad] = quadrant_counts.get(quad, 0) + 1

    # Console output
    print("Task 2 Chunk 3 completed.")
    print(f"Run ID: {run_id}")
    print(f"Matrix image saved to {output_dir / 'strategic_fit_matrix.png'}")
    if numbered_mode and legend_path:
        print(f"Legend file saved to {legend_path}")
    print(f"Narrative markdown saved to {narrative_path}")
    print(f"Audit log saved to {audit_path}")
    print(f"Total placements: {total_placements}")
    print(f"Plotted points: {plotted_count}, Unplotted (missing scores): {unplotted}")
    print("Quadrant counts:")
    for quad, count in quadrant_counts.items():
        print(f"  {quad}: {count}")

    # Log summary
    logger.info(
        "Task 2 Chunk 3 summary: total=%d, plotted=%d, unplotted=%d, numbered_mode=%s",
        total_placements,
        plotted_count,
        unplotted,
        numbered_mode,
    )

    return 0


def main(argv: list[str] | None = None) -> int:
    """
    Main entry point invoked by the command line.

    This function parses command‑line arguments to determine which task
    and chunk to run.  Two tasks are supported:

    * **Task 1:** Audit‑trail pipeline (scaffold, indexing, final outputs).
    * **Task 2:** Strategic fit matrix (visual + narrative + audit).

    For Task 1 the valid chunks are 1, 3 and 5.  For Task 2 the only
    valid chunk is 3.  Additional arguments apply to certain chunks
    (e.g., ``--query`` and ``--force`` for Task 1, Chunk 3).

    Parameters
    ----------
    argv : list of str, optional
        Command‑line arguments; defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status code (0 for success, non‑zero for errors).
    """
    parser = argparse.ArgumentParser(
        description="Audit‑trail and strategic matrix CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=int,
        default=1,
        help="Task to execute (1=audit‑trail pipeline, 2=strategic matrix).",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        required=True,
        help=(
            "Chunk number to execute. For task 1 use 1, 3 or 5. "
            "For task 2 use 3."
        ),
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="If provided with task 1, chunk 3, perform a query against the built index.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top results to return for a query (task 1, chunk 3).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild of embeddings and index (task 1, chunk 3).",
    )
    args = parser.parse_args(argv)

    base_dir = Path(__file__).resolve().parent

    if args.task == 1:
        # Audit‑trail pipeline tasks
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
            print(f"Unsupported chunk {args.chunk} for task 1", file=sys.stderr)
            return 1
    elif args.task == 2:
        # Strategic matrix tasks
        if args.chunk == 3:
            return run_task2_chunk3(base_dir)
        else:
            print(f"Unsupported chunk {args.chunk} for task 2", file=sys.stderr)
            return 1
    else:
        print(f"Unsupported task: {args.task}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())