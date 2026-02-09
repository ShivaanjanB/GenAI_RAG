"""
Run context utilities for the audit‑trail pipeline.

This module provides helper functions to manage run identifiers,
manifest files, directory scaffolding and notes logging. A run represents
a single execution of a pipeline chunk and is identified by a unique
``run_id`` derived from the current UTC timestamp.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Tuple


def ensure_directories(base_dir: Path) -> None:
    """
    Ensure that the expected directory hierarchy exists within the repository.

    This function creates the following directories under ``base_dir`` if
    they do not already exist:

    * ``config``
    * ``data/raw``
    * ``data/manual_inputs``
    * ``data/processed``
    * ``data/output``
    * ``logs``
    * ``logs/runs``
    * ``src/pipeline``
    * ``src/utils``
    """
    (base_dir / "config").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "manual_inputs").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "output").mkdir(parents=True, exist_ok=True)
    (base_dir / "logs" / "runs").mkdir(parents=True, exist_ok=True)
    # Note: ``logs`` directory itself is created by ``logs/runs`` call
    (base_dir / "src" / "pipeline").mkdir(parents=True, exist_ok=True)
    (base_dir / "src" / "utils").mkdir(parents=True, exist_ok=True)


def init_run(as_of_date: date, logs_root: Path) -> Tuple[str, Path, dict]:
    """
    Initialise a new run directory and manifest.

    Generates a unique ``run_id`` based on the current UTC timestamp. The
    run directory will be created under ``logs_root/runs/<run_id>`` and a
    ``manifest.json`` file will be written containing metadata about the
    run.

    Parameters
    ----------
    as_of_date : date
        Cutoff date for document processing.
    logs_root : Path
        Root directory for logs (typically ``base_dir / 'logs'``).

    Returns
    -------
    Tuple[str, Path, dict]
        A tuple containing the run_id, the run directory path, and the
        manifest dictionary.
    """
    now = datetime.utcnow()
    run_id = now.strftime("%Y%m%d_%H%M%S_%f")
    run_dir = logs_root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "started_at": now.isoformat(),
        "as_of_date": as_of_date.isoformat(),
    }
    # Write manifest file
    manifest_path = run_dir / "manifest.json"
    try:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except Exception:
        # Best effort; ignore errors
        pass
    return run_id, run_dir, manifest


def append_notes_log(
    logs_root: Path,
    run_id: str,
    started_at: str,
    as_of_date: date,
    chunk_desc: str,
) -> None:
    """
    Append an entry to the notes log summarising a run.

    The notes log is stored at ``logs_root/notes.log``. Each entry is a
    JSON object on its own line, containing the run_id, started_at,
    as_of_date and a description of the chunk executed.

    Parameters
    ----------
    logs_root : Path
        Directory under which the notes log is stored (e.g. ``base_dir / 'logs'``).
    run_id : str
        Unique identifier for the run.
    started_at : str
        ISO timestamp of when the run started.
    as_of_date : date
        Cutoff date used for the run.
    chunk_desc : str
        Description of the pipeline chunk executed.
    """
    notes_path = logs_root / "notes.log"
    entry = {
        "run_id": run_id,
        "started_at": started_at,
        "as_of_date": as_of_date.isoformat() if hasattr(as_of_date, 'isoformat') else str(as_of_date),
        "chunk": chunk_desc,
    }
    logs_root.mkdir(parents=True, exist_ok=True)
    try:
        with notes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        # Best effort; ignore errors
        pass
