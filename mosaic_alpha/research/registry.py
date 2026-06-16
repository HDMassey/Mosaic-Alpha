"""Experiment registry: local research memory for MosaicAlpha runs.

Each call to save_experiment() creates a timestamped folder under the
registry root (default: ``memory/experiments/``) containing three files:

- ``metadata.json``  -- experiment configuration and provenance
- ``metrics.json``   -- key numeric results
- ``report.md``      -- human-readable summary

All files are plain text so they are easy to read in a terminal and
friendly to Git (diffs are meaningful).

Directory layout
----------------
memory/
  experiments/
    20240601_120000_run_baseline/
      metadata.json
      metrics.json
      report.md
    20240601_130045_run_sector_baseline/
      ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path("memory/experiments")


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class ExperimentRecord:
    """Snapshot of a single MosaicAlpha experiment run."""

    experiment_id: str          # e.g. "20240601_120000_run_baseline"
    name: str                   # human-friendly name, e.g. "run_baseline"
    created_at: str             # ISO-8601 UTC timestamp
    command: str                # CLI invocation string (best-effort)
    start_date: str             # data window start, e.g. "2015-01-01"
    end_date: str               # data window end,   e.g. "2024-12-31"
    universe: list[str] = field(default_factory=list)        # tickers / "SPY"
    data_sources: list[str] = field(default_factory=list)    # e.g. ["yfinance"]
    feature_sets: list[str] = field(default_factory=list)    # e.g. ["price", "macro"]
    model_type: str = "ridge"
    validation_method: str = "walk_forward"
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    output_files: list[str] = field(default_factory=list)    # relative paths
    limitations: list[str] = field(default_factory=list)
    notes: str = ""


# ── Writer ─────────────────────────────────────────────────────────────────────

def _make_experiment_id(name: str) -> tuple[str, str]:
    """Return (experiment_id, created_at_iso) based on the current UTC time."""
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # Sanitise name: replace spaces and hyphens with underscores
    safe_name = name.replace("-", "_").replace(" ", "_")
    experiment_id = f"{timestamp}_{safe_name}"
    created_at = now.isoformat()
    return experiment_id, created_at


def save_experiment(
    record: ExperimentRecord,
    *,
    registry_root: Path | None = None,
    report_text: str = "",
) -> Path:
    """Persist an ExperimentRecord to the local registry.

    Parameters
    ----------
    record:
        Populated :class:`ExperimentRecord` to save.
    registry_root:
        Root directory for experiment folders.  Defaults to
        ``memory/experiments`` relative to the current working directory.
    report_text:
        Optional Markdown text to write as ``report.md``.

    Returns
    -------
    Path to the newly created experiment folder.
    """
    root = Path(registry_root) if registry_root is not None else _DEFAULT_ROOT
    exp_dir = root / record.experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── metadata.json ─────────────────────────────────────────────────────────
    metadata_path = exp_dir / "metadata.json"
    metadata_dict = asdict(record)
    # Remove metrics_summary from metadata (it goes in metrics.json)
    metrics_dict = metadata_dict.pop("metrics_summary", {})
    metadata_path.write_text(
        json.dumps(metadata_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── metrics.json ──────────────────────────────────────────────────────────
    metrics_path = exp_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── report.md ─────────────────────────────────────────────────────────────
    if report_text:
        report_path = exp_dir / "report.md"
        report_path.write_text(report_text, encoding="utf-8")

    logger.info("Experiment saved to %s", exp_dir)
    return exp_dir


# ── Reader helpers ──────────────────────────────────────────────────────────────

def _load_record(exp_dir: Path) -> ExperimentRecord | None:
    """Load one ExperimentRecord from an experiment folder. Returns None on error."""
    metadata_path = exp_dir / "metadata.json"
    metrics_path = exp_dir / "metrics.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata["metrics_summary"] = metrics
        return ExperimentRecord(**metadata)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load experiment from %s: %s", exp_dir, exc)
        return None


def list_experiments(registry_root: Path | None = None) -> list[ExperimentRecord]:
    """Return all saved ExperimentRecords sorted newest-first.

    Parameters
    ----------
    registry_root:
        Root directory.  Defaults to ``memory/experiments``.
    """
    root = Path(registry_root) if registry_root is not None else _DEFAULT_ROOT
    if not root.exists():
        return []

    records: list[ExperimentRecord] = []
    for exp_dir in sorted(root.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue
        rec = _load_record(exp_dir)
        if rec is not None:
            records.append(rec)

    return records


def show_experiment(
    experiment_id: str,
    registry_root: Path | None = None,
) -> ExperimentRecord | None:
    """Load and return a specific ExperimentRecord by its ID.

    Returns ``None`` if the experiment is not found.
    """
    root = Path(registry_root) if registry_root is not None else _DEFAULT_ROOT
    exp_dir = root / experiment_id
    if not exp_dir.exists():
        return None
    return _load_record(exp_dir)


def build_record(
    name: str,
    *,
    command: str = "",
    start_date: str = "",
    end_date: str = "",
    universe: list[str] | None = None,
    data_sources: list[str] | None = None,
    feature_sets: list[str] | None = None,
    model_type: str = "ridge",
    validation_method: str = "walk_forward",
    metrics_summary: dict[str, Any] | None = None,
    output_files: list[str] | None = None,
    limitations: list[str] | None = None,
    notes: str = "",
) -> ExperimentRecord:
    """Convenience constructor that fills in experiment_id and created_at."""
    experiment_id, created_at = _make_experiment_id(name)
    return ExperimentRecord(
        experiment_id=experiment_id,
        name=name,
        created_at=created_at,
        command=command,
        start_date=start_date,
        end_date=end_date,
        universe=universe or [],
        data_sources=data_sources or [],
        feature_sets=feature_sets or [],
        model_type=model_type,
        validation_method=validation_method,
        metrics_summary=metrics_summary or {},
        output_files=output_files or [],
        limitations=limitations or [],
        notes=notes,
    )
