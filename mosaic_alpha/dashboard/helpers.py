"""Data-loading helpers for the MosaicAlpha Streamlit dashboard.

All functions are pure Python (no Streamlit imports) so they can be unit-tested
without launching a browser session.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY = Path("memory/experiments")
_DEFAULT_GRAPH = Path("memory/research_graph.json")
_DEFAULT_REPORTS = Path("reports/generated")


# ── Experiments ────────────────────────────────────────────────────────────────

def load_experiments(
    registry_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return a list of experiment dicts loaded from the registry.

    Each dict contains all metadata fields plus ``metrics`` (from
    ``metrics.json``) and ``report_text`` (from ``report.md`` if present).

    Returns an empty list if the registry directory does not exist.
    """
    root = Path(registry_root) if registry_root is not None else _DEFAULT_REGISTRY
    if not root.exists():
        return []

    experiments: list[dict[str, Any]] = []
    for exp_dir in sorted(root.iterdir(), reverse=True):
        if not exp_dir.is_dir():
            continue
        detail = load_experiment_detail(exp_dir)
        if detail:
            experiments.append(detail)

    return experiments


def load_experiment_detail(exp_dir: Path) -> dict[str, Any] | None:
    """Load metadata, metrics, and report text for one experiment directory.

    Returns ``None`` if ``metadata.json`` is missing or malformed.
    """
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        return None

    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse metadata.json in %s: %s", exp_dir, exc)
        return None

    # Load metrics
    metrics_path = exp_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse metrics.json in %s: %s", exp_dir, exc)

    # Load report text
    report_text: str = ""
    report_path = exp_dir / "report.md"
    if report_path.exists():
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read report.md in %s: %s", exp_dir, exc)

    return {
        **metadata,
        "metrics": metrics,
        "report_text": report_text,
        "exp_dir": str(exp_dir),
    }


def extract_key_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Extract the most display-relevant scalar metrics from a metrics dict.

    Returns a flat dict of ``{label: value}`` suitable for a metrics table.
    Values that cannot be expressed as a float are omitted.
    """
    result: dict[str, float | None] = {}

    def _try_float(v: Any) -> float | None:
        try:
            f = float(v)
            import math
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    # Backtest top-level scalars
    for key in ("sharpe_net", "sharpe_gross", "ann_net_return", "ann_gross_return",
                "max_drawdown_net", "hit_rate", "avg_turnover", "n_periods"):
        if key in metrics:
            v = _try_float(metrics[key])
            if v is not None:
                result[key] = v

    # IC-level scalars (baseline / sector baselines)
    for key in ("mean_ic", "ic_t_stat", "mean_rank_ic"):
        if key in metrics:
            v = _try_float(metrics[key])
            if v is not None:
                result[key] = v

    # Nested: first label in "labels" list
    labels_list = metrics.get("labels", [])
    if labels_list and isinstance(labels_list, list) and isinstance(labels_list[0], dict):
        first = labels_list[0]
        for key in ("mean_ic", "ic_t_stat"):
            if key in first and key not in result:
                v = _try_float(first[key])
                if v is not None:
                    result[f"label0_{key}"] = v

    # Nested: first comparison in "comparisons" list
    comparisons = metrics.get("comparisons", [])
    if comparisons and isinstance(comparisons, list) and isinstance(comparisons[0], dict):
        first = comparisons[0]
        for key in ("price_macro_news_mean_ic", "price_macro_mean_ic", "price_only_mean_ic"):
            if key in first:
                v = _try_float(first[key])
                if v is not None:
                    result[key] = v
                    break  # only take the richest available

    return result


# ── Graph ──────────────────────────────────────────────────────────────────────

def load_graph_json(path: Path | None = None) -> dict[str, Any] | None:
    """Load the research graph JSON file.

    Returns the parsed dict, or ``None`` if the file does not exist or is
    not valid JSON.
    """
    p = Path(path) if path is not None else _DEFAULT_GRAPH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse graph JSON at %s: %s", p, exc)
        return None


def graph_node_table(graph_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a flat list of node dicts suitable for a DataFrame."""
    rows = []
    for node in graph_data.get("nodes", []):
        rows.append({
            "node_id": node.get("node_id", ""),
            "node_type": node.get("node_type", ""),
            "label": node.get("label", ""),
        })
    return rows


def graph_edge_table(graph_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a flat list of edge dicts suitable for a DataFrame."""
    rows = []
    for edge in graph_data.get("edges", []):
        rows.append({
            "source": edge.get("source_id", ""),
            "edge_type": edge.get("edge_type", ""),
            "target": edge.get("target_id", ""),
        })
    return rows


# ── Reports ────────────────────────────────────────────────────────────────────

def find_reports(
    reports_dir: Path | None = None,
    registry_root: Path | None = None,
) -> list[dict[str, str]]:
    """Return all Markdown report paths from two sources.

    1. ``reports/generated/`` (static CLI outputs)
    2. ``memory/experiments/<id>/report.md`` (experiment-scoped reports)

    Each entry is a dict with keys ``"label"`` and ``"path"``.
    """
    rpt_root = Path(reports_dir) if reports_dir is not None else _DEFAULT_REPORTS
    reg_root = Path(registry_root) if registry_root is not None else _DEFAULT_REGISTRY

    reports: list[dict[str, str]] = []

    if rpt_root.exists():
        for md in sorted(rpt_root.glob("*.md")):
            reports.append({"label": md.name, "path": str(md)})

    if reg_root.exists():
        for exp_dir in sorted(reg_root.iterdir(), reverse=True):
            rpt = exp_dir / "report.md"
            if rpt.exists():
                reports.append({
                    "label": f"{exp_dir.name}/report.md",
                    "path": str(rpt),
                })

    return reports


def load_report_text(path: str | Path) -> str | None:
    """Read and return the content of a Markdown report file.

    Returns ``None`` if the file does not exist or cannot be read.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read report at %s: %s", p, exc)
        return None


def find_latest_backtest(experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the most recent experiment with name 'run_backtest', or None."""
    for exp in experiments:
        if exp.get("name") == "run_backtest":
            return exp
    return None
