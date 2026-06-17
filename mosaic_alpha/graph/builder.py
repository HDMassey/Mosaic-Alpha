"""Build a ResearchGraph from the local experiment registry.

Each ExperimentRecord is translated into a cluster of nodes and edges:

  dataset node(s)   --[produces]-->  experiment node
  experiment node   --[uses]-->      feature node(s)
  experiment node   --[uses]-->      model node
  experiment node   --[has_metric]--> metric node(s)
  experiment node   --[reports]-->   report node(s)
  experiment node   --[has_limitation]--> limitation node(s)

Shared infrastructure nodes (datasets, features, models) are deduplicated
across experiments so that the graph captures the common lineage.

Usage
-----
    from mosaic_alpha.graph.builder import build_graph
    from mosaic_alpha.research.registry import list_experiments

    records = list_experiments()
    graph = build_graph(records)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from mosaic_alpha.graph.schema import GraphEdge, GraphNode, ResearchGraph
from mosaic_alpha.research.registry import ExperimentRecord, list_experiments

logger = logging.getLogger(__name__)


# ── ID helpers ─────────────────────────────────────────────────────────────────

def _dataset_id(name: str) -> str:
    return f"dataset:{name}"

def _feature_id(name: str) -> str:
    return f"feature:{name}"

def _model_id(name: str) -> str:
    return f"model:{name}"

def _experiment_id(exp_id: str) -> str:
    return f"experiment:{exp_id}"

def _metric_id(exp_id: str, metric_name: str) -> str:
    return f"metric:{exp_id}:{metric_name}"

def _report_id(exp_id: str, index: int) -> str:
    return f"report:{exp_id}:{index}"

def _limitation_id(text: str) -> str:
    # Use a short hash so identical limitation texts share a node across
    # experiments, helping surface systemic issues.
    digest = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()[:8]
    return f"limitation:{digest}"


# ── Per-experiment graph population ────────────────────────────────────────────

def _add_experiment(graph: ResearchGraph, rec: ExperimentRecord) -> None:
    """Add all nodes and edges for one ExperimentRecord into *graph*."""

    exp_node_id = _experiment_id(rec.experiment_id)

    # ── Experiment node ────────────────────────────────────────────────────────
    graph.add_node(GraphNode(
        node_id=exp_node_id,
        node_type="experiment",
        label=rec.name,
        attributes={
            "experiment_id": rec.experiment_id,
            "created_at": rec.created_at,
            "command": rec.command,
            "start_date": rec.start_date,
            "end_date": rec.end_date,
            "universe": rec.universe,
            "validation_method": rec.validation_method,
            "notes": rec.notes,
        },
    ))

    # ── Dataset nodes (data_sources) → experiment ──────────────────────────────
    for src in rec.data_sources:
        src_node_id = _dataset_id(src)
        graph.add_node(GraphNode(
            node_id=src_node_id,
            node_type="dataset",
            label=src,
            attributes={"source": src},
        ))
        graph.add_edge(GraphEdge(
            source_id=src_node_id,
            target_id=exp_node_id,
            edge_type="produces",
        ))

    # ── Feature nodes → experiment ─────────────────────────────────────────────
    for feat in rec.feature_sets:
        feat_node_id = _feature_id(feat)
        graph.add_node(GraphNode(
            node_id=feat_node_id,
            node_type="feature",
            label=feat,
            attributes={"feature_family": feat},
        ))
        graph.add_edge(GraphEdge(
            source_id=exp_node_id,
            target_id=feat_node_id,
            edge_type="uses",
        ))

    # ── Model node ─────────────────────────────────────────────────────────────
    if rec.model_type:
        model_node_id = _model_id(rec.model_type)
        graph.add_node(GraphNode(
            node_id=model_node_id,
            node_type="model",
            label=rec.model_type,
            attributes={"model_type": rec.model_type},
        ))
        graph.add_edge(GraphEdge(
            source_id=exp_node_id,
            target_id=model_node_id,
            edge_type="uses",
        ))

    # ── Metric nodes ───────────────────────────────────────────────────────────
    _add_metrics(graph, rec, exp_node_id)

    # ── Report nodes ───────────────────────────────────────────────────────────
    for idx, path_str in enumerate(rec.output_files):
        rpt_node_id = _report_id(rec.experiment_id, idx)
        graph.add_node(GraphNode(
            node_id=rpt_node_id,
            node_type="report",
            label=Path(path_str).name,
            attributes={"path": path_str},
        ))
        graph.add_edge(GraphEdge(
            source_id=exp_node_id,
            target_id=rpt_node_id,
            edge_type="reports",
        ))

    # ── Limitation nodes ───────────────────────────────────────────────────────
    for lim in rec.limitations:
        lim_node_id = _limitation_id(lim)
        graph.add_node(GraphNode(
            node_id=lim_node_id,
            node_type="limitation",
            label=lim[:80],          # truncate for display
            attributes={"text": lim},
        ))
        graph.add_edge(GraphEdge(
            source_id=exp_node_id,
            target_id=lim_node_id,
            edge_type="has_limitation",
        ))


def _add_metrics(
    graph: ResearchGraph,
    rec: ExperimentRecord,
    exp_node_id: str,
) -> None:
    """Extract numeric leaves from metrics_summary and add metric nodes."""
    metrics = rec.metrics_summary
    if not metrics:
        return

    # Flatten the metrics dict into (name, value) pairs of numeric scalars.
    flat = _flatten_metrics(metrics)

    for metric_name, value in flat:
        m_node_id = _metric_id(rec.experiment_id, metric_name)
        graph.add_node(GraphNode(
            node_id=m_node_id,
            node_type="metric",
            label=metric_name,
            attributes={
                "metric_name": metric_name,
                "value": value,
                "experiment_id": rec.experiment_id,
            },
        ))
        graph.add_edge(GraphEdge(
            source_id=exp_node_id,
            target_id=m_node_id,
            edge_type="has_metric",
        ))


def _flatten_metrics(
    obj: Any,
    prefix: str = "",
    depth: int = 0,
) -> list[tuple[str, float]]:
    """Recursively extract (name, value) pairs of numeric scalars from a dict/list."""
    if depth > 4:
        return []
    pairs: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else k
            pairs.extend(_flatten_metrics(v, key, depth + 1))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            pairs.extend(_flatten_metrics(v, key, depth + 1))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        pairs.append((prefix, float(obj)))
    return pairs


# ── Public entry point ─────────────────────────────────────────────────────────

def build_graph(
    records: list[ExperimentRecord] | None = None,
    *,
    registry_root: Path | None = None,
) -> ResearchGraph:
    """Build a ResearchGraph from a list of ExperimentRecords.

    Parameters
    ----------
    records:
        Pre-loaded list of ExperimentRecords.  If ``None``, the registry
        is read from *registry_root* (or the default ``memory/experiments``).
    registry_root:
        Registry directory.  Ignored when *records* is provided.

    Returns
    -------
    :class:`~mosaic_alpha.graph.schema.ResearchGraph`
    """
    if records is None:
        records = list_experiments(registry_root=registry_root)

    graph = ResearchGraph()

    for rec in records:
        try:
            _add_experiment(graph, rec)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping experiment %s due to error: %s",
                rec.experiment_id, exc,
            )

    logger.info(
        "Graph built: %d nodes, %d edges from %d experiments",
        len(graph.nodes), len(graph.edges), len(records),
    )
    return graph
