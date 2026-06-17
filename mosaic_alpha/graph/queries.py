"""Query helpers for the MosaicAlpha research knowledge graph.

All functions accept a :class:`~mosaic_alpha.graph.schema.ResearchGraph`
and return plain Python objects (lists of dicts or lists of node IDs) so
that results are easy to format in the CLI or process programmatically.

Usage
-----
    from mosaic_alpha.graph.builder import build_graph
    from mosaic_alpha.graph.queries import (
        find_experiments_using_dataset,
        find_features_used_by_experiment,
        find_best_experiments_by_metric,
        find_experiments_with_limitation,
    )

    graph = build_graph()
    print(find_experiments_using_dataset(graph, "GDELT"))
"""

from __future__ import annotations

import math
from typing import Any

from mosaic_alpha.graph.schema import GraphNode, ResearchGraph


# ── Helpers ────────────────────────────────────────────────────────────────────

def _experiment_nodes(graph: ResearchGraph) -> list[GraphNode]:
    return graph.nodes_by_type("experiment")


def _node_label(graph: ResearchGraph, node_id: str) -> str:
    node = graph.get_node(node_id)
    return node.label if node else node_id


# ── Public query functions ─────────────────────────────────────────────────────

def find_experiments_using_dataset(
    graph: ResearchGraph,
    dataset_name: str,
) -> list[dict[str, Any]]:
    """Return experiments that used a given dataset (data source).

    Matching is case-insensitive substring matching on the dataset label so
    that ``"gdelt"`` matches ``"GDELT"``.

    Parameters
    ----------
    graph:
        The research graph to query.
    dataset_name:
        Name (or substring) of the dataset node to search for.

    Returns
    -------
    List of dicts with keys ``experiment_id``, ``name``, ``created_at``,
    ``start_date``, ``end_date``.
    """
    target = dataset_name.lower()

    # Find all dataset nodes whose label matches
    matching_dataset_ids = {
        n.node_id
        for n in graph.nodes_by_type("dataset")
        if target in n.label.lower()
    }

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for edge in graph.edges_by_type("produces"):
        if edge.source_id not in matching_dataset_ids:
            continue
        exp_node = graph.get_node(edge.target_id)
        if exp_node is None or exp_node.node_type != "experiment":
            continue
        exp_id = exp_node.node_id
        if exp_id in seen:
            continue
        seen.add(exp_id)
        results.append({
            "experiment_id": exp_node.attributes.get("experiment_id", ""),
            "name": exp_node.label,
            "created_at": exp_node.attributes.get("created_at", ""),
            "start_date": exp_node.attributes.get("start_date", ""),
            "end_date": exp_node.attributes.get("end_date", ""),
        })

    return results


def find_features_used_by_experiment(
    graph: ResearchGraph,
    experiment_id: str,
) -> list[str]:
    """Return the feature families used by a specific experiment.

    Parameters
    ----------
    graph:
        The research graph to query.
    experiment_id:
        The experiment's ``experiment_id`` string (not the graph node_id).

    Returns
    -------
    Sorted list of feature family names (e.g. ``["macro", "news", "price"]``).
    """
    exp_node_id = f"experiment:{experiment_id}"
    features: list[str] = []

    for edge in graph.outgoing_edges(exp_node_id):
        if edge.edge_type != "uses":
            continue
        target = graph.get_node(edge.target_id)
        if target is not None and target.node_type == "feature":
            features.append(target.label)

    return sorted(set(features))


def find_best_experiments_by_metric(
    graph: ResearchGraph,
    metric_name: str,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-N experiments ranked by a named numeric metric.

    Matching is case-insensitive substring matching on the metric label so
    that ``"sharpe"`` matches ``"sharpe_net"`` and ``"sharpe_gross"``.

    Parameters
    ----------
    graph:
        The research graph to query.
    metric_name:
        Metric name substring to search for.
    top_n:
        Maximum number of results to return (default 5).

    Returns
    -------
    List of dicts (highest metric value first) with keys
    ``experiment_id``, ``name``, ``metric_label``, ``value``.
    """
    target = metric_name.lower()
    rows: list[dict[str, Any]] = []

    for metric_node in graph.nodes_by_type("metric"):
        label = metric_node.label.lower()
        if target not in label:
            continue
        value = metric_node.attributes.get("value")
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue

        exp_id_raw = metric_node.attributes.get("experiment_id", "")
        exp_node = graph.get_node(f"experiment:{exp_id_raw}")
        exp_name = exp_node.label if exp_node else exp_id_raw

        rows.append({
            "experiment_id": exp_id_raw,
            "name": exp_name,
            "metric_label": metric_node.label,
            "value": float(value),
        })

    rows.sort(key=lambda r: r["value"], reverse=True)
    return rows[:top_n]


def find_experiments_with_limitation(
    graph: ResearchGraph,
    keyword: str,
) -> list[dict[str, Any]]:
    """Return experiments that have a limitation containing *keyword*.

    Matching is case-insensitive substring matching on the limitation text.

    Parameters
    ----------
    graph:
        The research graph to query.
    keyword:
        Word or phrase to search for in limitation texts.

    Returns
    -------
    List of dicts with keys ``experiment_id``, ``name``, ``limitation``.
    """
    target = keyword.lower()

    # Find matching limitation node IDs
    matching_lim_ids = {
        n.node_id
        for n in graph.nodes_by_type("limitation")
        if target in n.attributes.get("text", "").lower()
    }

    results: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for edge in graph.edges_by_type("has_limitation"):
        if edge.target_id not in matching_lim_ids:
            continue
        exp_node = graph.get_node(edge.source_id)
        lim_node = graph.get_node(edge.target_id)
        if exp_node is None or lim_node is None:
            continue

        key = (exp_node.node_id, lim_node.node_id)
        if key in seen:
            continue
        seen.add(key)

        results.append({
            "experiment_id": exp_node.attributes.get("experiment_id", ""),
            "name": exp_node.label,
            "limitation": lim_node.attributes.get("text", lim_node.label),
        })

    return results
