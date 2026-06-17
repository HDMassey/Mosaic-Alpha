"""Export a ResearchGraph to portable file formats.

Supported formats
-----------------
JSON
    Human-readable, Git-friendly.  Written to ``memory/research_graph.json``.
    The schema is a plain dict with ``nodes`` and ``edges`` lists — no
    external dependencies required.

GraphML
    XML-based graph format readable by Gephi, Cytoscape, and networkx.
    Written to ``memory/research_graph.graphml``.
    Requires networkx (already a project dependency).

Usage
-----
    from mosaic_alpha.graph.builder import build_graph
    from mosaic_alpha.graph.export import export_json, export_graphml

    g = build_graph()
    export_json(g)
    export_graphml(g)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mosaic_alpha.graph.schema import ResearchGraph

logger = logging.getLogger(__name__)

_DEFAULT_JSON = Path("memory/research_graph.json")
_DEFAULT_GRAPHML = Path("memory/research_graph.graphml")


# ── JSON export ────────────────────────────────────────────────────────────────

def export_json(
    graph: ResearchGraph,
    path: Path | None = None,
) -> Path:
    """Serialise the graph to JSON and write it to *path*.

    The output is a dict with two keys:

    - ``nodes``: list of node dicts (node_id, node_type, label, attributes)
    - ``edges``: list of edge dicts (source_id, target_id, edge_type, attributes)

    Returns the path written.
    """
    out = Path(path) if path is not None else _DEFAULT_JSON
    out.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "nodes": [n.to_dict() for n in graph.nodes],
        "edges": [e.to_dict() for e in graph.edges],
        "summary": graph.summary(),
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Graph exported to JSON: %s (%d nodes, %d edges)", out, len(graph.nodes), len(graph.edges))
    return out


# ── GraphML export ─────────────────────────────────────────────────────────────

def export_graphml(
    graph: ResearchGraph,
    path: Path | None = None,
) -> Path | None:
    """Serialise the graph to GraphML via networkx and write it to *path*.

    Returns the path written, or ``None`` if networkx is unavailable.

    Attribute values that are not serialisable as strings are converted via
    ``str()`` so that lists, dicts, and None values do not cause errors.
    """
    try:
        import networkx as nx  # noqa: PLC0415
    except ImportError:
        logger.warning("networkx not available; skipping GraphML export.")
        return None

    out = Path(path) if path is not None else _DEFAULT_GRAPHML
    out.parent.mkdir(parents=True, exist_ok=True)

    G = nx.DiGraph()

    for node in graph.nodes:
        # networkx requires string-valued attributes for clean GraphML
        attrs = {
            "node_type": node.node_type,
            "label": node.label,
        }
        for k, v in node.attributes.items():
            attrs[k] = _to_graphml_value(v)
        G.add_node(node.node_id, **attrs)

    for edge in graph.edges:
        attrs = {"edge_type": edge.edge_type}
        for k, v in edge.attributes.items():
            attrs[k] = _to_graphml_value(v)
        G.add_edge(edge.source_id, edge.target_id, **attrs)

    nx.write_graphml(G, str(out), encoding="utf-8")
    logger.info("Graph exported to GraphML: %s", out)
    return out


def _to_graphml_value(v: Any) -> str:
    """Convert a value to a GraphML-safe string representation."""
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


# ── Convenience: export both ───────────────────────────────────────────────────

def export_all(
    graph: ResearchGraph,
    *,
    json_path: Path | None = None,
    graphml_path: Path | None = None,
) -> dict[str, Path | None]:
    """Export the graph to JSON and GraphML in one call.

    Returns a dict with keys ``"json"`` and ``"graphml"`` mapping to the
    paths written (``None`` if a format was skipped).
    """
    return {
        "json": export_json(graph, json_path),
        "graphml": export_graphml(graph, graphml_path),
    }
