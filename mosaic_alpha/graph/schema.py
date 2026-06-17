"""Research knowledge graph schema for MosaicAlpha.

The graph captures research lineage: which datasets fed which experiments,
which features were used, how models were evaluated, and what was found.

Node types
----------
dataset     : a raw data source (yfinance, FRED, GDELT, ...)
feature     : a computed feature family (price, macro, news, ...)
model       : a model class (ridge, ...)
experiment  : a single experiment run (keyed by experiment_id)
metric      : a numeric performance metric attached to one experiment
report      : an output file (Markdown report path)
limitation  : a documented limitation of an experiment
finding     : a high-level research conclusion derived from metrics

Edge types
----------
produces        : dataset → experiment  (data feeds the experiment)
uses            : experiment → feature | model  (experiment uses these inputs)
evaluated_by    : experiment → metric   (how the experiment was assessed)
has_metric      : experiment → metric   (alias, same semantics as evaluated_by)
reports         : experiment → report   (experiment generated this file)
has_limitation  : experiment → limitation
supports        : metric → finding      (a good metric supports a finding)
weakens         : metric → finding      (a bad metric weakens a finding)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# ── Allowed vocabularies ───────────────────────────────────────────────────────

NODE_TYPES = frozenset({
    "dataset",
    "feature",
    "model",
    "experiment",
    "metric",
    "report",
    "limitation",
    "finding",
})

EDGE_TYPES = frozenset({
    "produces",
    "uses",
    "evaluated_by",
    "has_metric",
    "reports",
    "has_limitation",
    "supports",
    "weakens",
})


# ── Core schema ────────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    """A node in the research knowledge graph.

    Parameters
    ----------
    node_id:
        Unique identifier, e.g. ``"dataset:GDELT"`` or
        ``"experiment:20240601_120000_run_baseline"``.
    node_type:
        One of the values in :data:`NODE_TYPES`.
    label:
        Human-readable display name.
    attributes:
        Arbitrary key-value metadata (dates, metric values, …).
    """

    node_id: str
    node_type: str
    label: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.node_type not in NODE_TYPES:
            raise ValueError(
                f"Unknown node_type {self.node_type!r}. "
                f"Allowed: {sorted(NODE_TYPES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphEdge:
    """A directed edge in the research knowledge graph.

    Parameters
    ----------
    source_id:
        node_id of the source node.
    target_id:
        node_id of the target node.
    edge_type:
        One of the values in :data:`EDGE_TYPES`.
    attributes:
        Arbitrary key-value metadata.
    """

    source_id: str
    target_id: str
    edge_type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(
                f"Unknown edge_type {self.edge_type!r}. "
                f"Allowed: {sorted(EDGE_TYPES)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResearchGraph:
    """Container for the full research knowledge graph.

    Nodes and edges are stored as plain lists for easy serialisation.
    Use the helper methods for fast lookup by ID or type.
    """

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    # ── Mutation helpers ───────────────────────────────────────────────────────

    def add_node(self, node: GraphNode) -> GraphNode:
        """Add a node; silently skip duplicates (same node_id)."""
        if not self._has_node(node.node_id):
            self.nodes.append(node)
        return node

    def add_edge(self, edge: GraphEdge) -> GraphEdge:
        """Add an edge; silently skip exact duplicates."""
        if not self._has_edge(edge.source_id, edge.target_id, edge.edge_type):
            self.edges.append(edge)
        return edge

    def get_node(self, node_id: str) -> GraphNode | None:
        """Return the node with the given ID, or None."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    # ── Lookup helpers ─────────────────────────────────────────────────────────

    def nodes_by_type(self, node_type: str) -> list[GraphNode]:
        return [n for n in self.nodes if n.node_type == node_type]

    def edges_by_type(self, edge_type: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.edge_type == edge_type]

    def outgoing_edges(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.source_id == node_id]

    def incoming_edges(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if e.target_id == node_id]

    # ── Summary ────────────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Return a dict of node/edge counts by type."""
        node_counts: dict[str, int] = {}
        for n in self.nodes:
            node_counts[n.node_type] = node_counts.get(n.node_type, 0) + 1
        edge_counts: dict[str, int] = {}
        for e in self.edges:
            edge_counts[e.edge_type] = edge_counts.get(e.edge_type, 0) + 1
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_by_type": node_counts,
            "edges_by_type": edge_counts,
        }

    # ── Private ────────────────────────────────────────────────────────────────

    def _has_node(self, node_id: str) -> bool:
        return any(n.node_id == node_id for n in self.nodes)

    def _has_edge(self, source_id: str, target_id: str, edge_type: str) -> bool:
        return any(
            e.source_id == source_id
            and e.target_id == target_id
            and e.edge_type == edge_type
            for e in self.edges
        )
