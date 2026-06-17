"""Tests for Milestone 7: research knowledge graph.

Categories
----------
1. schema: GraphNode, GraphEdge, ResearchGraph construction and validation.
2. builder: graph is built from experiment records with correct node/edge types.
3. export: JSON export writes valid, loadable JSON; GraphML if networkx present.
4. queries: dataset / metric / limitation queries return correct results.
5. CLI: build-graph, graph-summary, graph-query commands run without error.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic_alpha.cli import app
from mosaic_alpha.graph.builder import build_graph
from mosaic_alpha.graph.export import export_json
from mosaic_alpha.graph.queries import (
    find_best_experiments_by_metric,
    find_experiments_using_dataset,
    find_experiments_with_limitation,
    find_features_used_by_experiment,
)
from mosaic_alpha.graph.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    GraphEdge,
    GraphNode,
    ResearchGraph,
)
from mosaic_alpha.research.registry import build_record, save_experiment

runner = CliRunner()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_registry(tmp_path: Path) -> Path:
    """Populate a temporary registry with two experiment records."""
    reg = tmp_path / "experiments"

    rec1 = build_record(
        name="run_baseline",
        command="mosaic run-baseline --ticker SPY",
        start_date="2015-01-01",
        end_date="2024-12-31",
        universe=["SPY"],
        data_sources=["yfinance"],
        feature_sets=["price"],
        model_type="ridge",
        metrics_summary={"mean_ic": 0.12, "ic_t_stat": 3.1},
        output_files=["reports/generated/baseline_spy.md"],
        limitations=["Price features only; no alternative data."],
    )
    save_experiment(rec1, registry_root=reg)

    time.sleep(1.01)  # ensure different timestamp

    rec2 = build_record(
        name="run_backtest",
        command="mosaic run-backtest --offline-sample",
        start_date="2020-01-01",
        end_date="2024-12-31",
        universe=["XLK", "XLE", "XLF"],
        data_sources=["yfinance", "GDELT", "FRED"],
        feature_sets=["price", "macro", "news"],
        model_type="ridge",
        metrics_summary={
            "sharpe_net": 1.25,
            "ann_net_return": 0.08,
            "n_periods": 50,
        },
        output_files=["reports/generated/backtest.md"],
        limitations=[
            "SAMPLE DATA: backtest uses synthetic news counts.",
            "Universe limited to 11 sector ETFs; Sharpe ratios have wide CIs.",
        ],
    )
    save_experiment(rec2, registry_root=reg)

    return reg


def _build_test_graph(tmp_path: Path) -> ResearchGraph:
    reg = _make_registry(tmp_path)
    return build_graph(registry_root=reg)


# ── 1. Schema ───────────────────────────────────────────────────────────────────

def test_node_types_are_defined():
    """NODE_TYPES must include the required vocabulary."""
    required = {"dataset", "feature", "model", "experiment", "metric", "report", "limitation", "finding"}
    assert required <= NODE_TYPES


def test_edge_types_are_defined():
    """EDGE_TYPES must include the required vocabulary."""
    required = {"produces", "uses", "evaluated_by", "has_metric", "reports", "has_limitation", "supports", "weakens"}
    assert required <= EDGE_TYPES


def test_graph_node_valid():
    node = GraphNode(node_id="dataset:GDELT", node_type="dataset", label="GDELT")
    assert node.node_id == "dataset:GDELT"
    assert node.node_type == "dataset"


def test_graph_node_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown node_type"):
        GraphNode(node_id="foo:bar", node_type="unknown_type", label="oops")


def test_graph_edge_valid():
    edge = GraphEdge(source_id="a", target_id="b", edge_type="produces")
    assert edge.edge_type == "produces"


def test_graph_edge_rejects_unknown_type():
    with pytest.raises(ValueError, match="Unknown edge_type"):
        GraphEdge(source_id="a", target_id="b", edge_type="invented_edge")


def test_research_graph_add_node_deduplication():
    """Adding the same node_id twice must not create duplicates."""
    g = ResearchGraph()
    g.add_node(GraphNode("n1", "dataset", "DS"))
    g.add_node(GraphNode("n1", "dataset", "DS"))
    assert len(g.nodes) == 1


def test_research_graph_add_edge_deduplication():
    """Adding the same (source, target, type) edge twice must not create duplicates."""
    g = ResearchGraph()
    g.add_edge(GraphEdge("a", "b", "produces"))
    g.add_edge(GraphEdge("a", "b", "produces"))
    assert len(g.edges) == 1


def test_research_graph_get_node():
    g = ResearchGraph()
    node = GraphNode("n1", "dataset", "DS")
    g.add_node(node)
    assert g.get_node("n1") is node
    assert g.get_node("missing") is None


def test_research_graph_nodes_by_type():
    g = ResearchGraph()
    g.add_node(GraphNode("d1", "dataset", "A"))
    g.add_node(GraphNode("d2", "dataset", "B"))
    g.add_node(GraphNode("e1", "experiment", "X"))
    assert len(g.nodes_by_type("dataset")) == 2
    assert len(g.nodes_by_type("experiment")) == 1


def test_research_graph_summary():
    g = ResearchGraph()
    g.add_node(GraphNode("d1", "dataset", "A"))
    g.add_node(GraphNode("e1", "experiment", "X"))
    g.add_edge(GraphEdge("d1", "e1", "produces"))
    s = g.summary()
    assert s["total_nodes"] == 2
    assert s["total_edges"] == 1
    assert s["nodes_by_type"]["dataset"] == 1
    assert s["edges_by_type"]["produces"] == 1


# ── 2. Builder ─────────────────────────────────────────────────────────────────

def test_build_graph_creates_experiment_nodes(tmp_path):
    g = _build_test_graph(tmp_path)
    exp_nodes = g.nodes_by_type("experiment")
    assert len(exp_nodes) == 2
    names = {n.label for n in exp_nodes}
    assert "run_baseline" in names
    assert "run_backtest" in names


def test_build_graph_creates_dataset_nodes(tmp_path):
    g = _build_test_graph(tmp_path)
    ds_nodes = g.nodes_by_type("dataset")
    ds_labels = {n.label for n in ds_nodes}
    assert "yfinance" in ds_labels
    assert "GDELT" in ds_labels
    assert "FRED" in ds_labels


def test_build_graph_creates_feature_nodes(tmp_path):
    g = _build_test_graph(tmp_path)
    feat_nodes = g.nodes_by_type("feature")
    feat_labels = {n.label for n in feat_nodes}
    assert "price" in feat_labels
    assert "macro" in feat_labels
    assert "news" in feat_labels


def test_build_graph_creates_model_node(tmp_path):
    g = _build_test_graph(tmp_path)
    model_nodes = g.nodes_by_type("model")
    assert len(model_nodes) >= 1
    assert any(n.label == "ridge" for n in model_nodes)


def test_build_graph_creates_metric_nodes(tmp_path):
    g = _build_test_graph(tmp_path)
    metric_nodes = g.nodes_by_type("metric")
    assert len(metric_nodes) >= 1


def test_build_graph_creates_report_nodes(tmp_path):
    g = _build_test_graph(tmp_path)
    report_nodes = g.nodes_by_type("report")
    assert len(report_nodes) == 2   # one per experiment


def test_build_graph_creates_limitation_nodes(tmp_path):
    g = _build_test_graph(tmp_path)
    lim_nodes = g.nodes_by_type("limitation")
    assert len(lim_nodes) >= 2   # at least one per experiment


def test_build_graph_produces_edges_exist(tmp_path):
    g = _build_test_graph(tmp_path)
    produces = g.edges_by_type("produces")
    assert len(produces) >= 1   # at least dataset→experiment


def test_build_graph_uses_edges_exist(tmp_path):
    g = _build_test_graph(tmp_path)
    uses = g.edges_by_type("uses")
    assert len(uses) >= 1   # experiment→feature and experiment→model


def test_build_graph_has_metric_edges_exist(tmp_path):
    g = _build_test_graph(tmp_path)
    hm = g.edges_by_type("has_metric")
    assert len(hm) >= 1


def test_build_graph_has_limitation_edges_exist(tmp_path):
    g = _build_test_graph(tmp_path)
    hl = g.edges_by_type("has_limitation")
    assert len(hl) >= 1


def test_build_graph_reports_edges_exist(tmp_path):
    g = _build_test_graph(tmp_path)
    rp = g.edges_by_type("reports")
    assert len(rp) == 2


def test_build_graph_empty_registry(tmp_path):
    """build_graph on an empty registry must return an empty graph."""
    empty = tmp_path / "empty"
    g = build_graph(registry_root=empty)
    assert g.nodes == []
    assert g.edges == []


def test_build_graph_deduplicates_shared_datasets(tmp_path):
    """Both experiments use yfinance; there must be only one yfinance dataset node."""
    g = _build_test_graph(tmp_path)
    yf_nodes = [n for n in g.nodes_by_type("dataset") if n.label == "yfinance"]
    assert len(yf_nodes) == 1


# ── 3. Export ──────────────────────────────────────────────────────────────────

def test_export_json_writes_file(tmp_path):
    g = _build_test_graph(tmp_path)
    out = tmp_path / "graph.json"
    export_json(g, out)
    assert out.exists()


def test_export_json_is_valid_json(tmp_path):
    g = _build_test_graph(tmp_path)
    out = tmp_path / "graph.json"
    export_json(g, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "nodes" in data
    assert "edges" in data
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)


def test_export_json_round_trips_nodes(tmp_path):
    g = _build_test_graph(tmp_path)
    out = tmp_path / "graph.json"
    export_json(g, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data["nodes"]) == len(g.nodes)
    assert len(data["edges"]) == len(g.edges)


def test_export_json_contains_summary(tmp_path):
    g = _build_test_graph(tmp_path)
    out = tmp_path / "graph.json"
    export_json(g, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "summary" in data
    assert data["summary"]["total_nodes"] == len(g.nodes)


def test_export_graphml_produces_file_if_networkx_available(tmp_path):
    try:
        import networkx  # noqa: F401
    except ImportError:
        pytest.skip("networkx not installed")

    from mosaic_alpha.graph.export import export_graphml
    g = _build_test_graph(tmp_path)
    out = tmp_path / "graph.graphml"
    result = export_graphml(g, out)
    assert result is not None
    assert out.exists()
    # GraphML files are XML; check minimal structure
    content = out.read_text(encoding="utf-8")
    assert "<graphml" in content
    assert "<node" in content


# ── 4. Queries ─────────────────────────────────────────────────────────────────

def test_query_dataset_gdelt(tmp_path):
    g = _build_test_graph(tmp_path)
    results = find_experiments_using_dataset(g, "GDELT")
    assert len(results) == 1
    assert results[0]["name"] == "run_backtest"


def test_query_dataset_yfinance(tmp_path):
    """yfinance is used by both experiments."""
    g = _build_test_graph(tmp_path)
    results = find_experiments_using_dataset(g, "yfinance")
    assert len(results) == 2


def test_query_dataset_case_insensitive(tmp_path):
    g = _build_test_graph(tmp_path)
    results_upper = find_experiments_using_dataset(g, "GDELT")
    results_lower = find_experiments_using_dataset(g, "gdelt")
    assert len(results_upper) == len(results_lower)


def test_query_dataset_no_match(tmp_path):
    g = _build_test_graph(tmp_path)
    results = find_experiments_using_dataset(g, "NOAA")
    assert results == []


def test_query_features_by_experiment(tmp_path):
    reg = _make_registry(tmp_path)
    g = build_graph(registry_root=reg)

    # Find the run_backtest experiment_id
    bt_node = next(n for n in g.nodes_by_type("experiment") if n.label == "run_backtest")
    exp_id = bt_node.attributes["experiment_id"]

    features = find_features_used_by_experiment(g, exp_id)
    assert "price" in features
    assert "macro" in features
    assert "news" in features


def test_query_features_baseline_only_price(tmp_path):
    reg = _make_registry(tmp_path)
    g = build_graph(registry_root=reg)

    bl_node = next(n for n in g.nodes_by_type("experiment") if n.label == "run_baseline")
    exp_id = bl_node.attributes["experiment_id"]

    features = find_features_used_by_experiment(g, exp_id)
    assert features == ["price"]


def test_query_metric_sharpe(tmp_path):
    g = _build_test_graph(tmp_path)
    results = find_best_experiments_by_metric(g, "sharpe_net")
    assert len(results) >= 1
    assert results[0]["name"] == "run_backtest"
    assert results[0]["value"] == pytest.approx(1.25)


def test_query_metric_mean_ic(tmp_path):
    g = _build_test_graph(tmp_path)
    results = find_best_experiments_by_metric(g, "mean_ic")
    assert len(results) >= 1
    # Highest mean_ic should be baseline (0.12)
    assert results[0]["value"] == pytest.approx(0.12)


def test_query_metric_no_match(tmp_path):
    g = _build_test_graph(tmp_path)
    results = find_best_experiments_by_metric(g, "nonexistent_metric_xyz")
    assert results == []


def test_query_metric_top_n_limit(tmp_path):
    g = _build_test_graph(tmp_path)
    results = find_best_experiments_by_metric(g, "ic", top_n=1)
    assert len(results) <= 1


def test_query_limitation_keyword(tmp_path):
    g = _build_test_graph(tmp_path)
    results = find_experiments_with_limitation(g, "SAMPLE")
    assert len(results) >= 1
    names = {r["name"] for r in results}
    assert "run_backtest" in names


def test_query_limitation_no_match(tmp_path):
    g = _build_test_graph(tmp_path)
    results = find_experiments_with_limitation(g, "quantum_computing_xyz")
    assert results == []


def test_query_limitation_case_insensitive(tmp_path):
    g = _build_test_graph(tmp_path)
    results_upper = find_experiments_with_limitation(g, "SAMPLE")
    results_lower = find_experiments_with_limitation(g, "sample")
    assert len(results_upper) == len(results_lower)


# ── 5. CLI commands ────────────────────────────────────────────────────────────

def test_cli_build_graph(tmp_path):
    """build-graph must run without error and write a JSON file."""
    reg = _make_registry(tmp_path)
    json_out = tmp_path / "graph.json"
    result = runner.invoke(
        app,
        [
            "build-graph",
            "--registry-root", str(reg),
            "--json-path", str(json_out),
            "--no-graphml",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json_out.exists()
    data = json.loads(json_out.read_text())
    assert len(data["nodes"]) > 0


def test_cli_graph_summary_from_json(tmp_path):
    """graph-summary must load an existing JSON and print counts."""
    reg = _make_registry(tmp_path)
    json_out = tmp_path / "graph.json"
    # Build first
    runner.invoke(app, ["build-graph", "--registry-root", str(reg), "--json-path", str(json_out), "--no-graphml"])

    result = runner.invoke(
        app,
        ["graph-summary", "--json-path", str(json_out)],
    )
    assert result.exit_code == 0, result.output
    assert "experiment" in result.output
    assert "dataset" in result.output


def test_cli_graph_summary_live(tmp_path):
    """graph-summary must build the graph live if JSON is absent."""
    reg = _make_registry(tmp_path)
    result = runner.invoke(
        app,
        [
            "graph-summary",
            "--json-path", str(tmp_path / "nonexistent.json"),
            "--registry-root", str(reg),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "experiment" in result.output


def test_cli_graph_query_dataset(tmp_path):
    reg = _make_registry(tmp_path)
    json_out = tmp_path / "graph.json"
    runner.invoke(app, ["build-graph", "--registry-root", str(reg), "--json-path", str(json_out), "--no-graphml"])

    result = runner.invoke(
        app,
        ["graph-query", "--dataset", "GDELT", "--json-path", str(json_out)],
    )
    assert result.exit_code == 0, result.output
    assert "run_backtest" in result.output


def test_cli_graph_query_metric(tmp_path):
    reg = _make_registry(tmp_path)
    json_out = tmp_path / "graph.json"
    runner.invoke(app, ["build-graph", "--registry-root", str(reg), "--json-path", str(json_out), "--no-graphml"])

    result = runner.invoke(
        app,
        ["graph-query", "--metric", "sharpe_net", "--json-path", str(json_out)],
    )
    assert result.exit_code == 0, result.output
    assert "run_backtest" in result.output


def test_cli_graph_query_limitation(tmp_path):
    reg = _make_registry(tmp_path)
    json_out = tmp_path / "graph.json"
    runner.invoke(app, ["build-graph", "--registry-root", str(reg), "--json-path", str(json_out), "--no-graphml"])

    result = runner.invoke(
        app,
        ["graph-query", "--limitation", "sample", "--json-path", str(json_out)],
    )
    assert result.exit_code == 0, result.output
    assert "run_backtest" in result.output


def test_cli_graph_query_no_option(tmp_path):
    """graph-query with no query option must exit with code 1."""
    result = runner.invoke(app, ["graph-query"])
    assert result.exit_code == 1


def test_cli_graph_query_multiple_options(tmp_path):
    """graph-query with two query options must exit with code 1."""
    result = runner.invoke(
        app,
        ["graph-query", "--dataset", "GDELT", "--metric", "sharpe_net"],
    )
    assert result.exit_code == 1
