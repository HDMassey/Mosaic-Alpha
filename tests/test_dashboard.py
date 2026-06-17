"""Tests for Milestone 8: Streamlit research dashboard helpers.

We test only the pure-Python helper functions (no Streamlit imports required).
The CLI `dashboard` command is smoke-tested via typer's test runner without
actually starting the Streamlit server.

Categories
----------
1. load_experiments: returns list from populated registry; empty list when absent.
2. load_experiment_detail: parses metadata, metrics, report; handles missing files.
3. extract_key_metrics: extracts scalars from various metric dict shapes.
4. load_graph_json: loads valid JSON; returns None for missing/corrupt file.
5. graph_node_table / graph_edge_table: produce correct flat rows.
6. find_reports: discovers reports from both generated/ and registry folders.
7. load_report_text: reads content; None when file absent.
8. find_latest_backtest: picks the most recent run_backtest experiment.
9. CLI dashboard command: smoke test (no server started).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic_alpha.cli import app
from mosaic_alpha.dashboard.helpers import (
    extract_key_metrics,
    find_latest_backtest,
    find_reports,
    find_vendor_repos,
    format_metric,
    graph_edge_table,
    graph_node_table,
    load_experiment_detail,
    load_experiments,
    load_graph_json,
    load_report_text,
    metric_explanation,
    metric_label,
    summarize_vendor_repos,
)
from mosaic_alpha.research.registry import build_record, save_experiment

runner = CliRunner()


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_registry(tmp_path: Path) -> Path:
    """Create two experiment records in a temp registry."""
    reg = tmp_path / "experiments"

    rec1 = build_record(
        name="run_baseline",
        command="mosaic run-baseline",
        start_date="2015-01-01",
        end_date="2024-12-31",
        universe=["SPY"],
        data_sources=["yfinance"],
        feature_sets=["price"],
        metrics_summary={"mean_ic": 0.12, "ic_t_stat": 3.1, "n_rows": 2500},
        output_files=["reports/generated/baseline_spy.md"],
        limitations=["Price features only."],
    )
    save_experiment(rec1, registry_root=reg, report_text="# Baseline\n\nSample report.")
    time.sleep(1.01)

    rec2 = build_record(
        name="run_backtest",
        command="mosaic run-backtest --offline-sample",
        start_date="2020-01-01",
        end_date="2024-12-31",
        universe=["XLK", "XLE"],
        data_sources=["yfinance", "GDELT"],
        feature_sets=["price", "news"],
        metrics_summary={
            "sharpe_net": 1.3,
            "ann_net_return": 0.09,
            "hit_rate": 0.55,
            "n_periods": 40,
        },
        output_files=["reports/generated/backtest.md"],
        limitations=["SAMPLE DATA."],
    )
    save_experiment(rec2, registry_root=reg, report_text="# Backtest\n\nSample backtest report.")

    return reg


def _make_graph_json(tmp_path: Path) -> Path:
    """Write a minimal research_graph.json."""
    data = {
        "nodes": [
            {"node_id": "dataset:GDELT", "node_type": "dataset", "label": "GDELT", "attributes": {}},
            {"node_id": "experiment:abc", "node_type": "experiment", "label": "run_backtest", "attributes": {}},
        ],
        "edges": [
            {"source_id": "dataset:GDELT", "target_id": "experiment:abc", "edge_type": "produces", "attributes": {}},
        ],
        "summary": {
            "total_nodes": 2,
            "total_edges": 1,
            "nodes_by_type": {"dataset": 1, "experiment": 1},
            "edges_by_type": {"produces": 1},
        },
    }
    p = tmp_path / "research_graph.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return p


# ── 1. load_experiments ────────────────────────────────────────────────────────

def test_load_experiments_returns_records(tmp_path):
    reg = _make_registry(tmp_path)
    experiments = load_experiments(reg)
    assert len(experiments) == 2


def test_load_experiments_newest_first(tmp_path):
    reg = _make_registry(tmp_path)
    experiments = load_experiments(reg)
    # Folder names are timestamped; newest is returned first
    assert experiments[0]["name"] == "run_backtest"
    assert experiments[1]["name"] == "run_baseline"


def test_load_experiments_empty_when_dir_absent(tmp_path):
    experiments = load_experiments(tmp_path / "no_such_dir")
    assert experiments == []


def test_load_experiments_includes_metrics(tmp_path):
    reg = _make_registry(tmp_path)
    experiments = load_experiments(reg)
    bt = next(e for e in experiments if e["name"] == "run_backtest")
    assert bt["metrics"]["sharpe_net"] == pytest.approx(1.3)


def test_load_experiments_includes_report_text(tmp_path):
    reg = _make_registry(tmp_path)
    experiments = load_experiments(reg)
    bt = next(e for e in experiments if e["name"] == "run_backtest")
    assert "Backtest" in bt["report_text"]


# ── 2. load_experiment_detail ─────────────────────────────────────────────────

def test_load_experiment_detail_returns_dict(tmp_path):
    reg = _make_registry(tmp_path)
    exp_dirs = sorted(reg.iterdir())
    detail = load_experiment_detail(exp_dirs[0])
    assert isinstance(detail, dict)
    assert "experiment_id" in detail


def test_load_experiment_detail_missing_metadata(tmp_path):
    empty_dir = tmp_path / "empty_exp"
    empty_dir.mkdir()
    assert load_experiment_detail(empty_dir) is None


def test_load_experiment_detail_no_report_md(tmp_path):
    """Experiments without report.md should still load cleanly."""
    reg = tmp_path / "experiments"
    rec = build_record(name="no_report", start_date="2022-01-01", end_date="2022-12-31")
    save_experiment(rec, registry_root=reg, report_text="")  # no report
    experiments = load_experiments(reg)
    assert len(experiments) == 1
    assert experiments[0]["report_text"] == ""


def test_load_experiment_detail_corrupted_metadata(tmp_path):
    """A directory with invalid metadata.json must return None gracefully."""
    bad_dir = tmp_path / "bad_exp"
    bad_dir.mkdir()
    (bad_dir / "metadata.json").write_text("NOT VALID JSON", encoding="utf-8")
    assert load_experiment_detail(bad_dir) is None


# ── 3. extract_key_metrics ────────────────────────────────────────────────────

def test_extract_key_metrics_backtest():
    metrics = {"sharpe_net": 1.25, "ann_net_return": 0.08, "hit_rate": 0.55, "n_periods": 40}
    km = extract_key_metrics(metrics)
    assert "sharpe_net" in km
    assert km["sharpe_net"] == pytest.approx(1.25)
    assert "ann_net_return" in km


def test_extract_key_metrics_baseline():
    metrics = {"mean_ic": 0.12, "ic_t_stat": 3.1}
    km = extract_key_metrics(metrics)
    assert "mean_ic" in km
    assert km["mean_ic"] == pytest.approx(0.12)


def test_extract_key_metrics_nested_labels():
    metrics = {"labels": [{"mean_ic": 0.05, "ic_t_stat": 2.0}]}
    km = extract_key_metrics(metrics)
    assert "label0_mean_ic" in km
    assert km["label0_mean_ic"] == pytest.approx(0.05)


def test_extract_key_metrics_nested_comparisons():
    metrics = {"comparisons": [{"price_macro_news_mean_ic": 0.03}]}
    km = extract_key_metrics(metrics)
    assert "price_macro_news_mean_ic" in km


def test_extract_key_metrics_empty():
    assert extract_key_metrics({}) == {}


def test_extract_key_metrics_ignores_nan():
    import math
    metrics = {"sharpe_net": float("nan"), "ann_net_return": 0.05}
    km = extract_key_metrics(metrics)
    assert "sharpe_net" not in km
    assert "ann_net_return" in km


def test_extract_key_metrics_ignores_none():
    metrics = {"sharpe_net": None, "hit_rate": 0.55}
    km = extract_key_metrics(metrics)
    assert "sharpe_net" not in km
    assert "hit_rate" in km


# ── 4. load_graph_json ────────────────────────────────────────────────────────

def test_load_graph_json_loads_valid_file(tmp_path):
    p = _make_graph_json(tmp_path)
    data = load_graph_json(p)
    assert data is not None
    assert "nodes" in data
    assert "edges" in data


def test_load_graph_json_returns_none_when_absent(tmp_path):
    assert load_graph_json(tmp_path / "nonexistent.json") is None


def test_load_graph_json_returns_none_for_corrupt_file(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("THIS IS NOT JSON", encoding="utf-8")
    assert load_graph_json(bad) is None


def test_load_graph_json_contains_summary(tmp_path):
    p = _make_graph_json(tmp_path)
    data = load_graph_json(p)
    assert data is not None
    assert data["summary"]["total_nodes"] == 2
    assert data["summary"]["total_edges"] == 1


# ── 5. graph_node_table / graph_edge_table ────────────────────────────────────

def test_graph_node_table_returns_rows(tmp_path):
    p = _make_graph_json(tmp_path)
    data = load_graph_json(p)
    rows = graph_node_table(data)
    assert len(rows) == 2
    assert rows[0]["node_id"] == "dataset:GDELT"
    assert rows[0]["node_type"] == "dataset"


def test_graph_edge_table_returns_rows(tmp_path):
    p = _make_graph_json(tmp_path)
    data = load_graph_json(p)
    rows = graph_edge_table(data)
    assert len(rows) == 1
    assert rows[0]["edge_type"] == "produces"
    assert rows[0]["source"] == "dataset:GDELT"
    assert rows[0]["target"] == "experiment:abc"


def test_graph_node_table_empty_graph():
    data = {"nodes": [], "edges": []}
    assert graph_node_table(data) == []


def test_graph_edge_table_empty_graph():
    data = {"nodes": [], "edges": []}
    assert graph_edge_table(data) == []


# ── 6. find_reports ──────────────────────────────────────────────────────────

def test_find_reports_from_generated_dir(tmp_path):
    rpt_dir = tmp_path / "reports" / "generated"
    rpt_dir.mkdir(parents=True)
    (rpt_dir / "baseline_spy.md").write_text("# Report", encoding="utf-8")
    (rpt_dir / "sector_baseline.md").write_text("# Report", encoding="utf-8")

    # Pass an empty registry_root so no experiment reports bleed in
    reports = find_reports(reports_dir=rpt_dir, registry_root=tmp_path / "no_registry")
    assert len(reports) == 2
    assert all(r["path"].endswith(".md") for r in reports)


def test_find_reports_from_registry(tmp_path):
    reg = _make_registry(tmp_path)  # both experiments have report.md
    reports = find_reports(registry_root=reg)
    registry_reports = [r for r in reports if "experiments" in r["path"]]
    assert len(registry_reports) == 2


def test_find_reports_both_sources(tmp_path):
    reg = _make_registry(tmp_path)
    rpt_dir = tmp_path / "generated"
    rpt_dir.mkdir()
    (rpt_dir / "test.md").write_text("# Test")

    reports = find_reports(reports_dir=rpt_dir, registry_root=reg)
    paths = [r["path"] for r in reports]
    assert any("test.md" in p for p in paths)
    assert any("experiments" in p for p in paths)


def test_find_reports_empty_when_dirs_absent(tmp_path):
    reports = find_reports(
        reports_dir=tmp_path / "no_reports",
        registry_root=tmp_path / "no_registry",
    )
    assert reports == []


# ── 7. load_report_text ──────────────────────────────────────────────────────

def test_load_report_text_reads_content(tmp_path):
    p = tmp_path / "test.md"
    p.write_text("# Hello\n\nWorld.", encoding="utf-8")
    text = load_report_text(p)
    assert text == "# Hello\n\nWorld."


def test_load_report_text_none_when_absent(tmp_path):
    assert load_report_text(tmp_path / "missing.md") is None


# ── 8. find_latest_backtest ──────────────────────────────────────────────────

def test_find_latest_backtest_returns_most_recent(tmp_path):
    reg = _make_registry(tmp_path)
    experiments = load_experiments(reg)
    bt = find_latest_backtest(experiments)
    assert bt is not None
    assert bt["name"] == "run_backtest"


def test_find_latest_backtest_returns_none_when_no_backtest(tmp_path):
    reg = tmp_path / "experiments"
    rec = build_record(name="run_baseline", start_date="2022-01-01", end_date="2022-12-31")
    save_experiment(rec, registry_root=reg)
    experiments = load_experiments(reg)
    assert find_latest_backtest(experiments) is None


def test_find_latest_backtest_returns_none_for_empty_list():
    assert find_latest_backtest([]) is None


# ── 9. CLI dashboard smoke test ──────────────────────────────────────────────

def test_dashboard_cmd_help():
    """dashboard --help must exit 0 and mention Streamlit."""
    result = runner.invoke(app, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "streamlit" in result.output.lower() or "dashboard" in result.output.lower()


# ── 10. format_metric ────────────────────────────────────────────────────────

def test_format_metric_sharpe():
    assert format_metric(1.25, "sharpe_net") == "+1.25"


def test_format_metric_return_as_percent():
    """ann_net_return should be multiplied by 100 and suffixed with %."""
    result = format_metric(0.09, "ann_net_return")
    assert result.endswith("%")
    assert "9" in result


def test_format_metric_drawdown_as_percent():
    result = format_metric(-0.15, "max_drawdown_net")
    assert result.endswith("%")
    assert "-" in result


def test_format_metric_none_returns_dash():
    assert format_metric(None, "sharpe_net") == "—"


def test_format_metric_unknown_key_falls_back():
    """Unknown metric names should not raise; return a string."""
    result = format_metric(0.123456, "some_unknown_metric")
    assert isinstance(result, str)
    assert "0.1235" in result


# ── 11. metric_label / metric_explanation ────────────────────────────────────

def test_metric_label_known():
    assert metric_label("sharpe_net") == "Sharpe Ratio (net)"


def test_metric_label_unknown_uses_title_case():
    label = metric_label("my_custom_metric")
    assert label == "My Custom Metric"


def test_metric_explanation_known():
    explanation = metric_explanation("mean_ic")
    assert len(explanation) > 10
    assert "correlation" in explanation.lower() or "ic" in explanation.lower()


def test_metric_explanation_unknown_returns_empty():
    assert metric_explanation("nonexistent_key") == ""


# ── 12. find_vendor_repos / summarize_vendor_repos ───────────────────────────

def test_find_vendor_repos_absent_vendor_dir(tmp_path):
    """When vendor dir does not exist, repos are returned with present=False."""
    repos = find_vendor_repos(vendor_base=tmp_path / "no_vendor")
    assert len(repos) == 4  # 4 known vendor repos
    assert all(r["present"] is False for r in repos)


def test_find_vendor_repos_required_keys(tmp_path):
    """Each repo dict must contain the expected keys."""
    repos = find_vendor_repos(vendor_base=tmp_path / "no_vendor")
    required = {"name", "path", "present", "readme_snippet", "display_name", "tagline",
                "influence", "description"}
    for repo in repos:
        assert required.issubset(repo.keys()), f"Missing keys in {repo['name']}"


def test_find_vendor_repos_present_with_readme(tmp_path):
    """When a vendor repo dir exists with a README, present=True and snippet populated."""
    vendor_dir = tmp_path / "vendor"
    repo_dir = vendor_dir / "agent-reach"
    repo_dir.mkdir(parents=True)
    readme_content = "\n".join([f"Line {i}" for i in range(25)])
    (repo_dir / "README.md").write_text(readme_content, encoding="utf-8")

    repos = find_vendor_repos(vendor_base=vendor_dir)
    agent_reach = next(r for r in repos if r["name"] == "agent-reach")
    assert agent_reach["present"] is True
    assert "Line 0" in agent_reach["readme_snippet"]
    # snippet should be at most 20 lines
    assert len(agent_reach["readme_snippet"].splitlines()) <= 20


def test_find_vendor_repos_present_without_readme(tmp_path):
    """A vendor repo dir present but without README should have empty snippet."""
    vendor_dir = tmp_path / "vendor"
    repo_dir = vendor_dir / "G0DM0D3"
    repo_dir.mkdir(parents=True)

    repos = find_vendor_repos(vendor_base=vendor_dir)
    g0d = next(r for r in repos if r["name"] == "G0DM0D3")
    assert g0d["present"] is True
    assert g0d["readme_snippet"] == ""


def test_summarize_vendor_repos_is_alias(tmp_path):
    """summarize_vendor_repos should return the same result as find_vendor_repos."""
    base = tmp_path / "no_vendor"
    assert summarize_vendor_repos(vendor_base=base) == find_vendor_repos(vendor_base=base)
