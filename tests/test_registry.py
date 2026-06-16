"""Tests for Milestone 6: experiment registry and local research memory.

Categories
----------
1. ExperimentRecord: field defaults, round-trip serialisation.
2. save_experiment: directory layout, metadata.json, metrics.json, report.md.
3. list_experiments: finds saved records, empty registry, newest-first order.
4. show_experiment: finds a specific record, returns None for unknown ID.
5. build_record: generates unique IDs, fills defaults.
6. CLI --no-save-memory: commands run without writing to the registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mosaic_alpha.cli import app
from mosaic_alpha.research.registry import (
    ExperimentRecord,
    build_record,
    list_experiments,
    save_experiment,
    show_experiment,
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _minimal_record(name: str = "test_experiment") -> ExperimentRecord:
    """Return a populated ExperimentRecord for unit tests."""
    return build_record(
        name=name,
        command=f"mosaic {name}",
        start_date="2020-01-01",
        end_date="2024-12-31",
        universe=["SPY", "XLK"],
        data_sources=["yfinance"],
        feature_sets=["price"],
        metrics_summary={"mean_ic": 0.05, "ic_t_stat": 2.1},
        output_files=["reports/generated/test.md"],
        limitations=["Small universe."],
        notes="Unit test record.",
    )


# ── 1. ExperimentRecord ────────────────────────────────────────────────────────

def test_experiment_record_defaults():
    """ExperimentRecord built via build_record has all required fields."""
    rec = build_record(name="my_experiment", start_date="2022-01-01", end_date="2023-12-31")
    assert rec.experiment_id != ""
    assert "my_experiment" in rec.experiment_id
    assert rec.name == "my_experiment"
    assert rec.created_at != ""
    assert rec.start_date == "2022-01-01"
    assert rec.end_date == "2023-12-31"
    assert isinstance(rec.universe, list)
    assert isinstance(rec.data_sources, list)
    assert isinstance(rec.feature_sets, list)
    assert isinstance(rec.metrics_summary, dict)
    assert isinstance(rec.output_files, list)
    assert isinstance(rec.limitations, list)
    assert rec.model_type == "ridge"
    assert rec.validation_method == "walk_forward"


def test_experiment_record_id_contains_name():
    """The experiment_id must embed the sanitised experiment name."""
    rec = build_record(name="run-baseline")
    # hyphens are replaced with underscores in the ID
    assert "run_baseline" in rec.experiment_id


def test_build_record_unique_ids():
    """Two successive build_record calls must produce different IDs."""
    import time
    rec1 = build_record(name="x")
    time.sleep(1.01)          # ensure different second
    rec2 = build_record(name="x")
    assert rec1.experiment_id != rec2.experiment_id


# ── 2. save_experiment ──────────────────────────────────────────────────────────

def test_save_creates_experiment_directory(tmp_path: Path):
    """save_experiment must create a directory named after the experiment_id."""
    rec = _minimal_record()
    exp_dir = save_experiment(rec, registry_root=tmp_path)
    assert exp_dir.exists()
    assert exp_dir.is_dir()
    assert exp_dir.name == rec.experiment_id


def test_save_writes_metadata_json(tmp_path: Path):
    """metadata.json must be valid JSON and contain key fields."""
    rec = _minimal_record("meta_test")
    save_experiment(rec, registry_root=tmp_path)
    metadata_path = tmp_path / rec.experiment_id / "metadata.json"
    assert metadata_path.exists()
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["experiment_id"] == rec.experiment_id
    assert data["name"] == rec.name
    assert data["start_date"] == rec.start_date
    assert data["end_date"] == rec.end_date
    assert "metrics_summary" not in data  # moved to metrics.json


def test_save_writes_metrics_json(tmp_path: Path):
    """metrics.json must be valid JSON and contain the metrics_summary values."""
    rec = _minimal_record("metrics_test")
    save_experiment(rec, registry_root=tmp_path)
    metrics_path = tmp_path / rec.experiment_id / "metrics.json"
    assert metrics_path.exists()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert data["mean_ic"] == pytest.approx(0.05)
    assert data["ic_t_stat"] == pytest.approx(2.1)


def test_save_writes_report_md_when_provided(tmp_path: Path):
    """When report_text is provided, report.md must be written."""
    rec = _minimal_record()
    save_experiment(rec, registry_root=tmp_path, report_text="# Test Report\n\nHello.")
    report_path = tmp_path / rec.experiment_id / "report.md"
    assert report_path.exists()
    assert "Test Report" in report_path.read_text(encoding="utf-8")


def test_save_no_report_md_when_empty(tmp_path: Path):
    """When report_text is empty, report.md must NOT be written."""
    rec = _minimal_record()
    save_experiment(rec, registry_root=tmp_path, report_text="")
    report_path = tmp_path / rec.experiment_id / "report.md"
    assert not report_path.exists()


def test_save_metadata_is_valid_json(tmp_path: Path):
    """Round-trip: parsed metadata.json must match original record fields."""
    rec = _minimal_record("roundtrip_test")
    save_experiment(rec, registry_root=tmp_path)
    metadata_path = tmp_path / rec.experiment_id / "metadata.json"
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert data["universe"] == rec.universe
    assert data["data_sources"] == rec.data_sources
    assert data["feature_sets"] == rec.feature_sets
    assert data["limitations"] == rec.limitations


# ── 3. list_experiments ─────────────────────────────────────────────────────────

def test_list_experiments_finds_saved_records(tmp_path: Path):
    """list_experiments must return all saved records."""
    for name in ["exp_a", "exp_b", "exp_c"]:
        save_experiment(_minimal_record(name), registry_root=tmp_path)

    records = list_experiments(registry_root=tmp_path)
    assert len(records) == 3
    names = {r.name for r in records}
    assert names == {"exp_a", "exp_b", "exp_c"}


def test_list_experiments_empty_registry(tmp_path: Path):
    """list_experiments on an empty/non-existent directory must return []."""
    empty_dir = tmp_path / "no_experiments"
    records = list_experiments(registry_root=empty_dir)
    assert records == []


def test_list_experiments_newest_first(tmp_path: Path):
    """list_experiments must return records sorted newest-first by folder name."""
    import time
    rec1 = _minimal_record("first")
    save_experiment(rec1, registry_root=tmp_path)
    time.sleep(1.01)
    rec2 = _minimal_record("second")
    save_experiment(rec2, registry_root=tmp_path)

    records = list_experiments(registry_root=tmp_path)
    assert len(records) == 2
    # Newest folder name sorts lexicographically last → first in list
    assert records[0].name == "second"
    assert records[1].name == "first"


def test_list_experiments_ignores_non_directories(tmp_path: Path):
    """Stray files inside the registry root must be silently skipped."""
    # Put a file at root level (not a valid experiment dir)
    stray = tmp_path / "stray_file.txt"
    stray.write_text("not an experiment")

    save_experiment(_minimal_record("real_exp"), registry_root=tmp_path)
    records = list_experiments(registry_root=tmp_path)
    assert len(records) == 1


# ── 4. show_experiment ──────────────────────────────────────────────────────────

def test_show_experiment_finds_specific_record(tmp_path: Path):
    """show_experiment must return the correct record by ID."""
    rec = _minimal_record("specific_exp")
    save_experiment(rec, registry_root=tmp_path)

    found = show_experiment(rec.experiment_id, registry_root=tmp_path)
    assert found is not None
    assert found.experiment_id == rec.experiment_id
    assert found.name == rec.name
    assert found.start_date == rec.start_date


def test_show_experiment_returns_none_for_unknown(tmp_path: Path):
    """show_experiment must return None when the experiment ID does not exist."""
    result = show_experiment("nonexistent_20240101_120000", registry_root=tmp_path)
    assert result is None


def test_show_experiment_restores_metrics(tmp_path: Path):
    """Loaded record must include the original metrics_summary values."""
    rec = _minimal_record("metrics_restore")
    save_experiment(rec, registry_root=tmp_path)

    found = show_experiment(rec.experiment_id, registry_root=tmp_path)
    assert found is not None
    assert found.metrics_summary["mean_ic"] == pytest.approx(0.05)
    assert found.metrics_summary["ic_t_stat"] == pytest.approx(2.1)


# ── 5. CLI --no-save-memory ────────────────────────────────────────────────────

runner = CliRunner()


def test_run_baseline_no_save_memory_does_not_write_registry(tmp_path: Path, monkeypatch):
    """run-baseline --no-save-memory must not create any experiment folder."""
    import mosaic_alpha.research.baseline as baseline_mod
    import mosaic_alpha.research.metrics as metrics_mod
    from dataclasses import dataclass, field

    # Minimal stub for AggregateMetrics
    @dataclass
    class _FakeAgg:
        label: str = "fwd_ret_5"
        mean_ic: float = 0.02
        std_ic: float = 0.01
        ic_t_stat: float = 1.5
        mean_rank_ic: float = 0.02
        std_rank_ic: float = 0.01
        mean_hit_rate: float = 0.52
        std_hit_rate: float = 0.05
        mean_ls_decile: float = 0.01
        std_ls_decile: float = 0.005

    @dataclass
    class _FakeBaselineResult:
        ticker: str = "SPY"
        start: str = "2020-01-01"
        end: str = "2024-12-31"
        n_rows: int = 100
        n_folds: int = 3
        aggregate: list = field(default_factory=lambda: [_FakeAgg()])

    monkeypatch.setattr(baseline_mod, "run_baseline", lambda **kw: _FakeBaselineResult())
    monkeypatch.setattr(baseline_mod, "render_report", lambda result, path: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text("# Report"))

    reg_root = tmp_path / "experiments"
    result = runner.invoke(
        app,
        [
            "run-baseline",
            "--ticker", "SPY",
            "--start", "2020-01-01",
            "--end", "2022-12-31",
            "--no-save-memory",
            "--report-path", str(tmp_path / "report.md"),
        ],
    )
    assert result.exit_code == 0, result.output
    # Registry root must not have been created
    assert not reg_root.exists()


def test_run_sector_baseline_no_save_memory(tmp_path: Path, monkeypatch):
    """run-sector-baseline --no-save-memory must not write to registry."""
    import mosaic_alpha.research.sector_baseline as sb_mod
    from dataclasses import dataclass, field

    @dataclass
    class _FakeCS:
        label_col: str = "fwd_ret_5"
        mean_ic: float = 0.01
        ic_t_stat: float = 0.8
        mean_rank_ic: float = 0.01
        ic_hit_rate: float = 0.5
        mean_ls_spread: float = 0.005

    @dataclass
    class _FakeSectorResult:
        tickers: list = field(default_factory=lambda: ["XLK", "XLE"])
        n_folds: int = 2
        n_panel_rows: int = 200
        pooled: list = field(default_factory=lambda: [_FakeCS()])

    monkeypatch.setattr(sb_mod, "run_sector_baseline", lambda **kw: _FakeSectorResult())
    monkeypatch.setattr(sb_mod, "render_report", lambda result, path: path.parent.mkdir(parents=True, exist_ok=True) or path.write_text("# Report"))

    result = runner.invoke(
        app,
        [
            "run-sector-baseline",
            "--start", "2020-01-01",
            "--end", "2022-12-31",
            "--no-save-memory",
            "--report-path", str(tmp_path / "report.md"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (tmp_path / "experiments").exists()


# ── 6. CLI list-experiments and show-experiment ────────────────────────────────

def test_list_experiments_cmd_empty(tmp_path: Path):
    """list-experiments on an empty dir must print a 'no experiments' message."""
    result = runner.invoke(
        app,
        ["list-experiments", "--registry-root", str(tmp_path / "empty")],
    )
    assert result.exit_code == 0
    assert "No experiments" in result.output


def test_list_experiments_cmd_shows_saved(tmp_path: Path):
    """list-experiments must show saved experiment IDs."""
    rec = _minimal_record("my_test_run")
    save_experiment(rec, registry_root=tmp_path)

    result = runner.invoke(
        app,
        ["list-experiments", "--registry-root", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert rec.experiment_id in result.output


def test_show_experiment_cmd_found(tmp_path: Path):
    """show-experiment must print metadata for a known ID."""
    rec = _minimal_record("show_me")
    save_experiment(rec, registry_root=tmp_path)

    result = runner.invoke(
        app,
        ["show-experiment", rec.experiment_id, "--registry-root", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert rec.experiment_id in result.output
    assert "show_me" in result.output
    assert "2020-01-01" in result.output


def test_show_experiment_cmd_not_found(tmp_path: Path):
    """show-experiment with an unknown ID must exit with code 1."""
    result = runner.invoke(
        app,
        ["show-experiment", "nonexistent_id", "--registry-root", str(tmp_path)],
    )
    assert result.exit_code == 1
