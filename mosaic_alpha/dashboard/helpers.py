"""Data-loading helpers for the MosaicAlpha Streamlit dashboard.

All functions are pure Python (no Streamlit imports) so they can be
unit-tested without launching a browser session.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY = Path("memory/experiments")
_DEFAULT_GRAPH = Path("memory/research_graph.json")
_DEFAULT_REPORTS = Path("reports/generated")

# Vendor repo metadata: name → description of its influence on MosaicAlpha
_VENDOR_INFLUENCE: dict[str, dict[str, str]] = {
    "agent-reach": {
        "display_name": "agent-reach",
        "tagline": "Web-scraping and internet-access layer for AI agents",
        "influence": "Data connector philosophy",
        "description": (
            "agent-reach demonstrated a clean pattern for giving AI agents stable, "
            "modular access to external data sources. MosaicAlpha's data layer "
            "(yfinance, FRED, GDELT connectors) follows the same philosophy: each "
            "connector is a thin adapter that handles authentication, caching, and "
            "rate limiting, exposing a uniform DataFrame interface to the rest of "
            "the pipeline."
        ),
    },
    "G0DM0D3": {
        "display_name": "G0DM0D3",
        "tagline": "Multi-model AI chat interface with 50+ models via OpenRouter",
        "influence": "Multi-model evaluation and research-review idea",
        "description": (
            "G0DM0D3 demonstrated the value of side-by-side model comparison in a "
            "research workflow. MosaicAlpha's four-way ablation study (price-only vs "
            "price+macro vs price+news vs price+macro+news) borrows this comparative "
            "mindset. The future LLM research-reviewer layer on the roadmap is "
            "directly inspired by G0DM0D3's multi-model evaluation approach."
        ),
    },
    "openhuman": {
        "display_name": "openhuman",
        "tagline": "Open-source AI harness with human-in-the-loop principles",
        "influence": "Local memory and experiment registry",
        "description": (
            "openhuman emphasised local memory, managed services, and a simple-but-"
            "powerful design philosophy. MosaicAlpha's experiment registry "
            "(memory/experiments/) is a direct application of this idea: every "
            "research run is persisted in a human-readable JSON format that can be "
            "inspected, diffed, and queried without a database. The registry keeps "
            "the researcher in the loop on every decision."
        ),
    },
    "Understand-Anything": {
        "display_name": "Understand Anything",
        "tagline": "Interactive knowledge-graph generator for codebases and docs",
        "influence": "Research knowledge graph and explainability layer",
        "description": (
            "Understand-Anything demonstrated that unstructured content — code, docs, "
            "research notes — can be turned into a searchable, queryable knowledge "
            "graph. MosaicAlpha's graph layer (mosaic_alpha/graph/) applies the same "
            "idea to experiment lineage: datasets, features, models, experiments, "
            "metrics, and limitations are linked in a directed graph that makes the "
            "research provenance explicit and queryable."
        ),
    },
}


# ── Experiments ────────────────────────────────────────────────────────────────

def load_experiments(registry_root: Path | None = None) -> list[dict[str, Any]]:
    """Return all experiment dicts from the registry, newest first."""
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
    """Load metadata, metrics, and report text for one experiment directory."""
    metadata_path = exp_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse metadata.json in %s: %s", exp_dir, exc)
        return None

    metrics_path = exp_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse metrics.json in %s: %s", exp_dir, exc)

    report_text: str = ""
    report_path = exp_dir / "report.md"
    if report_path.exists():
        try:
            report_text = report_path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read report.md in %s: %s", exp_dir, exc)

    return {**metadata, "metrics": metrics, "report_text": report_text, "exp_dir": str(exp_dir)}


def extract_key_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
    """Flatten metrics dict to displayable scalar values (NaN/None omitted)."""
    result: dict[str, float | None] = {}

    def _try_float(v: Any) -> float | None:
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    for key in ("sharpe_net", "sharpe_gross", "ann_net_return", "ann_gross_return",
                "max_drawdown_net", "hit_rate", "avg_turnover", "n_periods"):
        if key in metrics:
            v = _try_float(metrics[key])
            if v is not None:
                result[key] = v

    for key in ("mean_ic", "ic_t_stat", "mean_rank_ic"):
        if key in metrics:
            v = _try_float(metrics[key])
            if v is not None:
                result[key] = v

    labels_list = metrics.get("labels", [])
    if labels_list and isinstance(labels_list, list) and isinstance(labels_list[0], dict):
        first = labels_list[0]
        for key in ("mean_ic", "ic_t_stat"):
            if key in first and key not in result:
                v = _try_float(first[key])
                if v is not None:
                    result[f"label0_{key}"] = v

    comparisons = metrics.get("comparisons", [])
    if comparisons and isinstance(comparisons, list) and isinstance(comparisons[0], dict):
        first = comparisons[0]
        for key in ("price_macro_news_mean_ic", "price_macro_mean_ic", "price_only_mean_ic"):
            if key in first:
                v = _try_float(first[key])
                if v is not None:
                    result[key] = v
                    break

    return result


def find_latest_backtest(experiments: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the most recent run_backtest experiment, or None."""
    for exp in experiments:
        if exp.get("name") == "run_backtest":
            return exp
    return None


# ── Formatting ─────────────────────────────────────────────────────────────────

# Human-readable labels and plain-English explanations for quant metrics
METRIC_META: dict[str, dict[str, str]] = {
    "sharpe_net": {
        "label": "Sharpe Ratio (net)",
        "unit": "",
        "fmt": "+.2f",
        "explanation": (
            "Risk-adjusted return after transaction costs. "
            "Measures how much return you get per unit of volatility. "
            "Higher is better; above 1.0 is generally considered good."
        ),
    },
    "sharpe_gross": {
        "label": "Sharpe Ratio (gross)",
        "unit": "",
        "fmt": "+.2f",
        "explanation": "Same as net Sharpe but before subtracting transaction costs.",
    },
    "ann_net_return": {
        "label": "Annualised Return (net)",
        "unit": "%",
        "fmt": "+.2f",
        "explanation": (
            "Average yearly return of the strategy after transaction costs, "
            "expressed as a percentage."
        ),
    },
    "ann_gross_return": {
        "label": "Annualised Return (gross)",
        "unit": "%",
        "fmt": "+.2f",
        "explanation": "Average yearly return before subtracting transaction costs.",
    },
    "max_drawdown_net": {
        "label": "Max Drawdown (net)",
        "unit": "%",
        "fmt": "+.2f",
        "explanation": (
            "The largest peak-to-trough decline in portfolio value, net of costs. "
            "A value of -15% means the portfolio fell 15% from its peak before recovering. "
            "Smaller (less negative) is better."
        ),
    },
    "hit_rate": {
        "label": "Hit Rate",
        "unit": "",
        "fmt": ".3f",
        "explanation": (
            "Fraction of rebalancing periods where the strategy made a positive gross return. "
            "0.5 = 50% of periods were profitable."
        ),
    },
    "avg_turnover": {
        "label": "Avg Turnover",
        "unit": "",
        "fmt": ".3f",
        "explanation": (
            "Average fraction of the portfolio that changes hands at each rebalance. "
            "Higher turnover means higher transaction costs."
        ),
    },
    "n_periods": {
        "label": "Periods",
        "unit": "",
        "fmt": ".0f",
        "explanation": "Number of non-overlapping rebalancing periods in the backtest.",
    },
    "mean_ic": {
        "label": "Mean IC",
        "unit": "",
        "fmt": "+.4f",
        "explanation": (
            "Information Coefficient — the average Pearson correlation between model "
            "predictions and actual returns. Ranges from -1 to +1. "
            "Values above +0.05 with a significant t-stat indicate useful signal."
        ),
    },
    "ic_t_stat": {
        "label": "IC t-stat",
        "unit": "",
        "fmt": "+.2f",
        "explanation": (
            "Statistical significance of the mean IC. "
            "A t-stat above ±2.0 suggests the signal is unlikely to be noise "
            "(assuming independent observations, which is an approximation)."
        ),
    },
    "mean_rank_ic": {
        "label": "Mean Rank IC",
        "unit": "",
        "fmt": "+.4f",
        "explanation": (
            "Spearman rank correlation — like IC but based on rank order rather than "
            "raw values. More robust to outliers."
        ),
    },
}


def format_metric(value: float | None, metric_name: str) -> str:
    """Format a metric value for display using its metadata, or '—' if absent."""
    if value is None:
        return "—"
    meta = METRIC_META.get(metric_name, {})
    unit = meta.get("unit", "")
    fmt = meta.get("fmt", ".4f")
    try:
        formatted = format(value * 100 if unit == "%" else value, fmt)
        return f"{formatted}{unit}"
    except (ValueError, TypeError):
        return str(value)


def metric_label(metric_name: str) -> str:
    """Return a human-friendly label for a metric name."""
    return METRIC_META.get(metric_name, {}).get("label", metric_name.replace("_", " ").title())


def metric_explanation(metric_name: str) -> str:
    """Return a plain-English explanation for a metric name."""
    return METRIC_META.get(metric_name, {}).get("explanation", "")


# ── Graph ──────────────────────────────────────────────────────────────────────

def load_graph_json(path: Path | None = None) -> dict[str, Any] | None:
    """Load the research graph JSON, returning None if absent or corrupt."""
    p = Path(path) if path is not None else _DEFAULT_GRAPH
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not parse graph JSON at %s: %s", p, exc)
        return None


# alias
safe_read_json = load_graph_json


def graph_node_table(graph_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat list of node dicts for a DataFrame."""
    return [
        {
            "node_id": n.get("node_id", ""),
            "node_type": n.get("node_type", ""),
            "label": n.get("label", ""),
        }
        for n in graph_data.get("nodes", [])
    ]


def graph_edge_table(graph_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flat list of edge dicts for a DataFrame."""
    return [
        {
            "source": e.get("source_id", ""),
            "edge_type": e.get("edge_type", ""),
            "target": e.get("target_id", ""),
        }
        for e in graph_data.get("edges", [])
    ]


# ── Reports ────────────────────────────────────────────────────────────────────

def find_reports(
    reports_dir: Path | None = None,
    registry_root: Path | None = None,
) -> list[dict[str, str]]:
    """Discover Markdown reports from reports/generated/ and experiment registry."""
    rpt_root = Path(reports_dir) if reports_dir is not None else _DEFAULT_REPORTS
    reg_root = Path(registry_root) if registry_root is not None else _DEFAULT_REGISTRY

    reports: list[dict[str, str]] = []
    if rpt_root.exists():
        for md in sorted(rpt_root.glob("*.md")):
            reports.append({"label": md.name, "path": str(md)})
    if reg_root.exists():
        for exp_dir in sorted(reg_root.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue
            rpt = exp_dir / "report.md"
            if rpt.exists():
                reports.append({"label": f"{exp_dir.name}/report.md", "path": str(rpt)})
    return reports


def load_report_text(path: str | Path) -> str | None:
    """Read a Markdown report; return None if absent."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read report at %s: %s", p, exc)
        return None


# alias
safe_read_markdown = load_report_text


# ── Vendor repos ───────────────────────────────────────────────────────────────

def _vendor_base() -> Path:
    """Return the path to the vendor directory (sibling of the project root)."""
    # Prefer cwd-relative path (works when running from project root)
    cwd_relative = Path("../vendor")
    if cwd_relative.exists():
        return cwd_relative
    # Fall back to path relative to this source file
    src_relative = Path(__file__).parent.parent.parent.parent / "vendor"
    return src_relative


def find_vendor_repos(vendor_base: Path | None = None) -> list[dict[str, Any]]:
    """Return a list of vendor repo dicts found in the sibling vendor directory.

    Each dict has keys: ``name``, ``path``, ``present``, ``readme_snippet``,
    ``display_name``, ``tagline``, ``influence``, ``description``.
    """
    base = Path(vendor_base) if vendor_base is not None else _vendor_base()

    results: list[dict[str, Any]] = []
    for repo_name, meta in _VENDOR_INFLUENCE.items():
        repo_path = base / repo_name
        present = repo_path.is_dir()
        readme_snippet = ""
        if present:
            for readme_name in ("README.md", "readme.md", "README.rst", "README.txt"):
                readme_path = repo_path / readme_name
                if readme_path.exists():
                    try:
                        lines = readme_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        readme_snippet = "\n".join(lines[:20])
                    except Exception:  # noqa: BLE001
                        pass
                    break
        results.append({
            "name": repo_name,
            "path": str(repo_path),
            "present": present,
            "readme_snippet": readme_snippet,
            **meta,
        })
    return results


def summarize_vendor_repos(vendor_base: Path | None = None) -> list[dict[str, Any]]:
    """Same as find_vendor_repos; named alias for clarity in tests."""
    return find_vendor_repos(vendor_base=vendor_base)
