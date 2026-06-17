"""MosaicAlpha local research dashboard (Streamlit).

Launch with:
    mosaic dashboard
or directly:
    streamlit run mosaic_alpha/dashboard/app.py

The dashboard reads from the local file system:
  - memory/experiments/   (experiment registry)
  - memory/research_graph.json  (knowledge graph export)
  - reports/generated/    (CLI-generated Markdown reports)

No network calls are made; all data is local.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="MosaicAlpha Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from mosaic_alpha.dashboard.helpers import (  # noqa: E402
    extract_key_metrics,
    find_latest_backtest,
    find_reports,
    graph_edge_table,
    graph_node_table,
    load_experiment_detail,
    load_experiments,
    load_graph_json,
    load_report_text,
)

# ── Constants ───────────────────────────────────────────────────────────────

_REGISTRY_ROOT = Path("memory/experiments")
_GRAPH_JSON = Path("memory/research_graph.json")
_REPORTS_DIR = Path("reports/generated")

_PAGES = ["Overview", "Experiments", "Backtest", "Research Graph", "Reports"]


# ── Sidebar navigation ───────────────────────────────────────────────────────

def _sidebar() -> str:
    with st.sidebar:
        st.title("MosaicAlpha")
        st.caption("Local quantitative research workbench")
        st.divider()
        page = st.radio("Navigate", _PAGES, label_visibility="collapsed")
        st.divider()
        st.caption(
            "⚠️ Research infrastructure only.  "
            "Not financial advice."
        )
    return page  # type: ignore[return-value]


# ── Page A: Overview ─────────────────────────────────────────────────────────

def _page_overview() -> None:
    st.title("MosaicAlpha — Research Overview")

    st.warning(
        "**This is research infrastructure, not trading advice.**  "
        "All results are backtests on historical data and do not predict future performance.",
        icon="⚠️",
    )

    st.markdown("""
MosaicAlpha is a **local-first alternative-data research workbench** for
discovering, validating, and explaining quantitative trading signals from
public datasets.

It combines rigorous walk-forward validation, transaction-cost-aware
backtesting, alternative-data ingestion, an experiment registry, and a
research knowledge graph.
""")

    st.divider()

    # Milestones
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Milestones completed")
        milestones = [
            ("M1", "Price-only SPY baseline"),
            ("M2", "Sector ETF cross-sectional baseline"),
            ("M3", "FRED macro regime features"),
            ("M4", "GDELT news intensity features"),
            ("M5", "Transaction-cost-aware backtesting"),
            ("M6", "Experiment registry & research memory"),
            ("M7", "Research knowledge graph"),
            ("M8", "Streamlit research dashboard"),
        ]
        for code, desc in milestones:
            st.markdown(f"✅ **{code}** — {desc}")

    with col2:
        st.subheader("Key CLI commands")
        st.code(
            "mosaic run-baseline\n"
            "mosaic run-sector-baseline\n"
            "mosaic run-macro-sector\n"
            "mosaic run-news-sector\n"
            "mosaic run-backtest --offline-sample\n"
            "mosaic list-experiments\n"
            "mosaic show-experiment <ID>\n"
            "mosaic build-graph\n"
            "mosaic graph-query --dataset GDELT\n"
            "mosaic dashboard",
            language="bash",
        )

    # Live counts from registry
    st.divider()
    experiments = load_experiments(_REGISTRY_ROOT)
    graph_data = load_graph_json(_GRAPH_JSON)
    reports = find_reports(_REPORTS_DIR, _REGISTRY_ROOT)

    m1, m2, m3 = st.columns(3)
    m1.metric("Experiments saved", len(experiments))
    m2.metric(
        "Graph nodes",
        graph_data["summary"]["total_nodes"] if graph_data else "—",
    )
    m3.metric("Reports available", len(reports))


# ── Page B: Experiments ──────────────────────────────────────────────────────

def _page_experiments() -> None:
    st.title("Experiments")

    experiments = load_experiments(_REGISTRY_ROOT)

    if not experiments:
        st.info(
            "No experiments found in `memory/experiments/`.\n\n"
            "Run any pipeline command to save an experiment:\n"
            "```bash\n"
            "mosaic run-baseline --ticker SPY\n"
            "```",
        )
        return

    # Build summary rows
    import pandas as pd  # noqa: PLC0415

    rows = []
    for exp in experiments:
        km = extract_key_metrics(exp.get("metrics", {}))
        # Pick best single metric to show
        metric_str = "—"
        if "sharpe_net" in km:
            metric_str = f"Sharpe(net)={km['sharpe_net']:+.2f}"
        elif "mean_ic" in km:
            metric_str = f"IC={km['mean_ic']:+.4f}"
        elif "label0_mean_ic" in km:
            metric_str = f"IC={km['label0_mean_ic']:+.4f}"

        rows.append({
            "experiment_id": exp.get("experiment_id", ""),
            "name": exp.get("name", ""),
            "created_at": exp.get("created_at", "")[:19].replace("T", " "),
            "start_date": exp.get("start_date", ""),
            "end_date": exp.get("end_date", ""),
            "key metric": metric_str,
        })

    df = pd.DataFrame(rows)

    st.markdown(f"**{len(experiments)} experiment(s)** in registry (newest first)")
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Experiment detail")

    options = [f"{e.get('experiment_id', '')} — {e.get('name', '')}" for e in experiments]
    selected = st.selectbox("Select experiment", options)
    idx = options.index(selected)
    exp = experiments[idx]

    _render_experiment_detail(exp)


def _render_experiment_detail(exp: dict) -> None:
    """Render the full detail view for one experiment."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**ID:** `{exp.get('experiment_id', '')}`")
        st.markdown(f"**Name:** {exp.get('name', '')}")
        st.markdown(f"**Created:** {exp.get('created_at', '')}")
        st.markdown(f"**Command:** `{exp.get('command', '')}`")
        st.markdown(f"**Window:** {exp.get('start_date', '')} → {exp.get('end_date', '')}")
        st.markdown(f"**Model:** {exp.get('model_type', '')} / {exp.get('validation_method', '')}")

    with col2:
        st.markdown(f"**Universe:** {', '.join(exp.get('universe', [])) or '—'}")
        st.markdown(f"**Data sources:** {', '.join(exp.get('data_sources', [])) or '—'}")
        st.markdown(f"**Feature sets:** {', '.join(exp.get('feature_sets', [])) or '—'}")
        lims = exp.get("limitations", [])
        if lims:
            st.markdown("**Limitations:**")
            for lim in lims:
                st.markdown(f"- {lim}")

    metrics = exp.get("metrics", {})
    if metrics:
        st.subheader("Metrics")
        st.json(metrics, expanded=False)

    report_text = exp.get("report_text", "")
    if report_text:
        with st.expander("📄 Full report"):
            st.markdown(report_text)


# ── Page C: Backtest ─────────────────────────────────────────────────────────

def _page_backtest() -> None:
    st.title("Backtest")

    experiments = load_experiments(_REGISTRY_ROOT)
    bt = find_latest_backtest(experiments)

    if bt is None:
        st.info(
            "No backtest experiment found in the registry.\n\n"
            "Run the backtest pipeline:\n"
            "```bash\n"
            "mosaic run-backtest \\\n"
            "  --experiment news-sector \\\n"
            "  --start 2020-01-01 \\\n"
            "  --end 2024-12-31 \\\n"
            "  --offline-sample \\\n"
            "  --cost-bps 5\n"
            "```",
        )
        return

    st.markdown(f"**Latest backtest:** `{bt.get('experiment_id', '')}`")
    st.markdown(
        f"**Period:** {bt.get('start_date', '')} → {bt.get('end_date', '')}  "
        f"| **Command:** `{bt.get('command', '')}`"
    )

    metrics = bt.get("metrics", {})
    km = extract_key_metrics(metrics)

    if km:
        st.subheader("Performance summary")
        cols = st.columns(len(km) if len(km) <= 5 else 5)
        items = list(km.items())
        for i, (label, value) in enumerate(items[:5]):
            with cols[i]:
                if label in ("ann_net_return", "ann_gross_return", "max_drawdown_net"):
                    display = f"{value*100:+.2f}%"
                elif label in ("sharpe_net", "sharpe_gross"):
                    display = f"{value:+.2f}"
                else:
                    display = f"{value:.4f}"
                st.metric(label.replace("_", " "), display)

    # Limitations
    lims = bt.get("limitations", [])
    if lims:
        with st.expander("⚠️ Limitations"):
            for lim in lims:
                st.markdown(f"- {lim}")

    # Full report
    report_text = bt.get("report_text", "")
    if report_text:
        with st.expander("📄 Full backtest report"):
            st.markdown(report_text)
    else:
        # Try the static file
        static_rpt = _REPORTS_DIR / "backtest_news_sector.md"
        text = load_report_text(static_rpt)
        if text:
            with st.expander("📄 Full backtest report (from reports/generated)"):
                st.markdown(text)


# ── Page D: Research Graph ────────────────────────────────────────────────────

def _page_graph() -> None:
    st.title("Research Graph")

    graph_data = load_graph_json(_GRAPH_JSON)

    if graph_data is None:
        st.info(
            "`memory/research_graph.json` not found.\n\n"
            "Build the knowledge graph:\n"
            "```bash\n"
            "mosaic build-graph\n"
            "```",
        )
        return

    summary = graph_data.get("summary", {})
    c1, c2 = st.columns(2)
    c1.metric("Total nodes", summary.get("total_nodes", 0))
    c2.metric("Total edges", summary.get("total_edges", 0))

    # Node/edge type counts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nodes by type")
        nbt = summary.get("nodes_by_type", {})
        if nbt:
            import pandas as pd  # noqa: PLC0415
            st.dataframe(
                pd.DataFrame(
                    [{"type": k, "count": v} for k, v in sorted(nbt.items())]
                ),
                hide_index=True,
                use_container_width=True,
            )

    with col2:
        st.subheader("Edges by type")
        ebt = summary.get("edges_by_type", {})
        if ebt:
            import pandas as pd  # noqa: PLC0415
            st.dataframe(
                pd.DataFrame(
                    [{"type": k, "count": v} for k, v in sorted(ebt.items())]
                ),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()

    # Searchable node table
    st.subheader("Nodes")
    node_rows = graph_node_table(graph_data)
    if node_rows:
        import pandas as pd  # noqa: PLC0415
        df_nodes = pd.DataFrame(node_rows)
        search_node = st.text_input("Filter nodes (label / type)", key="node_search")
        if search_node:
            mask = df_nodes.apply(
                lambda col: col.astype(str).str.contains(search_node, case=False)
            ).any(axis=1)
            df_nodes = df_nodes[mask]
        st.dataframe(df_nodes, use_container_width=True, hide_index=True)

    st.subheader("Edges")
    edge_rows = graph_edge_table(graph_data)
    if edge_rows:
        import pandas as pd  # noqa: PLC0415
        df_edges = pd.DataFrame(edge_rows)
        search_edge = st.text_input("Filter edges (source / type / target)", key="edge_search")
        if search_edge:
            mask = df_edges.apply(
                lambda col: col.astype(str).str.contains(search_edge, case=False)
            ).any(axis=1)
            df_edges = df_edges[mask]
        st.dataframe(df_edges, use_container_width=True, hide_index=True)


# ── Page E: Reports ──────────────────────────────────────────────────────────

def _page_reports() -> None:
    st.title("Reports")

    reports = find_reports(_REPORTS_DIR, _REGISTRY_ROOT)

    if not reports:
        st.info(
            "No Markdown reports found.\n\n"
            "Run any pipeline command to generate a report:\n"
            "```bash\n"
            "mosaic run-baseline\n"
            "```",
        )
        return

    labels = [r["label"] for r in reports]
    selected_label = st.selectbox("Select report", labels)
    idx = labels.index(selected_label)
    selected_path = reports[idx]["path"]

    st.caption(f"Path: `{selected_path}`")

    text = load_report_text(selected_path)
    if text is None:
        st.error(f"Could not read report at `{selected_path}`.")
    else:
        st.markdown(text)


# ── Router ────────────────────────────────────────────────────────────────────

def main() -> None:
    page = _sidebar()

    if page == "Overview":
        _page_overview()
    elif page == "Experiments":
        _page_experiments()
    elif page == "Backtest":
        _page_backtest()
    elif page == "Research Graph":
        _page_graph()
    elif page == "Reports":
        _page_reports()


main()
