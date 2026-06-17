"""MosaicAlpha research dashboard (Streamlit) — 8-page UX overhaul.

Launch with:
    mosaic dashboard
or directly:
    streamlit run mosaic_alpha/dashboard/app.py

All data is read from the local filesystem:
  - memory/experiments/         experiment registry
  - memory/research_graph.json  knowledge-graph export
  - reports/generated/          CLI-generated Markdown reports
  - ../vendor/                  sibling vendor repos (display only)

No network calls are made. No API keys required.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# ── Page config (MUST be the first Streamlit call) ───────────────────────────
st.set_page_config(
    page_title="MosaicAlpha Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

from mosaic_alpha.dashboard.helpers import (  # noqa: E402
    METRIC_META,
    extract_key_metrics,
    find_latest_backtest,
    find_reports,
    find_vendor_repos,
    format_metric,
    graph_edge_table,
    graph_node_table,
    load_experiments,
    load_graph_json,
    load_report_text,
    metric_explanation,
    metric_label,
)

# ── Constants ─────────────────────────────────────────────────────────────────

_REGISTRY_ROOT = Path("memory/experiments")
_GRAPH_JSON = Path("memory/research_graph.json")
_REPORTS_DIR = Path("reports/generated")

_PAGES = [
    "Overview",
    "Experiments",
    "Backtest",
    "Research Graph",
    "Reports",
    "Architecture",
    "Vendor Inspirations",
    "Reproduce Demo",
]


# ── Sidebar navigation ────────────────────────────────────────────────────────

def _sidebar() -> str:
    with st.sidebar:
        st.markdown("## 📊 MosaicAlpha")
        st.caption("Local quantitative research workbench")
        st.divider()
        page = st.radio("Navigate", _PAGES, label_visibility="collapsed")
        st.divider()

        # Live status mini-panel
        experiments = load_experiments(_REGISTRY_ROOT)
        graph_data = load_graph_json(_GRAPH_JSON)
        reports = find_reports(_REPORTS_DIR, _REGISTRY_ROOT)

        st.markdown("**Status**")
        st.markdown(f"- Experiments: **{len(experiments)}**")
        node_count = graph_data["summary"]["total_nodes"] if graph_data else 0
        st.markdown(f"- Graph nodes: **{node_count}**")
        st.markdown(f"- Reports: **{len(reports)}**")

        st.divider()
        st.caption(
            "⚠️ Research infrastructure only. "
            "Not investment advice."
        )
    return page  # type: ignore[return-value]


# ── Page 1: Overview ─────────────────────────────────────────────────────────

def _page_overview() -> None:
    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <h1 style="margin-bottom:0.2em;">MosaicAlpha</h1>
        <p style="font-size:1.15em; color:#666; margin-top:0;">
        A local-first alternative-data research workbench for discovering, validating,
        and explaining quantitative signals from public datasets.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.warning(
        "**Research infrastructure — not investment advice.** "
        "All numbers are backtests on historical data. Past performance does not predict "
        "future returns. This project is built for learning and demonstration purposes only.",
        icon="⚠️",
    )

    # ── Status cards ─────────────────────────────────────────────────────────
    experiments = load_experiments(_REGISTRY_ROOT)
    graph_data = load_graph_json(_GRAPH_JSON)
    reports = find_reports(_REPORTS_DIR, _REGISTRY_ROOT)
    bt = find_latest_backtest(experiments)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experiments saved", len(experiments))
    node_count = graph_data["summary"]["total_nodes"] if graph_data else 0
    c2.metric("Graph nodes", node_count)
    c3.metric("Reports available", len(reports))

    # Show latest Sharpe in hero card if available
    if bt:
        km = extract_key_metrics(bt.get("metrics", {}))
        sharpe = km.get("sharpe_net")
        c4.metric(
            "Latest Sharpe (net)",
            f"{sharpe:+.2f}" if sharpe is not None else "—",
            help="Net of transaction costs from the most recent backtest experiment.",
        )
    else:
        c4.metric("Latest Sharpe (net)", "—", help="Run mosaic run-backtest to populate.")

    st.divider()

    # ── What this project demonstrates ───────────────────────────────────────
    st.subheader("What this project demonstrates")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Data pipeline**
- Daily price data via `yfinance` (no key required)
- Macro regime features from FRED (free API key required for live pulls)
- News intensity features from GDELT (no key, or use `--offline-sample`)

**Signal research**
- Walk-forward expanding-window validation (no look-ahead bias)
- Cross-sectional Information Coefficient (IC) measures signal quality
- Four-way ablation: price-only vs price+macro vs price+news vs all three
""")

    with col_b:
        st.markdown("""
**Backtesting**
- Dollar-neutral long/short portfolio on 11 SPDR sector ETFs
- Transaction-cost-aware: configurable basis-point cost per trade
- Sharpe ratio, drawdown, hit rate, turnover all reported net of costs

**Research infrastructure**
- Experiment registry: every run persisted as human-readable JSON
- Knowledge graph: datasets, features, models, experiments, limitations linked
- This dashboard: browse results without touching the command line
""")

    st.divider()

    # ── Milestones ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Milestones completed")
        milestones = [
            ("M1", "Price-only SPY baseline signal"),
            ("M2", "Sector ETF cross-sectional baseline"),
            ("M3", "FRED macro regime features"),
            ("M4", "GDELT news intensity features"),
            ("M5", "Transaction-cost-aware backtesting"),
            ("M6", "Experiment registry & local research memory"),
            ("M7", "Research knowledge graph"),
            ("M8", "Streamlit research dashboard"),
            ("M9", "Repository polish & reproducibility docs"),
        ]
        for code, desc in milestones:
            st.markdown(f"✅ **{code}** — {desc}")

    with col2:
        st.subheader("Quick-start commands")
        st.code(
            "# Run the full pipeline (offline / no API keys)\n"
            "mosaic run-backtest --offline-sample\n\n"
            "# Explore results\n"
            "mosaic list-experiments\n"
            "mosaic build-graph\n"
            "mosaic dashboard",
            language="bash",
        )


# ── Page 2: Experiments ───────────────────────────────────────────────────────

def _page_experiments() -> None:
    st.title("Experiments")
    st.caption(
        "Every pipeline run is automatically saved to `memory/experiments/` as a "
        "human-readable JSON folder. Browse the full history here."
    )

    experiments = load_experiments(_REGISTRY_ROOT)

    if not experiments:
        st.info(
            "No experiments found in `memory/experiments/`.\n\n"
            "Run any pipeline command to save an experiment:\n"
            "```bash\n"
            "mosaic run-backtest --offline-sample\n"
            "```",
        )
        return

    import pandas as pd  # noqa: PLC0415

    # ── Filter bar ────────────────────────────────────────────────────────────
    col_f1, col_f2 = st.columns([2, 3])
    with col_f1:
        all_names = sorted({e.get("name", "") for e in experiments})
        name_filter = st.multiselect("Filter by pipeline name", all_names)
    with col_f2:
        search_text = st.text_input("Search by ID, data sources, or feature sets", "")

    filtered = experiments
    if name_filter:
        filtered = [e for e in filtered if e.get("name", "") in name_filter]
    if search_text:
        q = search_text.lower()
        filtered = [
            e for e in filtered
            if q in e.get("experiment_id", "").lower()
            or any(q in s.lower() for s in e.get("data_sources", []))
            or any(q in s.lower() for s in e.get("feature_sets", []))
        ]

    st.markdown(f"**{len(filtered)} of {len(experiments)} experiment(s)** shown (newest first)")

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for exp in filtered:
        km = extract_key_metrics(exp.get("metrics", {}))
        metric_str = "—"
        if "sharpe_net" in km:
            metric_str = f"Sharpe(net) {km['sharpe_net']:+.2f}"
        elif "mean_ic" in km:
            metric_str = f"IC {km['mean_ic']:+.4f}"
        elif "label0_mean_ic" in km:
            metric_str = f"IC {km['label0_mean_ic']:+.4f}"

        rows.append({
            "ID": exp.get("experiment_id", ""),
            "Pipeline": exp.get("name", ""),
            "Created": exp.get("created_at", "")[:19].replace("T", " "),
            "Period": f"{exp.get('start_date', '')} → {exp.get('end_date', '')}",
            "Key metric": metric_str,
            "Data sources": ", ".join(exp.get("data_sources", [])),
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Detail viewer ─────────────────────────────────────────────────────────
    st.subheader("Experiment detail")

    if not filtered:
        st.caption("No experiments match the current filters.")
        return

    options = [f"{e.get('experiment_id', '')} — {e.get('name', '')}" for e in filtered]
    selected = st.selectbox("Select experiment", options)
    idx = options.index(selected)
    exp = filtered[idx]

    _render_experiment_detail(exp)


def _render_experiment_detail(exp: dict) -> None:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**ID:** `{exp.get('experiment_id', '')}`")
        st.markdown(f"**Pipeline:** {exp.get('name', '')}")
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

    # ── Metrics with plain-English explanations ───────────────────────────────
    metrics = exp.get("metrics", {})
    km = extract_key_metrics(metrics)
    if km:
        st.subheader("Key metrics")
        cols = st.columns(min(len(km), 4))
        for i, (key, val) in enumerate(list(km.items())[:4]):
            with cols[i % 4]:
                label = metric_label(key)
                display = format_metric(val, key)
                explanation = metric_explanation(key)
                st.metric(label, display, help=explanation)

        if len(km) > 4:
            with st.expander("All metrics"):
                for key, val in km.items():
                    label = metric_label(key)
                    display = format_metric(val, key)
                    explanation = metric_explanation(key)
                    st.markdown(
                        f"**{label}:** {display}"
                        + (f" — {explanation}" if explanation else "")
                    )

    elif metrics:
        with st.expander("Raw metrics JSON"):
            st.json(metrics, expanded=False)

    report_text = exp.get("report_text", "")
    if report_text:
        with st.expander("📄 Full report"):
            st.markdown(report_text)


# ── Page 3: Backtest ──────────────────────────────────────────────────────────

def _page_backtest() -> None:
    st.title("Backtest")
    st.caption(
        "The most recent `run_backtest` experiment from the registry is displayed here. "
        "Metrics are shown net **and** gross of transaction costs so you can see the "
        "impact of trading friction."
    )

    experiments = load_experiments(_REGISTRY_ROOT)
    bt = find_latest_backtest(experiments)

    if bt is None:
        st.info(
            "No backtest experiment found in the registry.\n\n"
            "Run the full pipeline:\n"
            "```bash\n"
            "mosaic run-backtest \\\n"
            "  --start 2020-01-01 \\\n"
            "  --end 2024-12-31 \\\n"
            "  --offline-sample \\\n"
            "  --cost-bps 5\n"
            "```",
        )
        return

    st.markdown(
        f"**Experiment:** `{bt.get('experiment_id', '')}`  "
        f"| **Period:** {bt.get('start_date', '')} → {bt.get('end_date', '')}  "
        f"| **Command:** `{bt.get('command', '')}`"
    )

    metrics = bt.get("metrics", {})
    km = extract_key_metrics(metrics)

    # ── Net vs Gross split ────────────────────────────────────────────────────
    net_keys = ["sharpe_net", "ann_net_return", "max_drawdown_net"]
    gross_keys = ["sharpe_gross", "ann_gross_return"]
    other_keys = ["hit_rate", "avg_turnover", "n_periods", "mean_ic", "ic_t_stat"]

    st.subheader("After transaction costs (net)")
    net_cols = st.columns(len(net_keys))
    for col, key in zip(net_cols, net_keys):
        if key in km:
            col.metric(
                metric_label(key),
                format_metric(km[key], key),
                help=metric_explanation(key),
            )

    st.subheader("Before transaction costs (gross)")
    gross_cols = st.columns(len(gross_keys))
    for col, key in zip(gross_cols, gross_keys):
        if key in km:
            col.metric(
                metric_label(key),
                format_metric(km[key], key),
                help=metric_explanation(key),
            )

    other_present = [k for k in other_keys if k in km]
    if other_present:
        st.subheader("Other statistics")
        other_cols = st.columns(min(len(other_present), 4))
        for col, key in zip(other_cols, other_present):
            col.metric(
                metric_label(key),
                format_metric(km[key], key),
                help=metric_explanation(key),
            )

    # ── Plain-English summary ─────────────────────────────────────────────────
    st.divider()
    st.subheader("How to read these numbers")
    with st.expander("Metric glossary", expanded=False):
        for key in list(METRIC_META.keys()):
            label = metric_label(key)
            explanation = metric_explanation(key)
            if explanation:
                st.markdown(f"**{label}** — {explanation}")

    # ── Limitations ───────────────────────────────────────────────────────────
    lims = bt.get("limitations", [])
    if lims:
        with st.expander("⚠️ Limitations of this backtest"):
            for lim in lims:
                st.markdown(f"- {lim}")

    # ── Report ────────────────────────────────────────────────────────────────
    report_text = bt.get("report_text", "")
    if not report_text:
        static_rpt = _REPORTS_DIR / "backtest_news_sector.md"
        report_text = load_report_text(static_rpt) or ""
    if report_text:
        with st.expander("📄 Full backtest report"):
            st.markdown(report_text)


# ── Page 4: Research Graph ────────────────────────────────────────────────────

def _page_graph() -> None:
    st.title("Research Graph")
    st.caption(
        "The knowledge graph links datasets, features, models, experiments, metrics, "
        "and limitations into a searchable directed graph. Build it with `mosaic build-graph`."
    )

    graph_data = load_graph_json(_GRAPH_JSON)

    if graph_data is None:
        st.info(
            "`memory/research_graph.json` not found.\n\n"
            "Build the knowledge graph first:\n"
            "```bash\n"
            "mosaic build-graph\n"
            "```",
        )
        return

    import pandas as pd  # noqa: PLC0415

    summary = graph_data.get("summary", {})

    c1, c2 = st.columns(2)
    c1.metric("Total nodes", summary.get("total_nodes", 0))
    c2.metric("Total edges", summary.get("total_edges", 0))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Nodes by type")
        nbt = summary.get("nodes_by_type", {})
        if nbt:
            st.dataframe(
                pd.DataFrame([{"type": k, "count": v} for k, v in sorted(nbt.items())]),
                hide_index=True,
                use_container_width=True,
            )
    with col2:
        st.subheader("Edges by type")
        ebt = summary.get("edges_by_type", {})
        if ebt:
            st.dataframe(
                pd.DataFrame([{"type": k, "count": v} for k, v in sorted(ebt.items())]),
                hide_index=True,
                use_container_width=True,
            )

    st.divider()

    # ── Searchable node table ─────────────────────────────────────────────────
    st.subheader("Nodes")
    node_rows = graph_node_table(graph_data)
    if node_rows:
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
        df_edges = pd.DataFrame(edge_rows)
        search_edge = st.text_input("Filter edges (source / type / target)", key="edge_search")
        if search_edge:
            mask = df_edges.apply(
                lambda col: col.astype(str).str.contains(search_edge, case=False)
            ).any(axis=1)
            df_edges = df_edges[mask]
        st.dataframe(df_edges, use_container_width=True, hide_index=True)


# ── Page 5: Reports ───────────────────────────────────────────────────────────

def _page_reports() -> None:
    st.title("Reports")
    st.caption(
        "Markdown reports are generated automatically after each pipeline run "
        "and saved to `reports/generated/` and the experiment registry."
    )

    reports = find_reports(_REPORTS_DIR, _REGISTRY_ROOT)

    if not reports:
        st.info(
            "No Markdown reports found.\n\n"
            "Run any pipeline command to generate a report:\n"
            "```bash\n"
            "mosaic run-backtest --offline-sample\n"
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


# ── Page 6: Architecture ──────────────────────────────────────────────────────

def _page_architecture() -> None:
    st.title("Architecture")
    st.caption(
        "A layer-by-layer overview of how MosaicAlpha is built. "
        "See `docs/architecture.md` for the full written description."
    )

    st.markdown("""
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CLI  (mosaic_alpha/cli.py)                      │
│  run-baseline  run-sector-baseline  run-macro-sector  run-news-sector   │
│  run-backtest  list-experiments  show-experiment  build-graph           │
│  graph-summary  graph-query  dashboard                                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
         ┌─────────────────────────┼──────────────────────────┐
         ▼                         ▼                          ▼
┌────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│  Data Layer    │    │  Feature Layer       │    │  Backtest Layer      │
│                │    │                     │    │                     │
│ yfinance       │    │ price_features      │    │ portfolio.py        │
│ FRED (macro)   │    │ macro_features      │    │ BacktestResult      │
│ GDELT (news)   │    │ news_features       │    │ cost-aware L/S      │
└────────────────┘    └─────────────────────┘    └─────────────────────┘
         │                         │                          │
         └─────────────────────────┼──────────────────────────┘
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Research / Validation Layer                        │
│   walk_forward_splits  •  cross_sectional_ic  •  four_way_ablation   │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  Registry Layer  │  │   Graph Layer    │  │   Dashboard Layer    │
│                  │  │                  │  │                      │
│ ExperimentRecord │  │ ResearchGraph    │  │ app.py (Streamlit)   │
│ save_experiment  │  │ build_graph      │  │ helpers.py (pure Py) │
│ list_experiments │  │ export_json      │  │                      │
│ memory/          │  │ export_graphml   │  │                      │
│   experiments/   │  │ queries.py       │  │                      │
└──────────────────┘  └──────────────────┘  └──────────────────────┘
```
""")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Key design decisions")
        st.markdown("""
**Local-first**
All data is cached on disk. No database server required. The pipeline can run
fully offline (using `--offline-sample` for GDELT and cached FRED/price data).

**Walk-forward validation**
Training windows only expand forward in time — the model never sees future data
during training. Each test period immediately follows its training window with
no gap or overlap.

**Pure-Python helper layer**
`dashboard/helpers.py` contains zero Streamlit imports. Every helper function
can be unit-tested with plain pytest without launching a browser.

**Experiment registry as files**
Every run writes human-readable JSON to `memory/experiments/<id>/`. The registry
can be inspected with any text editor, diffed with git, and queried with the CLI.
""")

    with col2:
        st.subheader("Design philosophy")
        st.markdown("""
**Legible over clever**
Code is optimised for a reader who is unfamiliar with the codebase. Variable
names are explicit. Each layer has a single responsibility.

**Uncertainty-first**
Limitations are recorded alongside results in the experiment registry and the
knowledge graph. The dashboard surfaces them with equal prominence.

**No black boxes**
The knowledge graph makes the full research lineage explicit: which datasets
produced which features, which features trained which models, which experiments
produced which metrics, and what limitations apply.

**Reproducible from a fresh clone**
`mosaic run-backtest --offline-sample` produces deterministic results from
synthetic data with no API keys or network access required.
""")

    # Link to docs
    docs_path = Path("docs/architecture.md")
    if docs_path.exists():
        with st.expander("📄 Full architecture documentation"):
            text = load_report_text(docs_path)
            if text:
                st.markdown(text)


# ── Page 7: Vendor Inspirations ───────────────────────────────────────────────

def _page_vendor() -> None:
    st.title("Vendor Inspirations")
    st.caption(
        "MosaicAlpha was built alongside four open-source repositories that each "
        "influenced a specific aspect of the design. No code was copied; these repos "
        "are used as **design references** only."
    )

    st.info(
        "The vendor repos are expected at `../vendor/<repo-name>/` (a sibling of this "
        "project root). Clone them there to see their README previews here. "
        "They are not required to run MosaicAlpha.",
        icon="ℹ️",
    )

    repos = find_vendor_repos()

    for repo in repos:
        present = repo["present"]
        status = "✅ Found locally" if present else "📂 Not cloned (display-only)"

        with st.expander(
            f"**{repo['display_name']}** — {repo['tagline']}  |  {status}",
            expanded=True,
        ):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"**Repo name:** `{repo['name']}`")
                st.markdown(f"**Influence area:** {repo['influence']}")
                if present:
                    st.markdown(f"**Local path:** `{repo['path']}`")
                else:
                    st.markdown(
                        f"**Expected at:** `{repo['path']}`  \n"
                        f"*(Clone to see README preview)*"
                    )

            with col2:
                st.markdown(f"**How it shaped MosaicAlpha:**  \n{repo['description']}")

            if present and repo["readme_snippet"]:
                with st.expander("README preview (first 20 lines)"):
                    st.code(repo["readme_snippet"], language="markdown")

    st.divider()
    st.subheader("Attribution and licensing")
    st.markdown("""
Each vendor repo is an independent open-source project with its own license.
MosaicAlpha does not redistribute or incorporate any code from these projects.
They are listed here solely to credit the design ideas that influenced this
project's architecture.

To clone the vendor repos (optional):
```bash
mkdir -p ../vendor
cd ../vendor
git clone https://github.com/HunterMRocha/agent-reach
git clone https://github.com/HunterMRocha/G0DM0D3
git clone https://github.com/HunterMRocha/openhuman
git clone https://github.com/HunterMRocha/Understand-Anything
```

See `docs/vendor_inspirations.md` for the full written attribution.
""")


# ── Page 8: Reproduce Demo ────────────────────────────────────────────────────

def _page_reproduce() -> None:
    st.title("Reproduce Demo")
    st.caption(
        "Exact commands to reproduce every result from a fresh clone, "
        "on both Windows and macOS/Linux."
    )

    st.markdown("""
The full pipeline can run without any API keys using `--offline-sample` mode.
GDELT news data is replaced with deterministic synthetic samples; price and
macro data use cached parquet files generated on the first run.
""")

    tab_win, tab_mac = st.tabs(["Windows (PowerShell / bash)", "macOS / Linux"])

    _WINDOWS_COMMANDS = """\
# 1. Clone and enter the repo
git clone https://github.com/HunterMRocha/mosaic-alpha
cd mosaic-alpha

# 2. Create and activate the virtual environment
python -m venv .venv
.venv\\Scripts\\activate

# 3. Install the package (editable)
pip install -e ".[dev]"

# 4. (Optional) Add your FRED API key for live macro pulls
#    Copy .env.example → .env and fill in FRED_API_KEY
copy .env.example .env

# 5. Run all experiments (offline mode — no API keys needed)
mosaic run-baseline
mosaic run-sector-baseline
mosaic run-macro-sector --offline-sample
mosaic run-news-sector --offline-sample
mosaic run-backtest --offline-sample

# 6. Build the knowledge graph and open the dashboard
mosaic build-graph
mosaic dashboard

# 7. Run the test suite
pytest tests/ -v
"""

    _MAC_COMMANDS = """\
# 1. Clone and enter the repo
git clone https://github.com/HunterMRocha/mosaic-alpha
cd mosaic-alpha

# 2. Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the package (editable)
pip install -e ".[dev]"

# 4. (Optional) Add your FRED API key for live macro pulls
#    Copy .env.example → .env and fill in FRED_API_KEY
cp .env.example .env

# 5. Run all experiments (offline mode — no API keys needed)
mosaic run-baseline
mosaic run-sector-baseline
mosaic run-macro-sector --offline-sample
mosaic run-news-sector --offline-sample
mosaic run-backtest --offline-sample

# 6. Build the knowledge graph and open the dashboard
mosaic build-graph
mosaic dashboard

# 7. Run the test suite
pytest tests/ -v
"""

    with tab_win:
        st.code(_WINDOWS_COMMANDS, language="bash")

    with tab_mac:
        st.code(_MAC_COMMANDS, language="bash")

    st.divider()

    # ── What each command produces ────────────────────────────────────────────
    st.subheader("What each command produces")

    commands = [
        ("mosaic run-baseline", "SPY price-only signal; IC and t-stat on 1-day forward return"),
        ("mosaic run-sector-baseline", "Cross-sectional IC across 11 SPDR sector ETFs"),
        ("mosaic run-macro-sector", "Adds FRED macro regime features; 2-way IC comparison"),
        ("mosaic run-news-sector", "Adds GDELT news intensity; 4-way ablation (price/macro/news/all)"),
        ("mosaic run-backtest", "Dollar-neutral L/S portfolio with transaction costs; Sharpe, drawdown, hit rate"),
        ("mosaic list-experiments", "Table of all saved experiments in the registry"),
        ("mosaic build-graph", "Exports research_graph.json and research_graph.graphml"),
        ("mosaic dashboard", "Opens this dashboard in your browser"),
        ("pytest tests/ -v", "Runs the full test suite (should pass 210+ tests)"),
    ]

    import pandas as pd  # noqa: PLC0415
    st.dataframe(
        pd.DataFrame(commands, columns=["Command", "Produces"]),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Environment variables")
    st.markdown("""
| Variable | Required | Purpose |
|---|---|---|
| `FRED_API_KEY` | Optional | Live macro pulls from FRED. Free at fred.stlouisfed.org. Omit to use `--offline-sample`. |

GDELT and Yahoo Finance do not require API keys.
""")

    st.subheader("Offline-sample mode explained")
    st.markdown("""
`--offline-sample` replaces live GDELT network calls with deterministic synthetic
news intensity counts seeded by date and ticker. Results are **reproducible across
machines** but do not reflect real news data.

Price data is fetched from Yahoo Finance on the first run and cached as Parquet
files in `data/`. Subsequent runs read from cache. The cache directory is listed
in `.gitignore` and must be regenerated after a fresh clone.

FRED macro data follows the same pattern: fetched once, cached locally.
""")


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
    elif page == "Architecture":
        _page_architecture()
    elif page == "Vendor Inspirations":
        _page_vendor()
    elif page == "Reproduce Demo":
        _page_reproduce()


main()
