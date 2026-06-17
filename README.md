# MosaicAlpha

A local-first quantitative research workbench for discovering, validating, and explaining signals from public alternative data.

---

## Motivation

Most open-source quant projects either stop at a backtest number or hide the entire workflow inside a black box. MosaicAlpha takes a different approach: every stage of the research process — data ingestion, feature engineering, walk-forward validation, transaction-cost-aware backtesting, experiment logging, and lineage tracking — is explicit, testable, and reproducible from a blank machine.

The goal is to demonstrate what rigorous quant research infrastructure looks like, not to claim live alpha.

---

## What this project demonstrates

- **Walk-forward validation** with strictly non-overlapping train/test splits and no lookahead leakage
- **Cross-sectional information coefficient (IC)** as the primary signal evaluation metric for panel data
- **Alternative-data integration** (FRED macro series, GDELT news intensity) with ablation studies comparing price-only vs. price+macro vs. price+news vs. price+macro+news
- **Transaction-cost-aware backtesting** with explicit turnover tracking and basis-point cost charges
- **Experiment provenance** via a local file-based registry (JSON, human-readable, Git-friendly)
- **Research lineage** via a directed knowledge graph linking datasets → features → models → experiments → metrics → limitations
- **Reproducibility**: a single offline-sample flag removes all network dependencies so the full pipeline can be run on any machine without API keys

---

## Current capabilities

| Milestone | Component | Status |
|---|---|---|
| M1 | Price-only SPY baseline (walk-forward ridge) | ✅ |
| M2 | Sector ETF cross-sectional baseline | ✅ |
| M3 | FRED macro regime features | ✅ |
| M4 | GDELT news intensity features (+ offline-sample mode) | ✅ |
| M5 | Transaction-cost-aware L/S portfolio backtest | ✅ |
| M6 | Experiment registry and local research memory | ✅ |
| M7 | Research knowledge graph (build, export, query) | ✅ |
| M8 | Streamlit research dashboard | ✅ |

---

## Architecture overview

```
mosaic_alpha/
├── data/           # Data connectors: yfinance, FRED, GDELT, caching
├── features/       # Feature families: price, macro, news
├── research/       # Labels, models, walk-forward validation, metrics,
│                   # baseline orchestrators, backtest engine, registry
├── graph/          # Knowledge graph: schema, builder, export, queries
├── dashboard/      # Streamlit app and pure-Python data helpers
└── cli.py          # Typer CLI: mosaic <command>
```

See [`docs/architecture.md`](docs/architecture.md) for a full layer-by-layer description.

---

## Installation

### Prerequisites

- Python 3.11 or later
- Git

### Windows (PowerShell)

```powershell
# 1. Clone
git clone https://github.com/HDMassey/Mosaic-Alpha.git
cd Mosaic-Alpha

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install
python -m pip install --upgrade pip
pip install -e .

# 4. Verify
mosaic hello
python -m pytest
```

### macOS / Linux

```bash
# 1. Clone
git clone https://github.com/HDMassey/Mosaic-Alpha.git
cd Mosaic-Alpha

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install
pip install --upgrade pip
pip install -e .

# 4. Verify
mosaic hello
python -m pytest
```

---

## Environment variables

Copy the example file and fill in the values you need:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `FRED_API_KEY` | Optional | Needed only for live FRED macro pulls. Free at [fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html). |

**GDELT** does not require an API key. Use `--offline-sample` to skip live GDELT calls entirely (see below).

The `.env` file is listed in `.gitignore` and is never committed.

---

## Reproducing the demo from a blank machine

The complete sequence below runs the entire pipeline with no API keys and no network calls beyond the initial price download from Yahoo Finance.

```bash
# Price-only single-asset baseline
mosaic run-baseline --ticker SPY --start 2015-01-01 --end 2024-12-31

# Cross-sectional sector ETF baseline (price features only)
mosaic run-sector-baseline --start 2015-01-01 --end 2024-12-31

# Price vs. price+macro comparison (requires FRED_API_KEY for live data;
# set to offline-sample via --no-fred or use pre-cached series)
mosaic run-macro-sector --start 2015-01-01 --end 2024-12-31

# 4-way comparison with synthetic news data (no GDELT network calls)
mosaic run-news-sector --start 2020-01-01 --end 2024-12-31 --offline-sample

# Transaction-cost-aware backtest on the price+macro+news signal
mosaic run-backtest \
  --experiment news-sector \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --offline-sample \
  --cost-bps 5

# Inspect the experiment registry
mosaic list-experiments

# Build the research knowledge graph
mosaic build-graph

# Query the graph
mosaic graph-summary
mosaic graph-query --dataset GDELT
mosaic graph-query --metric sharpe_net
mosaic graph-query --limitation sample

# Launch the local dashboard
mosaic dashboard
```

All commands save results to `memory/experiments/` (registry) and `reports/generated/` (Markdown reports). These directories are listed in `.gitignore`.

See [`docs/reproducibility.md`](docs/reproducibility.md) for a full explanation of the offline-sample mode, caching policy, and how to obtain a FRED API key.

---

## Core CLI reference

```
mosaic hello                    # Smoke test
mosaic run-baseline             # Price-only SPY baseline
mosaic run-sector-baseline      # Cross-sectional sector ETF baseline
mosaic run-macro-sector         # Price vs. price+macro comparison
mosaic run-news-sector          # 4-way price/macro/news/macro+news comparison
mosaic run-backtest             # Transaction-cost-aware L/S backtest
mosaic list-experiments         # List saved experiment records
mosaic show-experiment <ID>     # Detail view for one experiment
mosaic build-graph              # Build and export the knowledge graph
mosaic graph-summary            # Print node/edge counts
mosaic graph-query              # Query graph by dataset / metric / limitation
mosaic dashboard                # Launch the Streamlit dashboard
```

Each command accepts `--help` for full options.

---

## Dashboard

```bash
mosaic dashboard
```

Opens a local Streamlit server at `http://localhost:8501` with five pages:

| Page | Content |
|---|---|
| Overview | Project description, milestone status, live experiment/graph counts |
| Experiments | Searchable table of all saved experiments; detail view on selection |
| Backtest | Latest backtest metrics; full report in an expandable section |
| Research Graph | Node/edge type counts; searchable node and edge tables |
| Reports | Browse and view all generated Markdown reports |

---

## Data sources

| Source | Access | Notes |
|---|---|---|
| **Yahoo Finance** | Free, no key | Via `yfinance`; OHLCV prices cached as Parquet files |
| **FRED** | Free API key | Macro series (Federal Funds Rate, CPI, yield curve, …). Optional — offline macro is not yet supported separately. |
| **GDELT** | Free, no key | News intensity counts via GDELT GKG API. Use `--offline-sample` to skip live calls. |

---

## Generated files and Git policy

The following paths are generated at runtime and are excluded from version control:

| Path | Contents |
|---|---|
| `data/` | Price Parquet files, GDELT cache |
| `reports/generated/` | Markdown experiment reports |
| `memory/experiments/*/` | Experiment registry folders |
| `memory/research_graph.json` | Exported knowledge graph JSON |
| `memory/research_graph.graphml` | Exported knowledge graph GraphML |
| `.streamlit/` | Streamlit server cache |
| `.env` | Local environment variables |

The directory skeletons (`memory/experiments/.gitkeep`) are tracked so the repository clones with the expected structure.

---

## Testing

```bash
python -m pytest            # Run all 210 tests
python -m pytest -v         # Verbose output
python -m pytest tests/test_registry.py   # Single module
```

The test suite covers:

- Timestamp alignment and lookahead leakage (features and labels)
- Walk-forward split correctness
- Cross-sectional IC computation
- Macro feature construction
- News feature construction (offline-sample mode)
- Backtest weight construction, turnover, and P&L
- Experiment registry round-trips
- Knowledge graph schema, builder, export, and queries
- Dashboard helper functions

---

## Limitations

- The universe is restricted to 11 SPDR sector ETFs. Cross-sectional IC is statistically weak at this size.
- All model results are in-sample to the walk-forward framework; they have not been validated on a held-out out-of-sample period.
- GDELT news intensity is a raw count proxy; it captures volume of coverage, not sentiment.
- FRED macro data introduces point-in-time issues that are not fully corrected (series are available with publication lag).
- No market-impact model. Transaction costs are a flat basis-point charge on one-way turnover.
- Short selling is assumed frictionless beyond the explicit cost charge.
- These results do not constitute a trading strategy recommendation.

See [`docs/results.md`](docs/results.md) for a candid summary of all experiment results.

---

## Roadmap

See [`docs/roadmap.md`](docs/roadmap.md).

---

## License and disclaimer

This project is released for educational and research purposes.

**Nothing in this repository constitutes financial or investment advice.** All backtest results are historical simulations. Past performance of a research pipeline does not imply future trading profitability.
