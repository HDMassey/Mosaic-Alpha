# Architecture

MosaicAlpha is organised as a set of vertical layers. Each layer has a clear responsibility, takes clean inputs, and produces clean outputs. Layers communicate through plain Python dataclasses and JSON files rather than a shared database, which keeps them independently testable and replaceable.

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI  (mosaic_alpha/cli.py)              │
│            Typer commands wire the layers together           │
└──────────────────────────────┬──────────────────────────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         ▼                     ▼                      ▼
┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Dashboard     │  │  Registry        │  │  Graph           │
│  (Streamlit)   │  │  (memory/)       │  │  (knowledge      │
│                │  │                  │  │   graph)         │
└────────────────┘  └──────────────────┘  └──────────────────┘
         │                     │                      │
         └─────────────────────┼──────────────────────┘
                               │
         ┌─────────────────────┼──────────────────────┐
         ▼                     ▼                      ▼
┌────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  Research /    │  │  Backtest        │  │  Metrics         │
│  Orchestrators │  │  Engine          │  │  (CS-IC, IC,     │
│                │  │                  │  │   Sharpe, …)     │
└────────────────┘  └──────────────────┘  └──────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│                   Feature Layer                             │
│        price_features   macro_features   news_features      │
└──────────────────────────────┬─────────────────────────────┘
                               │
                               ▼
┌────────────────────────────────────────────────────────────┐
│                    Data Layer                               │
│         yfinance loader    FRED connector    GDELT fetcher  │
└────────────────────────────────────────────────────────────┘
```

---

## Data layer (`mosaic_alpha/data/`)

| Module | Responsibility |
|---|---|
| `loader.py` | Download and cache daily OHLCV prices via `yfinance`. Stores one Parquet file per ticker under `data/`. Reads from cache on subsequent calls. |
| `fred.py` | Fetch FRED macro series by series ID using the FRED REST API. Requires `FRED_API_KEY` in the environment. Returns a wide-format DataFrame indexed by date. |
| `gdelt.py` | Fetch GDELT GKG news intensity counts per sector keyword bucket. Supports live HTTP calls and an offline-sample mode that substitutes deterministic synthetic counts. |

All connectors return pandas DataFrames. Caching is handled at the connector level; callers do not manage cache invalidation.

---

## Feature layer (`mosaic_alpha/features/`)

| Module | Feature family | Inputs | Outputs |
|---|---|---|---|
| `price_features.py` | Price | OHLCV DataFrame | 7 features: `log_return`, `rolling_vol_5`, `rolling_vol_20`, `momentum_5`, `momentum_20`, `momentum_60`, `volume_zscore_20` |
| `macro_features.py` | Macro | FRED macro panel | 6 regime features: `fed_rate_chg`, `yield_slope`, `cpi_yoy`, `ind_prod_yoy`, `unemp_chg`, `yield_above_fed` |
| `news_features.py` | News | GDELT count panel | 5 news features: `news_intensity_raw`, `news_intensity_ma5`, `news_intensity_ma20`, `news_intensity_zscore`, `news_intensity_chg` |

All features are constructed with strict no-lookahead discipline: every value at row *t* uses only information available at the close of day *t*.

---

## Research and validation layer (`mosaic_alpha/research/`)

| Module | Responsibility |
|---|---|
| `labels.py` | Construct forward-return labels for 1-, 5-, and 20-day horizons. |
| `models.py` | Build and fit a ridge regression pipeline (StandardScaler → Ridge). |
| `validation.py` | Walk-forward expanding-window splits: training grows from a minimum of 252 days; each test fold covers 63 trading days. Splits are generated once and shared across all models in a comparison. |
| `metrics.py` | Compute IC (Pearson), rank IC (Spearman), hit rate, MSE, and L/S decile spread per fold. Aggregate across folds. |
| `cross_sectional.py` | Cross-sectional IC per date across the universe. Aggregate with t-statistics. |
| `baseline.py` | End-to-end orchestrator for the price-only single-ticker experiment (Milestone 1). |
| `sector_baseline.py` | Cross-sectional sector panel experiment (Milestone 2). |
| `macro_sector_baseline.py` | Price-only vs. price+macro ablation (Milestone 3). |
| `news_sector_baseline.py` | Four-way ablation: price / price+macro / price+news / price+macro+news (Milestone 4). |
| `backtest.py` | Transaction-cost-aware L/S backtest (Milestone 5). |
| `registry.py` | Experiment registry: read/write `ExperimentRecord` to `memory/experiments/` (Milestone 6). |

---

## Backtest layer (`mosaic_alpha/research/backtest.py`)

The backtest engine takes a pooled walk-forward prediction DataFrame and simulates a dollar-neutral L/S sector portfolio:

1. **Signal**: model scores from the `price+macro+news` walk-forward predictions.
2. **Construction**: on each rebalance date, the top `quantile` fraction of tickers by score are longed (equal weight, sum = +1); the bottom `quantile` are shorted (equal weight, sum = -1).
3. **Turnover**: one-way turnover is computed as `sum(|w_t - w_{t-1}|) / 2`. The first period has turnover 1.0.
4. **Cost**: `cost_t = (cost_bps / 10_000) × turnover_t`.
5. **Returns**: gross and net period log-returns are accumulated. Performance metrics (Sharpe, max drawdown, hit rate) are annualised.

---

## Registry and memory layer (`mosaic_alpha/research/registry.py`)

Each experiment run can be persisted to a timestamped folder:

```
memory/experiments/
  YYYYMMDD_HHMMSS_run_baseline/
    metadata.json    # ExperimentRecord fields (name, dates, universe, sources, …)
    metrics.json     # Numeric performance summary
    report.md        # Copy of the Markdown report
```

The registry is human-readable, Git-friendly (JSON diffs are meaningful), and independent of any database. `list_experiments()` and `show_experiment()` provide programmatic access. The CLI commands `list-experiments` and `show-experiment` expose these to the terminal.

---

## Graph layer (`mosaic_alpha/graph/`)

| Module | Responsibility |
|---|---|
| `schema.py` | `GraphNode`, `GraphEdge`, `ResearchGraph` dataclasses with validated vocabularies. |
| `builder.py` | `build_graph()` reads all `ExperimentRecord`s and materialises dataset / feature / model / experiment / metric / report / limitation nodes and produces / uses / has_metric / reports / has_limitation edges. Shared nodes (e.g., the `yfinance` dataset) are deduplicated. |
| `export.py` | `export_json()` (no dependencies) and `export_graphml()` (via networkx). |
| `queries.py` | `find_experiments_using_dataset()`, `find_features_used_by_experiment()`, `find_best_experiments_by_metric()`, `find_experiments_with_limitation()`. |

The graph captures research lineage: which datasets fed which experiments, which features were used, and where limitations appear across runs. It is exported to `memory/research_graph.json` for use by the dashboard and external tools.

---

## Dashboard layer (`mosaic_alpha/dashboard/`)

| Module | Responsibility |
|---|---|
| `helpers.py` | Pure-Python data-loading functions with no Streamlit dependency. All file I/O is here so it can be unit-tested without a browser. |
| `app.py` | Streamlit page renderer. Five pages: Overview, Experiments, Backtest, Research Graph, Reports. Reads entirely from the local file system; no network calls. |

The dashboard is launched via `mosaic dashboard`, which invokes `streamlit run` in a subprocess. The working directory is wherever the user runs the command, so relative paths to `memory/` and `reports/` resolve correctly.

---

## CLI (`mosaic_alpha/cli.py`)

Built with [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/). Each command:

1. Configures logging.
2. Imports the relevant module lazily (only when that command is invoked).
3. Runs the computation.
4. Prints a Rich table to the terminal.
5. Writes the Markdown report.
6. Optionally saves an `ExperimentRecord` to the registry (`--save-memory` / `--no-save-memory`).
