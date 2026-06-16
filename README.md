# MosaicAlpha

MosaicAlpha is a local-first alternative-data research workbench for discovering, validating, and explaining quantitative trading signals from noisy public datasets.

It combines:

- public data ingestion
- feature engineering
- walk-forward validation
- leakage-aware backtesting
- alternative-data signal research
- research memory
- knowledge-graph-based signal lineage

The goal is not to build a black-box trading bot. The goal is to demonstrate a rigorous quantitative research workflow: data provenance, statistical validation, transaction-cost-aware backtesting, ablation, and clear communication.

## Milestone 1 — Price-Only Baseline

### What the baseline does

The first milestone establishes a price-only research pipeline as a reproducible foundation before any alternative data is introduced.

**Data.** Daily OHLCV prices are downloaded via yfinance and cached locally as parquet files.

**Features.** Seven features are computed from price and volume history, all constructed strictly from information available at the close of day *t* with no forward-looking inputs:

| Feature | Description |
|---|---|
| `log_return` | Log return: log(close_t / close_{t-1}) |
| `rolling_vol_5` | 5-day realised volatility (annualised) |
| `rolling_vol_20` | 20-day realised volatility (annualised) |
| `momentum_5` | Cumulative log return over the prior 5 trading days |
| `momentum_20` | Cumulative log return over the prior 20 trading days |
| `momentum_60` | Cumulative log return over the prior 60 trading days |
| `volume_zscore_20` | Volume z-score relative to 20-day rolling mean and std |

**Labels.** Forward log-return labels are constructed for 1-day, 5-day, and 20-day horizons.

**Validation.** Walk-forward expanding-window splits are used: training sets grow from a minimum of one year; each test fold covers approximately one quarter (63 trading days). No data from the test fold is ever visible during training.

**Model.** A ridge regression with standard scaling is fit independently on each training fold and evaluated on the held-out test fold.

**Metrics.** Each fold reports IC (Pearson correlation), rank IC (Spearman correlation), hit rate (directional accuracy), MSE, and a simple long/short decile spread.

### SPY walk-forward results (2015-01-01 to 2024-12-31, 34 folds)

| Label | Mean IC | IC t-stat | Mean Rank IC | Hit Rate | L/S Decile |
|---|---|---|---|---|---|
| 1-day forward return  | +0.059 | +2.87 | +0.054 | 51.4% | +0.28% |
| 5-day forward return  | +0.118 | +3.07 | +0.102 | 51.1% | +1.30% |
| 20-day forward return | +0.225 | +3.52 | +0.224 | 57.3% | +3.56% |

Positive IC t-statistics across all horizons indicate that the price features carry statistically detectable signal. Predictive strength increases with horizon, consistent with the well-documented persistence of momentum over intermediate holding periods.

> **Note.** These results are a validation of the research pipeline, not a trading claim. A single asset, no transaction costs, no slippage, and no position sizing are modelled. The purpose is to confirm that the walk-forward harness is implemented correctly and that features are free of lookahead leakage before any alternative data is layered on top.

### Tests

14/14 tests pass, covering:

- Timestamp alignment between features, labels, and the source price index.
- Correctness of forward-return label offsets at every row.
- No lookahead leakage in log returns, rolling statistics, or momentum features.
- Strict disjointness and temporal ordering of train and test sets across all folds.

Run the test suite with:

    pytest

## Project structure

- `mosaic_alpha/data`: data connectors and cache logic
- `mosaic_alpha/features`: feature engineering
- `mosaic_alpha/research`: labels, models, validation, backtests, metrics
- `mosaic_alpha/graph`: research knowledge graph
- `mosaic_alpha/llm`: optional LLM-assisted research review
- `mosaic_alpha/dashboard`: Streamlit dashboard

## Development

Create and activate the environment:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install the project:

    pip install -e .

Run tests:

    pytest

Run the CLI smoke test:

    mosaic hello
