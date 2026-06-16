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

## Initial MVP

The first milestone is a price-only baseline:

- load daily price data
- build returns, volatility, momentum, and volume z-score features
- create forward return labels
- run walk-forward validation
- train a ridge regression baseline
- report IC, rank IC, hit rate, MSE, and simple long/short metrics

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
