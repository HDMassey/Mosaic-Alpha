# Reproducibility

MosaicAlpha is designed to be fully reproducible from a blank machine. This document explains the mechanisms that make that possible and the policies that keep the repository clean.

---

## Offline-sample mode

The most important reproducibility feature is **offline-sample mode**, available on any command that calls GDELT:

```bash
mosaic run-news-sector --offline-sample
mosaic run-backtest --offline-sample
```

When this flag is set:

- No HTTP calls are made to the GDELT GKG API.
- Synthetic news intensity counts are generated deterministically using a seeded pseudorandom number generator keyed on the ticker and date.
- The synthetic series has the same shape and column layout as real GDELT output.
- Every metric produced in this mode is labelled **SAMPLE DATA** in the terminal output, the Markdown report, and the experiment registry.

This means the entire pipeline — including the four-way ablation study and the transaction-cost-aware backtest — can be run on any machine without network access beyond Yahoo Finance and without a FRED API key.

> **Important**: results produced in offline-sample mode are not meaningful for drawing research conclusions. The mode exists for pipeline validation, CI, and demonstration only.

---

## Why generated outputs are excluded from Git

All files produced at runtime — price caches, model predictions, Markdown reports, experiment registry entries, and graph exports — are excluded from version control via `.gitignore`. This policy exists because:

1. **Size**: price Parquet files and GDELT caches can be tens of megabytes.
2. **Determinism**: any developer can regenerate them exactly by running the commands in this document.
3. **Cleanliness**: committing generated files obscures meaningful code diffs.
4. **Privacy**: experiment results may embed assumptions or data that should not be shared.

The directory skeleton (`.gitkeep` files) is tracked so the repository clones with the expected structure and no post-clone mkdir is required.

---

## Regenerating all reports

Run the commands below in order. Each writes a Markdown report to `reports/generated/` and saves an experiment record to `memory/experiments/`.

```bash
# 1. Price-only baseline
mosaic run-baseline --ticker SPY --start 2015-01-01 --end 2024-12-31

# 2. Sector ETF baseline
mosaic run-sector-baseline --start 2015-01-01 --end 2024-12-31

# 3. Macro comparison (requires FRED_API_KEY; see below)
mosaic run-macro-sector --start 2015-01-01 --end 2024-12-31

# 4. News comparison (offline: no GDELT calls)
mosaic run-news-sector --start 2020-01-01 --end 2024-12-31 --offline-sample

# 5. Backtest (offline)
mosaic run-backtest \
  --experiment news-sector \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --offline-sample \
  --cost-bps 5

# 6. Build and export the knowledge graph
mosaic build-graph
```

After these six commands, all pages of `mosaic dashboard` will have data to display.

---

## Obtaining a FRED API key

The FRED macro experiment (`mosaic run-macro-sector`) downloads series from the Federal Reserve Economic Data API. The key is free:

1. Register at [fred.stlouisfed.org](https://fred.stlouisfed.org).
2. Navigate to **My Account → API Keys**.
3. Request a key (instant approval).
4. Copy the key into your `.env` file:

```bash
# .env
FRED_API_KEY=your_key_here
```

The `.env` file is listed in `.gitignore` and is never committed. If you run `mosaic run-macro-sector` without a key, the command raises `FredApiKeyError` with a clear message explaining how to provide one.

---

## Caching policy

**Price data** (yfinance): downloaded once and cached as Parquet files under `data/prices/`. Subsequent runs use the cache unless the file is deleted. Use `--force-refresh` (where available) to re-download.

**GDELT data**: cached under `data/gdelt/` as JSON files keyed by ticker and date range. Use `--force-refresh` on any GDELT-dependent command to bypass the cache.

**FRED data**: not currently cached between runs. Each `run-macro-sector` call makes fresh API requests. This is intentional for correctness; FRED series can be revised.

---

## Environment checklist for a clean reproduction

```
✅  Python 3.11+ installed
✅  .venv created and activated
✅  pip install -e . completed
✅  mosaic hello prints "MosaicAlpha is ready."
✅  python -m pytest passes all 210 tests
✅  .env created (optional: FRED_API_KEY for live macro)
❌  .env is NOT committed to Git
❌  data/, reports/generated/, memory/experiments/* are NOT committed
```

---

## CI and testing without credentials

The test suite uses monkeypatching and offline-sample mode throughout so that no API keys are required to run the tests. The GDELT tests use synthetic data. The FRED tests are skipped or use a mock if no key is present.

Running `python -m pytest` from the repository root on a blank machine (no `.env`, no cached data) should produce 210 passing tests.
