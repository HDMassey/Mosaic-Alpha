# Roadmap

This document describes planned future work for MosaicAlpha. Items are grouped by theme and ordered approximately by implementation dependency, not by priority.

---

## Data sources

### SEC filing features (Milestone 9+)

Integrate structured features derived from SEC EDGAR filings:

- Earnings surprise (reported EPS vs. consensus)
- Revenue growth YoY from 10-Q/10-K
- Accruals ratio as a quality signal
- Short interest from FINRA data

The EDGAR full-text search API and XBRL financial data are freely available. The primary challenge is point-in-time correctness: features must use only the filing data that was publicly available on the signal date, which requires tracking filing and amendment dates carefully.

### NOAA / weather features

Weather data from NOAA can be used to construct sector-relevant features for energy, utilities, and consumer staples:

- Heating degree days (relevant for XLE, XLU)
- Cooling degree days
- Precipitation anomalies

NOAA provides free API access to climate data. These features are expected to have small and regime-dependent marginal IC, but they extend the alternative-data scope of the project.

### Additional macro series

Extend the FRED macro feature set:

- Credit spreads (ICE BofA High Yield OAS)
- Consumer sentiment (University of Michigan)
- Housing starts
- ISM Manufacturing PMI

These are all available via FRED and would require only extending `configs/macro.yaml`.

---

## Feature engineering

### Sentiment-based news features

Replace raw GDELT article count with a sentiment score:

- Use a pre-trained financial sentiment model (FinBERT or similar) to score GDELT article headlines.
- Aggregate daily sentiment by sector keyword bucket.
- Compare marginal IC of sentiment vs. volume-based news features.

This requires processing GDELT GKG headline text rather than just count metadata.

### Factor exposure features

Add standard cross-sectional equity factor exposures:

- Size (market cap)
- Value (P/B, P/E)
- Momentum (already partially covered by price features)
- Quality (ROE, gross margin stability)

These require price and fundamental data aligned per ticker.

---

## Models

### Stronger model comparison

The current model is ridge regression. Add a systematic comparison:

- Lasso (sparse feature selection)
- Elastic net
- Gradient boosted trees (LightGBM)
- A simple neural network baseline

All models would be evaluated on the same walk-forward splits with the same IC metrics to ensure a fair comparison.

### Ensemble methods

Combine predictions from multiple models trained on different feature sets. Simple equal-weight or IC-weighted averaging of model scores before ranking.

---

## Portfolio construction

### Risk-adjusted weighting

Replace equal-weight L/S legs with weights derived from signal confidence:

- IC-weighted: higher weight to predictions with higher historical IC in recent folds
- Volatility-scaled: normalise position sizes by recent realised volatility

### Constraints

Add basic portfolio constraints:

- Maximum single-position weight
- Minimum number of holdings per leg
- Sector neutrality constraint (relevant when universe is expanded beyond sector ETFs)

### Expanded universe

Extend beyond 11 SPDR sector ETFs to individual large-cap equities (S&P 500 components). This meaningfully increases cross-sectional rank dispersion and makes IC estimates more statistically reliable.

---

## Infrastructure

### Real GDELT rate-limit-aware collection

The current GDELT fetcher uses a configurable sleep between requests to avoid HTTP 429 errors. Replace this with a proper retry-with-backoff strategy:

- Exponential backoff on 429 responses
- Persistent queue so interrupted collection can be resumed
- Per-session rate limit tracking

### Point-in-time FRED data

FRED series are subject to revision. The current implementation fetches the latest vintage, which introduces a look-ahead bias for series that are revised significantly. Address this by:

- Using FRED's vintage date API to fetch data as-of a specific date
- Documenting the publication lag for each series in `configs/macro.yaml`

### FRED offline-sample mode

Add an offline-sample mode for macro features analogous to the GDELT offline mode, so the entire pipeline including macro features can run without any network calls or API keys.

---

## Research tooling

### Optional LLM research reviewer

Add an optional LLM-assisted experiment reviewer:

- Given an `ExperimentRecord`, generate a structured critique highlighting potential issues (lookahead risk, statistical concerns, limitation caveats)
- Use the `anthropic` SDK; disable by default so no API key is required for normal use
- Output a review section appended to the Markdown report

This fits naturally into the existing registry and report infrastructure.

### Knowledge graph visualisation

Export the research graph in a format suitable for visual exploration:

- Static HTML using a JavaScript graph library (D3 or vis.js) that can be opened in a browser without a server
- Optionally: a dedicated Streamlit graph page with interactive node selection

### Dashboard screenshots and hosted documentation

Generate static screenshots of the dashboard for inclusion in the README and GitHub Pages documentation. This makes the project immediately legible to viewers who have not installed it.

---

## Validation

### Holdout period

Designate the most recent year of data as a strict hold-out period that is never used during development. Evaluate all models on this period only after the research phase is complete. Document the hold-out protocol formally.

### Monte Carlo permutation tests

Replace t-statistic significance tests with permutation-based p-values. Randomly shuffle the cross-sectional label ranking on each date and compute the null IC distribution. This is more robust than the Gaussian t-test for small universes.

### Multiple testing correction

Apply Bonferroni or Benjamini-Hochberg correction when comparing IC across multiple feature combinations and horizons simultaneously.
