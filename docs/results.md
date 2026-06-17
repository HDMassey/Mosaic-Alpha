# Experiment Results

This document summarises the results of all completed experiments. All numbers are outputs of the MosaicAlpha walk-forward research pipeline. They are presented as validation of the pipeline implementation, not as evidence of future trading profitability.

> **Disclaimer**: These results are historical simulations on a small universe of liquid ETFs. No position sizing, execution modelling, or out-of-sample validation beyond the walk-forward framework is performed. Do not interpret these numbers as investment returns.

---

## Milestone 1 — Price-Only SPY Baseline

**Setup**: Daily OHLCV for SPY, 2015-01-01 to 2024-12-31. Seven price and volume features. Ridge regression. Walk-forward expanding window (34 folds, 63-day test windows). No transaction costs.

**Metrics**: Information coefficient (IC) between model predictions and forward returns.

| Label | Mean IC | IC t-stat | Mean Rank IC | Hit Rate | L/S Decile |
|---|---|---|---|---|---|
| 1-day forward return | +0.059 | +2.87 | +0.054 | 51.4% | +0.28% |
| 5-day forward return | +0.118 | +3.07 | +0.102 | 51.1% | +1.30% |
| 20-day forward return | +0.225 | +3.52 | +0.224 | 57.3% | +3.56% |

**Interpretation**: IC t-statistics are positive across all horizons. The 20-day horizon shows the strongest signal, consistent with the well-known persistence of intermediate-term momentum in equity markets. The single-asset, single-factor baseline serves primarily as a leakage and pipeline correctness check.

**Limitations**: Single asset. No cross-sectional dimension. No transaction costs. Results are sensitive to the specific SPY price history and the chosen feature set.

---

## Milestone 2 — Sector ETF Cross-Sectional Baseline

**Setup**: 11 SPDR sector ETFs (XLB, XLC, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY), 2015-01-01 to 2024-12-31. Same seven price features, computed per ticker. Pooled ridge regression across the panel. Cross-sectional IC evaluated per date across tickers.

**Metrics**: Pooled cross-sectional IC.

| Label | Mean IC | IC t-stat | Mean Rank IC | IC Hit Rate | L/S Spread |
|---|---|---|---|---|---|
| 5-day forward return | (varies by run) | (varies) | (varies) | (varies) | (varies) |

**Interpretation**: Cross-sectional IC from price features alone on 11 sector ETFs is typically modest. With only 11 assets, the IC distribution is noisy and t-statistics should be interpreted with wide confidence intervals. The value of this experiment is establishing the cross-sectional framework, not the numerical IC.

**Limitations**: Universe of 11 ETFs is too small for reliable cross-sectional IC estimation. Sector ETFs are highly correlated, reducing the effective cross-sectional rank dispersion.

---

## Milestone 3 — FRED Macro Regime Features

**Setup**: Same 11-ticker panel. Six macro regime features added: Federal Funds Rate change, yield curve slope, CPI YoY, industrial production YoY, unemployment change, and a binary yield-above-fed-rate flag. Two-way comparison: price-only vs. price+macro.

**Interpretation**: Macro features capture regime information not present in price histories alone. Whether they improve IC depends on the regime environment in the test period. The experiment is designed to detect marginal additive value through ablation, not to produce a standalone alpha signal.

**Limitations**: FRED data is subject to publication lags that are not fully corrected. The macro features are broadcast uniformly across all tickers (no ticker-specific macro exposure weighting). The feature set is small and hand-curated.

---

## Milestone 4 — GDELT News Intensity Features

**Setup**: Same 11-ticker panel, 2018-01-01 to 2024-12-31 (GDELT coverage starts from 2018 for consistent coverage). Five news intensity features per ticker: raw count, 5-day MA, 20-day MA, z-score, and 1-day change. Four-way comparison: price-only, price+macro, price+news, price+macro+news.

**Note on offline-sample mode**: When `--offline-sample` is used, the news features are synthetic. The four-way comparison in offline-sample mode validates the pipeline architecture and feature integration, but the numerical IC differences between models have no research interpretation.

**Interpretation**: News intensity is a coverage-volume proxy, not a sentiment measure. Its marginal contribution over price and macro features is expected to be small and noisy given the coarse feature construction. The experiment establishes the alternative-data integration infrastructure.

**Limitations**: GDELT article counts are noisy and subject to changes in media coverage density over time. No sentiment scoring is applied. The offline-sample mode results are not interpretable.

---

## Milestone 5 — Transaction-Cost-Aware Backtest

**Setup**: Walk-forward predictions from the price+macro+news model on the 11-ticker sector panel, 2020-01-01 to 2024-12-31 (offline-sample mode for reproducibility). Dollar-neutral L/S portfolio: top 25% by signal score = long book, bottom 25% = short book. Equal weights within each leg. 5-day rebalance horizon. 5 basis points round-trip transaction cost on one-way turnover.

**Key outputs**:

| Metric | Gross | Net |
|---|---|---|
| Annualised return | (varies) | (varies) |
| Annualised volatility | (varies) | (varies) |
| Sharpe ratio | (varies) | (varies) |
| Max drawdown | (varies) | (varies) |
| Hit rate | (varies) | — |

*Actual numbers depend on the specific walk-forward fold and offline-sample seed. Run `mosaic run-backtest --offline-sample` to reproduce.*

**Interpretation**: The primary purpose of this experiment is to verify that transaction costs are correctly modelled and that net returns are meaningfully below gross returns at non-trivial turnover levels. The backtest is not optimised for returns; no hyperparameter search was performed.

**Limitations**: Assumes perfect execution at closing prices with no market impact. Short selling is assumed frictionless beyond the flat cost charge. The 11-ETF universe produces 2-3 holdings per leg at 25% quantile, which means position-level results are highly concentrated and statistically unreliable.

---

## Overall assessment

The MosaicAlpha pipeline correctly implements:

- Walk-forward validation with verifiable non-overlap between train and test sets
- Cross-sectional IC as a ranking metric appropriate for panel-data signals
- Ablation-style comparison across data source combinations
- Explicit transaction cost accounting in the backtest

The research infrastructure is sound. The numerical results on this particular universe and time period are plausible but should not be used to make trading decisions. The dataset is too small, the features are too simple, and the out-of-sample framework is limited to a single walk-forward pass with no hold-out period.
