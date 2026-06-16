import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    help="MosaicAlpha quantitative research CLI.",
    no_args_is_help=True,
)

console = Console()


@app.callback()
def main() -> None:
    """MosaicAlpha quantitative research CLI."""
    return None


@app.command()
def hello() -> None:
    """Verify that the MosaicAlpha CLI is installed correctly."""
    typer.echo("MosaicAlpha is ready.")


@app.command("run-baseline")
def run_baseline(
    ticker: str = typer.Option("SPY", help="Ticker symbol to analyse."),
    start: str = typer.Option("2015-01-01", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2024-12-31", help="End date (YYYY-MM-DD)."),
    ridge_alpha: float = typer.Option(1.0, help="Ridge regularisation strength."),
    report_path: Path = typer.Option(
        Path("reports/generated/baseline_spy.md"),
        help="Output path for the Markdown report.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run the price-only ridge-regression baseline and write a report."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from mosaic_alpha.research.baseline import render_report, run_baseline as _run

    console.print(
        f"[bold cyan]MosaicAlpha Baseline[/bold cyan] - {ticker} "
        f"[dim]{start} to {end}[/dim]"
    )

    result = _run(ticker=ticker, start=start, end=end, ridge_alpha=ridge_alpha)

    table = Table(title=f"Aggregate Metrics - {ticker}", show_lines=True)
    table.add_column("Label", style="bold")
    table.add_column("IC (mean+/-std)", justify="right")
    table.add_column("IC t-stat", justify="right")
    table.add_column("Rank IC", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("L/S Decile", justify="right")

    for agg in result.aggregate:
        table.add_row(
            agg.label,
            f"{agg.mean_ic:+.4f} +/- {agg.std_ic:.4f}",
            f"{agg.ic_t_stat:+.2f}",
            f"{agg.mean_rank_ic:+.4f} +/- {agg.std_rank_ic:.4f}",
            f"{agg.mean_hit_rate:.3f} +/- {agg.std_hit_rate:.3f}",
            f"{agg.mean_ls_decile:+.4f} +/- {agg.std_ls_decile:.4f}",
        )

    console.print(table)
    console.print(f"[dim]Folds: {result.n_folds} | Clean rows: {result.n_rows:,}[/dim]")

    render_report(result, report_path)
    console.print(f"[green]Report written to[/green] {report_path}")


@app.command("run-sector-baseline")
def run_sector_baseline(
    start: str = typer.Option("2015-01-01", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2024-12-31", help="End date (YYYY-MM-DD)."),
    ridge_alpha: float = typer.Option(1.0, help="Ridge regularisation strength."),
    universe_config: Path = typer.Option(
        Path("configs/universe.yaml"),
        help="Path to universe YAML config.",
    ),
    report_path: Path = typer.Option(
        Path("reports/generated/sector_baseline.md"),
        help="Output path for the Markdown report.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run the cross-sectional sector ETF baseline and write a report."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from mosaic_alpha.research.sector_baseline import (
        render_report,
        run_sector_baseline as _run,
    )

    console.print(
        f"[bold cyan]MosaicAlpha Sector Baseline[/bold cyan] "
        f"[dim]{start} to {end}[/dim]"
    )

    result = _run(
        start=start,
        end=end,
        universe_config=universe_config,
        ridge_alpha=ridge_alpha,
    )

    # ── Pooled metrics table ──────────────────────────────────────────────────
    table = Table(title="Pooled Cross-Sectional Metrics", show_lines=True)
    table.add_column("Label", style="bold")
    table.add_column("Mean IC", justify="right")
    table.add_column("IC t-stat", justify="right")
    table.add_column("Mean Rank IC", justify="right")
    table.add_column("IC Hit Rate", justify="right")
    table.add_column("Mean L/S Spread", justify="right")

    for cs in result.pooled:
        table.add_row(
            cs.label_col,
            f"{cs.mean_ic:+.4f}",
            f"{cs.ic_t_stat:+.2f}",
            f"{cs.mean_rank_ic:+.4f}",
            f"{cs.ic_hit_rate:.3f}",
            f"{cs.mean_ls_spread:+.4f}",
        )

    console.print(table)
    console.print(
        f"[dim]Universe: {len(result.tickers)} tickers | "
        f"Folds: {result.n_folds} | "
        f"Panel rows: {result.n_panel_rows:,}[/dim]"
    )

    render_report(result, report_path)
    console.print(f"[green]Report written to[/green] {report_path}")


@app.command("run-macro-sector")
def run_macro_sector(
    start: str = typer.Option("2015-01-01", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2024-12-31", help="End date (YYYY-MM-DD)."),
    ridge_alpha: float = typer.Option(1.0, help="Ridge regularisation strength."),
    universe_config: Path = typer.Option(
        Path("configs/universe.yaml"),
        help="Path to universe YAML config.",
    ),
    macro_config: Path = typer.Option(
        Path("configs/macro.yaml"),
        help="Path to macro YAML config.",
    ),
    report_path: Path = typer.Option(
        Path("reports/generated/macro_sector_baseline.md"),
        help="Output path for the Markdown report.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run price-only vs price+macro sector comparison and write a report."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from mosaic_alpha.research.macro_sector_baseline import (
        render_report,
        run_macro_sector_baseline as _run,
    )

    console.print(
        f"[bold cyan]MosaicAlpha Macro Sector Baseline[/bold cyan] "
        f"[dim]{start} to {end}[/dim]"
    )

    result = _run(
        start=start,
        end=end,
        universe_config=universe_config,
        macro_config=macro_config,
        ridge_alpha=ridge_alpha,
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    table = Table(title="Price-Only vs Price+Macro (Pooled CS-IC)", show_lines=True)
    table.add_column("Label", style="bold")
    table.add_column("Model", style="dim")
    table.add_column("Mean IC", justify="right")
    table.add_column("IC t-stat", justify="right")
    table.add_column("Rank IC", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("L/S Spread", justify="right")

    for cmp in result.comparisons:
        for label, cs in [("price-only", cmp.price_only), ("price+macro", cmp.price_macro)]:
            table.add_row(
                cmp.label_col if label == "price-only" else "",
                label,
                f"{cs.mean_ic:+.4f}",
                f"{cs.ic_t_stat:+.2f}",
                f"{cs.mean_rank_ic:+.4f}",
                f"{cs.ic_hit_rate:.3f}",
                f"{cs.mean_ls_spread:+.4f}",
            )

    console.print(table)
    console.print(
        f"[dim]Universe: {len(result.tickers)} tickers | "
        f"Macro series: {len(result.macro_series)} | "
        f"Folds: {result.n_folds} | "
        f"Panel rows: {result.n_panel_rows:,}[/dim]"
    )

    render_report(result, report_path)
    console.print(f"[green]Report written to[/green] {report_path}")


@app.command("run-news-sector")
def run_news_sector(
    start: str = typer.Option("2018-01-01", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2024-12-31", help="End date (YYYY-MM-DD)."),
    ridge_alpha: float = typer.Option(1.0, help="Ridge regularisation strength."),
    universe_config: Path = typer.Option(
        Path("configs/universe.yaml"),
        help="Path to universe YAML config.",
    ),
    macro_config: Path = typer.Option(
        Path("configs/macro.yaml"),
        help="Path to macro YAML config.",
    ),
    gdelt_config: Path = typer.Option(
        Path("configs/gdelt.yaml"),
        help="Path to GDELT sector keyword YAML config.",
    ),
    report_path: Path = typer.Option(
        Path("reports/generated/news_sector_baseline.md"),
        help="Output path for the Markdown report.",
    ),
    offline_sample: bool = typer.Option(
        False,
        "--offline-sample",
        help=(
            "Use deterministic synthetic news counts instead of live GDELT calls. "
            "Useful for pipeline testing or when GDELT is rate-limiting."
        ),
    ),
    sectors: str = typer.Option(
        "",
        "--sectors",
        help=(
            "Comma-separated subset of sector tickers to fetch news for, "
            "e.g. XLE,XLK. When empty, all tickers from gdelt_config are used."
        ),
    ),
    sleep_seconds: float = typer.Option(
        10.0,
        "--sleep-seconds",
        help="Seconds to sleep between live GDELT requests (rate-limit compliance).",
    ),
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Ignore the local GDELT cache and re-download from the API.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run 4-way price / macro / news / macro+news comparison and write a report.

    Use --offline-sample to run without live GDELT calls (safe for testing).
    Use --sectors XLE,XLK to limit which sectors are fetched (fewer API calls).
    Use --sleep-seconds 30 if you hit HTTP 429 rate limits.
    Use --force-refresh to bypass the local cache and re-download.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from mosaic_alpha.research.news_sector_baseline import (
        render_report,
        run_news_sector_baseline as _run,
    )

    sectors_list = [s.strip() for s in sectors.split(",") if s.strip()] or None

    mode_label = "[yellow]SAMPLE DATA[/yellow]" if offline_sample else "[green]live GDELT[/green]"
    console.print(
        f"[bold cyan]MosaicAlpha News Sector Baseline[/bold cyan] "
        f"[dim]{start} to {end}[/dim] | mode: {mode_label}"
    )

    if offline_sample:
        console.print(
            "[yellow]WARNING: --offline-sample is active. "
            "News features use synthetic data, not real GDELT counts.[/yellow]"
        )

    result = _run(
        start=start,
        end=end,
        universe_config=universe_config,
        macro_config=macro_config,
        gdelt_config=gdelt_config,
        ridge_alpha=ridge_alpha,
        gdelt_sleep_secs=sleep_seconds,
        offline_sample=offline_sample,
        sectors=sectors_list,
        force_refresh=force_refresh,
    )

    # ── 4-way comparison table ────────────────────────────────────────────────
    table = Table(title="4-Way Comparison (Pooled CS-IC)", show_lines=True)
    table.add_column("Label", style="bold")
    table.add_column("Model", style="dim")
    table.add_column("Mean IC", justify="right")
    table.add_column("IC t-stat", justify="right")
    table.add_column("Rank IC", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("L/S Spread", justify="right")

    model_keys = [
        ("price-only", "price_only"),
        ("price+macro", "price_macro"),
        ("price+news", "price_news"),
        ("price+macro+news", "price_macro_news"),
    ]

    for cmp in result.comparisons:
        for i, (model_name, attr) in enumerate(model_keys):
            cs = getattr(cmp, attr)
            table.add_row(
                cmp.label_col if i == 0 else "",
                model_name,
                f"{cs.mean_ic:+.4f}",
                f"{cs.ic_t_stat:+.2f}",
                f"{cs.mean_rank_ic:+.4f}",
                f"{cs.ic_hit_rate:.3f}",
                f"{cs.mean_ls_spread:+.4f}",
            )

    console.print(table)
    console.print(
        f"[dim]Universe: {len(result.tickers)} tickers | "
        f"News: {len(result.news_tickers)} tickers | "
        f"Folds: {result.n_folds} | "
        f"Panel rows: {result.n_panel_rows:,}[/dim]"
    )

    render_report(result, report_path)
    console.print(f"[green]Report written to[/green] {report_path}")


@app.command("run-backtest")
def run_backtest_cmd(
    experiment: str = typer.Option(
        "news-sector",
        "--experiment",
        help="Which experiment to run before backtesting. Only 'news-sector' is supported.",
    ),
    start: str = typer.Option("2020-01-01", help="Start date (YYYY-MM-DD)."),
    end: str = typer.Option("2024-12-31", help="End date (YYYY-MM-DD)."),
    ridge_alpha: float = typer.Option(1.0, help="Ridge regularisation strength."),
    universe_config: Path = typer.Option(
        Path("configs/universe.yaml"),
        help="Path to universe YAML config.",
    ),
    macro_config: Path = typer.Option(
        Path("configs/macro.yaml"),
        help="Path to macro YAML config.",
    ),
    gdelt_config: Path = typer.Option(
        Path("configs/gdelt.yaml"),
        help="Path to GDELT sector keyword YAML config.",
    ),
    offline_sample: bool = typer.Option(
        False,
        "--offline-sample",
        help="Use synthetic news counts instead of live GDELT (no network calls).",
    ),
    sectors: str = typer.Option(
        "",
        "--sectors",
        help="Comma-separated sector tickers to include, e.g. XLE,XLK.",
    ),
    sleep_seconds: float = typer.Option(
        10.0,
        "--sleep-seconds",
        help="Seconds between live GDELT requests.",
    ),
    force_refresh: bool = typer.Option(
        False,
        "--force-refresh",
        help="Ignore GDELT cache and re-download.",
    ),
    cost_bps: float = typer.Option(
        5.0,
        "--cost-bps",
        help="Round-trip transaction cost in basis points applied to one-way turnover.",
    ),
    quantile: float = typer.Option(
        0.25,
        "--quantile",
        help="Fraction of tickers in each leg of the L/S portfolio.",
    ),
    horizon: int = typer.Option(
        5,
        "--horizon",
        help="Rebalance horizon in trading days (should match the label horizon).",
    ),
    report_path: Path = typer.Option(
        Path("reports/generated/backtest_news_sector.md"),
        help="Output path for the backtest Markdown report.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Run the news-sector experiment then backtest the price+macro+news signal.

    Steps:
    1. Run the 4-way news-sector experiment to generate walk-forward predictions.
    2. Extract pooled price+macro+news predictions (pmn_fwd_ret_<horizon>).
    3. Build a dollar-neutral L/S sector portfolio with transaction costs.
    4. Write a performance report to --report-path.

    Quick start (no network needed):
        mosaic run-backtest --offline-sample --start 2020-01-01 --end 2024-12-31
    """
    if experiment != "news-sector":
        console.print(f"[red]Unknown experiment: {experiment!r}. Only 'news-sector' is supported.[/red]")
        raise SystemExit(1)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from mosaic_alpha.research.backtest import render_report as bt_render, run_backtest
    from mosaic_alpha.research.news_sector_baseline import run_news_sector_baseline

    sectors_list = [s.strip() for s in sectors.split(",") if s.strip()] or None
    mode_label = "[yellow]SAMPLE DATA[/yellow]" if offline_sample else "[green]live GDELT[/green]"

    console.print(
        f"[bold cyan]MosaicAlpha Backtest[/bold cyan] | {experiment} | "
        f"[dim]{start} to {end}[/dim] | mode: {mode_label}"
    )

    if offline_sample:
        console.print(
            "[yellow]WARNING: --offline-sample active. "
            "Backtest news signal is based on synthetic data.[/yellow]"
        )

    # ── Step 1: run the news-sector experiment ────────────────────────────────
    console.print("[dim]Running 4-way experiment to generate walk-forward predictions...[/dim]")
    ns_result = run_news_sector_baseline(
        start=start,
        end=end,
        universe_config=universe_config,
        macro_config=macro_config,
        gdelt_config=gdelt_config,
        ridge_alpha=ridge_alpha,
        gdelt_sleep_secs=sleep_seconds,
        offline_sample=offline_sample,
        sectors=sectors_list,
        force_refresh=force_refresh,
    )

    if ns_result.pooled_predictions.empty:
        console.print("[red]No pooled predictions returned from the experiment. Cannot run backtest.[/red]")
        raise SystemExit(1)

    # ── Step 2: backtest the pmn signal ───────────────────────────────────────
    score_col = f"pmn_fwd_ret_{horizon}"
    label_col = f"fwd_ret_{horizon}"

    if score_col not in ns_result.pooled_predictions.columns:
        available = [c for c in ns_result.pooled_predictions.columns if c.startswith("pmn_")]
        console.print(
            f"[red]Score column {score_col!r} not found. Available: {available}[/red]"
        )
        raise SystemExit(1)

    console.print(
        f"[dim]Running backtest: score={score_col}, label={label_col}, "
        f"quantile={quantile:.0%}, cost={cost_bps:.1f}bps...[/dim]"
    )
    bt_result = run_backtest(
        ns_result.pooled_predictions,
        score_col=score_col,
        label_col=label_col,
        quantile=quantile,
        cost_bps=cost_bps,
        horizon=horizon,
    )

    # ── Step 3: print summary table ───────────────────────────────────────────
    table = Table(title="Backtest Performance Summary", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Gross", justify="right")
    table.add_column("Net", justify="right")

    def _pct(v: float) -> str:
        import math
        return f"{v*100:+.2f}%" if not math.isnan(v) else "N/A"

    def _f2(v: float) -> str:
        import math
        return f"{v:+.2f}" if not math.isnan(v) else "N/A"

    table.add_row("Ann. Return", _pct(bt_result.ann_gross_return), _pct(bt_result.ann_net_return))
    table.add_row("Ann. Volatility", _pct(bt_result.ann_vol_gross), _pct(bt_result.ann_vol_net))
    table.add_row("Sharpe Ratio", _f2(bt_result.sharpe_gross), _f2(bt_result.sharpe_net))
    table.add_row("Max Drawdown", _pct(bt_result.max_drawdown_gross), _pct(bt_result.max_drawdown_net))
    table.add_row("Hit Rate", f"{bt_result.hit_rate:.3f}", "--")
    table.add_row("Avg Turnover", f"{bt_result.avg_turnover:.3f}", "--")
    table.add_row("Periods", str(bt_result.n_periods), "--")

    console.print(table)

    # ── Step 4: write report ──────────────────────────────────────────────────
    bt_render(bt_result, report_path, data_mode=ns_result.data_mode)
    console.print(f"[green]Backtest report written to[/green] {report_path}")


if __name__ == "__main__":
    app()
