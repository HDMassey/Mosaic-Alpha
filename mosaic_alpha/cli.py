import logging
import sys
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
    save_memory: bool = typer.Option(
        True,
        "--save-memory/--no-save-memory",
        help="Save this experiment to the local registry (memory/experiments/).",
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

    if save_memory:
        _save_baseline_experiment(
            result=result,
            ticker=ticker,
            start=start,
            end=end,
            ridge_alpha=ridge_alpha,
            report_path=report_path,
        )


def _save_baseline_experiment(result, *, ticker, start, end, ridge_alpha, report_path) -> None:
    """Build and persist an ExperimentRecord for run-baseline."""
    from mosaic_alpha.research.registry import build_record, save_experiment

    metrics_summary = {}
    if result.aggregate:
        agg = result.aggregate[0]
        metrics_summary = {
            "label": agg.label,
            "mean_ic": round(agg.mean_ic, 6),
            "ic_t_stat": round(agg.ic_t_stat, 4),
            "mean_rank_ic": round(agg.mean_rank_ic, 6),
            "mean_hit_rate": round(agg.mean_hit_rate, 4),
            "mean_ls_decile": round(agg.mean_ls_decile, 6),
            "n_folds": result.n_folds,
            "n_rows": result.n_rows,
        }

    command = (
        f"mosaic run-baseline --ticker {ticker} --start {start} --end {end} "
        f"--ridge-alpha {ridge_alpha}"
    )
    record = build_record(
        name="run_baseline",
        command=command,
        start_date=start,
        end_date=end,
        universe=[ticker],
        data_sources=["yfinance"],
        feature_sets=["price"],
        model_type="ridge",
        validation_method="walk_forward",
        metrics_summary=metrics_summary,
        output_files=[str(report_path)],
        limitations=[
            "Single-ticker time-series only; no cross-sectional IC.",
            "Price features only; no macro or alternative data.",
        ],
    )
    exp_dir = save_experiment(record, report_text=Path(report_path).read_text(encoding="utf-8") if Path(report_path).exists() else "")
    console.print(f"[dim]Experiment saved to[/dim] {exp_dir}")


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
    save_memory: bool = typer.Option(
        True,
        "--save-memory/--no-save-memory",
        help="Save this experiment to the local registry (memory/experiments/).",
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

    if save_memory:
        _save_sector_experiment(
            result=result,
            start=start,
            end=end,
            ridge_alpha=ridge_alpha,
            universe_config=universe_config,
            report_path=report_path,
        )


def _save_sector_experiment(result, *, start, end, ridge_alpha, universe_config, report_path) -> None:
    from mosaic_alpha.research.registry import build_record, save_experiment

    metrics_summary: dict = {"labels": []}
    for cs in result.pooled:
        metrics_summary["labels"].append({
            "label": cs.label_col,
            "mean_ic": round(cs.mean_ic, 6),
            "ic_t_stat": round(cs.ic_t_stat, 4),
            "mean_rank_ic": round(cs.mean_rank_ic, 6),
            "ic_hit_rate": round(cs.ic_hit_rate, 4),
            "mean_ls_spread": round(cs.mean_ls_spread, 6),
        })
    metrics_summary["n_tickers"] = len(result.tickers)
    metrics_summary["n_folds"] = result.n_folds
    metrics_summary["n_panel_rows"] = result.n_panel_rows

    command = (
        f"mosaic run-sector-baseline --start {start} --end {end} "
        f"--ridge-alpha {ridge_alpha} --universe-config {universe_config}"
    )
    record = build_record(
        name="run_sector_baseline",
        command=command,
        start_date=start,
        end_date=end,
        universe=list(result.tickers),
        data_sources=["yfinance"],
        feature_sets=["price"],
        model_type="ridge",
        validation_method="walk_forward",
        metrics_summary=metrics_summary,
        output_files=[str(report_path)],
        limitations=[
            "Price features only; no macro or alternative data.",
            "Cross-sectional IC is weak with only 11 sector ETFs.",
        ],
    )
    exp_dir = save_experiment(record, report_text=Path(report_path).read_text(encoding="utf-8") if Path(report_path).exists() else "")
    console.print(f"[dim]Experiment saved to[/dim] {exp_dir}")


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
    save_memory: bool = typer.Option(
        True,
        "--save-memory/--no-save-memory",
        help="Save this experiment to the local registry (memory/experiments/).",
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

    if save_memory:
        _save_macro_experiment(
            result=result,
            start=start,
            end=end,
            ridge_alpha=ridge_alpha,
            universe_config=universe_config,
            macro_config=macro_config,
            report_path=report_path,
        )


def _save_macro_experiment(result, *, start, end, ridge_alpha, universe_config, macro_config, report_path) -> None:
    from mosaic_alpha.research.registry import build_record, save_experiment

    metrics_summary: dict = {"comparisons": [], "n_tickers": len(result.tickers), "n_folds": result.n_folds}
    for cmp in result.comparisons:
        metrics_summary["comparisons"].append({
            "label": cmp.label_col,
            "price_only_mean_ic": round(cmp.price_only.mean_ic, 6),
            "price_macro_mean_ic": round(cmp.price_macro.mean_ic, 6),
            "price_only_ic_t_stat": round(cmp.price_only.ic_t_stat, 4),
            "price_macro_ic_t_stat": round(cmp.price_macro.ic_t_stat, 4),
        })

    command = (
        f"mosaic run-macro-sector --start {start} --end {end} "
        f"--ridge-alpha {ridge_alpha} --universe-config {universe_config} "
        f"--macro-config {macro_config}"
    )
    record = build_record(
        name="run_macro_sector",
        command=command,
        start_date=start,
        end_date=end,
        universe=list(result.tickers),
        data_sources=["yfinance", "FRED"],
        feature_sets=["price", "macro"],
        model_type="ridge",
        validation_method="walk_forward",
        metrics_summary=metrics_summary,
        output_files=[str(report_path)],
        limitations=[
            "FRED data requires a valid API key.",
            "Cross-sectional IC is weak with only 11 sector ETFs.",
        ],
    )
    exp_dir = save_experiment(record, report_text=Path(report_path).read_text(encoding="utf-8") if Path(report_path).exists() else "")
    console.print(f"[dim]Experiment saved to[/dim] {exp_dir}")


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
    save_memory: bool = typer.Option(
        True,
        "--save-memory/--no-save-memory",
        help="Save this experiment to the local registry (memory/experiments/).",
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

    if save_memory:
        _save_news_experiment(
            result=result,
            start=start,
            end=end,
            ridge_alpha=ridge_alpha,
            universe_config=universe_config,
            macro_config=macro_config,
            gdelt_config=gdelt_config,
            offline_sample=offline_sample,
            report_path=report_path,
        )


def _save_news_experiment(result, *, start, end, ridge_alpha, universe_config, macro_config, gdelt_config, offline_sample, report_path) -> None:
    from mosaic_alpha.research.registry import build_record, save_experiment

    metrics_summary: dict = {"comparisons": [], "n_tickers": len(result.tickers), "n_folds": result.n_folds, "data_mode": result.data_mode}
    for cmp in result.comparisons:
        metrics_summary["comparisons"].append({
            "label": cmp.label_col,
            "price_only_mean_ic": round(cmp.price_only.mean_ic, 6),
            "price_macro_mean_ic": round(cmp.price_macro.mean_ic, 6),
            "price_news_mean_ic": round(cmp.price_news.mean_ic, 6),
            "price_macro_news_mean_ic": round(cmp.price_macro_news.mean_ic, 6),
        })

    data_sources = ["yfinance", "GDELT"]
    if not offline_sample:
        data_sources.append("FRED")

    limitations = [
        "Cross-sectional IC is weak with only 11 sector ETFs.",
        "News intensity is a simple count proxy, not sentiment.",
    ]
    if offline_sample:
        limitations.insert(0, "SAMPLE DATA: news features use synthetic counts, not real GDELT.")

    command = (
        f"mosaic run-news-sector --start {start} --end {end} "
        f"--ridge-alpha {ridge_alpha}"
        + (" --offline-sample" if offline_sample else "")
    )
    record = build_record(
        name="run_news_sector",
        command=command,
        start_date=start,
        end_date=end,
        universe=list(result.tickers),
        data_sources=data_sources,
        feature_sets=["price", "macro", "news"],
        model_type="ridge",
        validation_method="walk_forward",
        metrics_summary=metrics_summary,
        output_files=[str(report_path)],
        limitations=limitations,
    )
    exp_dir = save_experiment(record, report_text=Path(report_path).read_text(encoding="utf-8") if Path(report_path).exists() else "")
    console.print(f"[dim]Experiment saved to[/dim] {exp_dir}")


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
    save_memory: bool = typer.Option(
        True,
        "--save-memory/--no-save-memory",
        help="Save this experiment to the local registry (memory/experiments/).",
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

    if save_memory:
        _save_backtest_experiment(
            bt_result=bt_result,
            ns_result=ns_result,
            start=start,
            end=end,
            score_col=score_col,
            label_col=label_col,
            quantile=quantile,
            cost_bps=cost_bps,
            horizon=horizon,
            offline_sample=offline_sample,
            report_path=report_path,
        )


def _save_backtest_experiment(*, bt_result, ns_result, start, end, score_col, label_col, quantile, cost_bps, horizon, offline_sample, report_path) -> None:
    import math
    from mosaic_alpha.research.registry import build_record, save_experiment

    def _safe(v):
        return None if math.isnan(v) else round(v, 6)

    metrics_summary = {
        "score_col": score_col,
        "label_col": label_col,
        "quantile": quantile,
        "cost_bps": cost_bps,
        "horizon": horizon,
        "n_periods": bt_result.n_periods,
        "ann_gross_return": _safe(bt_result.ann_gross_return),
        "ann_net_return": _safe(bt_result.ann_net_return),
        "sharpe_gross": _safe(bt_result.sharpe_gross),
        "sharpe_net": _safe(bt_result.sharpe_net),
        "max_drawdown_gross": _safe(bt_result.max_drawdown_gross),
        "max_drawdown_net": _safe(bt_result.max_drawdown_net),
        "hit_rate": _safe(bt_result.hit_rate),
        "avg_turnover": _safe(bt_result.avg_turnover),
        "data_mode": ns_result.data_mode,
    }

    limitations = [
        "Assumes perfect fill at closing prices (no market impact).",
        "Short selling assumed frictionless beyond the flat cost_bps charge.",
        "Universe limited to 11 sector ETFs; Sharpe ratios have wide CIs.",
    ]
    if offline_sample:
        limitations.insert(0, "SAMPLE DATA: backtest uses synthetic news counts.")

    command = (
        f"mosaic run-backtest --start {start} --end {end} "
        f"--cost-bps {cost_bps} --quantile {quantile} --horizon {horizon}"
        + (" --offline-sample" if offline_sample else "")
    )
    record = build_record(
        name="run_backtest",
        command=command,
        start_date=start,
        end_date=end,
        universe=list(ns_result.tickers),
        data_sources=["yfinance", "GDELT", "FRED"],
        feature_sets=["price", "macro", "news"],
        model_type="ridge",
        validation_method="walk_forward",
        metrics_summary=metrics_summary,
        output_files=[str(report_path)],
        limitations=limitations,
    )
    exp_dir = save_experiment(record, report_text=Path(report_path).read_text(encoding="utf-8") if Path(report_path).exists() else "")
    console.print(f"[dim]Experiment saved to[/dim] {exp_dir}")


# ── Registry commands ──────────────────────────────────────────────────────────

@app.command("list-experiments")
def list_experiments_cmd(
    registry_root: Path = typer.Option(
        Path("memory/experiments"),
        "--registry-root",
        help="Root directory of the experiment registry.",
    ),
) -> None:
    """List all saved experiments in the local registry."""
    from mosaic_alpha.research.registry import list_experiments

    records = list_experiments(registry_root=registry_root)

    if not records:
        console.print("[yellow]No experiments found in[/yellow] " + str(registry_root))
        return

    table = Table(title=f"Experiments ({len(records)} found)", show_lines=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Created (UTC)", justify="right")
    table.add_column("Start", justify="right")
    table.add_column("End", justify="right")
    table.add_column("Key Metrics", justify="left")

    for rec in records:
        key_metrics = _format_key_metrics(rec.metrics_summary)
        created_short = rec.created_at[:19].replace("T", " ")  # "YYYY-MM-DD HH:MM:SS"
        table.add_row(
            rec.experiment_id,
            rec.name,
            created_short,
            rec.start_date,
            rec.end_date,
            key_metrics,
        )

    console.print(table)


def _format_key_metrics(metrics: dict) -> str:
    """Return a compact one-line summary of the most important metrics."""
    parts: list[str] = []

    # Backtest metrics
    if "sharpe_net" in metrics and metrics["sharpe_net"] is not None:
        parts.append(f"Sharpe(net)={metrics['sharpe_net']:+.2f}")
    if "ann_net_return" in metrics and metrics["ann_net_return"] is not None:
        parts.append(f"Ret(net)={metrics['ann_net_return']*100:+.1f}%")

    # IC metrics (sector baselines)
    if "labels" in metrics and metrics["labels"]:
        first = metrics["labels"][0]
        if "mean_ic" in first:
            parts.append(f"IC={first['mean_ic']:+.4f}")
        if "ic_t_stat" in first:
            parts.append(f"t={first['ic_t_stat']:+.2f}")

    # Comparison metrics (macro/news baselines)
    if "comparisons" in metrics and metrics["comparisons"]:
        first = metrics["comparisons"][0]
        for key in ("price_macro_news_mean_ic", "price_macro_mean_ic", "price_only_mean_ic"):
            if key in first:
                short = key.replace("_mean_ic", "").replace("price_", "")
                parts.append(f"IC({short})={first[key]:+.4f}")
                break

    # Simple baseline
    if "mean_ic" in metrics:
        parts.append(f"IC={metrics['mean_ic']:+.4f}")
    if "ic_t_stat" in metrics:
        parts.append(f"t={metrics['ic_t_stat']:+.2f}")

    return "  ".join(parts) if parts else "—"


@app.command("show-experiment")
def show_experiment_cmd(
    experiment_id: str = typer.Argument(help="Experiment ID to display (e.g. 20240601_120000_run_baseline)."),
    registry_root: Path = typer.Option(
        Path("memory/experiments"),
        "--registry-root",
        help="Root directory of the experiment registry.",
    ),
) -> None:
    """Show metadata and report path for a specific experiment."""
    from mosaic_alpha.research.registry import show_experiment

    rec = show_experiment(experiment_id, registry_root=registry_root)

    if rec is None:
        console.print(f"[red]Experiment not found:[/red] {experiment_id}")
        console.print(f"[dim]Registry root:[/dim] {registry_root}")
        raise SystemExit(1)

    exp_dir = Path(registry_root) / rec.experiment_id

    console.print(f"\n[bold cyan]Experiment:[/bold cyan] {rec.experiment_id}")
    console.print(f"  [bold]Name:[/bold]        {rec.name}")
    console.print(f"  [bold]Created:[/bold]     {rec.created_at}")
    console.print(f"  [bold]Command:[/bold]     {rec.command}")
    console.print(f"  [bold]Window:[/bold]      {rec.start_date} to {rec.end_date}")
    console.print(f"  [bold]Universe:[/bold]    {', '.join(rec.universe) if rec.universe else '—'}")
    console.print(f"  [bold]Data sources:[/bold] {', '.join(rec.data_sources) if rec.data_sources else '—'}")
    console.print(f"  [bold]Features:[/bold]    {', '.join(rec.feature_sets) if rec.feature_sets else '—'}")
    console.print(f"  [bold]Model:[/bold]       {rec.model_type}")
    console.print(f"  [bold]Validation:[/bold]  {rec.validation_method}")

    if rec.metrics_summary:
        console.print("\n  [bold]Metrics:[/bold]")
        _print_metrics(rec.metrics_summary, indent="    ")

    if rec.limitations:
        console.print("\n  [bold]Limitations:[/bold]")
        for lim in rec.limitations:
            console.print(f"    - {lim}")

    if rec.notes:
        console.print(f"\n  [bold]Notes:[/bold] {rec.notes}")

    report_md = exp_dir / "report.md"
    console.print(f"\n  [bold]Files:[/bold]")
    console.print(f"    metadata: {exp_dir / 'metadata.json'}")
    console.print(f"    metrics:  {exp_dir / 'metrics.json'}")
    if report_md.exists():
        console.print(f"    report:   {report_md}")
    console.print()


def _print_metrics(metrics: dict, indent: str = "") -> None:
    """Recursively print a metrics dict in a readable format."""
    for key, value in metrics.items():
        if isinstance(value, list):
            console.print(f"{indent}[dim]{key}:[/dim]")
            for item in value:
                if isinstance(item, dict):
                    parts = "  ".join(f"{k}={v}" for k, v in item.items())
                    console.print(f"{indent}  {parts}")
                else:
                    console.print(f"{indent}  {item}")
        elif isinstance(value, dict):
            console.print(f"{indent}[dim]{key}:[/dim]")
            _print_metrics(value, indent=indent + "  ")
        elif isinstance(value, float):
            console.print(f"{indent}[dim]{key}:[/dim] {value:+.6f}")
        else:
            console.print(f"{indent}[dim]{key}:[/dim] {value}")


# ── Graph commands ─────────────────────────────────────────────────────────────

@app.command("build-graph")
def build_graph_cmd(
    registry_root: Path = typer.Option(
        Path("memory/experiments"),
        "--registry-root",
        help="Root directory of the experiment registry.",
    ),
    json_path: Path = typer.Option(
        Path("memory/research_graph.json"),
        "--json-path",
        help="Output path for the JSON graph export.",
    ),
    graphml_path: Path = typer.Option(
        Path("memory/research_graph.graphml"),
        "--graphml-path",
        help="Output path for the GraphML export.",
    ),
    no_graphml: bool = typer.Option(
        False,
        "--no-graphml",
        help="Skip the GraphML export (JSON only).",
    ),
) -> None:
    """Build the research knowledge graph from saved experiments and export it.

    Reads all experiment records from the local registry, constructs a directed
    graph of datasets → experiments → features / models / metrics / reports /
    limitations, and writes it to JSON (always) and GraphML (if networkx is
    available and --no-graphml is not set).
    """
    from mosaic_alpha.graph.builder import build_graph
    from mosaic_alpha.graph.export import export_all, export_json

    records_root = Path(registry_root)
    console.print(f"[bold cyan]MosaicAlpha Build Graph[/bold cyan] | registry: {records_root}")

    graph = build_graph(registry_root=records_root)
    summary = graph.summary()

    console.print(
        f"[dim]Built graph: {summary['total_nodes']} nodes, "
        f"{summary['total_edges']} edges[/dim]"
    )

    if no_graphml:
        out = export_json(graph, json_path)
        console.print(f"[green]Graph JSON written to[/green] {out}")
    else:
        paths = export_all(graph, json_path=json_path, graphml_path=graphml_path)
        console.print(f"[green]Graph JSON written to[/green] {paths['json']}")
        if paths["graphml"]:
            console.print(f"[green]Graph GraphML written to[/green] {paths['graphml']}")
        else:
            console.print("[dim]GraphML skipped (networkx unavailable or error).[/dim]")

    # Print a summary table
    table = Table(title="Graph Summary", show_lines=True)
    table.add_column("Node / Edge Type", style="bold")
    table.add_column("Count", justify="right")

    for ntype, count in sorted(summary.get("nodes_by_type", {}).items()):
        table.add_row(f"node: {ntype}", str(count))
    for etype, count in sorted(summary.get("edges_by_type", {}).items()):
        table.add_row(f"edge: {etype}", str(count))

    console.print(table)


@app.command("graph-summary")
def graph_summary_cmd(
    json_path: Path = typer.Option(
        Path("memory/research_graph.json"),
        "--json-path",
        help="Path to the exported JSON graph.",
    ),
    registry_root: Path = typer.Option(
        Path("memory/experiments"),
        "--registry-root",
        help="Registry root (used to build graph on-the-fly if JSON not found).",
    ),
) -> None:
    """Print a summary of the research knowledge graph.

    Loads the graph from the JSON export (or builds it live from the registry
    if the JSON file does not exist yet).
    """
    import json as _json

    from mosaic_alpha.graph.builder import build_graph
    from mosaic_alpha.graph.schema import GraphEdge, GraphNode, ResearchGraph

    if Path(json_path).exists():
        data = _json.loads(Path(json_path).read_text(encoding="utf-8"))
        graph = ResearchGraph(
            nodes=[GraphNode(**n) for n in data.get("nodes", [])],
            edges=[GraphEdge(**e) for e in data.get("edges", [])],
        )
        console.print(f"[dim]Loaded graph from[/dim] {json_path}")
    else:
        console.print(f"[dim]{json_path} not found — building graph from registry...[/dim]")
        graph = build_graph(registry_root=registry_root)

    summary = graph.summary()

    table = Table(title="Research Knowledge Graph Summary", show_lines=True)
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")

    table.add_row("[bold]Total nodes[/bold]", str(summary["total_nodes"]))
    table.add_row("[bold]Total edges[/bold]", str(summary["total_edges"]))

    for ntype, count in sorted(summary.get("nodes_by_type", {}).items()):
        table.add_row(f"  node: {ntype}", str(count))
    for etype, count in sorted(summary.get("edges_by_type", {}).items()):
        table.add_row(f"  edge: {etype}", str(count))

    console.print(table)


@app.command("graph-query")
def graph_query_cmd(
    dataset: str = typer.Option(
        "",
        "--dataset",
        help="Find experiments that used this dataset (e.g. GDELT, FRED, yfinance).",
    ),
    metric: str = typer.Option(
        "",
        "--metric",
        help="Find experiments ranked by this metric (e.g. sharpe_net, mean_ic).",
    ),
    limitation: str = typer.Option(
        "",
        "--limitation",
        help="Find experiments whose limitations contain this keyword.",
    ),
    top_n: int = typer.Option(
        5,
        "--top-n",
        help="Maximum results to return for --metric queries.",
    ),
    registry_root: Path = typer.Option(
        Path("memory/experiments"),
        "--registry-root",
        help="Registry root (used to build graph on-the-fly).",
    ),
    json_path: Path = typer.Option(
        Path("memory/research_graph.json"),
        "--json-path",
        help="Pre-built JSON graph (used if it exists).",
    ),
) -> None:
    """Query the research knowledge graph.

    Exactly one of --dataset, --metric, or --limitation must be provided.

    Examples
    --------
        mosaic graph-query --dataset GDELT
        mosaic graph-query --metric sharpe_net
        mosaic graph-query --limitation sample
    """
    import json as _json

    from mosaic_alpha.graph.builder import build_graph
    from mosaic_alpha.graph.queries import (
        find_best_experiments_by_metric,
        find_experiments_using_dataset,
        find_experiments_with_limitation,
    )
    from mosaic_alpha.graph.schema import GraphEdge, GraphNode, ResearchGraph

    n_opts = sum(bool(x) for x in [dataset, metric, limitation])
    if n_opts == 0:
        console.print("[red]Provide exactly one of --dataset, --metric, or --limitation.[/red]")
        raise SystemExit(1)
    if n_opts > 1:
        console.print("[red]Only one of --dataset, --metric, --limitation may be used at a time.[/red]")
        raise SystemExit(1)

    # Load or build graph
    if Path(json_path).exists():
        data = _json.loads(Path(json_path).read_text(encoding="utf-8"))
        graph = ResearchGraph(
            nodes=[GraphNode(**n) for n in data.get("nodes", [])],
            edges=[GraphEdge(**e) for e in data.get("edges", [])],
        )
    else:
        console.print(f"[dim]{json_path} not found — building graph from registry...[/dim]")
        graph = build_graph(registry_root=registry_root)

    if dataset:
        results = find_experiments_using_dataset(graph, dataset)
        if not results:
            console.print(f"[yellow]No experiments found using dataset:[/yellow] {dataset!r}")
            return
        table = Table(title=f"Experiments using dataset: {dataset!r}", show_lines=True)
        table.add_column("Experiment ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Window")
        for r in results:
            table.add_row(r["experiment_id"], r["name"], f"{r['start_date']} → {r['end_date']}")
        console.print(table)

    elif metric:
        results = find_best_experiments_by_metric(graph, metric, top_n=top_n)
        if not results:
            console.print(f"[yellow]No metric found matching:[/yellow] {metric!r}")
            return
        table = Table(title=f"Top experiments by metric: {metric!r}", show_lines=True)
        table.add_column("Rank", justify="right")
        table.add_column("Experiment ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Metric", style="dim")
        table.add_column("Value", justify="right")
        for i, r in enumerate(results, 1):
            table.add_row(
                str(i),
                r["experiment_id"],
                r["name"],
                r["metric_label"],
                f"{r['value']:+.6f}",
            )
        console.print(table)

    elif limitation:
        results = find_experiments_with_limitation(graph, limitation)
        if not results:
            console.print(f"[yellow]No experiments found with limitation containing:[/yellow] {limitation!r}")
            return
        table = Table(title=f"Experiments with limitation containing: {limitation!r}", show_lines=True)
        table.add_column("Experiment ID", style="dim")
        table.add_column("Name", style="bold")
        table.add_column("Limitation")
        for r in results:
            table.add_row(r["experiment_id"], r["name"], r["limitation"])
        console.print(table)


# ── Dashboard command ─────────────────────────────────────────────────────────

@app.command("dashboard")
def dashboard_cmd(
    port: int = typer.Option(8501, "--port", help="Port for the Streamlit server."),
    host: str = typer.Option("localhost", "--host", help="Host address to bind to."),
    browser: bool = typer.Option(True, "--browser/--no-browser", help="Open browser automatically."),
) -> None:
    """Launch the MosaicAlpha research dashboard in a local Streamlit server.

    The dashboard reads from the local file system:

    \\b
      memory/experiments/        experiment registry
      memory/research_graph.json knowledge graph (run: mosaic build-graph)
      reports/generated/         CLI-generated Markdown reports

    Quick start:
        mosaic dashboard
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    app_path = Path(__file__).parent / "dashboard" / "app.py"
    if not app_path.exists():
        console.print(f"[red]Dashboard app not found at[/red] {app_path}")
        raise SystemExit(1)

    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "false" if browser else "true",
        "--theme.base", "light",
    ]

    console.print(
        f"[bold cyan]MosaicAlpha Dashboard[/bold cyan] "
        f"→ http://{host}:{port}  (Ctrl+C to stop)"
    )

    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard stopped.[/dim]")


if __name__ == "__main__":
    app()
