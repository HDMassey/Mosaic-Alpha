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

    # ── Pretty-print summary table ────────────────────────────────────────────
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
    console.print(
        f"[dim]Folds: {result.n_folds} | Clean rows: {result.n_rows:,}[/dim]"
    )

    # ── Write report ──────────────────────────────────────────────────────────
    render_report(result, report_path)
    console.print(f"[green]Report written to[/green] {report_path}")


if __name__ == "__main__":
    app()
