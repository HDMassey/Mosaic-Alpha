import typer

app = typer.Typer(
    help="MosaicAlpha quantitative research CLI.",
    no_args_is_help=True,
)


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
    ticker: str = typer.Option("SPY", help="Ticker to analyze."),
    start: str = typer.Option("2015-01-01", help="Start date."),
    end: str = typer.Option("2024-12-31", help="End date."),
) -> None:
    """Placeholder for the first price-only baseline experiment."""
    typer.echo(f"Baseline placeholder: ticker={ticker}, start={start}, end={end}")
    typer.echo("Next step: Claude Code will implement the research pipeline.")


if __name__ == "__main__":
    app()
