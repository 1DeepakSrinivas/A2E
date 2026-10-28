import typer

app = typer.Typer(help="A2E: tool-calling profiling harness")


@app.command()
def baseline(model: str, quant: str, device: str) -> None:
    """1 warm-up + 3 measured runs (implemented in Phase 5)."""
    raise typer.Exit("not implemented yet")


if __name__ == "__main__":
    app()
