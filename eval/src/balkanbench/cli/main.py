"""Root typer app for balkanbench."""
from __future__ import annotations

import typer

from balkanbench import __version__

app = typer.Typer(
    name="balkanbench",
    help="BalkanBench: reproducible evaluation benchmark for BCMS language models.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show balkanbench version and exit.",
    ),
) -> None:
    """BalkanBench CLI root."""


if __name__ == "__main__":
    app()
