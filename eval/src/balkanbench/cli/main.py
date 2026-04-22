"""Root typer app for balkanbench."""
from __future__ import annotations

import typer

from balkanbench import __version__
from balkanbench.cli import validate as validate_cmds

app = typer.Typer(
    name="balkanbench",
    help="BalkanBench: reproducible evaluation benchmark for BCMS language models.",
    add_completion=False,
    no_args_is_help=True,
)

app.command("validate-env", help="Check Python + deps + env vars.")(validate_cmds.validate_env)
app.command("validate-config", help="Validate a YAML config against a JSON Schema.")(
    validate_cmds.validate_config
)
app.command("validate-data", help="Validate a dataset manifest JSON.")(
    validate_cmds.validate_data
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
