"""Root typer app for balkanbench."""

from __future__ import annotations

import typer

from balkanbench import __version__
from balkanbench.cli import validate as validate_cmds
from balkanbench.cli.eval import eval_cmd
from balkanbench.cli.hp_search import hp_search_cmd
from balkanbench.cli.leaderboard import leaderboard_app
from balkanbench.cli.listcmd import list_app
from balkanbench.cli.predict import predict_cmd
from balkanbench.cli.publish import publish_dataset_cmd
from balkanbench.cli.run import run_cmd
from balkanbench.cli.score import score_cmd
from balkanbench.cli.throughput import throughput_cmd

app = typer.Typer(
    name="balkanbench",
    help="BalkanBench: reproducible evaluation benchmark for BCMS language models.",
    add_completion=False,
    no_args_is_help=True,
)

app.add_typer(list_app, name="list")
app.add_typer(leaderboard_app, name="leaderboard")

app.command("validate-env", help="Check Python + deps + env vars.")(validate_cmds.validate_env)
app.command("validate-config", help="Validate a YAML config against a JSON Schema.")(
    validate_cmds.validate_config
)
app.command("validate-data", help="Validate a dataset manifest JSON.")(validate_cmds.validate_data)
app.command(
    "publish-dataset",
    help="Normalise a source HF dataset and publish the public BalkanBench variant.",
)(publish_dataset_cmd)
app.command("eval", help="Train + evaluate a model, emit a result artifact.")(eval_cmd)
app.command(
    "predict",
    help="Predict on the public test split; emit predictions.jsonl + run_metadata.json.",
)(predict_cmd)
app.command(
    "score",
    help="Score predictions.jsonl against private test labels (requires HF_OFFICIAL_TOKEN).",
)(score_cmd)
app.command(
    "hp-search",
    help="Optuna TPE search on train -> validation; writes the winning model YAML.",
)(hp_search_cmd)
app.command(
    "throughput",
    help="Measure inference throughput on reference hardware; writes per-task + aggregate JSON.",
)(throughput_cmd)
app.command(
    "run",
    help="End-to-end pipeline: HP search per ranked task -> multi-seed eval -> leaderboard export.",
)(run_cmd)


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
