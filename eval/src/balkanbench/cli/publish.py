"""``balkanbench publish-dataset`` subcommand."""

from __future__ import annotations

import typer

from balkanbench.data.publish import PublishError, publish_dataset


def _green(text: str) -> str:
    return typer.style(text, fg=typer.colors.GREEN, bold=True)


def _red(text: str) -> str:
    return typer.style(text, fg=typer.colors.RED, bold=True)


def publish_dataset_cmd(
    source_repo: str = typer.Option(
        "permitt/superglue",
        "--source-repo",
        help="Source HuggingFace dataset to download from.",
    ),
    public_repo: str = typer.Option(
        ...,
        "--public-repo",
        help="Destination public HF dataset repo (e.g. permitt/superglue-serbian).",
    ),
    private_repo: str | None = typer.Option(
        None,
        "--private-repo",
        help="Optional private HF repo holding the hidden test labels.",
    ),
    language: str = typer.Option(
        ...,
        "--language",
        help="BCMS language code (sr, hr, cnr, bs).",
    ),
    license: str = typer.Option(
        "CC-BY-4.0",
        "--license",
        help="SPDX-style license identifier for the published dataset.",
    ),
    dataset_revision: str = typer.Option(
        ...,
        "--dataset-revision",
        help="Tag recorded on the published dataset (e.g. v0.1.0-data).",
    ),
    configs: list[str] = typer.Option(
        ...,
        "--config",
        "-c",
        help="Repeatable: config name to publish (boolq, cb, copa, ...).",
    ),
    benchmark: str = typer.Option(
        "superglue",
        "--benchmark",
        help="Benchmark identifier recorded in the manifest and task_id column.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build + validate manifest and dataset card without pushing to HF.",
    ),
) -> None:
    """Publish a balkanbench dataset to Hugging Face.

    Normalises every config (renaming COPA ``dev`` → ``validation``), strips
    test labels, builds a manifest validated against ``dataset_manifest.json``,
    renders a dataset card, and pushes to ``--public-repo``. Requires
    ``HF_OFFICIAL_TOKEN`` in env.
    """
    try:
        report = publish_dataset(
            source_repo=source_repo,
            public_repo=public_repo,
            private_repo=private_repo,
            language=language,
            license=license,
            dataset_revision=dataset_revision,
            configs_to_publish=list(configs),
            benchmark=benchmark,
            dry_run=dry_run,
        )
    except PublishError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc

    if report.pushed:
        typer.echo(_green(f"Pushed {len(report.configs)} configs to {report.public_repo}"))
    else:
        typer.echo(_green(f"Dry run complete for {len(report.configs)} configs"))

    typer.echo("")
    typer.echo("Configs:")
    for name, details in report.manifest["configs"].items():
        splits = details["splits"]
        line = f"  {name}: " + ", ".join(
            f"{s}={info['num_rows']}" + ("*" if info["has_labels"] and s != "test" else "")
            for s, info in splits.items()
        )
        typer.echo(line)

    typer.echo("")
    typer.echo(f"License: {report.manifest['license']}")
    typer.echo(f"Revision: {report.manifest['dataset_revision']}")
    typer.echo("Sponsor: Recrewty")

    if dry_run:
        typer.echo("")
        typer.echo(_green("--- dataset card preview ---"))
        typer.echo(report.dataset_card[:500] + ("..." if len(report.dataset_card) > 500 else ""))
