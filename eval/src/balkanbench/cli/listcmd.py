"""``balkanbench list ...`` discovery commands."""

from __future__ import annotations

import os
from pathlib import Path

import typer

V01_LANGUAGES: tuple[str, ...] = ("sr",)
V01_ROADMAP_LANGUAGES: tuple[str, ...] = ("hr", "cnr", "bs")

list_app = typer.Typer(
    name="list",
    help="Discover configured benchmarks, tasks, models, languages.",
    no_args_is_help=True,
    add_completion=False,
)


def _configs_root() -> Path:
    override = os.environ.get("BALKANBENCH_CONFIGS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "configs"


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*.yaml"))


@list_app.command("benchmarks")
def list_benchmarks() -> None:
    """List known benchmarks."""
    root = _configs_root() / "benchmarks"
    if not root.is_dir():
        typer.echo("no benchmarks configured yet")
        return
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not names:
        typer.echo("no benchmarks configured yet")
        return
    for name in names:
        typer.echo(name)


@list_app.command("tasks")
def list_tasks() -> None:
    """List tasks across all benchmarks."""
    root = _configs_root() / "benchmarks"
    if not root.is_dir():
        typer.echo("no tasks configured yet")
        return
    task_ids: list[str] = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for task_yaml in _yaml_files(bench_dir / "tasks"):
            task_ids.append(f"{bench_dir.name}.{task_yaml.stem}")
    if not task_ids:
        typer.echo("no tasks configured yet")
        return
    for tid in task_ids:
        typer.echo(tid)


@list_app.command("models")
def list_models() -> None:
    """List model configs."""
    root = _configs_root() / "models"
    if not root.is_dir():
        typer.echo("no models configured yet")
        return
    files = _yaml_files(root)
    if not files:
        typer.echo("no models configured yet")
        return
    for f in files:
        tier = f.parent.name if f.parent.name in {"official", "experimental"} else "-"
        typer.echo(f"{f.stem}\t{tier}")


@list_app.command("languages")
def list_languages() -> None:
    """List languages in scope for v0.1 plus the roadmap."""
    for lang in V01_LANGUAGES:
        typer.echo(f"{lang}\tavailable")
    for lang in V01_ROADMAP_LANGUAGES:
        typer.echo(f"{lang}\troadmap")
